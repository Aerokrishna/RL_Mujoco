"""Peg-in-hole comparison: spiral search with the scripted catch rule, FORGE and TacDiffusion.

All settings live in a YAML config (default `compare.yaml` next to this file); see its comments for every key.
Every combination of the condition lists (shape x clearance x hole error x tilt x friction) is one condition, and
every method runs the same reset seeds in every condition (common random numbers): socket poses, hole-error
directions and sizes, tilt directions and sizes and friction draws are identical across methods.

Common to all methods:
- 8 mm peg (diameter / side / across flats, `peg_half_size_mm`), 25 mm deep hole; non-round holes are turned to the
  gripper yaw (no yaw error).
- Hole-estimate error in xy (random direction); every method aims at the estimate.
- Tilt: the arm holds the peg (rigid in the grip) at the tilt angle to the socket normal (random horizontal axis). The
  tilted pose is every method's own nominal "down" orientation, so no method knows the socket is tilted relative to it.
- Measured per physics tick: success (peg tip within 2.5 mm of the axis and 1 mm of the floor; the episode stops at
  the first success, so a peg that is later pushed back out still counts), first contact (|F| > 1 N), contact force.
  F_max = max of the 50 ms moving average of the peg contact force (after a 3-tick median that removes one-tick
  collision glitches). Outcome of a failure: `jammed` (tip got into the hole) or `not_found`.

Method start: spiral_rule: tip 2 mm above the estimated opening; FORGE: its reset pose 3.7-5.7 cm above the hole;
TacDiffusion: the FORGE reset pose, then its scripted approach, guarded descent and diffusion phase (all within
`time_max`).

Usage (`conda activate mjtest`, from `mujoco_rl_bed/`):
    python projects/benchmark/compare.py                                     # everything in compare.yaml
    python projects/benchmark/compare.py config=my.yaml                      # another config
    python projects/benchmark/compare.py episodes=5 methods=[spiral_rule,forge] conditions.shapes=[hexagon]
    python projects/benchmark/compare.py conditions.clearance_mm=[0.1,0.2] conditions.hole_error_mm=["0-4"]
    python projects/benchmark/compare.py view.enabled=true view.method=forge conditions.shapes=[cuboid] episodes=5
    python projects/benchmark/compare.py out=projects/benchmark/results/compare_<time>   # resume an interrupted run

Overrides are dotted keys with YAML values. Writes `episodes.jsonl` (one line per episode, appended as jobs finish;
finished (method, condition) jobs are skipped on resume), `summary.md`, `summary.csv` and `config.yaml` into the
output folder.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
BED = PROJECTS.parent                           # .../mujoco_rl_bed (relative paths in the config resolve here)
NAMES = {"spiral_rule": "Spiral+rule", "forge": "FORGE", "tacdiff": "TacDiff"}
ORDER = {"tacdiff": 0, "forge": 1, "spiral_rule": 2}   # slowest first (load balance)
SHAPES = ("cylinder", "cuboid", "hexagon", "octagon")


# ---------------------------------------------------------------------------------- config
def load_config(argv: list[str]) -> dict:
    """Read the YAML config (`config=<path>`, default compare.yaml) and apply dotted key=value overrides.

    Args:
        argv: Command-line arguments.

    Returns:
        The config dict.
    """
    args = dict(a.split("=", 1) for a in argv)
    path = Path(args.pop("config", HERE / "compare.yaml"))
    cfg = yaml.safe_load(path.read_text())
    for key, raw in args.items():
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                raise KeyError(f"unknown key '{key}'")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown key '{key}' (valid here: {sorted(node)})")
        node[parts[-1]] = yaml.safe_load(raw)
    for m in cfg["methods"]:
        if m not in NAMES:
            raise ValueError(f"unknown method '{m}' (valid: {list(NAMES)})")
    for s in cfg["conditions"]["shapes"]:
        if s not in SHAPES:
            raise ValueError(f"unknown shape '{s}' (valid: {list(SHAPES)})")
    return cfg


def path_of(p: str) -> Path:
    """A config path: absolute, or relative to mujoco_rl_bed/."""
    q = Path(p)
    return q if q.is_absolute() else BED / q


def level(label) -> tuple[str, float, float]:
    """A condition level: "a" (exactly a) or "a-b" (U(a, b)) -> (label, lo, hi)."""
    s = str(label).strip()
    lo, _, hi = s.partition("-")
    return s, float(lo), float(hi or lo)


def build_conditions(cfg: dict) -> list[dict]:
    """Full factorial of the condition lists."""
    c = cfg["conditions"]
    out = []
    for shape in c["shapes"]:
        for clr in c["clearance_mm"]:
            for err in c["hole_error_mm"]:
                for tilt in c["tilt_deg"]:
                    for mu in c["friction"]:
                        e, t, f = level(err), level(tilt), level(mu)
                        out.append({"shape": shape, "clearance_mm": float(clr),
                                    "err": e[0], "err_lo": e[1], "err_hi": e[2],
                                    "tilt": t[0], "tilt_lo": t[1], "tilt_hi": t[2],
                                    "friction": f[0], "mu_lo": f[1], "mu_hi": f[2]})
    return out


# ---------------------------------------------------------------------------------- scene and shared draws
def part_assets(shape: str, clearance_mm: float, half: float) -> list:
    """Peg and hole assets for a shape and radial clearance (friction is set per episode)."""
    from mujoco_rl_bed.sim.assets import BoxPeg, HexHole, HexPeg, Peg, RectHole, RoundHole

    h = half + 1e-3 * clearance_mm
    if shape == "cylinder":
        return [Peg(name="peg", radius=half), RoundHole(name="hole", inner_radius=h)]
    if shape == "cuboid":
        return [BoxPeg(name="peg", half_x=half, half_y=half), RectHole(name="hole", inner_half_x=h, inner_half_y=h)]
    if shape == "hexagon":
        return [HexPeg(name="peg", apothem=half, n_sides=6), HexHole(name="hole", inner_radius=h)]
    return [HexPeg(name="peg", apothem=half, n_sides=8), RoundHole(name="hole", inner_radius=h, n_walls=8)]


def _register_events() -> None:
    """Register the benchmark reset events (once per process)."""
    from mujoco_rl_bed.env.managers.events import event_term

    if getattr(_register_events, "done", False):
        return

    @event_term("bench_hole_error")
    def bench_hole_error(ctx, lo: float = 0.0, hi: float = 0.0) -> None:
        """Reset: shift the hole estimate (anchor) by U(lo, hi) [m] in a random horizontal direction.

        Size fraction and direction come from `ctx.state["bench"]` (set per seed, shared by all methods). Must run
        after `forge_reset_fixed` and before the end-effector reset.
        """
        b = ctx.state["bench"]
        r = lo + (hi - lo) * b["err_u"]
        st = ctx.state["forge"]
        st.anchor[0] += r * math.cos(b["err_phi"])
        st.anchor[1] += r * math.sin(b["err_phi"])
        b["err_applied_mm"] = 1e3 * r

    @event_term("bench_friction")
    def bench_friction(ctx, lo: float = 0.5, hi: float = 0.5) -> None:
        """Reset: peg and socket sliding friction U(lo, hi), draw shared by all methods."""
        st = ctx.state["forge"]
        st.friction = lo + (hi - lo) * ctx.state["bench"]["mu_u"]
        ctx.model.geom_friction[st.part_geoms, 0] = st.friction

    @event_term("bench_arm_tilt")
    def bench_arm_tilt(ctx, tilt_lo: float = 0.0, tilt_hi: float = 0.0, fix_anchor: bool = True) -> None:
        """Reset: the arm holds the peg tilted against the socket plane by U(tilt_lo, tilt_hi) deg.

        The tilted orientation replaces the task's nominal "gripper down" orientation (`ForgeState.home_quat`), so
        every method takes it as its own reference. With `fix_anchor` the anchor is corrected so the peg tip, not the
        TCP, aims at the estimate (the robot knows its own kinematics). Must run after the hole-error event and
        before the end-effector reset.
        """
        import mujoco

        st = ctx.state["forge"]
        b = ctx.state["bench"]
        if "bench_home0" not in ctx.state:
            ctx.state["bench_home0"] = st.home_quat.copy()
        ang = math.radians(tilt_lo + (tilt_hi - tilt_lo) * b["tilt_u"])
        qt = np.array([1.0, 0.0, 0.0, 0.0])
        if ang > 0.0:
            mujoco.mju_axisAngle2Quat(qt, np.array([math.cos(b["tilt_phi"]), math.sin(b["tilt_phi"]), 0.0]), ang)
        mujoco.mju_mulQuat(st.home_quat, qt, ctx.state["bench_home0"])
        mujoco.mju_negQuat(st.home_quat_inv, st.home_quat)
        b["tilt_applied_deg"] = math.degrees(ang)
        if fix_anchor and ang > 0.0:
            Rt = np.zeros(9)
            mujoco.mju_quat2Mat(Rt, qt)
            st.anchor += st.tcp_to_tip * (Rt.reshape(3, 3)[:, 2] - np.array([0.0, 0.0, 1.0]))

    @event_term("bench_paper_error")
    def bench_paper_error(ctx, tol_xy: float = 0.002, tol_deg: float = 2.0, fix_anchor: bool = True) -> None:
        """Reset: TacDiffusion-paper hole-pose error (`mios/bricks.py::error_simulate`): U(+-tol_xy) in x, y and
        U(+-tol_deg) in roll, pitch, yaw of the hole frame the robot aims at (shared draws). Runs after the tilt."""
        import mujoco

        st = ctx.state["forge"]
        u = np.asarray(ctx.state["bench"]["paper_u"])
        dx, dy = tol_xy * (2.0 * u[:2] - 1.0)
        rpy = np.radians(tol_deg * (2.0 * u[2:5] - 1.0))
        yaw_h = float(ctx.state.get("hole_yaw", 0.0))
        qh = np.array([math.cos(0.5 * yaw_h), 0.0, 0.0, math.sin(0.5 * yaw_h)])
        qe, tmp = np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(4)
        for axis, ang in ((2, rpy[2]), (1, rpy[1]), (0, rpy[0])):
            qa = np.array([math.cos(0.5 * ang), 0.0, 0.0, 0.0])
            qa[1 + axis] = math.sin(0.5 * ang)
            mujoco.mju_mulQuat(tmp, qe, qa)
            qe[:] = tmp
        qhi, qw, base = np.zeros(4), np.zeros(4), st.home_quat.copy()
        mujoco.mju_negQuat(qhi, qh)
        mujoco.mju_mulQuat(tmp, qh, qe)
        mujoco.mju_mulQuat(qw, tmp, qhi)
        mujoco.mju_mulQuat(st.home_quat, qw, base)
        mujoco.mju_negQuat(st.home_quat_inv, st.home_quat)
        c, sn = math.cos(yaw_h), math.sin(yaw_h)
        st.anchor[0] += c * dx - sn * dy
        st.anchor[1] += sn * dx + c * dy
        if fix_anchor:
            Rb, Rn = np.zeros(9), np.zeros(9)
            mujoco.mju_quat2Mat(Rb, base)
            mujoco.mju_quat2Mat(Rn, st.home_quat)
            st.anchor -= st.tcp_to_tip * (Rn.reshape(3, 3)[:, 2] - Rb.reshape(3, 3)[:, 2])

    _register_events.done = True


def set_shared(env, seed: int) -> None:
    """Per-seed draws shared by all methods: hole-error size/direction, tilt size/direction, friction, paper error."""
    r = np.random.default_rng(seed + 31337)
    env.ctx.state["bench"] = {"tilt_phi": float(r.uniform(0.0, 2.0 * math.pi)), "tilt_u": float(r.uniform()),
                              "err_phi": float(r.uniform(0.0, 2.0 * math.pi)), "err_u": float(r.uniform()),
                              "mu_u": float(r.uniform()), "paper_u": r.uniform(size=5)}


def patch_task(task, cond: dict, cfg: dict, replace: str | None = None, fix_anchor: bool = True) -> None:
    """Parts, hole error, friction, yaw alignment, arm tilt (and paper error) on a forge-style task config.

    Args:
        task: `TaskCfg` (modified in place).
        cond: Condition.
        cfg: Benchmark config.
        replace: Event replaced by the tilt event (None: insert it before `reset_ee`).
        fix_anchor: Correct the anchor for the tilt (methods that aim the TCP through the anchor).
    """
    from mujoco_rl_bed.env.cfg import EventTermCfg as E

    _register_events()
    task.scene.assets = part_assets(cond["shape"], cond["clearance_mm"], 1e-3 * float(cfg["peg_half_size_mm"]))
    pe = cfg["paper_error"]
    tilt = [E(mode="reset", func="bench_arm_tilt",
              params={"tilt_lo": cond["tilt_lo"], "tilt_hi": cond["tilt_hi"], "fix_anchor": fix_anchor})]
    if pe["enabled"]:
        tilt.append(E(mode="reset", func="bench_paper_error", params={
            "tol_xy": 1e-3 * float(pe["tol_mm"]), "tol_deg": float(pe["tol_deg"]), "fix_anchor": fix_anchor}))
    events = {}
    for k, v in task.events.items():
        if k in ("tilt_ee", "shift_ee"):
            continue
        if k == "randomize_friction":
            events["friction"] = E(mode="reset", func="bench_friction", params={"lo": cond["mu_lo"], "hi": cond["mu_hi"]})
            continue
        if k == replace or (k == "reset_ee" and replace is None):
            events.update({f"bench_{i}": t for i, t in enumerate(tilt)})
            if k == replace:
                continue
        events[k] = v
        if k == "reset_fixed":
            v.params.update({"pos_noise_std": 0.0, "pos_noise_xy": 0.0})
            events["hole_error"] = E(mode="reset", func="bench_hole_error",
                                     params={"lo": 1e-3 * cond["err_lo"], "hi": 1e-3 * cond["err_hi"]})
        if k == "init" and cond["shape"] != "cylinder":
            events["align_hole_yaw"] = E(mode="startup", func="cr_align_hole_yaw")
    task.events = events


def _view_ov(cfg: dict) -> dict:
    v = cfg["view"]
    return {"render": str(bool(v["enabled"])).lower(), "realtime": str(bool(v["realtime"])).lower()}


# ---------------------------------------------------------------------------------- envs
def make_forge_env(cond: dict, cfg: dict):
    """FORGE env with the checkpoint's training settings plus the condition."""
    import compliant_RL  # noqa: F401  (cr_align_hole_yaw)
    from forge.task import NO_DR
    from mujoco_rl_bed.env.torque_env import TorqueEnv
    from mujoco_rl_bed.tasks.registry import make_env_cfg

    f = cfg["forge"]
    # Controller settings of the checkpoint's training run (config.json); a run without them (older defaults) gets
    # decimation 25, gains 600/100, no EMA prediction and `forge.ema`.
    run_cfg = path_of(cfg["models"]["forge_run"]) / "config.json"
    env_c = json.loads(run_cfg.read_text())["env"] if run_cfg.is_file() else {}
    act_c, task_c = env_c.get("task", {}).get("action", {}), env_c.get("task", {})
    rc = task_c.get("events", {}).get("randomize_controller", {}).get("params", {})
    kp = rc.get("kp_default") or task_c.get("controller", {}).get("kp") or [600.0, 600.0, 600.0, 100.0, 100.0, 100.0]
    ema = act_c.get("ema_factor") or f["ema"]
    ov = {**NO_DR, "decimation": str(env_c.get("decimation") or 25),
          "task.episode_length_s": str(float(cfg["time_max"]) + 1.0),
          "task.action.ema_prediction": str(bool(act_c.get("ema_prediction"))).lower(),
          "task.events.randomize_controller.params.ema_range": f"({ema},{ema})",
          "task.events.randomize_controller.params.kp_default": "(" + ",".join(str(float(k)) for k in kp) + ")",
          "task.events.randomize_controller.params.lam_default": str(rc.get("lam_default", 0.02)),
          "task.events.reset_threshold.params.lo": str(f["f_th"]),
          "task.events.reset_threshold.params.hi": str(f["f_th"]), **_view_ov(cfg)}
    ecfg = make_env_cfg("forge_peg", ov)
    patch_task(ecfg.task, cond, cfg)
    return TorqueEnv(ecfg)


def make_tacdiff_env(cond: dict, cfg: dict, td):
    """`forge_peg` env as in `tacdiff/run.py` (1 ms physics, no FORGE randomization) plus the condition."""
    import compliant_RL  # noqa: F401
    from forge.task import NO_DR
    from mujoco_rl_bed.env.torque_env import TorqueEnv
    from mujoco_rl_bed.tasks.registry import make_env_cfg

    ov = {**NO_DR, "sim_dt": str(td.SIM_DT), **_view_ov(cfg), "task.episode_length_s": str(float(cfg["time_max"]) + 15.0)}
    ecfg = make_env_cfg("forge_peg", ov)
    patch_task(ecfg.task, cond, cfg, fix_anchor=False)   # its approach aims the peg tip itself
    return TorqueEnv(ecfg)


def make_spiral_env(cond: dict, cfg: dict):
    """`compliant_peg_s2g` (spiral base + try-insert action) with the condition; [policy | critic] observations."""
    import compliant_RL  # noqa: F401
    from mujoco_rl_bed.env.torque_env import TorqueEnv
    from mujoco_rl_bed.tasks.registry import make_env_cfg

    s = cfg["spiral"]
    ecfg = make_env_cfg("compliant_peg_s2g", {"task.episode_length_s": str(float(cfg["time_max"]) + 1.0),
                                              "obs_mode": "asymmetric", **_view_ov(cfg)})
    ecfg.task.action.params.update({"f_search": float(s["f_search_n"]), "pitch": 1e-3 * float(s["pitch_mm"]),
                                    "r_max": 1e-3 * float(s["r_max_mm"]), "v": 1e-3 * float(s["speed_mm_s"])})
    patch_task(ecfg.task, cond, cfg, replace="randomize_grasp")
    ecfg.task.events["reset_ee"].params["xy_jitter"] = 0.0
    return TorqueEnv(ecfg)


# ---------------------------------------------------------------------------------- per-tick recorder
class _Stop(Exception):
    """Raised from inside a physics tick at success or timeout (ends the episode for every method alike)."""


class Recorder:
    """Wraps `plant.control_step` and `env.reset`: per-tick force, first contact, success, entry, timeout."""

    def __init__(self, env, time_max: float) -> None:
        self.env, self.st, self.d = env, env.ctx.state["forge"], env.ctx.data
        self.dt = float(env.ctx.model.opt.timestep)
        self.n_max = int(round(time_max / self.dt))
        self._tick, self._reset = env.plant.control_step, env.reset
        env.plant.control_step, env.reset = self.tick, self.reset
        self.f: list[float] = []

    def reset(self, *args, **kwargs):
        out = self._reset(*args, **kwargs)
        self.f, self.n, self.t_contact, self.t_success, self.entered = [], 0, -1.0, -1.0, False
        self.min_xy = float("inf")
        return out

    def tick(self, fn) -> None:
        self._tick(fn)
        st, d = self.st, self.d
        fv = d.cfrc_ext[st.peg_body, 3:6]
        f = math.sqrt(float(fv @ fv))
        self.f.append(f)
        self.n += 1
        t = self.n * self.dt
        if self.t_contact < 0.0 and f > 1.0:
            self.t_contact = t
        tip = d.site_xpos[st.peg_tip]
        xy = math.hypot(tip[0] - st.hole_tip[0], tip[1] - st.hole_tip[1])
        self.min_xy = min(self.min_xy, xy)
        if xy < st.place_xy:
            if tip[2] < st.hole_tip[2] - 0.001:
                self.entered = True
            if tip[2] - st.hole_floor[2] < st.success_dist:
                self.t_success = t
                raise _Stop
        if self.n >= self.n_max or (self.n % 50 == 0 and not self.env._viewer.is_running()):
            raise _Stop

    def result(self, unstable: bool = False) -> dict:
        f = np.asarray(self.f) if self.f else np.zeros(1)
        fm = np.median(np.stack((f[:-2], f[1:-1], f[2:])), axis=0) if len(f) >= 3 else f
        w = max(1, min(len(fm), int(round(0.05 / self.dt))))
        f50 = np.convolve(fm, np.ones(w) / w, "valid")
        ok = self.t_success > 0.0
        b = self.env.ctx.state.get("bench", {})
        return {"success": ok, "t": self.t_success, "t_contact": self.t_contact,
                "t_from_contact": self.t_success - self.t_contact if ok and self.t_contact >= 0 else -1.0,
                "f_max": float(f50.max()), "f_peak": float(f.max()), "f_mean": float(f.mean()),
                "outcome": "success" if ok else "unstable" if unstable else "jammed" if self.entered else "not_found",
                "min_xy_mm": 1e3 * self.min_xy, "err_mm": b.get("err_applied_mm", float("nan")),
                "tilt_deg": b.get("tilt_applied_deg", float("nan")), "friction": float(self.st.friction)}


# ---------------------------------------------------------------------------------- policies
def _load_sb3(run: Path, ckpt: str = "final_model.zip"):
    """PPO or RecurrentPPO model (the run's `algo`) + its observation normalization (mean, std, clip) from a run folder.

    Call it before creating the env: building and first running the policy imports parts of torch
    (torch._dynamo), and that import segfaults once the MuJoCo viewer's OpenGL thread is running.
    """
    import re

    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    conf = run / "config.json"
    algo = json.loads(conf.read_text())["train"]["algo"] if conf.is_file() else "recurrent_ppo"
    zp = run / ckpt
    vn = run / "vecnormalize.pkl"
    m = re.search(r"model_(\d+)_steps\.zip$", zp.name)
    if m:
        vn = zp.parent / f"model_vecnormalize_{m.group(1)}_steps.pkl"
    model = (RecurrentPPO if algo == "recurrent_ppo" else PPO).load(str(zp), device="cpu")
    with open(vn, "rb") as f:
        v = pickle.load(f)
    # Warm-up prediction: the first forward pass imports more of torch, which segfaults once the viewer runs.
    model.predict(np.zeros((1,) + model.observation_space.shape, dtype=np.float32), state=None,
                  episode_start=np.ones((1,), dtype=bool), deterministic=True)
    return model, v.obs_rms.mean, np.sqrt(v.obs_rms.var + v.epsilon), v.clip_obs


def _episodes(env, seeds: list[int], cfg: dict, step_fn, reset_fn=None) -> list[dict]:
    """Run episodes with `step_fn(obs) -> action` (gym loop) and record them."""
    rec = Recorder(env, float(cfg["time_max"]))
    rows = []
    for seed in seeds:
        set_shared(env, seed)
        obs, _ = env.reset(seed=seed)
        if reset_fn is not None:
            reset_fn(seed)
        unstable = False
        try:
            while True:
                obs, _, term, trunc, info = env.step(step_fn(obs))
                if term or trunc:
                    unstable = info.get("termination") == "unstable"
                    break
        except _Stop:
            pass
        rows.append({"seed": seed, **rec.result(unstable)})
        if not _after_episode(env, rows, cfg):
            break
    return rows


def job_spiral_rule(cond: dict, seeds: list[int], cfg: dict, method: str = "spiral_rule") -> list[dict]:
    """Spiral search + scripted catch rule (|F_xy| > 3.5 N for 3 steps -> try insert for 0.6 s)."""
    from compliant_RL.scripted import CatchRule

    env = make_spiral_env(cond, cfg)
    p = env.policy_obs_dim
    rule = CatchRule()
    try:
        return _episodes(env, seeds, cfg, lambda o: np.array([rule(o[:p])]), lambda s: rule.reset())
    finally:
        _close(env, cfg)


def job_forge(cond: dict, seeds: list[int], cfg: dict, method: str = "forge") -> list[dict]:
    import torch

    model, mean, std, clip = _load_sb3(path_of(cfg["models"]["forge_run"]))
    n_obs = model.observation_space.shape[0]
    env = make_forge_env(cond, cfg)
    p = env.policy_obs_dim
    mem = {}

    def reset(seed):
        torch.manual_seed(seed)
        mem.update(state=None, start=np.ones((1,), dtype=bool), full=np.zeros(n_obs, dtype=np.float32))

    def act(o):
        mem["full"][:p] = o
        x = np.clip((mem["full"] - mean) / std, -clip, clip).astype(np.float32)
        a, mem["state"] = model.predict(x[None], state=mem["state"], episode_start=mem["start"],
                                        deterministic=bool(cfg["forge"]["deterministic"]))
        mem["start"] = np.zeros((1,), dtype=bool)
        return a[0]

    try:
        return _episodes(env, seeds, cfg, act, reset)
    finally:
        _close(env, cfg)


def job_tacdiff(cond: dict, seeds: list[int], cfg: dict, method: str = "tacdiff") -> list[dict]:
    spec = importlib.util.spec_from_file_location("tacdiff_run", PROJECTS / "tacdiff" / "run.py")
    td = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(td)
    tcfg = dict(td.DEFAULTS)
    tcfg.update(model=str(path_of(cfg["models"]["tacdiff_model"])), render=bool(cfg["view"]["enabled"]),
                realtime=bool(cfg["view"]["realtime"]), time_max=float(cfg["time_max"]),
                xy_err=0.0, xy_err_from_env=True, threads=1)
    model = td.load_model(tcfg["model"], tcfg["repo"], int(tcfg["n_hidden"]), int(tcfg["n_T"]), int(tcfg["extra_steps"]), 1)
    env = make_tacdiff_env(cond, cfg, td)
    rec = Recorder(env, float(cfg["time_max"]))
    rows = []
    try:
        for seed in seeds:
            set_shared(env, seed)
            sens = td.WrenchSensing(env, 0.0, 0.0, 0.0, np.random.default_rng(seed + 7919))
            try:
                td.run_episode(env, model, sens, tcfg, np.random.default_rng(seed), env._viewer, seed=seed)
            except _Stop:
                pass
            rows.append({"seed": seed, **rec.result()})
            if not _after_episode(env, rows, cfg):
                break
    finally:
        _close(env, cfg)
    return rows


JOBS = {"spiral_rule": job_spiral_rule, "forge": job_forge, "tacdiff": job_tacdiff}


def _after_episode(env, rows: list[dict], cfg: dict) -> bool:
    """Viewer mode: print the episode and hold the final pose; False when the window was closed."""
    if not cfg["view"]["enabled"]:
        return True
    r = rows[-1]
    print(f"  seed {r['seed']}: error {r['err_mm']:.2f} mm, tilt {r['tilt_deg']:.1f}°, friction {r['friction']:.2f}  ->  "
          f"{'INSERTED in %.2f s' % r['t'] if r['success'] else r['outcome']}  (F_max {r['f_max']:.1f} N)", flush=True)
    t_end = time.time() + float(cfg["view"]["pause_s"])
    while time.time() < t_end and env._viewer.is_running():
        env._viewer.sync()
        time.sleep(0.02)
    return env._viewer.is_running()


def _close(env, cfg: dict) -> None:
    """Close the viewer window (also on Ctrl+C), or X11 aborts with GLXBadDrawable at exit."""
    if cfg["view"]["enabled"]:
        env.close()


def run_job(task: tuple) -> tuple:
    """Worker entry: all episodes of one (method, condition)."""
    method, ci, cond, seeds, cfg = task
    import torch

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    rows = JOBS[method](cond, seeds, cfg, method)
    for r in rows:
        r.update(method=method, cond=ci, **{k: cond[k] for k in ("shape", "clearance_mm", "err", "tilt")},
                 friction_level=cond["friction"])
    return method, ci, rows, time.perf_counter() - t0


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(PROJECTS))


# ---------------------------------------------------------------------------------- summary
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(rows: list[dict], methods: list[str], conds: list[dict]) -> tuple[list[str], list[list[str]]]:
    """Per condition and method: success (95% CI), median time to success, median F_max, failure split."""
    header = ["#", "shape", "clearance/side [mm]", "hole error [mm]", "tilt [deg]", "friction"]
    for m in methods:
        n = NAMES[m]
        header += [f"{n} success (95% CI)", f"{n} time [s]", f"{n} F_max [N]", f"{n} fails: not found / jammed"]
    table = []
    for ci, c in enumerate(conds):
        line = [str(ci + 1), c["shape"], f"{c['clearance_mm']:g}", c["err"], c["tilt"], c["friction"]]
        for m in methods:
            rs = [r for r in rows if r["method"] == m and r["cond"] == ci]
            n, k = len(rs), sum(r["success"] for r in rs)
            if n == 0:
                line += ["-"] * 4
                continue
            lo, hi = wilson(k, n)
            ts = [r["t"] for r in rs if r["success"]]
            nf = sum(r["outcome"] == "not_found" for r in rs)
            jm = sum(r["outcome"] in ("jammed", "unstable") for r in rs)
            line += [f"{k}/{n} ({100 * lo:.0f}-{100 * hi:.0f}%)", f"{np.median(ts):.2f}" if ts else "-",
                     f"{np.median([r['f_max'] for r in rs]):.1f}", f"{nf} / {jm}"]
        table.append(line)
    return header, table


def write_summary(out: Path, rows: list[dict], methods: list[str], conds: list[dict]) -> str:
    header, table = summarize(rows, methods, conds)
    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + table)
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in table]
    tot = ["", "| method | success (all conditions) |", "|---|---|"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        k = sum(r["success"] for r in rs)
        tot.append(f"| {NAMES[m]} | {k}/{len(rs)} ({100 * k / max(len(rs), 1):.0f}%) |")
    text = "\n".join(md + tot)
    (out / "summary.md").write_text(text + "\n")
    return text


# ---------------------------------------------------------------------------------- main
def view(cfg: dict, conds: list[dict], seeds: list[int]) -> None:
    """Viewer mode: one method, condition by condition, in this process."""
    method = cfg["view"]["method"]
    if method not in NAMES:
        raise ValueError(f"view.method must be one of {list(NAMES)}")
    for ci, c in enumerate(conds):
        print(f"{NAMES[method]} | condition {ci + 1}/{len(conds)}: {c['shape']}, clearance {c['clearance_mm']:g} mm, "
              f"hole error {c['err']} mm, tilt {c['tilt']}°, friction {c['friction']}", flush=True)
        try:
            rows = JOBS[method](c, seeds, cfg, method)
        except KeyboardInterrupt:                           # the job has already closed the viewer
            print("stopped (Ctrl+C)")
            return
        k = sum(r["success"] for r in rows)
        print(f"  inserted {k}/{len(rows)}", flush=True)
        if len(rows) < len(seeds):
            print("viewer closed")
            return


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    conds = build_conditions(cfg)
    seeds = [int(cfg["seed"]) + i for i in range(int(cfg["episodes"]))]
    if cfg["view"]["enabled"]:
        view(cfg, conds, seeds)
        return
    methods = list(cfg["methods"])
    out = path_of(cfg["out"]) if cfg["out"] else HERE / "results" / time.strftime("compare_%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ep_file = out / "episodes.jsonl"
    rows = [json.loads(s) for s in ep_file.read_text().splitlines()] if ep_file.exists() else []
    done = {(r["method"], r["cond"]) for r in rows}
    tasks = [(m, ci, conds[ci], seeds, copy.deepcopy(cfg)) for m in sorted(methods, key=ORDER.get)
             for ci in range(len(conds)) if (m, ci) not in done]
    print(f"{len(conds)} conditions x {len(methods)} methods x {len(seeds)} episodes: {len(tasks)} jobs to run "
          f"({len(done)} done) -> {out}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn"),
                             initializer=_init_worker) as ex:
        futs = [ex.submit(run_job, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            method, ci, rs, wall = fut.result()
            with open(ep_file, "a") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
            rows += rs
            c = conds[ci]
            print(f"[{i + 1:3d}/{len(tasks)}] {NAMES[method]:11s} #{ci + 1:2d} {c['shape']:8s} clr {c['clearance_mm']:g} "
                  f"err {c['err']} tilt {c['tilt']} mu {c['friction']}: {sum(r['success'] for r in rs)}/{len(rs)}  "
                  f"({wall:.0f} s, elapsed {time.perf_counter() - t0:.0f} s)", flush=True)
    print("\n" + write_summary(out, rows, methods, conds))
    print(f"\n{len(rows)} episodes -> {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECTS))
    main(sys.argv[1:])
