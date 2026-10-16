"""End-to-end evaluation of the spiral + local (x, y, yaw) pipeline (`compliant_peg_s3`) with a trained local policy.

Methods, all on the same reset seeds (shared hole-error / friction draws from `projects/benchmark/compare.py`):
    current          scripted spiral + catch rule (`compliant_peg_s2g` + `CatchRule`): today's pipeline
    trigger_oracle   spiral -> hole trigger (`hole_trigger.HoleTrigger`) -> privileged `LocalOracle` (ceiling)
    trigger_policy   spiral -> hole trigger -> trained local policy (`run`)
    perfect_policy   spiral -> privileged trigger (true tip within 1 mm) -> trained local policy
Conditions: one shape and clearance, hole yaw error exactly each of `yaws` (random sign), hole-estimate error
U(err), friction U(friction), no tilt. The local policy acts only in the local phase; its LSTM state is reset at
every local-phase start, or runs through the whole episode with `lstm_continuous` (as in `compliant_peg_s3_handoff`
training; "auto" picks it from the run's task).

Usage (from `mujoco_rl_bed/`):
    python projects/compliant_RL/eval_s3.py run=<run_folder>
    python projects/compliant_RL/eval_s3.py run=<run_folder> episodes=50 yaws=[0,3,6,9] workers=8
    python projects/compliant_RL/eval_s3.py run=<run_folder> shape=hexagon yaws=[0,10,20,25]
    python projects/compliant_RL/eval_s3.py methods=[current,trigger_oracle]          # no policy needed

Keys are key=value with YAML values (see DEFAULTS). Writes summary.md and episodes.jsonl to `out` (default
<run>/eval_s3_<time>, or projects/compliant_RL/runs/eval_s3_<time> without a run).
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
BED = PROJECTS.parent
SYM = {"cylinder": 90.0, "cuboid": 90.0, "hexagon": 60.0, "octagon": 45.0}
DEFAULTS = {
    "run": "",                       # trained local-policy run (folder name in runs/ or a path)
    "checkpoint": "final_model.zip",
    "methods": ["current", "trigger_oracle", "trigger_policy", "perfect_policy"],
    "episodes": 50,
    "seed": 2000,
    "yaws": [0.0, 3.0, 6.0, 9.0],    # hole yaw error magnitudes [deg]
    "shape": "cuboid",
    "clearance_mm": 0.5,
    "err": "0-4",                    # hole-estimate error [mm]: "a" or "a-b"
    "friction": "0.5-1.0",
    "time_max": 20.0,                # [s] per episode
    "t_local": 6.0,                  # [s] per local phase before the spiral resumes
    "deterministic": True,           # policy mean (False: sampled actions)
    "lstm_continuous": "auto",       # True: LSTM over the whole episode; False: reset per local phase; "auto": by task
    "workers": 5,
    "out": "",
}


def load_config(argv: list[str]) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for a in argv:
        k, v = a.split("=", 1)
        if k not in cfg:
            raise KeyError(f"unknown key '{k}' (valid: {sorted(cfg)})")
        cfg[k] = yaml.safe_load(v)
    return cfg


def _paths() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    for p in (str(PROJECTS / "benchmark"), str(PROJECTS)):
        if p not in sys.path:
            sys.path.insert(0, p)


def bench(cfg: dict) -> tuple[dict, dict]:
    """compare.py condition and config dicts for the episode loop and `patch_task`."""
    from compare import level

    e, f = level(cfg["err"]), level(cfg["friction"])
    cond = {"shape": cfg["shape"], "clearance_mm": float(cfg["clearance_mm"]), "err": e[0], "err_lo": e[1],
            "err_hi": e[2], "tilt": "0", "tilt_lo": 0.0, "tilt_hi": 0.0, "friction": f[0], "mu_lo": f[1], "mu_hi": f[2]}
    bcfg = {"time_max": float(cfg["time_max"]), "peg_half_size_mm": 4.0, "paper_error": {"enabled": False},
            "spiral": {"f_search_n": 1.5, "pitch_mm": 0.8, "r_max_mm": 4.6, "speed_mm_s": 10.0},
            "view": {"enabled": False, "realtime": False, "pause_s": 0.0}}
    return cond, bcfg


def make_env(task: str, cfg: dict, yaw: float):
    import compare as C
    import compliant_RL  # noqa: F401
    from mujoco_rl_bed.env.cfg import EventTermCfg
    from mujoco_rl_bed.env.torque_env import TorqueEnv
    from mujoco_rl_bed.tasks.registry import make_env_cfg

    cond, bcfg = bench(cfg)
    ecfg = make_env_cfg(task, {"task.episode_length_s": str(float(cfg["time_max"]) + 1.0), "obs_mode": "asymmetric",
                               "render": "false", "realtime": "false"})
    if "hole_yaw_error" not in ecfg.task.events:             # compliant_peg_s2g: same event, same position as s3
        ev = {}
        for k, v in ecfg.task.events.items():
            ev[k] = v
            if k == "reset_fixed":
                ev["hole_yaw_error"] = EventTermCfg(mode="reset", func="cr_hole_yaw_error")
        ecfg.task.events = ev
    ecfg.task.events["hole_yaw_error"].params.update({"lo_deg": yaw, "hi_deg": yaw, "sym_deg": SYM[cfg["shape"]]})
    s = bcfg["spiral"]
    ecfg.task.action.params.update({"f_search": s["f_search_n"], "pitch": 1e-3 * s["pitch_mm"],
                                    "r_max": 1e-3 * s["r_max_mm"], "v": 1e-3 * s["speed_mm_s"]})
    if task == "compliant_peg_s3":
        ecfg.task.action.params["t_local"] = float(cfg["t_local"])
    C.patch_task(ecfg.task, cond, bcfg, replace="randomize_grasp")
    ecfg.task.events["reset_ee"].params["xy_jitter"] = 0.0
    return TorqueEnv(ecfg)


def run_dir(cfg: dict) -> Path:
    from mujoco_rl_bed.rl.evaluate import _find_run

    return _find_run(cfg["run"], str(HERE / "runs"))


def run_job(task: tuple) -> tuple:
    method, yaw, cfg = task
    _paths()
    import torch

    torch.set_num_threads(1)
    import compare as C
    from compliant_RL.hole_trigger import HoleTrigger
    from compliant_RL.scripted import CatchRule, LocalOracle

    seeds = [int(cfg["seed"]) + i for i in range(int(cfg["episodes"]))]
    _, bcfg = bench(cfg)
    t0 = time.perf_counter()
    if method == "current":
        env = make_env("compliant_peg_s2g", cfg, yaw)
        rule = CatchRule()
        p = env.policy_obs_dim
        rows = C._episodes(env, seeds, bcfg, lambda o: np.array([rule(o[:p])]), lambda s: rule.reset())
        return method, yaw, rows, time.perf_counter() - t0

    uses_policy = method in ("trigger_policy", "perfect_policy")
    continuous = False
    if uses_policy:
        model, mean, std, clip = C._load_sb3(run_dir(cfg), cfg["checkpoint"])
        continuous = cfg["lstm_continuous"]
        if continuous == "auto":
            task_name = json.loads((run_dir(cfg) / "config.json").read_text())["train"]["task"]
            continuous = task_name == "compliant_peg_s3_handoff"
    env = make_env("compliant_peg_s3", cfg, yaw)
    ctx = env.ctx
    st, sp = ctx.state["forge"], ctx.state["spiral"]
    orc = None if uses_policy else LocalOracle(env, sym_deg=SYM[cfg["shape"]])
    trig = HoleTrigger() if method.startswith("trigger") else None
    mem: dict = {}
    hand: list[list[dict]] = []

    def reset(seed):
        torch.manual_seed(seed)
        if orc is not None:
            orc.reset()
        if trig is not None:
            trig.reset()
        mem.update(prev=0, since=10**6, state=None, first=True)
        hand.append([])

    def act(o):
        ph = sp["phase"]
        if mem["prev"] != 3 and ph == 3:
            tip = ctx.data.site_xpos[st.peg_tip]
            hand[-1].append({"off_mm": 1e3 * math.hypot(tip[0] - st.hole_tip[0], tip[1] - st.hole_tip[1]),
                             "out": "end"})
            if not continuous:
                mem["state"] = None
        if mem["prev"] == 3 and ph != 3:
            hand[-1][-1]["out"] = "drop" if ph == 2 else "timeout"
            mem["since"] = 0
        mem["since"] += 1
        mem["prev"] = ph
        if ph == 1 and mem["since"] > 25:
            if trig is not None:
                if trig(env.compute_obs("trigger")):
                    sp["trigger"] = True
            else:
                tip = ctx.data.site_xpos[st.peg_tip]
                if math.hypot(tip[0] - st.hole_tip[0], tip[1] - st.hole_tip[1]) < 0.001:
                    sp["trigger"] = True
        if orc is not None:
            return orc(o)
        if ph != 3 and not continuous:
            return np.zeros(3)
        x = np.clip((o - mean) / std, -clip, clip).astype(np.float32)
        start = np.array([mem["first"] if continuous else mem["state"] is None])
        mem["first"] = False
        a, mem["state"] = model.predict(x[None], state=mem["state"], episode_start=start,
                                        deterministic=bool(cfg["deterministic"]))
        return a[0] if ph == 3 else np.zeros(3)

    rows = C._episodes(env, seeds, bcfg, act, reset)
    for r, h in zip(rows, hand):
        r["handoffs"] = h
    return method, yaw, rows, time.perf_counter() - t0


def summarize(res: dict, cfg: dict) -> str:
    yaws, methods = cfg["yaws"], cfg["methods"]
    n = int(cfg["episodes"])
    L = [f"## {cfg['shape']}, {cfg['clearance_mm']:g} mm clearance, hole error U({cfg['err']}) mm, friction "
         f"U({cfg['friction']}), {n} episodes per cell (seeds {cfg['seed']}+), {cfg['time_max']:g} s; run "
         f"`{cfg['run'] or '-'}` ({cfg['checkpoint']}, {'deterministic' if cfg['deterministic'] else 'sampled'}, "
         f"local phase {cfg['t_local']:g} s, LSTM {cfg['lstm_continuous']})", "",
         "| method | " + " | ".join(f"yaw {y:g}°" for y in yaws) + " | all |", "|---|" + "---|" * (len(yaws) + 1)]
    for m in methods:
        cells, tot = [], 0
        for y in yaws:
            rs = res[(m, y)]
            k = sum(r["success"] for r in rs)
            tot += k
            ts = [r["t"] for r in rs if r["success"]]
            cells.append(f"{k}/{len(rs)}" + (f" ({np.median(ts):.1f} s)" if ts else ""))
        L.append(f"| {m} | " + " | ".join(cells) + f" | {tot}/{len(yaws) * n} |")
    L += ["", "Failures (not found / jammed) and hand-offs (per episode; within 1.5 mm -> ended in a drop):", ""]
    for m in methods:
        parts = []
        for y in yaws:
            rs = res[(m, y)]
            nf = sum(r["outcome"] == "not_found" for r in rs)
            jm = sum(r["outcome"] in ("jammed", "unstable") for r in rs)
            hs = [h for r in rs for h in r.get("handoffs", [])]
            near = [h for h in hs if h["off_mm"] < 1.5]
            extra = (f"; {len(hs) / len(rs):.2f}/ep, {sum(h['out'] == 'drop' for h in near)}/{len(near)}"
                     if "handoffs" in rs[0] else "")
            parts.append(f"{y:g}°: {nf}/{jm}{extra}")
        L.append(f"- {m}: " + ", ".join(parts))
    return "\n".join(L)


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    _paths()
    if any(m in ("trigger_policy", "perfect_policy") for m in cfg["methods"]) and not cfg["run"]:
        raise ValueError("run=<run_folder> is needed for trigger_policy / perfect_policy")
    stamp = time.strftime("eval_s3_%Y%m%d-%H%M%S")
    out = Path(cfg["out"]) if cfg["out"] else (run_dir(cfg) / stamp if cfg["run"] else HERE / "runs" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(m, float(y), cfg) for m in cfg["methods"] for y in cfg["yaws"]]
    res = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn"),
                             initializer=_paths) as ex:
        for m, y, rows, wall in ex.map(run_job, tasks):
            res[(m, y)] = rows
            print(f"{m:15s} yaw {y:g}: {sum(r['success'] for r in rows)}/{len(rows)} ({wall:.0f} s, elapsed "
                  f"{time.perf_counter() - t0:.0f} s)", flush=True)
    with open(out / "episodes.jsonl", "w") as f:
        for (m, y), rows in res.items():
            for r in rows:
                f.write(json.dumps({"method": m, "yaw": y, **r}) + "\n")
    text = summarize(res, {**cfg, "yaws": [float(y) for y in cfg["yaws"]]})
    (out / "summary.md").write_text(text + "\n")
    print("\n" + text + f"\n\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
