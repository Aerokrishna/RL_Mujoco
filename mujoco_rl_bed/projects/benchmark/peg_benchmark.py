"""Peg-in-hole benchmark: TacDiffusion (authors' model) vs. a FORGE checkpoint under controlled perturbations.

One-factor-at-a-time design around a nominal condition (Kp 600 N/m, 0.5 mm diametrical clearance, exact
hole-estimate error 0 mm, no force noise), 10 conditions x `episodes` episodes per method. Every condition
uses the same reset seeds, so socket poses and error directions are identical across conditions and methods.

| # | Kp [N/m] | clearance (diam.) [mm] | hole error [mm] | force noise          |
|---|----------|------------------------|-----------------|----------------------|
| 1 | 600      | 0.5                    | 0               | off                  |
| 2-3 | 400, 800 | 0.5                  | 0               | off                  |
| 4-5 | 600    | 1.0, 0.25              | 0               | off                  |
| 6-9 | 600    | 0.5                    | 1, 2, 3, 4      | off                  |
| 10 | 600     | 0.5                    | 0               | N(0.5 N, var 0.2 N^2) per axis per sample |

Success: peg tip within 1 mm of the hole floor and within 2.5 mm of the axis, within `time_max` seconds
from when the policy takes control (FORGE: from reset, 3.7-5.7 cm above the hole; TacDiffusion: from the
handover after a scripted, privileged approach to the estimated hole and a guarded descent to contact).

Method details:
- FORGE: the checkpoint's own training settings (20 Hz, position EMA 0.2, λ 2 cm, rot. stiffness 100 Nm/rad,
  friction 0.75, no dead zone / in-hand offset / yaw / extra observation noise), F_th = 7.5 N, deterministic
  actions. Kp is the x/y/z stiffness. Force noise is added to its observed (step-averaged) contact force. The
  actor reads only its 21 inputs; the critic part of the observation (absent from today's task layout) is zero.
- TacDiffusion: `projects/tacdiff/run.py` (1 kHz wrench control, ONNX model, paper filter). Kp is the
  lateral spring stiffness of its insertion phase (it has no z spring: the model commands the push force).
  Force noise is added to the sensed contact force at 1 kHz.

`grid=full` replaces this table by all combinations of `kp_list` x `clearance_list` x `err_list`
(force noise off, or on for all rows with `full_force_noise=true`).

Usage (`conda activate dqn`, from `mujoco_rl_bed/`):
    python projects/benchmark/peg_benchmark.py                    # both methods, 10 episodes per condition
    python projects/benchmark/peg_benchmark.py grid=full kp_list=600,400 err_list=0,0.5,1,2,3 clearance_list=0.5,1,0.25
    python projects/benchmark/peg_benchmark.py methods=forge episodes=2 workers=4
    python projects/benchmark/peg_benchmark.py mode=sanity        # privileged scripted expert, feasibility check

Writes `projects/benchmark/results/<time>/summary.csv` and `summary.md`, and prints the table.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
REPO = PROJECTS.parents[1]                      # .../RL_Mujoco

DEFAULTS = {
    "mode": "bench",                            # bench | sanity
    "methods": "tacdiff,forge",
    "episodes": 10,                             # per condition and method
    "seed": 1000,                               # episode i uses reset seed `seed + i` in every condition
    "time_max": 15.0,                           # [s] from when the policy takes control
    "workers": 5,
    "forge_run": str(PROJECTS / "forge" / "runs" / "forge_peg_20261007-194131_s0_srv_ft_n1_5M_cont"),
    "tacdiff_model": str(REPO / "diffusion" / "TacDiffusion" / "output" / "TacDiffusion_model_512.onnx"),
    "tacdiff_threads": 2,
    "force_noise_mean": 0.5,                    # [N]
    "force_noise_var": 0.2,                     # [N^2]
    "f_th": 7.5,                                # FORGE force threshold input [N]
    "grid": "ofat",                             # ofat: CONDITIONS below | full: all combinations of the lists
    "kp_list": "600,400",                       # grid=full: stiffness levels [N/m]
    "clearance_list": "0.5,1,0.25",             # grid=full: diametrical clearances [mm]
    "err_list": "0,0.5,1,2,3",                  # grid=full: hole-estimate errors [mm]
    "full_force_noise": False,                  # grid=full: force noise on in every row
}

NOMINAL = {"kp": 600.0, "clearance_mm": 0.5, "err_mm": 0.0, "force_noise": False}
CONDITIONS = [dict(NOMINAL)]
CONDITIONS += [{**NOMINAL, "kp": k} for k in (400.0, 800.0)]
CONDITIONS += [{**NOMINAL, "clearance_mm": c} for c in (1.0, 0.25)]
CONDITIONS += [{**NOMINAL, "err_mm": e} for e in (1.0, 2.0, 3.0, 4.0)]
CONDITIONS += [{**NOMINAL, "force_noise": True}]


def build_conditions(cfg: dict) -> list[dict]:
    """The condition list of this run.

    Args:
        cfg: Benchmark config.

    Returns:
        CONDITIONS (grid=ofat) or the full factorial of the lists (grid=full), ordered Kp, clearance, error.
    """
    if cfg["grid"] == "ofat":
        return [dict(c) for c in CONDITIONS]
    if cfg["grid"] != "full":
        raise ValueError("grid must be 'ofat' or 'full'")
    vals = {k: [float(x) for x in str(cfg[k]).split(",") if x] for k in ("kp_list", "clearance_list", "err_list")}
    return [{"kp": k, "clearance_mm": c, "err_mm": e, "force_noise": bool(cfg["full_force_noise"])}
            for k in vals["kp_list"] for c in vals["clearance_list"] for e in vals["err_list"]]

PEG_RADIUS = 0.004


def hole_radius(clearance_mm: float) -> float:
    """Hole radius [m] for a diametrical clearance.

    Args:
        clearance_mm: Diametrical clearance [mm].

    Returns:
        Radius [m].
    """
    return PEG_RADIUS + 0.5e-3 * clearance_mm


# ---------------------------------------------------------------------------------- FORGE
_CACHE: dict = {}


def _forge_env(cond: dict, cfg: dict):
    """FORGE env with the checkpoint's training settings plus the condition.

    Args:
        cond: Condition.
        cfg: Benchmark config.

    Returns:
        The env.
    """
    import forge  # noqa: F401
    from forge.task import NO_DR
    from mujoco_rl_bed.tasks.registry import make_env

    k = cond["kp"]
    ov = {**NO_DR,
          "decimation": "25", "task.episode_length_s": str(cfg["time_max"]),
          "task.action.ema_prediction": "false",
          "task.events.randomize_controller.params.kp_default": f"({k},{k},{k},100.0,100.0,100.0)",
          "task.events.randomize_controller.params.lam_default": "0.02",
          "task.events.reset_threshold.params.lo": str(cfg["f_th"]),
          "task.events.reset_threshold.params.hi": str(cfg["f_th"]),
          "task.events.reset_fixed.params.pos_noise_std": "0.0",
          "task.events.reset_fixed.params.pos_noise_xy": str(1e-3 * cond["err_mm"]),
          "task.scene.assets.1.inner_radius": str(hole_radius(cond["clearance_mm"]))}
    return make_env("forge_peg", ov)


def _forge_policy(cfg: dict):
    """Load the checkpoint and its observation normalization (cached per worker).

    Args:
        cfg: Benchmark config.

    Returns:
        (model, obs_mean, obs_std, clip_obs, model_obs_dim).
    """
    if "forge" not in _CACHE:
        import torch
        from sb3_contrib import RecurrentPPO

        torch.set_num_threads(1)
        run = Path(cfg["forge_run"])
        model = RecurrentPPO.load(str(run / "final_model.zip"), device="cpu")
        with open(run / "vecnormalize.pkl", "rb") as f:
            vn = pickle.load(f)
        std = np.sqrt(vn.obs_rms.var + vn.epsilon)
        _CACHE["forge"] = (model, vn.obs_rms.mean, std, vn.clip_obs, model.observation_space.shape[0])
    return _CACHE["forge"]


def run_forge(cond_idx: int, cond: dict, seed: int, cfg: dict) -> dict:
    """One FORGE episode.

    Args:
        cond_idx: Condition index (cache key / table row).
        cond: Condition.
        seed: Reset seed.
        cfg: Benchmark config.

    Returns:
        Episode result.
    """
    model, mean, std, clip, n_obs = _forge_policy(cfg)
    key = ("forge_env", cond_idx)
    if key not in _CACHE:
        _CACHE[key] = _forge_env(cond, cfg)
    env = _CACHE[key]
    st = env.ctx.state["forge"]
    fsl = dict(env.obs_mgr.layout["policy"])["contact_force"]
    noise_rng = np.random.default_rng(seed + 7919)
    sd = math.sqrt(cfg["force_noise_var"])
    obs, _ = env.reset(seed=seed)
    full = np.zeros(n_obs, dtype=np.float32)
    p = env.policy_obs_dim
    state, start = None, np.ones((1,), dtype=bool)
    forces, t_succ = [], -1.0
    for k in range(env.ctx.max_episode_steps):
        full[:p] = obs
        if cond["force_noise"]:
            full[fsl] += noise_rng.normal(cfg["force_noise_mean"], sd, size=3)
        x = np.clip((full - mean) / std, -clip, clip).astype(np.float32)
        action, state = model.predict(x[None], state=state, episode_start=start, deterministic=True)
        start = np.zeros((1,), dtype=bool)
        obs, _, term, trunc, _ = env.step(action[0])
        forces.append(st.force_mean_norm)                 # contact force averaged over the 50 ms step
        if st.success:
            t_succ = (k + 1) * env.ctx.policy_dt
            break
        if term or trunc:
            break
    f = np.asarray(forces) if forces else np.zeros(1)
    return {"method": "forge", "cond": cond_idx, "seed": seed, "success": bool(st.success), "t": t_succ,
            "f_mean": float(f.mean()), "f_max": float(f.max()), "xy_mm": 1e3 * st.xy_dist,
            "h_mm": 1e3 * st.tip_height}


# ---------------------------------------------------------------------------------- TacDiffusion
def _tacdiff_module():
    """Import `projects/tacdiff/run.py` (not a package) once per worker.

    Returns:
        The module.
    """
    if "td_mod" not in _CACHE:
        spec = importlib.util.spec_from_file_location("tacdiff_run", PROJECTS / "tacdiff" / "run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["td_mod"] = mod
    return _CACHE["td_mod"]


def run_tacdiff(cond_idx: int, cond: dict, seed: int, cfg: dict) -> dict:
    """One TacDiffusion episode.

    Args:
        cond_idx: Condition index (cache key / table row).
        cond: Condition.
        seed: Reset seed.
        cfg: Benchmark config.

    Returns:
        Episode result.
    """
    td = _tacdiff_module()
    tcfg = dict(td.DEFAULTS)
    tcfg.update(model=cfg["tacdiff_model"], render=False, realtime=False, time_max=float(cfg["time_max"]),
                xy_err=0.0, xy_err_from_env=True, lat_kp=float(cond["kp"]),
                hole_radius=hole_radius(cond["clearance_mm"]), threads=int(cfg["tacdiff_threads"]))
    if "td_model" not in _CACHE:
        import torch

        torch.set_num_threads(1)
        _CACHE["td_model"] = td.load_model(tcfg["model"], tcfg["repo"], int(tcfg["n_hidden"]), int(tcfg["n_T"]),
                                           int(tcfg["extra_steps"]), int(tcfg["threads"]))
    key = ("td_env", cond_idx)
    if key not in _CACHE:
        _CACHE[key] = td.make_sim_env(tcfg, {"task.events.reset_fixed.params.pos_noise_xy": str(1e-3 * cond["err_mm"])})
    env = _CACHE[key]
    noise_rng = np.random.default_rng(seed + 7919)
    mean, sd = (cfg["force_noise_mean"], math.sqrt(cfg["force_noise_var"])) if cond["force_noise"] else (0.0, 0.0)
    sens = td.WrenchSensing(env, float(tcfg["frame_yaw"]), mean, sd, noise_rng)
    r = td.run_episode(env, _CACHE["td_model"], sens, tcfg, np.random.default_rng(seed), env._viewer, seed=seed)
    return {"method": "tacdiff", "cond": cond_idx, "seed": seed, "success": bool(r["success"]), "t": r["t_success"],
            "f_mean": r["f_mean"], "f_max": r["f_max_50ms"], "xy_mm": r["xy_err_mm"], "h_mm": r["tip_height_mm"]}


def _run(task: tuple) -> dict:
    """Worker entry point.

    Args:
        task: (method, cond_idx, cond, seed, cfg).

    Returns:
        Episode result.
    """
    method, ci, cond, seed, cfg = task
    t0 = time.perf_counter()
    r = (run_forge if method == "forge" else run_tacdiff)(ci, cond, seed, cfg)
    r["wall_s"] = time.perf_counter() - t0
    return r


# ---------------------------------------------------------------------------------- summary
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile (1.96 = 95%).

    Returns:
        (low, high).
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(rows: list[dict], methods: list[str], conditions: list[dict]) -> tuple[list[str], list[list[str]]]:
    """Per-condition table: condition columns + per method success (95% CI), time, F_mean, F_max.

    Args:
        rows: Episode results.
        methods: Methods in column order.
        conditions: Condition list (table rows).

    Returns:
        (header, table rows as strings).
    """
    names = {"tacdiff": "TacDiff", "forge": "FORGE"}
    header = ["#", "Kp [N/m]", "clearance [mm]", "hole error [mm]", "force noise"]
    for m in methods:
        header += [f"{names[m]} success (95% CI)", f"{names[m]} time [s]", f"{names[m]} F_mean [N]",
                   f"{names[m]} F_max [N]"]
    table = []
    for ci, c in enumerate(conditions):
        line = [str(ci + 1), f"{c['kp']:.0f}", f"{c['clearance_mm']:g}", f"{c['err_mm']:g}",
                "N(0.5, 0.2)" if c["force_noise"] else "off"]
        for m in methods:
            rs = [r for r in rows if r["method"] == m and r["cond"] == ci]
            n, k = len(rs), sum(r["success"] for r in rs)
            lo, hi = wilson(k, n)
            ts = [r["t"] for r in rs if r["success"]]
            line += [f"{k}/{n} ({100 * lo:.0f}-{100 * hi:.0f}%)",
                     f"{np.mean(ts):.2f}" if ts else "-",
                     f"{np.mean([r['f_mean'] for r in rs]):.1f}" if rs else "-",
                     f"{np.mean([r['f_max'] for r in rs]):.1f}" if rs else "-"]
        table.append(line)
    return header, table


def to_markdown(header: list[str], table: list[list[str]]) -> str:
    """Render a Markdown table.

    Args:
        header: Column names.
        table: Rows.

    Returns:
        Markdown text.
    """
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in table]
    return "\n".join(out)


# ---------------------------------------------------------------------------------- sanity
def sanity(cfg: dict) -> None:
    """Privileged scripted expert (true poses) per Kp and clearance: is every condition physically solvable?

    Args:
        cfg: Benchmark config.
    """
    import forge  # noqa: F401
    from forge.scripted import ScriptedPegInsert
    from forge.task import NO_DR
    from mujoco_rl_bed.tasks.registry import make_env

    n = int(cfg["episodes"])
    for c in CONDITIONS[:5]:
        k = c["kp"]
        env = make_env("forge_peg", {**NO_DR, "task.episode_length_s": str(cfg["time_max"]),
                                     "task.events.randomize_controller.params.kp_default": f"({k},{k},{k},28,28,28)",
                                     "task.events.reset_fixed.params.pos_noise_std": "0.0",
                                     "task.scene.assets.1.inner_radius": str(hole_radius(c["clearance_mm"]))})
        pol = ScriptedPegInsert(env)
        ok, ts = 0, []
        for i in range(n):
            obs, _ = env.reset(seed=int(cfg["seed"]) + i)
            pol.reset()
            st = env.ctx.state["forge"]
            for _ in range(env.ctx.max_episode_steps):
                obs, _, te, tr, _ = env.step(pol(obs))
                if st.success:
                    ok += 1
                    ts.append(env.ctx.episode_step * env.ctx.policy_dt)
                    break
                if te or tr:
                    break
        print(f"Kp {k:.0f}  clearance {c['clearance_mm']:g} mm: scripted expert {ok}/{n}"
              + (f", mean time {np.mean(ts):.2f} s" if ts else ""), flush=True)


# ---------------------------------------------------------------------------------- main
def main(argv: list[str]) -> None:
    """Run the benchmark (or the sanity check) and write the summary.

    Args:
        argv: key=value overrides of DEFAULTS.
    """
    from mujoco_rl_bed.utils.config import _literal, parse_cli

    cfg = dict(DEFAULTS)
    for k, v in parse_cli(argv).items():
        if k not in cfg:
            raise KeyError(f"unknown key '{k}' (valid: {sorted(cfg)})")
        cfg[k] = v if isinstance(DEFAULTS[k], str) else type(DEFAULTS[k])(_literal(v))
    if cfg["mode"] == "sanity":
        sanity(cfg)
        return
    methods = [m for m in cfg["methods"].split(",") if m]
    if any(m not in ("tacdiff", "forge") for m in methods):
        raise ValueError("methods: comma list of tacdiff, forge")
    conds = build_conditions(cfg)
    tasks = [(m, ci, conds[ci], int(cfg["seed"]) + i, cfg) for i in range(int(cfg["episodes"]))
             for ci in range(len(conds)) for m in methods]
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(_run, tasks, chunksize=1):
            rows.append(r)
            print(f"[{len(rows):4d}/{len(tasks)}] {r['method']:7s} cond {r['cond'] + 1:2d} seed {r['seed']}  "
                  f"success={r['success']!s:5}  t={r['t']:6.2f}s  F_max={r['f_max']:5.1f}N  "
                  f"xy={r['xy_mm']:.2f}mm h={r['h_mm']:.1f}mm  ({r['wall_s']:.0f}s)", flush=True)
    header, table = summarize(rows, methods, conds)
    out = HERE / "results" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + table)
    md = to_markdown(header, table)
    (out / "summary.md").write_text(md + "\n")
    (out / "config.json").write_text(json.dumps({"cfg": cfg, "conditions": conds}, indent=2))
    print("\n" + md)
    print(f"\n{len(rows)} episodes in {time.perf_counter() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
