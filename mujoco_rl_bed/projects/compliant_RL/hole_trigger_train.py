"""Collect plain-spiral data and train the hole-trigger classifier (`hole_trigger.HoleTrigger`).

Data: `compliant_peg_s3` with the spiral only (never triggered), recording the trigger's view (the noisy `trigger`
observation group) and the truth (tip-to-axis distance) every search step. Conditions (one env each, `eps`
episodes): shape U{cylinder, cuboid, hexagon, octagon}, radial clearance N(clr_mean, clr_std) clipped to
[clr_lo, clr_hi]; per episode hole-estimate error U(err), friction U(friction), hole yaw error U(0, min(yaw_cap, half
the shape's symmetry)) with random sign (cylinder: none), arm tilt U(tilt) in a share `1 - tilt_zero_share` of the
conditions (else none). Seeds `seed`+ (evaluation uses 2000+).

Model: gradient boosting on a 4-step window [x_t, mean, std, x_t - x_(t-3)] of the 21 inputs; label tip within 1 mm
vs farther than 1.5 mm (peg not yet dropping). 5-fold cross-validation grouped by condition gives held-out scores; the
firing rule (p > TH for K steps) is the one that fires near the hole in the most episodes with at most `max_false`
false firings (farther than 1.5 mm) per episode. Saved to models/<out> (fit on everything).

Usage (from `mujoco_rl_bed/`; OMP_NUM_THREADS=1 keeps the workers from oversubscribing the CPU):
    OMP_NUM_THREADS=1 python projects/compliant_RL/hole_trigger_train.py out=hole_trigger_classifier_tilt.pkl workers=24
    python projects/compliant_RL/hole_trigger_train.py n_cond=8 eps=1 workers=4 out=smoke.pkl     # smoke test
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
SHAPES = ["cylinder", "cuboid", "hexagon", "octagon"]
SYM = {"cylinder": 0.0, "cuboid": 90.0, "hexagon": 60.0, "octagon": 45.0}
W, COOL = 4, 100
DEFAULTS = {
    "n_cond": 400, "eps": 4, "seed": 4000, "workers": 16, "out": "hole_trigger_classifier_tilt.pkl",
    "clr_mean": 0.3, "clr_std": 0.1, "clr_lo": 0.1, "clr_hi": 0.5,   # radial clearance [mm]
    "err": "0-4", "friction": "0.3-1.0", "tilt": "0-5", "tilt_zero_share": 0.5, "yaw_cap": 12.0,
    "time_max": 20.0, "max_false": 0.15,
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


def conditions(cfg: dict) -> list[dict]:
    r = np.random.default_rng(12345)
    out = []
    for i in range(int(cfg["n_cond"])):
        shape = SHAPES[i % 4]
        clr = float(np.clip(r.normal(cfg["clr_mean"], cfg["clr_std"]), cfg["clr_lo"], cfg["clr_hi"]))
        tilt = "0" if r.uniform() < cfg["tilt_zero_share"] else str(cfg["tilt"])
        out.append({"id": i, "shape": shape, "clearance_mm": round(clr, 3), "tilt": tilt,
                    "yaw_max": min(float(cfg["yaw_cap"]), 0.5 * SYM[shape])})
    return out


def make_env(c: dict, cfg: dict):
    import compare as C
    import compliant_RL  # noqa: F401
    from mujoco_rl_bed.env.torque_env import TorqueEnv
    from mujoco_rl_bed.tasks.registry import make_env_cfg

    e, f, t = C.level(cfg["err"]), C.level(cfg["friction"]), C.level(c["tilt"])
    cond = {"shape": c["shape"], "clearance_mm": c["clearance_mm"], "err": e[0], "err_lo": e[1], "err_hi": e[2],
            "tilt": t[0], "tilt_lo": t[1], "tilt_hi": t[2], "friction": f[0], "mu_lo": f[1], "mu_hi": f[2]}
    bcfg = {"time_max": float(cfg["time_max"]), "peg_half_size_mm": 4.0, "paper_error": {"enabled": False},
            "view": {"enabled": False, "realtime": False, "pause_s": 0.0}}
    ecfg = make_env_cfg("compliant_peg_s3", {"task.episode_length_s": str(float(cfg["time_max"]) + 1.0),
                                             "obs_mode": "asymmetric", "render": "false", "realtime": "false"})
    ecfg.task.events["hole_yaw_error"].params.update({"lo_deg": 0.0, "hi_deg": c["yaw_max"],
                                                      "sym_deg": SYM[c["shape"]] or 90.0})
    ecfg.task.action.params.update({"f_search": 1.5, "pitch": 0.0008, "r_max": 0.0046, "v": 0.01})
    C.patch_task(ecfg.task, cond, bcfg, replace="randomize_grasp")
    ecfg.task.events["reset_ee"].params["xy_jitter"] = 0.0
    return TorqueEnv(ecfg)


def collect(task: tuple) -> dict:
    c, cfg = task
    _paths()
    import compare as C

    env = make_env(c, cfg)
    ctx = env.ctx
    st, sp, d = ctx.state["forge"], ctx.state["spiral"], ctx.data
    keys = ("obs", "dist", "depth", "ep", "cond", "yaw", "tilt", "mu", "clr", "shape")
    out = {k: [] for k in keys}
    for e in range(int(cfg["eps"])):
        seed = int(cfg["seed"]) + c["id"] * int(cfg["eps"]) + e
        C.set_shared(env, seed)
        env.reset(seed=seed)
        yaw = abs(math.degrees(ctx.state.get("hole_yaw_err", 0.0))) if SYM[c["shape"]] else 0.0
        tilt = float(ctx.state.get("bench", {}).get("tilt_applied_deg", 0.0))
        for _ in range(int(float(cfg["time_max"]) / env.policy_dt)):
            if sp["phase"] == 1:
                tip = d.site_xpos[st.peg_tip]
                out["obs"].append(env.compute_obs("trigger"))
                out["dist"].append(1e3 * math.hypot(tip[0] - st.hole_tip[0], tip[1] - st.hole_tip[1]))
                out["depth"].append(1e3 * sp["depth"])
                for k, v in (("ep", c["id"] * int(cfg["eps"]) + e), ("cond", c["id"]), ("yaw", yaw), ("tilt", tilt),
                             ("mu", float(st.friction)), ("clr", c["clearance_mm"]), ("shape", SHAPES.index(c["shape"]))):
                    out[k].append(v)
            _, _, term, trunc, _ = env.step(np.zeros(3))
            if term or trunc or st.first_success_step >= 0 or sp["phase"] == 2:
                break
    return {k: np.asarray(v) for k, v in out.items()}


def window_features(obs: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Same features as `HoleTrigger.prob` (try state and previous action zeroed)."""
    from compliant_RL.hole_trigger import ZEROED

    o = obs.astype(np.float64)
    o[:, ZEROED] = 0.0
    idx = np.arange(len(o))
    start = np.searchsorted(ep, ep, side="left")
    stack = np.stack([o[np.maximum(idx - k, start)] for k in range(W)])
    return np.concatenate([o, stack.mean(0), stack.std(0), o - stack[-1]], axis=1)


def make_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, class_weight="balanced", random_state=0)


def runs(p: np.ndarray, ep: np.ndarray, th: float, k: int) -> np.ndarray:
    above = p > th
    new_ep = np.r_[True, ep[1:] != ep[:-1]]
    run = np.zeros(len(p), dtype=int)
    for i in range(len(p)):
        run[i] = (0 if new_ep[i] else run[i - 1]) + 1 if above[i] else 0
    return run >= k


def events(fire: np.ndarray, ep: np.ndarray, dist: np.ndarray) -> dict:
    """Per episode: (fired with the tip within 1.5 mm, false firings, tip came within 1 mm)."""
    out = {}
    bounds = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        cool, near, false = 0, False, 0
        for i in range(a, b):
            if cool > 0:
                cool -= 1
            elif fire[i]:
                near |= dist[i] < 1.5
                false += dist[i] >= 1.5
                cool = COOL
        out[int(ep[a])] = (near, false, bool((dist[a:b] < 1.0).any()))
    return out


def score(ev: dict, eps) -> tuple[float, float, int]:
    sel = [ev[e] for e in eps if ev[e][2]]
    return (float(np.mean([s[0] for s in sel])) if sel else float("nan"), float(np.mean([ev[e][1] for e in eps])), len(sel))


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    _paths()
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    t0 = time.perf_counter()
    conds = conditions(cfg)
    parts = []
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn"),
                             initializer=_paths) as ex:
        for i, q in enumerate(ex.map(collect, [(c, cfg) for c in conds])):
            parts.append(q)
            if (i + 1) % 40 == 0 or i + 1 == len(conds):
                print(f"[collect] {i + 1}/{len(conds)} conditions, {time.perf_counter() - t0:.0f} s", flush=True)
    D = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    obs, dist, depth, ep, cond = D["obs"], D["dist"], D["depth"], D["ep"], D["cond"]
    X = window_features(obs, ep)
    lab = dist < 1.0
    m = (depth < 0.3) & ((dist < 1.0) | (dist > 1.5))
    print(f"[data] {len(obs)} steps, {len(np.unique(ep))} episodes, {len(conds)} conditions; training samples {m.sum()} "
          f"(within 1 mm {lab[m].mean():.3f})", flush=True)

    p_cv = np.zeros(len(obs))
    for f, (tr, te) in enumerate(GroupKFold(5).split(X, lab, cond)):
        trm = tr[m[tr]]
        p_cv[te] = make_clf().fit(X[trm], lab[trm]).predict_proba(X[te])[:, 1]
        print(f"[cv] fold {f + 1}/5 done, {time.perf_counter() - t0:.0f} s", flush=True)
    all_eps = np.unique(ep)
    cands = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97):
        for k in (1, 2, 3):
            ev = events(runs(p_cv, ep, th, k), ep, dist)
            near, false, _ = score(ev, all_eps)
            cands.append((near, false, th, k, ev))
    ok = [c for c in cands if c[1] <= float(cfg["max_false"])]
    near, false, TH, K, ev = max(ok, key=lambda c: c[0]) if ok else max(cands, key=lambda c: c[0] - 0.2 * c[1])
    old_p = None
    try:
        from compliant_RL.hole_trigger import MODEL
        with open(MODEL, "rb") as fh:
            old = pickle.load(fh)
        old_p = old["clf"].predict_proba(X)[:, 1]
        ev_old = events(runs(old_p, ep, float(old["TH"]), int(old["K"])), ep, dist)
    except FileNotFoundError:
        ev_old = None
    fxy = np.hypot(obs[:, 0], obs[:, 1])
    ev_rule = events(runs(fxy, ep, 3.5, 3), ep, dist)

    info = {}
    for e in all_eps:
        i = np.searchsorted(ep, e)
        info[int(e)] = (int(D["shape"][i]), float(D["clr"][i]), float(D["yaw"][i]), float(D["tilt"][i]), float(D["mu"][i]))
    G = {f"shape {s}": [e for e, v in info.items() if v[0] == i] for i, s in enumerate(SHAPES)}
    for lo, hi in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.51)):
        G[f"clearance {lo:g}-{min(hi, 0.5):g} mm"] = [e for e, v in info.items() if lo <= v[1] < hi]
    G["tilt 0"] = [e for e, v in info.items() if v[3] < 1e-6]
    for lo, hi in ((1e-6, 2.5), (2.5, 99)):
        G[f"tilt {0 if lo < 1 else lo:g}-{hi if hi < 99 else 'max'} deg"] = [e for e, v in info.items() if lo <= v[3] < hi]
    for lo, hi in ((0, 4), (4, 8), (8, 99)):
        G[f"yaw {lo}-{hi if hi < 99 else 'max'} deg (non-round)"] = [e for e, v in info.items() if v[0] > 0 and lo <= v[2] < hi]
    G["all"] = list(info)
    lines = [f"Hole trigger `{cfg['out']}`: p > {TH} for {K} steps; config {cfg}", "",
             "| group | episodes | AUC (held-out conditions) | new: fires near / false per ep | "
             "previous model: near / false | catch rule: near / false |", "|---|---|---|---|---|---|"]
    for name, eps in G.items():
        s = np.isin(ep, eps) & m
        auc = roc_auc_score(lab[s], p_cv[s]) if 0 < lab[s].sum() < s.sum() else float("nan")
        a, b = score(ev, eps), score(ev_rule, eps)
        o = score(ev_old, eps) if ev_old is not None else (float("nan"), float("nan"), 0)
        lines.append(f"| {name} | {len(eps)} | {auc:.2f} | {a[0]:.2f} / {a[1]:.2f} | {o[0]:.2f} / {o[1]:.2f} | "
                     f"{b[0]:.2f} / {b[1]:.2f} |")
    text = "\n".join(lines)
    print("\n" + text, flush=True)

    clf = make_clf().fit(X[m], lab[m])
    from compliant_RL.hole_trigger import MODELS

    out = MODELS / str(cfg["out"])
    with open(out, "wb") as fh:
        pickle.dump({"clf": clf, "W": W, "TH": TH, "K": K, "config": cfg,
                     "inputs": "4-step window of the 21 compliant_peg_s3 trigger inputs: [x, mean, std, x - x_(t-3)]"}, fh)
    out.with_suffix(".md").write_text(text + "\n")
    print(f"\nsaved {out} (+ .md report), {time.perf_counter() - t0:.0f} s total", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
