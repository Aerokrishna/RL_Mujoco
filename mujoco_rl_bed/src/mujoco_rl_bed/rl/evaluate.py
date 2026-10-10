"""Generic evaluation of a trained checkpoint (PPO or RecurrentPPO) or a scripted policy.

Used by `scripts/eval.py` (core tasks) and project entry points (e.g. `projects/forge/eval.py`).

When `run` is given, the task, algo, env overrides and task modules come from its
config.json, and VecNormalize statistics are loaded if present (`vecnormalize.pkl`, or
the checkpoint's matching `model_vecnormalize_<N>_steps.pkl`). Extra non-EvalCfg keys
are EnvCfg overrides. `policy=scripted` uses the `scripted_factory` the caller provides.

Reports per-episode and aggregate: success at the final step, success at any step,
time to success, mean/max contact force, return (when the task reports them).
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import mujoco_rl_bed.tasks  # noqa: F401  (registers tasks)
from mujoco_rl_bed.sim.scene import PROJECT_ROOT
from mujoco_rl_bed.tasks.registry import make_env
from mujoco_rl_bed.rl.train import import_task_modules
from mujoco_rl_bed.utils.config import apply_overrides, parse_cli, to_jsonable


@dataclass
class EvalCfg:
    """Evaluation options.

    Attributes:
        run: Run directory (from training); relative paths are relative to the current directory,
            then to `run_root`.
        run_root: Fallback base directory for relative `run` paths (set by project entry points).
        checkpoint: Model zip, relative to `run` (default final_model.zip) or absolute.
        policy: "checkpoint" or "scripted" (needs a `scripted_factory` from the caller).
        task: Task name (only needed without `run`).
        eval_task: Evaluate a `run` on this task instead of its training task (same observation and action
            sizes required), e.g. a different part geometry. Empty = the run's task.
        episodes: Number of episodes.
        seed: First episode seed (episode i uses seed + i).
        deterministic: Use the policy mean.
        render: Open the viewer.
        realtime: Pace the viewer to wall time.
        device: Torch device for the policy.
    """

    run: str = ""
    run_root: str = ""
    checkpoint: str = ""
    policy: str = "checkpoint"
    task: str = ""
    eval_task: str = ""
    episodes: int = 10
    seed: int = 1000
    deterministic: bool = True
    render: bool = False
    realtime: bool = True
    device: str = "cpu"


def _resolve(p: str, base: Path) -> Path:
    """Resolve a path against `base` unless absolute.

    Args:
        p: Path string.
        base: Base directory.

    Returns:
        Absolute path.
    """
    q = Path(p)
    return q if q.is_absolute() else base / q


def _find_run(run: str, run_root: str) -> Path:
    """Locate a run directory.

    Args:
        run: Absolute path, path relative to the cwd, or relative to `run_root` / the repository.
        run_root: Optional fallback base directory.

    Returns:
        Existing run directory.
    """
    cands = [Path(run)]
    if run_root:
        cands.append(Path(run_root) / run)
    cands.append(PROJECT_ROOT / run)
    for c in cands:
        if (c / "config.json").exists():
            return c.resolve()
    raise FileNotFoundError(f"No run with config.json at any of: {[str(c) for c in cands]}")


_IGNORE_DIFF = {"render", "realtime", "seed", "task_name"}


def _config_diff(trained: Any, current: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Leaf-level differences between two `to_jsonable` config trees.

    Args:
        trained: Config stored in the run's config.json ("env").
        current: Config of the env about to be evaluated.
        path: Dotted prefix (override syntax: dict keys, dataclass fields, list indices).

    Returns:
        List of (dotted key, trained value, current value).
    """
    if isinstance(trained, dict) and isinstance(current, dict):
        out = []
        for k in sorted(set(trained) | set(current)):
            if k == "_type" or (not path and k in _IGNORE_DIFF):
                continue
            out += _config_diff(trained.get(k), current.get(k), f"{path}.{k}" if path else k)
        return out
    if isinstance(trained, list) and isinstance(current, list) and len(trained) == len(current) \
            and any(isinstance(v, (dict, list)) for v in trained):
        out = []
        for i, (a, b) in enumerate(zip(trained, current)):
            out += _config_diff(a, b, f"{path}.{i}")
        return out
    return [] if trained == current else [(path, trained, current)]


def _warn_config_drift(run_conf: dict, env: Any) -> None:
    """Print overrides that would restore settings the run was trained with but today's defaults changed.

    Args:
        run_conf: Parsed config.json of the run.
        env: The env about to be evaluated.
    """
    if "env" not in run_conf:
        return
    diffs = _config_diff(run_conf["env"], to_jsonable(env.cfg))
    changed = [(k, old, new) for k, old, new in diffs if old is not None]
    added = [k for k, old, _ in diffs if old is None]
    if not diffs:
        return
    print("[eval] WARNING: the env differs from the one this run was trained with (task defaults changed since).")
    if changed:
        print("       To restore the trained values, add:")
        for key, old, new in changed:
            val = str(old).replace(" ", "") if isinstance(old, (list, tuple)) else old
            print(f"         {key}={val}      # now: {new}")
    if added:
        print("       Settings added since this run (did not exist then; disable if they change behaviour, "
              "e.g. task.action.success_prediction=false): " + ", ".join(added))


def _check_compatible(model: Any, env: Any) -> None:
    """Fail early, with a hint, if a checkpoint does not match the env built from today's task defaults.

    Evaluation re-applies only the run's CLI overrides, so a run trained before a task default
    changed (e.g. success prediction added an action/observation dim) rebuilds a different env.

    Args:
        model: Loaded SB3 model.
        env: Freshly built `TorqueEnv`.

    Raises:
        ValueError: On observation/action shape mismatch.
    """
    mo, eo = model.observation_space.shape, env.observation_space.shape
    ma, ea = model.action_space.shape, env.action_space.shape
    if mo == eo and ma == ea:
        return
    hints = []
    if ea[0] == ma[0] + 1:
        hints.append("task.action.success_prediction=false  (run predates success prediction)")
    if eo[0] != mo[0]:
        hints.append("check other task defaults changed since the run (e.g. task.scene.assets.1.inner_radius, "
                     "obs groups); compare with the run's config.json 'env' section")
    raise ValueError(
        f"Checkpoint does not match the current task: model obs {mo} / act {ma}, env obs {eo} / act {ea}. "
        "The run was likely trained with older task defaults. Try adding:\n  " + "\n  ".join(hints))


def main(argv: list[str], task_modules: Sequence[str] = (), defaults: dict[str, Any] | None = None,
         scripted_factory: Callable[[Any], Callable[[np.ndarray], np.ndarray]] | None = None) -> dict:
    """Run the evaluation.

    Args:
        argv: key=value overrides (highest priority).
        task_modules: Modules to import to register tasks (merged with the run's config).
        defaults: Override defaults applied before `argv`.
        scripted_factory: `factory(env) -> policy` for `policy=scripted`; the policy is called
            as `policy(obs) -> action` and may define `reset()`.

    Returns:
        Aggregate metrics.
    """
    import_task_modules(task_modules)
    raw = {**{k: str(v) for k, v in (defaults or {}).items()}, **parse_cli(argv)}
    names = {f.name for f in dataclasses.fields(EvalCfg)}
    cfg = apply_overrides(EvalCfg(), {k: v for k, v in raw.items() if k in names})
    env_ov = {k: v for k, v in raw.items() if k not in names}

    algo, task, run_dir = "ppo", cfg.task, None
    if cfg.run:
        run_dir = _find_run(cfg.run, cfg.run_root)
        conf = json.loads((run_dir / "config.json").read_text())
        algo, task = conf["train"]["algo"], conf["train"]["task"]
        import_task_modules(conf.get("task_modules", []))
        env_ov = {**conf.get("env_overrides", {}), **env_ov}
    if cfg.eval_task:
        task = cfg.eval_task
    if not task:
        raise ValueError("Give run=<dir> or task=<name>")
    env_ov.update({"render": str(cfg.render).lower(), "realtime": str(cfg.realtime).lower()})

    # Load the checkpoint BEFORE creating the env: building the policy lazily imports parts of
    # torch (torch._dynamo), and that import segfaults once the viewer's OpenGL thread is running.
    model, scripted, vn = None, None, None
    if cfg.policy != "scripted":
        if run_dir is None:
            raise ValueError("policy=checkpoint needs run=<dir>")
        ckpt = _resolve(cfg.checkpoint, run_dir) if cfg.checkpoint else run_dir / "final_model.zip"
        vn = run_dir / "vecnormalize.pkl"
        m = re.search(r"model_(\d+)_steps\.zip$", ckpt.name)
        if m:
            cand = ckpt.parent / f"model_vecnormalize_{m.group(1)}_steps.pkl"
            vn = cand if cand.exists() else vn
        if algo == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO as Algo
        else:
            from stable_baselines3 import PPO as Algo
        model = Algo.load(str(ckpt), device=cfg.device)
        print(f"[eval] {algo} checkpoint {ckpt.name}, vecnormalize={'yes' if vn.exists() else 'no'}")

    env = make_env(task, env_ov)
    try:
        if run_dir is not None:
            _warn_config_drift(conf, env)
        if model is not None:
            _check_compatible(model, env)
        venv = DummyVecEnv([lambda: env])
        if model is not None and vn is not None and vn.exists():
            venv = VecNormalize.load(str(vn), venv)
            venv.training = False
            venv.norm_reward = False
        if cfg.policy == "scripted":
            if scripted_factory is None:
                raise ValueError("policy=scripted needs a scripted_factory (use the project's eval.py)")
            scripted = scripted_factory(env)
    except BaseException:
        env.close()  # close the viewer before exiting, or the interpreter segfaults at exit
        raise

    rows = []
    stopped = False
    try:
        for ep in range(cfg.episodes):
            venv.seed(cfg.seed + ep)
            obs = venv.reset()
            if scripted is not None and hasattr(scripted, "reset"):
                scripted.reset()
            state, start = None, np.ones((1,), dtype=bool)
            ret, info = 0.0, {}
            while True:
                if scripted is not None:
                    action = scripted(obs[0])[None]
                else:
                    action, state = model.predict(obs, state=state, episode_start=start,
                                                  deterministic=cfg.deterministic)
                obs, reward, done, infos = venv.step(action)
                start = done
                ret += float(reward[0])
                if done[0]:
                    info = infos[0]
                    break
                if not env.viewer_is_running():  # window closed by the user
                    stopped = True
                    break
            if stopped:
                print("[eval] viewer closed, stopping")
                break
            row = {"episode": ep, "return": ret, "is_success": bool(info.get("is_success", False))}
            for k in ("ever_success", "time_to_success", "contact_force_mean", "contact_force_max", "force_threshold",
                      "et_triggered", "et_correct", "et_recall_hit", "et_delay"):
                if k in info:
                    row[k] = info[k]
            rows.append(row)
            print("[eval] " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    except KeyboardInterrupt:
        print("[eval] interrupted")
    finally:
        venv.close()  # closes the viewer cleanly (avoids a crash at interpreter exit)

    if not rows:
        return {}
    agg = {"success_rate": float(np.mean([r["is_success"] for r in rows])),
           "return_mean": float(np.mean([r["return"] for r in rows]))}
    if rows and "ever_success" in rows[0]:
        agg["ever_success_rate"] = float(np.mean([r["ever_success"] for r in rows]))
        t = [r["time_to_success"] for r in rows if r["time_to_success"] >= 0]
        agg["time_to_success_mean"] = float(np.mean(t)) if t else -1.0
        agg["contact_force_mean"] = float(np.mean([r["contact_force_mean"] for r in rows]))
        agg["contact_force_max_mean"] = float(np.mean([r["contact_force_max"] for r in rows]))
    if rows and "et_triggered" in rows[0]:
        # Paper Table I early-termination metrics (p > p_term).
        trig = [r for r in rows if r["et_triggered"]]
        succ = [r for r in rows if r.get("ever_success")]
        agg["et_precision"] = float(np.mean([r["et_correct"] for r in trig])) if trig else float("nan")
        agg["et_recall"] = float(np.mean([r["et_recall_hit"] for r in succ])) if succ else float("nan")
        d = [r["et_delay"] for r in rows if r["et_delay"] >= 0]
        agg["et_delay_mean"] = float(np.mean(d)) if d else -1.0
    print("[eval] summary: " + "  ".join(f"{k}={v:.3f}" for k, v in agg.items()))
    return agg

