"""Generic PPO / RecurrentPPO (LSTM, sb3-contrib) training entry point.

Used by `scripts/train_ppo.py` (core tasks) and by project entry points such as
`projects/forge/train.py`, which pass `task_modules` (packages whose import registers
their tasks/terms) and `defaults` (e.g. the project's default task and run folder).

The simulation runs on CPU subprocesses; the network runs on `device` (auto = CUDA if
available, else CPU).

CLI: keys that are `TrainCfg` fields (no dots) configure training. Every other key is
forwarded as an `EnvCfg` override (e.g. `render`, `sim_dt`, `decimation`, `task.*`).

Each run writes <run_root>/<task>_<YYYYmmdd-HHMMSS>_s<seed>/ with:
    config.json        resolved TrainCfg + EnvCfg + raw overrides + task_modules
    tb/                TensorBoard logs (SB3 scalars + episode/* task metrics)
    checkpoints/       periodic model_<steps>_steps.zip (+ model_vecnormalize_<steps>_steps.pkl)
    final_model.zip, vecnormalize.pkl
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

import mujoco_rl_bed.tasks  # noqa: F401  (registers tasks)
from mujoco_rl_bed.sim.scene import PROJECT_ROOT
from mujoco_rl_bed.tasks.registry import make_env_cfg
from mujoco_rl_bed.utils.config import apply_overrides, parse_cli, to_jsonable
from mujoco_rl_bed.utils.seeding import seed_everything
from mujoco_rl_bed.wrappers.sb3 import EpisodeInfoCallback, make_vec_env


@dataclass
class TrainCfg:
    """Training options (all overridable as key=value).

    Attributes:
        task: Registered task name.
        algo: "ppo" (MLP) or "recurrent_ppo" (sb3-contrib MlpLstmPolicy; actor and critic each
            have their own LSTM, the SB3 default).
        asymmetric: Asymmetric actor-critic. The env returns [policy | critic] observations
            (`obs_mode=asymmetric`), the actor sees only the policy group and the critic sees both
            (privileged state). Needs a `critic` group in the task. See `mujoco_rl_bed.rl.asymmetric`.
        seed: Run seed (env i uses seed + i).
        n_envs: Parallel environments (one subprocess each).
        vec: "subproc" or "dummy".
        total_timesteps: Total environment steps.
        device: Torch device for the network ("auto", "cpu", "cuda").
        torch_threads: torch.set_num_threads in the learner process (0 = PyTorch default, i.e. one
            thread per core). The networks are tiny, so extra threads only add synchronization
            overhead and steal cores from the env workers: with the default, a 10-env run
            measured ~690 steps/s with the learner at ~700% CPU and the workers mostly idle.
        n_steps: Rollout length per env per update.
        batch_size: Minibatch size (for RecurrentPPO: number of transitions per minibatch).
        n_epochs: Optimization epochs per update.
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        vf_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping.
        target_kl: Early-stop an update when the approx. KL exceeds this (0 = off).
        log_std_init: Initial log std of the Gaussian policy.
        net_arch: Hidden layers of the actor and critic MLP heads.
        lstm_hidden_size: LSTM width (recurrent_ppo).
        n_lstm_layers: LSTM depth (recurrent_ppo).
        normalize_obs: Running observation normalization (VecNormalize).
        normalize_reward: Running return normalization (VecNormalize).
        checkpoint_every: Save a checkpoint every this many env steps.
        run_root: Directory holding run folders (relative paths are relative to the repository root).
        run_name: Optional suffix for the run folder.
        verbose: SB3 verbosity.
    """

    task: str = "reach"
    algo: str = "ppo"
    asymmetric: bool = False
    seed: int = 0
    n_envs: int = 8
    vec: str = "subproc"
    total_timesteps: int = 50_000
    device: str = "auto"
    torch_threads: int = 1
    n_steps: int = 256
    batch_size: int = 512
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.0
    log_std_init: float = -0.5
    net_arch: tuple[int, ...] = (256, 128)
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1
    normalize_obs: bool = True
    normalize_reward: bool = True
    checkpoint_every: int = 100_000
    run_root: str = "runs"
    run_name: str = ""
    verbose: int = 1


def split_overrides(raw: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Separate TrainCfg keys from EnvCfg overrides.

    Args:
        raw: All CLI overrides.

    Returns:
        (train_overrides, env_overrides).
    """
    names = {f.name for f in dataclasses.fields(TrainCfg)}
    train = {k: v for k, v in raw.items() if k in names}
    env = {k: v for k, v in raw.items() if k not in names}
    return train, env


def build_model(cfg: TrainCfg, venv, tb_dir: Path, actor_dim: int | None = None):
    """Instantiate the SB3 model.

    Args:
        cfg: Training config.
        venv: Vectorized env.
        tb_dir: TensorBoard directory.
        actor_dim: Policy-part size of the observation (required when `cfg.asymmetric`).

    Returns:
        A `PPO` or `RecurrentPPO` model.
    """
    arch = list(cfg.net_arch)
    common = dict(n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                  learning_rate=cfg.learning_rate, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                  clip_range=cfg.clip_range, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
                  max_grad_norm=cfg.max_grad_norm, target_kl=cfg.target_kl or None, seed=cfg.seed,
                  device=cfg.device, tensorboard_log=str(tb_dir), verbose=cfg.verbose)
    pk = dict(net_arch=dict(pi=arch, vf=arch), log_std_init=cfg.log_std_init)
    if cfg.asymmetric:
        from mujoco_rl_bed.rl.asymmetric import AsymmetricActorCriticPolicy, AsymmetricRecurrentPolicy

        pk["actor_dim"] = int(actor_dim)
    if cfg.algo == "ppo":
        policy = AsymmetricActorCriticPolicy if cfg.asymmetric else "MlpPolicy"
        return PPO(policy, venv, policy_kwargs=pk, **common)
    if cfg.algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        pk.update(lstm_hidden_size=cfg.lstm_hidden_size, n_lstm_layers=cfg.n_lstm_layers)
        policy = AsymmetricRecurrentPolicy if cfg.asymmetric else "MlpLstmPolicy"
        return RecurrentPPO(policy, venv, policy_kwargs=pk, **common)
    raise ValueError(f"unknown algo '{cfg.algo}' (ppo | recurrent_ppo)")


def import_task_modules(modules: Sequence[str]) -> None:
    """Import packages whose import registers tasks/terms (e.g. "forge").

    Args:
        modules: Importable module names.
    """
    for name in modules:
        importlib.import_module(name)


def main(argv: list[str], task_modules: Sequence[str] = (), defaults: dict[str, Any] | None = None) -> Path:
    """Run training.

    Args:
        argv: key=value overrides (highest priority).
        task_modules: Modules to import (here and in every env worker) to register tasks.
        defaults: Override defaults applied before `argv` (same key=value semantics).

    Returns:
        The run directory.
    """
    task_modules = tuple(task_modules)
    import_task_modules(task_modules)
    raw = {**{k: str(v) for k, v in (defaults or {}).items()}, **parse_cli(argv)}
    train_ov, env_ov = split_overrides(raw)
    cfg = apply_overrides(TrainCfg(), train_ov)
    if cfg.asymmetric:
        env_ov.setdefault("obs_mode", "asymmetric")
        if env_ov["obs_mode"] != "asymmetric":
            raise ValueError("asymmetric=true requires obs_mode=asymmetric")
    env_cfg = make_env_cfg(cfg.task, env_ov)  # validates env overrides before spawning workers
    if (cfg.n_steps * cfg.n_envs) % cfg.batch_size != 0:
        raise ValueError(f"n_steps * n_envs ({cfg.n_steps * cfg.n_envs}) must be divisible by batch_size")

    seed_everything(cfg.seed)
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)

    root = Path(cfg.run_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{cfg.task}_{stamp}_s{cfg.seed}{'_' + cfg.run_name if cfg.run_name else ''}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({
        "train": to_jsonable(cfg), "env": to_jsonable(env_cfg),
        "env_overrides": env_ov, "argv": argv, "task_modules": list(task_modules),
    }, indent=2))

    venv = make_vec_env(cfg.task, cfg.n_envs, cfg.seed, env_ov, vec=cfg.vec, task_modules=task_modules)
    if cfg.normalize_obs or cfg.normalize_reward:
        venv = VecNormalize(venv, norm_obs=cfg.normalize_obs, norm_reward=cfg.normalize_reward, gamma=cfg.gamma)

    actor_dim = venv.get_attr("policy_obs_dim", indices=[0])[0]
    model = build_model(cfg, venv, run_dir / "tb", actor_dim=actor_dim)
    callbacks = CallbackList([
        EpisodeInfoCallback(),
        CheckpointCallback(save_freq=max(1, cfg.checkpoint_every // cfg.n_envs), save_path=str(run_dir / "checkpoints"),
                           name_prefix="model", save_vecnormalize=isinstance(venv, VecNormalize)),
    ])
    print(f"[train] run dir: {run_dir}")
    print(f"[train] algo={cfg.algo} task={cfg.task} n_envs={cfg.n_envs} device={model.device} "
          f"obs_dim={venv.observation_space.shape[0]} (actor {actor_dim}) act_dim={venv.action_space.shape[0]} "
          f"asymmetric={cfg.asymmetric} "
          f"policy_hz={env_cfg.policy_hz:.1f}")
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, tb_log_name="tb", progress_bar=False)
    finally:
        model.save(run_dir / "final_model.zip")
        if isinstance(venv, VecNormalize):
            venv.save(str(run_dir / "vecnormalize.pkl"))
        venv.close()
    dt = time.perf_counter() - t0
    print(f"[train] done: {model.num_timesteps} steps in {dt:.0f} s ({model.num_timesteps / max(dt, 1e-9):.0f} steps/s)")
    return run_dir

