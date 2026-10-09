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

Fine-tuning: `init_from=<run>` starts from that run's trained weights (policy, value function,
optimizer state) and VecNormalize statistics instead of a fresh network, e.g. to continue a
finished run with more steps or harder settings. The new run gets its own folder and counts
`total_timesteps` from zero; the network settings (algo, asymmetric, net_arch, LSTM) must match.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
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
        learning_rate: Adam learning rate (the initial value with `lr_schedule=adaptive`).
        lr_schedule: "constant", or "adaptive" (rl_games / Isaac Lab): after every PPO update the rate is
            divided by 1.5 if the update's approx. KL > 2 * `kl_threshold` and multiplied by 1.5 if it is
            < 0.5 * `kl_threshold`, within [1e-6, 1e-2].
        kl_threshold: Target KL of the adaptive schedule (Isaac Lab FORGE: 0.008).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        clip_range_vf: Value-function clip range (rl_games `clip_value`; 0 = off).
        ent_coef: Entropy coefficient.
        vf_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping.
        target_kl: Early-stop an update when the approx. KL exceeds this (0 = off).
        log_std_init: Initial log std of the Gaussian policy.
        net_arch: Hidden layers of the actor and critic MLP heads.
        activation: MLP activation: "tanh" (SB3 default), "elu" (Isaac Lab FORGE) or "relu".
        lstm_hidden_size: LSTM width (recurrent_ppo).
        n_lstm_layers: LSTM depth (recurrent_ppo).
        normalize_obs: Running observation normalization (VecNormalize).
        normalize_reward: Running return normalization (VecNormalize).
        checkpoint_every: Save a checkpoint every this many env steps.
        run_root: Directory holding run folders (relative paths are relative to the repository root).
        run_name: Optional suffix for the run folder.
        init_from: Run to initialize from (folder name in `run_root`, or a path): its weights, optimizer
            state and VecNormalize statistics are loaded before training. Empty = train from scratch.
        init_checkpoint: Model zip inside `init_from` (default final_model.zip), e.g.
            checkpoints/model_3000000_steps.zip (the matching VecNormalize file is used).
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
    lr_schedule: str = "constant"
    kl_threshold: float = 0.008
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.0
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.0
    log_std_init: float = -0.5
    net_arch: tuple[int, ...] = (256, 128)
    activation: str = "tanh"
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1
    normalize_obs: bool = True
    normalize_reward: bool = True
    checkpoint_every: int = 100_000
    run_root: str = "runs"
    run_name: str = ""
    init_from: str = ""
    init_checkpoint: str = ""
    verbose: int = 1


_ACTIVATIONS = {"tanh": torch.nn.Tanh, "elu": torch.nn.ELU, "relu": torch.nn.ReLU}


class AdaptiveLR:
    """KL-adaptive learning rate (rl_games `AdaptiveScheduler`), used as an SB3 learning-rate schedule.

    SB3 calls the schedule at the start of every update; `AdaptiveLRCallback` updates `lr` from the
    approx. KL of the previous update. Module-level so saved models unpickle.

    Attributes:
        lr: Current learning rate.
        kl_threshold: Target KL.
        min_lr: Lower bound.
        max_lr: Upper bound.
    """

    def __init__(self, lr: float, kl_threshold: float, min_lr: float = 1e-6, max_lr: float = 1e-2) -> None:
        """Store the initial rate and the bounds.

        Args:
            lr: Initial learning rate.
            kl_threshold: Target KL.
            min_lr: Lower bound.
            max_lr: Upper bound.
        """
        self.lr, self.kl_threshold, self.min_lr, self.max_lr = float(lr), float(kl_threshold), min_lr, max_lr

    def __call__(self, progress_remaining: float) -> float:
        """Return the current rate (SB3 schedule interface; progress is ignored).

        Args:
            progress_remaining: 1 -> 0 over training.

        Returns:
            Learning rate.
        """
        return self.lr

    def update(self, kl: float) -> None:
        """Adapt the rate to the KL of the last update.

        Args:
            kl: Mean approx. KL of the last PPO update.
        """
        if kl > 2.0 * self.kl_threshold:
            self.lr = max(self.lr / 1.5, self.min_lr)
        elif kl < 0.5 * self.kl_threshold:
            self.lr = min(self.lr * 1.5, self.max_lr)


class AdaptiveLRCallback(BaseCallback):
    """Feeds the last update's approx. KL into an `AdaptiveLR` schedule.

    SB3 order per iteration: collect rollouts -> dump logs -> train (records train/approx_kl). The KL of
    update k is therefore still in the logger at the start of rollout k+1, before update k+1 reads the rate.
    """

    def __init__(self, schedule: AdaptiveLR) -> None:
        """Bind the schedule.

        Args:
            schedule: The schedule passed to the model as `learning_rate`.
        """
        super().__init__()
        self.schedule = schedule
        self._seen_updates = 0

    def _on_rollout_start(self) -> None:
        """Adapt the rate once per completed update."""
        n = getattr(self.model, "_n_updates", 0)
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if n != self._seen_updates and kl is not None:
            self.schedule.update(float(kl))
            self._seen_updates = n

    def _on_step(self) -> bool:
        """No per-step work.

        Returns:
            True (continue training).
        """
        return True


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


def build_model(cfg: TrainCfg, venv, tb_dir: Path, actor_dim: int | None = None, learning_rate: Any = None):
    """Instantiate the SB3 model.

    Args:
        cfg: Training config.
        venv: Vectorized env.
        tb_dir: TensorBoard directory.
        actor_dim: Policy-part size of the observation (required when `cfg.asymmetric`).
        learning_rate: Rate or schedule (default `cfg.learning_rate`), e.g. an `AdaptiveLR`.

    Returns:
        A `PPO` or `RecurrentPPO` model.
    """
    arch = list(cfg.net_arch)
    if cfg.activation not in _ACTIVATIONS:
        raise ValueError(f"unknown activation '{cfg.activation}' ({' | '.join(_ACTIVATIONS)})")
    common = dict(n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                  learning_rate=cfg.learning_rate if learning_rate is None else learning_rate,
                  gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
                  clip_range_vf=cfg.clip_range_vf or None, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
                  max_grad_norm=cfg.max_grad_norm, target_kl=cfg.target_kl or None, seed=cfg.seed,
                  device=cfg.device, tensorboard_log=str(tb_dir), verbose=cfg.verbose)
    pk = dict(net_arch=dict(pi=arch, vf=arch), log_std_init=cfg.log_std_init,
              activation_fn=_ACTIVATIONS[cfg.activation])
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


# TrainCfg fields that define the network; they must match the run given by `init_from`.
_ARCH_FIELDS = ("algo", "asymmetric", "net_arch", "lstm_hidden_size", "n_lstm_layers", "activation")


def resolve_init(cfg: TrainCfg, run_root: Path) -> tuple[Path, Path | None]:
    """Locate the model zip and VecNormalize statistics of `cfg.init_from` and check the network matches.

    Args:
        cfg: Training config (`init_from` set).
        run_root: Absolute run root, searched for `init_from` by folder name.

    Returns:
        (model zip, VecNormalize pickle or None when the source run has none).
    """
    from mujoco_rl_bed.rl.evaluate import _find_run

    src = _find_run(cfg.init_from, str(run_root))
    conf = json.loads((src / "config.json").read_text())["train"]
    mine = to_jsonable(cfg)
    defaults = to_jsonable(TrainCfg())  # fields added after the source run was made have their default
    diff = [f"{k}: {conf.get(k, defaults[k])} (init_from) vs {mine[k]} (now)" for k in _ARCH_FIELDS
            if conf.get(k, defaults[k]) != mine[k]]
    if diff:
        raise ValueError("init_from network mismatch; use the source run's settings:\n  " + "\n  ".join(diff))
    ckpt = Path(cfg.init_checkpoint) if cfg.init_checkpoint else Path("final_model.zip")
    ckpt = ckpt if ckpt.is_absolute() else src / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"init_from checkpoint not found: {ckpt}")
    vn = src / "vecnormalize.pkl"
    m = re.search(r"model_(\d+)_steps\.zip$", ckpt.name)
    if m:
        vn = ckpt.parent / f"model_vecnormalize_{m.group(1)}_steps.pkl"
    return ckpt, (vn if vn.exists() else None)


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
    init_ckpt, init_vn = resolve_init(cfg, root) if cfg.init_from else (None, None)
    if init_ckpt is not None and (cfg.normalize_obs or cfg.normalize_reward) and init_vn is None:
        raise FileNotFoundError(f"init_from: no VecNormalize statistics next to {init_ckpt}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{cfg.task}_{stamp}_s{cfg.seed}{'_' + cfg.run_name if cfg.run_name else ''}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({
        "train": to_jsonable(cfg), "env": to_jsonable(env_cfg),
        "env_overrides": env_ov, "argv": argv, "task_modules": list(task_modules),
        "init_from": {"model": str(init_ckpt), "vecnormalize": str(init_vn)} if init_ckpt else None,
    }, indent=2))

    venv = make_vec_env(cfg.task, cfg.n_envs, cfg.seed, env_ov, vec=cfg.vec, task_modules=task_modules)
    if cfg.normalize_obs or cfg.normalize_reward:
        if init_vn is not None:
            # Keep the source run's running statistics (the loaded network expects them) and keep updating them.
            venv = VecNormalize.load(str(init_vn), venv)
            venv.training = True
            venv.norm_obs, venv.norm_reward = cfg.normalize_obs, cfg.normalize_reward
        else:
            venv = VecNormalize(venv, norm_obs=cfg.normalize_obs, norm_reward=cfg.normalize_reward, gamma=cfg.gamma)

    actor_dim = venv.get_attr("policy_obs_dim", indices=[0])[0]
    if cfg.lr_schedule not in ("constant", "adaptive"):
        raise ValueError(f"lr_schedule must be 'constant' or 'adaptive', got '{cfg.lr_schedule}'")
    schedule = AdaptiveLR(cfg.learning_rate, cfg.kl_threshold) if cfg.lr_schedule == "adaptive" else None
    model = build_model(cfg, venv, run_dir / "tb", actor_dim=actor_dim, learning_rate=schedule)
    if init_ckpt is not None:
        # Weights + optimizer state; hyperparameters (n_steps, learning rate, ...) come from this run's cfg.
        model.set_parameters(str(init_ckpt), exact_match=True, device=model.device)
        print(f"[train] initialized from {init_ckpt} (vecnormalize: {init_vn})")
    callbacks = CallbackList([
        EpisodeInfoCallback(),
        *([AdaptiveLRCallback(schedule)] if schedule is not None else []),
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

