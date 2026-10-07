# FORGE: command reference

All commands need the `dqn` conda env and work from any directory.

```bash
conda activate dqn
cd ~/mujoco_RL/RL_Mujoco/mujoco_rl_bed      # optional; paths below are relative to this folder
```

> **Always activate `dqn` first.** The plain `python` in a new shell is the base Miniconda
> interpreter, which fails with `ModuleNotFoundError: No module named 'mujoco_rl_bed'`.
> Check with `which python`: it should end in `envs/dqn/bin/python`.

Arguments are always `key=value` (no dashes). Lists or tuples must be quoted, e.g.
`net_arch="(256,128)"`. Unknown keys and wrongly typed values fail immediately with an
error that lists the valid keys.

---

## 0. View the scene in the MuJoCo viewer

```bash
python projects/forge/view.py                         # scripted expert inserting the peg (3 episodes)
python projects/forge/view.py mode=idle               # arm holds its reset pose: inspect the scene
python projects/forge/view.py mode=random             # random actions
python projects/forge/view.py run=<run_folder>        # a trained policy (folder name in projects/forge/runs/)
python projects/forge/view.py run=<run_folder> checkpoint=checkpoints/model_500000_steps.zip episodes=5
python projects/forge/view.py mode=scripted realtime=false   # run as fast as possible
```

- Close the window or press Ctrl+C to stop. Metrics are printed per episode.
- `view.py` accepts every `eval.py` key and environment override.
- Green/red/blue dots: hole tip, peg tip, hole floor. The TCP and flange sites are also shown.
- Viewer controls: double-click a body to select it, Ctrl+right-drag to push it, Tab to hide the side panel.
- Robot only (no task), to check the controller: `python scripts/view_scene.py` (see section 7).

Wayland desktops (Ubuntu default): the code automatically uses GLFW's X11 backend
(XWayland), because the Wayland backend hangs or crashes the viewer. A harmless
`GLFWError ... Wayland` warning means that fix was bypassed, e.g. `mujoco` was imported
before `mujoco_rl_bed` in your own script. In that case run with
`PYGLFW_LIBRARY_VARIANT=x11 python ...`.

## 1. Train with plain PPO (MLP)

```bash
python projects/forge/train.py algo=ppo n_envs=10 n_steps=256 batch_size=640 total_timesteps=3000000
```

- Runs are written to `projects/forge/runs/forge_peg_<YYYYmmdd-HHMMSS>_s<seed>/`.
- Ctrl+C stops training cleanly. `final_model.zip` and `vecnormalize.pkl` are still saved.
- `n_steps * n_envs` must be divisible by `batch_size` (here 256 × 10 = 2560 = 4 × 640).
- Keep `n_envs` ≤ number of CPU cores − 1 or 2 (this machine has 12 cores).

Short smoke run to check that everything starts (about 1 minute):

```bash
python projects/forge/train.py algo=ppo n_envs=4 total_timesteps=20000 run_name=smoke
```

## 2. Train with recurrent PPO (LSTM)

```bash
python projects/forge/train.py algo=recurrent_ppo n_envs=10 n_steps=256 batch_size=640 total_timesteps=3000000
```

Actor and critic see the same observation and each has its own LSTM (SB3 default).
This runs slower than plain PPO on CPU.

## 2b. Paper setup: recurrent PPO, asymmetric critic, hole-position noise

```bash
python projects/forge/train.py algo=recurrent_ppo asymmetric=true n_envs=4 n_steps=512 batch_size=512 \
    total_timesteps=3000000 log_std_init=-1.0 task.events.reset_fixed.params.pos_noise_std=0.0025 \
    run_name=rppo_asym_noise2p5
```

About 200–250 steps/s on this laptop (LSTM training on CPU dominates; `torch_threads=4` measured slower than 1).

## 2c. GPU machine (next run: paper clearance, smoothing on)

```bash
python projects/forge/train.py algo=recurrent_ppo asymmetric=true device=cuda n_envs=8 n_steps=512 batch_size=1024 \
    total_timesteps=5000000 log_std_init=-1.0 task.events.reset_fixed.params.pos_noise_std=0.001 \
    run_name=gpu_asym_n1_c025
```

- Clearance is the paper's 0.25 mm radial by default (no `inner_radius` override needed).
- Smoothing (`ema_factor=0.2`), the action-rate penalty (−0.02) and success-prediction weight
  (−0.1) are task defaults now.
- `pos_noise_std=0.001` is the 1 mm diagnostic noise; the paper uses `0.0025`.
- Set `n_envs` to about the number of CPU cores − 2 (the simulation runs on CPU; only the
  network uses the GPU) and keep `n_steps * n_envs` divisible by `batch_size`.
- Run it detached so it survives closing the terminal:
  `setsid -f python projects/forge/train.py ... > projects/forge/runs/gpu_run.log 2>&1 < /dev/null`

## 2d. Continue training a finished run (`init_from`)

Starts from a run's trained weights, optimizer state and observation-normalization statistics
instead of a fresh network. The new run gets its own folder and counts `total_timesteps` from 0.
Use the source run's network settings (`algo`, `asymmetric`, `net_arch`, LSTM size); training
hyperparameters (`n_steps`, `batch_size`, learning rate) and task overrides may change.
`log_std_init` is ignored (the trained std is loaded).

```bash
python projects/forge/train.py algo=recurrent_ppo asymmetric=true device=cpu \
    n_envs=24 n_steps=128 batch_size=768 total_timesteps=5000000 \
    task.events.reset_fixed.params.pos_noise_std=0.001 \
    init_from=<source_run_folder> run_name=ft
# from a specific checkpoint instead of final_model.zip:
#   init_from=<source_run_folder> init_checkpoint=checkpoints/model_3000000_steps.zip
```

## 3. TensorBoard

```bash
tensorboard --logdir projects/forge/runs
# open http://localhost:6006
```

To compare only some runs: `tensorboard --logdir projects/forge/runs/<run_folder>`.

Most useful scalars:

| Scalar | Meaning |
|---|---|
| `rollout/ep_rew_mean` | Mean episode return |
| `episode/is_success` | Fraction of episodes with the peg inserted at the final step |
| `episode/ever_success` | Fraction of episodes inserted at any step |
| `episode/ever_placed` | Fraction where the peg got centered and into the hole |
| `episode/time_to_success` | Seconds until first success (successful episodes only) |
| `episode/contact_force_mean`, `episode/contact_force_max` | Mean / max over the episode of the step-averaged peg contact force [N] (paper's F_mean, F_max). The force penalty uses the same step-averaged force. |
| `episode/contact_force_peak` | Max single-tick contact force [N], including short impact spikes (diagnostic only) |
| `episode/reward_terms/<name>` | Per-term weighted episode sums (`kp_coarse`, `kp_fine`, `place_bonus`, `success_bonus`, `contact_penalty`) |
| `episode/ik_err_max` | Non-zero only if the reset IK failed (should stay 0) |
| `train/approx_kl`, `train/clip_fraction`, `train/entropy_loss`, `train/explained_variance` | PPO health |
| `time/fps` | Training throughput (env steps/s) |

## 4. Evaluate a trained policy

```bash
# final model of a run (folder name inside projects/forge/runs/, or any path)
python projects/forge/eval.py run=forge_peg_20261006-130000_s0 episodes=20

# watch it in the viewer (same as view.py run=...)
python projects/forge/eval.py run=forge_peg_20261006-130000_s0 episodes=5 render=true

# a specific checkpoint
python projects/forge/eval.py run=forge_peg_20261006-130000_s0 checkpoint=checkpoints/model_1000000_steps.zip
```

Prints one line per episode, then a summary: `success_rate`, `ever_success_rate`,
`time_to_success_mean`, `contact_force_mean`, `contact_force_max_mean`, `return_mean`.
The task, algorithm, environment overrides and observation normalization are loaded
from the run folder automatically.

### Success prediction / early termination (paper Sec. III-C)

The policy has a 4th action a_ET, mapped to a predicted success probability p in [0, 1] and trained
with the reward -|p - y| (y = true success). Training always runs full episodes. To stop
episodes when the policy predicts success (deployment-style), evaluate with:

```bash
python projects/forge/eval.py run=<run_folder> episodes=20 \
    task.terminations.early_term.ends_episode=true task.terminations.early_term.params.p_term=0.9
```

The summary then also reports `et_precision` (fraction of early stops that were real successes),
`et_recall` (fraction of successful episodes that stopped correctly) and `et_delay_mean` (seconds
from success to stop), as in the paper's Table I. Runs trained before this change had 3D actions; evaluate
them with `task.action.success_prediction=false`.

## 5. Scripted expert (sanity check, no training)

```bash
python projects/forge/eval.py policy=scripted episodes=5 render=true
```

Uses privileged state (true hole and peg poses). It should succeed in every episode with
less than about 1 N of contact force. If it fails, the scene or contact settings are broken, not RL.

## 6. Tests

```bash
pytest projects/forge          # FORGE tests only (about 3 s); run pytest from mujoco_rl_bed/
pytest                         # everything (core + projects), from mujoco_rl_bed/
```

## 7. Robot / controller sanity check (core script)

```bash
python scripts/view_scene.py                    # impedance hold at the home pose (viewer)
python scripts/view_scene.py mode=gravcomp      # gravity compensation only, arm should stay still
python scripts/view_scene.py mode=zero          # zero torque, arm should fall
python scripts/view_scene.py headless=true duration=5   # no window, prints drift
```

---

## Training arguments (`train.py`)

| Key | Default | Meaning |
|---|---|---|
| `algo` | `ppo` | `ppo` (MLP) or `recurrent_ppo` (LSTM) |
| `asymmetric` | `false` | Asymmetric actor-critic (paper): the actor sees the 20 policy observations, the critic additionally gets privileged ground truth (true peg-tip offset from the hole, hole-estimate error). The run's `config.json` records it, so eval loads it automatically |
| `task` | `forge_peg` | Registered task name |
| `seed` | `0` | Run seed; env *i* uses `seed + i` |
| `n_envs` | `8` | Parallel environments (one CPU process each) |
| `vec` | `subproc` | `subproc` (parallel processes) or `dummy` (single process, for debugging) |
| `total_timesteps` | `50000` | Total environment steps (set this explicitly) |
| `device` | `auto` | Network device: `auto`, `cpu`, `cuda` (no GPU on this machine, so it runs on CPU) |
| `torch_threads` | `1` | `torch.set_num_threads` for the learner (0 = PyTorch default = one per core). 1 is recommended for these small networks on CPU; more threads steal cores from the env workers |
| `n_steps` | `256` | Rollout length per env per update |
| `batch_size` | `512` | Minibatch size; must divide `n_steps * n_envs` |
| `n_epochs` | `5` | Optimization passes per update |
| `learning_rate` | `3e-4` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE λ |
| `clip_range` | `0.2` | PPO clip range |
| `ent_coef` | `0.0` | Entropy bonus |
| `vf_coef` | `0.5` | Value loss weight |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `target_kl` | `0.0` | Stop an update early when KL exceeds this (0 = off) |
| `log_std_init` | `-0.5` | Initial log std of the action distribution |
| `net_arch` | `(256,128)` | Hidden layers of actor and critic heads |
| `lstm_hidden_size` | `256` | LSTM width (`recurrent_ppo` only) |
| `n_lstm_layers` | `1` | LSTM depth (`recurrent_ppo` only) |
| `normalize_obs` | `true` | Running observation normalization (saved as `vecnormalize.pkl`) |
| `normalize_reward` | `true` | Running return normalization |
| `checkpoint_every` | `100000` | Env steps between checkpoints |
| `run_root` | `projects/forge/runs` | Where run folders are created |
| `run_name` | `""` | Suffix appended to the run folder name |
| `init_from` | `""` | Run to continue from (folder name in `run_root` or a path): loads its weights, optimizer state and VecNormalize statistics (section 2d) |
| `init_checkpoint` | `""` | Model zip inside `init_from` (default `final_model.zip`), e.g. `checkpoints/model_3000000_steps.zip` |
| `verbose` | `1` | SB3 console output (0 = quiet) |

## Evaluation arguments (`eval.py`)

| Key | Default | Meaning |
|---|---|---|
| `run` | `""` | Run folder name (inside `projects/forge/runs/`) or path |
| `checkpoint` | `""` | Model zip relative to the run folder (default `final_model.zip`) |
| `policy` | `checkpoint` | `checkpoint` or `scripted` |
| `episodes` | `10` | Number of episodes |
| `seed` | `1000` | Episode *i* uses `seed + i` (different from training seeds) |
| `deterministic` | `true` | Use the mean action |
| `render` | `false` | Open the viewer |
| `realtime` | `true` | Pace the viewer to wall time |
| `device` | `cpu` | Device for the policy network |

## Environment / task overrides (train and eval)

Any key that is not a training/eval argument is applied to the environment config. Useful ones:

| Key | Default | Meaning |
|---|---|---|
| `sim_dt` | `0.002` | Physics timestep [s] |
| `decimation` | `25` | Physics steps per policy step (policy rate = 1 / (sim_dt × decimation) = 20 Hz) |
| `task.episode_length_s` | `7.5` | Episode length [s] (150 policy steps at 20 Hz; paper: 10 s) |
| `task.action.max_step` | `0.02` | λ: max target distance from the TCP per axis [m] |
| `task.action.ema_factor` | `0.2` | Action smoothing α (1.0 = off): applied = α·a + (1−α)·previous. Removes jitter (−76% TCP jerk vs. off) |
| `task.action.anchor_bounds` | `(0.05,0.05,0.05)` | Action range around the anchor (TCP pose with the peg tip at the hole opening; zero action = there) [m] |
| `task.controller.kp` | `(600,600,600,100,100,100)` | Cartesian stiffness (x, y, z [N/m], rx, ry, rz [Nm/rad]) |
| `task.rewards.<name>.weight` | see `task.py` | Reward weights: `kp_coarse`, `kp_fine`, `place_bonus`, `success_bonus`, `contact_penalty` (−β = −0.2), `success_pred` (−0.1), `action_rate` (−0.02, penalizes action changes) |
| `task.scene.assets.1.inner_radius` | `0.00425` | Hole radius [m]; peg radius is 4 mm, so 0.00425 = 0.25 mm radial clearance (paper), 0.0045 = 0.5 mm |
| `task.events.reset_fixed.params.lo` / `.hi` | `(0.55,-0.05,0.0)` / `(0.65,0.05,0.1)` | Socket position range [m] |
| `task.events.reset_fixed.params.pos_noise_std` | `0.0` | Hole-position estimate noise [m] (paper: 0.0025) |
| `task.events.reset_threshold.params.lo` / `.hi` | `5.0` / `10.0` | Force threshold range F_th [N] |
| `task.events.reset_ee.params.xy_range` | `0.02` | Initial TCP lateral offset from the hole [m] |
| `task.events.reset_ee.params.z_range` | `(0.037,0.057)` | Initial TCP height above the hole tip [m] |
| `render` | `false` | Viewer (forced off when `n_envs > 1`) |

Example: paper-length episodes, a softer penalty, and hole-position noise:

```bash
python projects/forge/train.py algo=ppo n_envs=10 total_timesteps=3000000 \
    task.episode_length_s=10 task.rewards.contact_penalty.weight=-0.1 \
    task.events.reset_fixed.params.pos_noise_std=0.0025 run_name=noise2p5
```

## Run folder contents

```
projects/forge/runs/forge_peg_<date>_s<seed>[_<run_name>]/
├── config.json        # every resolved setting + the CLI overrides (reproducibility)
├── tb/                # TensorBoard logs
├── checkpoints/       # model_<N>_steps.zip + model_vecnormalize_<N>_steps.pkl
├── final_model.zip
└── vecnormalize.pkl   # observation normalization stats (eval loads it automatically)
```

Resuming training from a checkpoint is not supported yet. A new run always starts from scratch.
