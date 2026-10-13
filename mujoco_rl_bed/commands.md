# Commands

Training and evaluation of the two learned peg-in-hole policies on the Franka MuJoCo test bed, and the comparison
against the scripted spiral and TacDiffusion:

- **FORGE** (`projects/forge`): end-to-end force-conditioned RL policy (position actions, LSTM, asymmetric critic).
- **Spiral residual RL** (`projects/compliant_RL`): a scripted spiral search on the axis compliance controller plus a
  learned "try insert" decision (pause the spiral, soft gains, 5 N push when the peg is caught on the hole edge).
  Background and results: `projects/compliant_RL/RL_NOTES.md`.

## 0. Setup

First-time setup (new machine): see "Quick start" in the repository README (conda env `mjtest`, pinned packages,
`scripts/fetch_models.sh` for the trained models, `git submodule update --init` for TacDiffusion).

```bash
conda activate mjtest                    # every command needs it; `which python` must end in envs/mjtest/bin/python
cd RL_Mujoco/mujoco_rl_bed               # all paths below are relative to this folder
pip install -e ".[tacdiff,dev]"          # once, and again after pulling new projects
bash scripts/fetch_models.sh             # once: trained models (GitHub release models-v1) into projects/*/runs/
python scripts/system_check.py           # optional: cores, RAM, GPU, sim steps/s vs n_envs (quick=true: ~3 min)
```

- MuJoCo is pinned to 3.10.0 (`requirements-lock.txt`): 3.15 changes the contact forces in these tasks 3-4x.

- Arguments are `key=value` (no dashes). Lists/tuples: quote them, e.g. `net_arch="[64,64]"`.
- Unknown keys fail immediately with the list of valid keys.
- Keep `n_envs` ≤ CPU cores − 2 (this laptop: 12 cores → 10) and `n_steps × n_envs` divisible by `batch_size`.
- Long runs: start them detached so they survive the terminal, e.g. in `screen` (tmux is not installed):
  `screen -dmS train bash -c "python projects/... > projects/<project>/runs/<name>.log 2>&1"`, then
  `screen -r train` to attach (Ctrl+A D to detach). Ctrl+C (or `kill -INT <pid>`) stops training cleanly and still
  saves `final_model.zip` and `vecnormalize.pkl`.

Kept models:

| Policy | Run folder | Notes |
|---|---|---|
| FORGE | `projects/forge/runs/jitter_but_works` | Cylinder, 0.5 mm clearance, sampled actions |
| Spiral catch policy | `projects/compliant_RL/runs/compliant_peg_s2g_20261012-122716_s0_s2g_bc` | Warm start (imitated catch rule): 51/60 at 0.25 mm vs 49/60 plain spiral |

FORGE and spiral models: `bash scripts/fetch_models.sh` (they are not in git).
| TacDiffusion | `../diffusion/TacDiffusion/output/TacDiffusion_model_256.onnx` | Authors' model (git submodule; the benchmark uses 256, the 512 model is ~3x slower) |

---

## 1. Evaluate and compare

One script compares every method on identical seeds; its settings are in `projects/benchmark/compare.yaml`
(shapes, episodes, clearance, hole error, tilt, friction, models, spiral and FORGE settings, viewer). Every key can be
overridden on the command line with dotted keys and YAML values:

```bash
python projects/benchmark/compare.py                                        # everything in compare.yaml
python projects/benchmark/compare.py config=my_config.yaml                  # a copy with your settings
python projects/benchmark/compare.py episodes=60 methods=[spiral,spiral_rl] \
    conditions.shapes=[cylinder,cuboid,hexagon,octagon] conditions.clearance_mm=[0.1,0.2] \
    conditions.hole_error_mm=["0-4"] conditions.tilt_deg=["0-8"] conditions.friction=["0.5-1.0"]
python projects/benchmark/compare.py methods=[spiral_rl] models.spiral_rl_run=projects/compliant_RL/runs/<run> \
    models.spiral_rl_checkpoint=checkpoints/model_1000000_steps.zip        # evaluate a new spiral checkpoint
python projects/benchmark/compare.py methods=[forge] models.forge_run=projects/forge/runs/<run>
python projects/benchmark/compare.py out=projects/benchmark/results/compare_<time>   # resume an interrupted run
```

- Methods: `spiral` (plain), `spiral_rule` (scripted catch rule), `spiral_rl` (learned catch policy), `forge`, `tacdiff`.
- Shapes: `cylinder`, `cuboid`, `hexagon`, `octagon` (8 mm across). Levels: `"a"` exact, `"a-b"` uniform.
- Success = peg tip within 1 mm of the floor and 2.5 mm of the axis at any time (the episode stops there).
- Output: `projects/benchmark/results/compare_<time>/summary.md` (success with 95 % CI, time, F_max, failures split
  into not found / jammed per method and condition), `summary.csv`, `episodes.jsonl`, `config.yaml`.
- `workers` parallel processes; TacDiffusion runs at 1 kHz and takes ~15 s per episode.
- 0.1 mm clearance: hexagon and octagon pegs can be squeezed back out of the hole after seating (contact softness
  wider than the gap); the stop-at-first-success rule counts them as inserted.

**Watch a method in the viewer** (one condition after another, nothing saved):

```bash
python projects/benchmark/compare.py view.enabled=true view.method=spiral_rl episodes=10 \
    conditions.shapes=[hexagon] conditions.clearance_mm=[0.2]
python projects/benchmark/compare.py view.enabled=true view.method=forge view.realtime=false
```

**Single-project tools:**

```bash
python projects/forge/view.py run=jitter_but_works episodes=5             # FORGE policy on its own task
python projects/forge/view.py mode=scripted                               # privileged scripted expert (sanity)
python projects/forge/eval.py run=<run_folder> episodes=20                # FORGE task metrics; render=true to watch
python projects/forge/eval.py run=<run_folder> episodes=20 \
    task.terminations.early_term.ends_episode=true task.terminations.early_term.params.p_term=0.9   # early stop
python projects/compliant_RL/eval.py run=<run_folder> episodes=20         # compliant task metrics
python projects/tacdiff/run.py mode=offline                               # TacDiffusion weights on the test set
python projects/tacdiff/run.py episodes=5                                 # TacDiffusion in the viewer
python scripts/view_scene.py                                              # robot/controller only (also mode=gravcomp|zero)
```

## 2. Train FORGE

The `forge_peg` task defaults are the paper / Isaac Lab FORGE environment: 15 Hz, 150-step episodes, gain/λ/EMA
randomization, controller dead zone, friction U(0.5, 1.0), peg offset in the hand, yaw, observation noise, 1 mm
hole-pose noise, delayed success-prediction penalty, 0.25 mm radial clearance.

**CPU (this laptop), recurrent PPO with asymmetric critic:**

```bash
python projects/forge/train.py algo=recurrent_ppo asymmetric=true n_envs=10 n_steps=256 batch_size=1280 \
    total_timesteps=3000000 log_std_init=-1.0 gamma=0.995 run_name=forge_cpu
```

About 200–250 steps/s (LSTM updates on CPU dominate); `torch_threads` above 1 is slower.

**GPU machine, Isaac Lab FORGE settings** (LSTM 2×1024, ELU MLP 512-128-64, KL-adaptive learning rate):

```bash
python projects/forge/train.py algo=recurrent_ppo asymmetric=true device=cuda n_envs=32 \
    n_steps=128 batch_size=512 n_epochs=4 learning_rate=1e-4 lr_schedule=adaptive kl_threshold=0.008 \
    gamma=0.995 gae_lambda=0.95 clip_range=0.2 clip_range_vf=0.2 vf_coef=1.0 ent_coef=0.0 \
    log_std_init=0.0 lstm_hidden_size=1024 n_lstm_layers=2 net_arch="(512,128,64)" activation=elu \
    total_timesteps=20000000 run_name=isaac
```

Pick `n_envs` from `scripts/system_check.py` (simulation runs in CPU processes; only the network uses the GPU).

**Common FORGE variations** (any env override can be appended):

```bash
... task.events.reset_fixed.params.pos_noise_std=0.0025       # paper's 2.5 mm hole-pose noise
... task.scene.assets.1.inner_radius=0.0045                   # 0.5 mm radial clearance (jitter_but_works)
... task.rewards.contact_penalty.weight=-0.1                  # softer force penalty
python projects/forge/train.py algo=ppo n_envs=4 total_timesteps=20000 run_name=smoke   # 1-minute smoke run
```

**Continue a run** (weights, optimizer state and normalization statistics; the new run counts from 0; the network
settings must match the source run, `log_std_init` is ignored):

```bash
python projects/forge/train.py algo=recurrent_ppo asymmetric=true n_envs=10 n_steps=256 batch_size=1280 \
    total_timesteps=5000000 init_from=<source_run_folder> run_name=ft
#   from a checkpoint: init_from=<source_run_folder> init_checkpoint=checkpoints/model_3000000_steps.zip
```

Runs go to `projects/forge/runs/forge_peg_<date>_s<seed>[_<run_name>]/`.

## 3. Train the spiral residual RL policy (catch-and-insert)

Task `compliant_peg_s2g`: cylinder, 0.25 mm clearance, hole-estimate error U(0, 4 mm), start tilt U(0, 8°), friction
U(0.5, 1.0), 20 s episodes. Base: approach at 1.5 N, Archimedean spiral (pitch 0.8 mm, 10 mm/s, out to 4.6 mm), drop
detection, soft insertion at 5 N. Action: try-insert level (dead zone below 0.3; 1 = pause the spiral, insert gains,
5 N). Actor (19, LSTM): contact wrench (noise 1 N / 0.05 Nm), spring deflection, TCP lateral velocity and roll/pitch,
spiral offset, depth, phase, try state, previous action. Critic: + true tip offset, peg axis, success, friction,
contact force, episode progress.

PPO from scratch learns to try everywhere and gets worse than the plain spiral, so training has two steps.

**Step 1: warm start** (imitate the scripted catch rule; ~5 min; creates `compliant_peg_s2g_<date>_s0_s2g_bc`):

```bash
python projects/compliant_RL/bc_warmstart.py
python projects/compliant_RL/bc_warmstart.py task=compliant_peg_s2h run_name=s2h_bc   # 0.1 mm clearance variant
```

Prints the held-out imitation quality (try recall/precision, critic R²) and the command for step 2.

**Step 2: PPO fine-tuning from the warm start** (~560 steps/s with 10 envs):

```bash
screen -dmS spiral bash -c "python projects/compliant_RL/train.py task=compliant_peg_s2g algo=recurrent_ppo \
    asymmetric=true net_arch='[64,64]' lstm_hidden_size=128 n_envs=10 n_steps=256 batch_size=1280 n_epochs=5 \
    learning_rate=1e-4 gamma=0.995 total_timesteps=3000000 \
    init_from=<warm_start_run_folder> run_name=spiral_ft > projects/compliant_RL/runs/spiral_ft.log 2>&1"
```

- The network keys (`algo`, `asymmetric`, `net_arch`, `lstm_hidden_size`) must match the warm start.
- Judge it on fixed seeds (section 1), not on the training success rate, which includes exploration noise.
- Watch `episode/try_onsets` and `episode/try_true_frac`: tries rising while the true fraction stays ~15 % means PPO
  is drifting toward blanket tries. That happened in runs H and I (see `RL_NOTES.md`): the warm start stayed best at
  0.25 mm. Stop when the fixed-seed evaluation falls below the warm start.

**Spiral-task variations** (append to step 1 and step 2 alike so both see the same task):

| Override | Default (s2g) | Meaning |
|---|---|---|
| `task=compliant_peg_s2h` | | 0.1 mm clearance, error U(0, 5 mm), spiral to 5.6 mm, + raw try-action cost |
| `task.scene.assets.1.inner_radius=0.0042` | `0.00425` | Hole radius [m] (peg 4 mm) → 0.2 mm clearance |
| `task.events.shift_ee.params.offset_max=0.005` | `0.004` | Max hole-estimate error [m] (also `offset_min`) |
| `task.events.tilt_ee.params.tilt_max_deg=5` | `8` | Max start tilt [deg] |
| `task.events.randomize_friction.params.lo=0.3` / `.hi` | `0.5` / `1.0` | Friction range |
| `task.action.params.r_max=0.0056` | `0.0046` | Spiral radius [m] (~0.6 mm above the max error) |
| `task.action.params.pitch=0.0006` | `0.0008` | Spiral pitch [m] |
| `task.action.params.f_search=1.0` | `1.5` | Search push [N] (0.5 N fails to drop; 1.5 N best) |
| `task.action.params.f_try=5.0` | `5.0` | Push during a try [N] |
| `task.rewards.try_cost.weight=-0.5` | `-0.5` | Cost per step of a full try |
| `task.episode_length_s=25` | `20` | Episode length [s] |

Runs go to `projects/compliant_RL/runs/<task>_<date>_s<seed>[_<run_name>]/`.

## 4. Monitor training

```bash
python scripts/watch_run.py run=projects/compliant_RL/runs/<run_folder> every=250000 interval=60
python scripts/watch_run.py run=latest                                     # newest FORGE run
tensorboard --logdir projects/compliant_RL/runs                             # or projects/forge/runs; http://localhost:6006
```

Useful scalars:

| Scalar | Meaning |
|---|---|
| `rollout/ep_rew_mean` | Mean episode return |
| `episode/is_success`, `episode/ever_success` | Inserted at the final step / at any step |
| `episode/time_to_success` | Seconds to first success (successful episodes) |
| `episode/contact_force_mean`, `contact_force_max` | Step-averaged peg contact force [N] (FORGE's F_mean, F_max) |
| `episode/reward_terms/<name>` | Per-term weighted episode sums |
| `episode/try_onsets`, `try_steps`, `try_true_frac` | Spiral RL: tries per episode, try steps, fraction of try steps with the tip within 0.5 mm of the axis |
| `episode/success_pred_scale`, `et_correct`, `et_delay` | FORGE success prediction / early termination |
| `episode/ctrl_kp`, `ctrl_lambda`, `ctrl_ema`, `dead_zone_force`, `part_friction` | FORGE randomized dynamics |
| `train/approx_kl`, `clip_fraction`, `explained_variance`, `std` | PPO health |
| `time/fps` | Env steps/s |

## 5. Tests

```bash
pytest                                  # everything (core + projects), from mujoco_rl_bed/
pytest projects/forge                   # FORGE only (~3 s)
pytest projects/compliant_RL            # compliant / spiral tasks
```

---

## Training arguments (`train.py`, both projects)

| Key | Default | Meaning |
|---|---|---|
| `task` | `forge_peg` / `compliant_peg` | Registered task |
| `algo` | `ppo` | `ppo` (MLP) or `recurrent_ppo` (LSTM) |
| `asymmetric` | `false` | Actor sees the policy observations, critic gets clean privileged state |
| `seed` | `0` | Run seed; env *i* uses `seed + i` |
| `n_envs` | `8` | Parallel environments (one CPU process each) |
| `vec` | `subproc` | `subproc` or `dummy` (single process, debugging) |
| `total_timesteps` | `50000` | Total env steps (set it explicitly) |
| `device` | `auto` | `auto`, `cpu`, `cuda` |
| `torch_threads` | `1` | Learner threads (1 is fastest for these networks on CPU) |
| `n_steps` | `256` | Rollout length per env per update |
| `batch_size` | `512` | Minibatch size (transitions); must divide `n_steps × n_envs` |
| `n_epochs` | `5` | Passes per update |
| `learning_rate` | `3e-4` | Adam learning rate (start value with `lr_schedule=adaptive`) |
| `lr_schedule`, `kl_threshold` | `constant`, `0.008` | `adaptive`: ÷1.5 if KL > 2×threshold, ×1.5 if KL < threshold/2 |
| `gamma`, `gae_lambda` | `0.99`, `0.95` | Discount, GAE λ |
| `clip_range`, `clip_range_vf` | `0.2`, `0` | PPO clip, value clip (0 = off) |
| `ent_coef`, `vf_coef`, `max_grad_norm`, `target_kl` | `0`, `0.5`, `1.0`, `0` | |
| `log_std_init` | `-0.5` | Initial log std (ignored with `init_from`) |
| `net_arch`, `activation` | `(256,128)`, `tanh` | MLP heads; `elu`, `relu` also available |
| `lstm_hidden_size`, `n_lstm_layers` | `256`, `1` | LSTM size (`recurrent_ppo`) |
| `normalize_obs`, `normalize_reward` | `true`, `true` | Running normalization (saved as `vecnormalize.pkl`) |
| `checkpoint_every` | `100000` | Env steps between checkpoints |
| `run_root`, `run_name` | project `runs/`, `""` | Run folder location and name suffix |
| `init_from`, `init_checkpoint` | `""` | Continue from a run (folder name or path) / a checkpoint zip inside it |

## Evaluation arguments (`projects/*/eval.py`)

| Key | Default | Meaning |
|---|---|---|
| `run`, `checkpoint` | `""` | Run folder (name or path); model zip relative to it (default `final_model.zip`) |
| `policy` | `checkpoint` | `checkpoint` or `scripted` |
| `episodes`, `seed` | `10`, `1000` | Episode *i* uses `seed + i` |
| `deterministic` | `true` | Mean action |
| `render`, `realtime`, `device` | `false`, `true`, `cpu` | Viewer, wall-clock pacing, policy device |

## FORGE environment overrides (train and eval)

| Key | Default | Meaning |
|---|---|---|
| `sim_dt`, `decimation` | `0.002`, `33` | Physics step [s], physics steps per policy step (15.15 Hz) |
| `task.episode_length_s` | `9.9` | Episode length [s] (150 policy steps) |
| `task.action.max_step`, `task.action.ema_factor` | `0.02`, `0.0625` | Initial λ [m], smoothing α (randomized per episode) |
| `task.controller.kp` | `(565,565,565,28,28,28)` | Nominal stiffness (randomized per episode) |
| `task.rewards.<name>.weight` | see `forge/task.py` | `kp_baseline`, `kp_coarse`, `kp_fine`, `place_bonus`, `success_bonus`, `contact_penalty` (−0.2), `success_pred` (−1), `action_grad` (−0.1), `action_asset` (−0.001) |
| `task.scene.assets.1.inner_radius` | `0.00425` | Hole radius [m] (peg 4 mm → 0.25 mm clearance) |
| `task.events.reset_fixed.params.pos_noise_std` | `0.001` | Hole-position estimate noise [m] |
| `task.events.reset_fixed.params.lo` / `.hi` | `(0.55,-0.05,0.0)` / `(0.65,0.05,0.1)` | Socket position range [m] |
| `task.events.reset_threshold.params.lo` / `.hi` | `5.0` / `10.0` | Force threshold F_th range [N] |
| `task.events.reset_ee.params.xy_range` / `z_range` / `yaw_range` | `0.02` / `(0.037,0.057)` / `0.785` | Initial TCP offset [m] and yaw [rad] |
| `task.events.randomize_controller.params.kp_noise` / `lam_noise` / `ema_range` | `(0.41,)*6` / `(0.25,)*3` / `(0.025,0.1)` | Gain, λ, α randomization |
| `task.events.dead_zone.params.max_dz` | `(5,5,5,1,1,1)` | Wrench dead zone bound [N, Nm] |
| `task.events.randomize_friction.params.lo` / `.hi` | `0.5` / `1.0` | Part friction range |
| `task.events.randomize_held.params.lo` / `.hi` | `(-0.003,0,-0.003)` / `(0.003,0,0.003)` | Peg offset in the hand [m] |
| `task.obs.term_cfg.<term>.noise_std` | see `forge/task.py` | Actor observation noise |
| `task.events.init.params.delay_until_ratio` | `0.25` | Success rate that switches the success-prediction penalty on |

## Run folder contents

```
projects/<project>/runs/<task>_<date>_s<seed>[_<run_name>]/
├── config.json        # every resolved setting + the CLI overrides
├── tb/                # TensorBoard logs
├── checkpoints/       # model_<N>_steps.zip + model_vecnormalize_<N>_steps.pkl
├── final_model.zip
└── vecnormalize.pkl   # observation normalization statistics (eval loads them automatically)
```
