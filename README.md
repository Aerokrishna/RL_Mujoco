# RL_Mujoco

A modular MuJoCo test bed for training, testing and comparing contact-rich manipulation policies on a
torque-controlled Franka Panda. The benchmark task is peg-in-hole insertion: an 8 mm peg goes into a 25 mm deep hole
whose position is only known approximately. Every policy runs in the same scene, and one script compares them on
identical episodes.

- **Commands** for setup, training, evaluation and comparison: [`mujoco_rl_bed/commands.md`](mujoco_rl_bed/commands.md)
- **Benchmark results**: [`mujoco_rl_bed/projects/benchmark/benchmark_comparisons.md`](mujoco_rl_bed/projects/benchmark/benchmark_comparisons.md)
  (raw per-episode results of any run: `mujoco_rl_bed/projects/benchmark/results/<run>/`)
- **Design notes for the spiral residual RL** (what was tried and why): [`mujoco_rl_bed/projects/compliant_RL/RL_NOTES.md`](mujoco_rl_bed/projects/compliant_RL/RL_NOTES.md)

## Quick start: reproduce the benchmark

Needs Linux, conda, git and ~3 GB of disk (~6 GB with CUDA PyTorch). The trained models are not in git: they are
downloaded from the [`models-v1` release](https://github.com/Aerokrishna/RL_Mujoco/releases/tag/models-v1).
TacDiffusion is a git submodule (the authors' repository with their models).

```bash
git clone --recursive -b torque-rl-forge https://github.com/Aerokrishna/RL_Mujoco.git
cd RL_Mujoco/mujoco_rl_bed
conda create -n mjtest python=3.10 -y && conda activate mjtest
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch (smaller)
pip install -r requirements-lock.txt && pip install -e ".[tacdiff,dev]"
bash scripts/fetch_models.sh             # FORGE + spiral RL models into projects/*/runs/
python -m pytest -q                      # optional; one known failure: test_scripted_corrector_with_randomization
```

Already cloned without `--recursive`? Run `git submodule update --init` in `RL_Mujoco/`.

Smoke test (~3 min, 2 episodes per method on one condition):

```bash
python projects/benchmark/compare.py config=projects/benchmark/benchmark_comparison.yaml episodes=2 workers=3 \
    "conditions.shapes=[cylinder]" "conditions.clearance_mm=[0.5]" "conditions.hole_error_mm=['2']" \
    "conditions.tilt_deg=['0']" "conditions.friction=['0.4']" out=projects/benchmark/results/smoke
```

It prints one line per method and a summary table (`results/smoke/summary.md`). Full comparison (48 conditions x
20 episodes; Spiral+RL and FORGE take ~20 min with 4 workers, TacDiffusion several hours):

```bash
python projects/benchmark/compare.py config=projects/benchmark/benchmark_comparison.yaml workers=4
```

- **Memory:** each worker needs ~1 GB of RAM; keep `workers` at or below (RAM in GB) / 1.5 and below the core count.
- **MuJoCo is pinned to 3.10.0.** Newer versions (3.15) change the contact forces in these tasks 3-4x, and the
  numbers no longer match.
- A run writes into `out` and resumes from there if interrupted; `methods=[spiral_rl,forge]` skips TacDiffusion.
- Our results for this config: `projects/benchmark/results/benchmark_comparison/` (episodes, `summary.md`).

## Policies

| Policy | Type | Where |
|---|---|---|
| **FORGE** | End-to-end RL: recurrent PPO with an asymmetric critic. It outputs position targets for a Cartesian impedance controller and is conditioned on a contact-force threshold (Noseworthy et al., *FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation under Uncertainty*, arXiv:2408.04587). | `projects/forge` |
| **Spiral residual RL** | Hybrid. A scripted Archimedean spiral search runs on an axis-compliance controller (force control along the peg, compliant sideways and in tilt). A learned LSTM policy decides when to "try insert": pause the spiral, soften the gains and push harder when the peg catches on the hole edge. It is trained in two steps: imitation of a scripted catch rule, then PPO. | `projects/compliant_RL` |
| **Scripted spiral / catch rule** | Non-learned baselines on the same controller. The catch rule tries to insert when the sideways contact force stays above 3.5 N for 3 steps. | `projects/compliant_RL` |
| **TacDiffusion** | Pretrained force-domain diffusion policy, used as published (no training here). See below. | `projects/tacdiff` |

Trained models (downloaded by `scripts/fetch_models.sh`, not in git): FORGE `projects/forge/runs/jitter_but_works` and
spiral RL `projects/compliant_RL/runs/compliant_peg_s2g_20261012-122716_s0_s2g_bc` (paths under `mujoco_rl_bed/`).

### TacDiffusion

Wu et al., *TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation*, arXiv:2409.11047
([code](https://github.com/popnut123/TacDiffusion)).
- **What it is:** a conditional denoising diffusion model that outputs a 6-D wrench (force and torque) command, not a
  pose. It was trained on demonstrations of an expert insertion skill on a real Franka.
- **Input:** the external wrench, the commanded wrench and the tool twist, now and 7 ms earlier.
- **Smoothing:** a dynamic-system filter bridges the gap between the model's inference rate (~140 Hz for the
  512-unit model) and the 1 kHz control loop.
- **Reported results:** 95.7 % zero-shot success on new parts on the real robot.

Here, `projects/tacdiff/run.py` runs the authors' model in the MuJoCo scene: a scripted approach to the estimated
hole, a guarded descent, then the diffusion policy at 1 kHz. The authors' repository is the git submodule
`diffusion/TacDiffusion` (pinned commit; their ONNX models are in its `output/`). The benchmark uses the 256-unit model
(14 ms per inference on a laptop CPU; the 512-unit model takes 45 ms).

## Layout

```
RL_Mujoco/
├── diffusion/TacDiffusion/         # git submodule: the authors' TacDiffusion repo, models in output/
└── mujoco_rl_bed/
    ├── commands.md                 # all commands
    ├── src/mujoco_rl_bed/          # core framework
    │   ├── sim/                    #   scene, Franka plant, assets (peg/hole shapes), IK, viewer
    │   ├── control/                #   joint / Cartesian impedance, axis compliance controllers
    │   ├── env/                    #   TorqueEnv + config dataclasses + action/obs/reward/event/termination managers
    │   ├── terms/                  #   generic observation / reward / event terms
    │   └── rl/                     #   PPO / recurrent PPO training (asymmetric actor-critic), evaluation
    ├── projects/
    │   ├── forge/                  # FORGE task, terms, train/eval/view
    │   ├── compliant_RL/           # spiral residual RL tasks, terms, catch rule, warm start, train/eval
    │   ├── tacdiff/                # TacDiffusion runner
    │   └── benchmark/              # comparison script, configs, results
    ├── scripts/                    # system check, training monitor, robot viewer
    └── tests/                      # core tests (project tests live in projects/*/tests)
```

## Important scripts

Run from `mujoco_rl_bed/` with the `mjtest` conda env (setup: quick start above). Arguments are `key=value`; any other key
is an environment override (e.g. `task.events.randomize_friction.params.lo=0.3`). Full lists: `commands.md`.

| Script | Purpose | Important arguments |
|---|---|---|
| `projects/benchmark/compare.py` | Compare spiral, spiral+rule, spiral RL, FORGE and TacDiffusion on identical seeds; writes `summary.md` / `.csv`. With `view.enabled=true` it shows one method in the viewer instead. | `config=<yaml>` (default `compare.yaml`), `methods=[...]`, `episodes`, `conditions.shapes` (cylinder, cuboid, hexagon, octagon), `conditions.clearance_mm`, `conditions.hole_error_mm`, `conditions.tilt_deg`, `conditions.friction` (`"a"` exact or `"a-b"` range), `models.*`, `view.method` |
| `projects/forge/train.py` | Train FORGE | `algo` (`ppo`/`recurrent_ppo`), `asymmetric`, `n_envs`, `n_steps`, `batch_size`, `total_timesteps`, `learning_rate`, `lstm_hidden_size`, `net_arch`, `init_from`, `run_name` |
| `projects/forge/eval.py`, `view.py` | Evaluate / watch a FORGE run | `run`, `checkpoint`, `episodes`, `render`, `mode` (`scripted`/`idle`/`random`, view only) |
| `projects/compliant_RL/bc_warmstart.py` | Spiral RL step 1: imitate the catch rule into the LSTM policy | `task` (default `compliant_peg_s2g`), `episodes_per_env`, `epochs`, `run_name` |
| `projects/compliant_RL/train.py` | Spiral RL step 2: PPO fine-tuning (same keys as FORGE's `train.py`) | `task=compliant_peg_s2g`, `init_from=<warm start run>`, `learning_rate` (1e-4), `total_timesteps` |
| `projects/compliant_RL/eval.py` | Evaluate a spiral RL run on its own task | `run`, `checkpoint`, `episodes`, `render` |
| `projects/tacdiff/run.py` | Run TacDiffusion alone (viewer or batch), or check its weights offline | `mode` (`sim`/`offline`), `episodes`, `model`, `xy_err`, `render` |
| `scripts/watch_run.py` | Print training progress from TensorBoard logs | `run`, `every`, `interval` |

## Where to change the training setup

| To change | FORGE | Spiral residual RL |
|---|---|---|
| Task definition: observations, rewards and weights, randomization ranges, episode length, controller gains | `projects/forge/task.py` (`forge_peg`) | `projects/compliant_RL/task.py` (`compliant_peg_s2g`; `s2h` for 0.1 mm clearance) |
| Term implementations (reward, observation, reset event functions) | `projects/forge/terms.py` | `projects/compliant_RL/terms.py` |
| Action: what the policy commands | `anchor_relative_pos` action term (core, `src/mujoco_rl_bed/env/managers/action.py`) | `spiral_residual` action term in `projects/compliant_RL/terms.py` (spiral base, try insert) |
| Controller | `src/mujoco_rl_bed/control/cartesian_impedance.py` | `src/mujoco_rl_bed/control/axis_compliance.py` |
| Scripted teacher | `projects/forge/scripted.py` (privileged expert) | `projects/compliant_RL/scripted.py` (`CatchRule`) |
| PPO hyperparameters, network | command line of `train.py` (defaults: `TrainCfg` in `src/mujoco_rl_bed/rl/train.py`) | same |
| Peg/hole shapes | `src/mujoco_rl_bed/sim/assets.py` | same |

Quick changes don't need code edits: task values can be overridden on the command line, e.g.
`task.rewards.try_cost.weight=-0.5` or `task.scene.assets.1.inner_radius=0.0042` (see the override tables in
`commands.md`).
