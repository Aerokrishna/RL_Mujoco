# FORGE (first pass)

Peg-in-hole policy learning following *FORGE: Force-Guided Exploration for Robust
Contact-Rich Manipulation under Uncertainty* (Noseworthy et al., arXiv:2408.04587),
built on the `mujoco_rl_bed` core (torque-controlled Franka, Cartesian impedance).

| File | Contents |
|---|---|
| `task.py` | `forge_peg` task config (values from the paper's Table II, 8 mm peg) |
| `terms.py` | FORGE terms: keypoint kernels, place/success bonuses, force-threshold penalty, contact force / F_th observations, socket + hand reset, episode metrics |
| `scripted.py` | Privileged scripted expert (validation, demos) |
| `train.py`, `eval.py`, `view.py` | Entry points (train / evaluate / MuJoCo viewer); runs are written to `runs/` here |
| `tests/` | Geometry/contact validation and term tests |

Reusable MuJoCo pieces live in the core: `Peg`/`RoundHole` assets (`mujoco_rl_bed.sim.assets`),
reset IK (`mujoco_rl_bed.sim.ik`), the `anchor_relative_pos` action term.

## Usage (`conda activate dqn`, from any directory; full reference in `commands.md`)

```bash
python projects/forge/train.py algo=recurrent_ppo asymmetric=true n_envs=4 n_steps=512 batch_size=512 total_timesteps=3000000 log_std_init=-1.0 task.events.reset_fixed.params.pos_noise_std=0.001
tensorboard --logdir projects/forge/runs
python projects/forge/eval.py run=<run_folder_name> episodes=20 render=true
python projects/forge/view.py                      # viewer: scripted expert (mode=idle|random, run=<folder>)
pytest projects/forge
```

Any `EnvCfg` field can be overridden, e.g. `task.episode_length_s=10`,
`task.action.max_step=0.015`, `task.rewards.contact_penalty.weight=-0.1`,
`task.events.reset_fixed.params.pos_noise_std=0.0025`.

## Implemented vs. paper

Implemented: 3D position action relative to the fixed-part tip clipped by λ (Eq. 5),
coarse+fine keypoint reward (App. B), place/success bonuses (Eq. 2), excessive-force
penalty with randomized F_th observed by the policy (Eq. 3), fixed-part and hand
initial-state randomization (Table II), hole-position estimate noise (opt-in, 2.5 mm in the paper),
success prediction a_ET with -|p - y| reward and optional early termination (Sec. III-C),
PPO / recurrent PPO, symmetric or asymmetric actor-critic (`asymmetric=true`: clean privileged critic, as in the paper).

Additions not in the paper: action smoothing (EMA, alpha 0.2) and an action-rate penalty against
jitter; success-prediction weight 0.1 (an untrained predictor at weight 1 cancels the success bonus).

Controller randomization (Table II): Kp ~ U[400, 800] N/m and λ ~ U[1.6, 2.5] cm per episode,
observed only by the critic.

Not yet: EE/force observation noise (hooks exist, default 0), part friction/mass randomization,
force dead zone, in-hand peg offset.
