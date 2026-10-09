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
python scripts/system_check.py                     # (from mujoco_rl_bed/) machine check before a long run
python projects/forge/train.py algo=recurrent_ppo asymmetric=true device=cuda n_envs=32 n_steps=128 \
    batch_size=512 n_epochs=4 learning_rate=1e-4 lr_schedule=adaptive kl_threshold=0.008 gamma=0.995 \
    clip_range_vf=0.2 vf_coef=1.0 log_std_init=0.0 lstm_hidden_size=1024 n_lstm_layers=2 \
    net_arch="(512,128,64)" activation=elu total_timesteps=20000000 run_name=isaac
tensorboard --logdir projects/forge/runs
python projects/forge/eval.py run=<run_folder_name> episodes=20 render=true
python projects/forge/view.py                      # viewer: scripted expert (mode=idle|random, run=<folder>)
pytest projects/forge
```

Any `EnvCfg` field can be overridden, e.g. `task.episode_length_s=10`,
`task.rewards.contact_penalty.weight=-0.1`, `task.events.reset_fixed.params.pos_noise_std=0.0025`.
`forge.task.NO_DR` lists the overrides that switch all randomization and observation noise off.

## Implemented vs. paper / Isaac Lab

Values follow the paper (Table II, 8 mm peg). Where the paper is silent they follow NVIDIA's reference
implementation in Isaac Lab (`isaaclab_tasks/direct/forge`, `.../factory`).

| | Paper | Isaac Lab | Here |
|---|---|---|---|
| Policy rate / episode | 15 Hz, 150 steps (10 s) | 15 Hz, 10 s | 15.15 Hz (2 ms x 33), 150 steps (9.9 s) |
| Action | x, y, z, yaw + a_ET | same, EMA-smoothed (all dims) | x, y, z + a_ET, EMA-smoothed (no yaw: round peg) |
| Kp / Kd | U[400, 800], 2 sqrt(Kp) | 565 x/÷ (1 + U(0, 0.41)) per axis (rot. 28) | as Isaac |
| λ | U[1.6, 2.5] cm | 2 cm x/÷ (1 + U(0, 0.25)) per axis | as Isaac |
| EMA α | – | U[0.025, 0.1] | as Isaac |
| Dead zone | U[0, 5] N per axis | U[0, (5 N, 1 Nm)], re-drawn every 2 s | as Isaac |
| Friction | U[0.5, 1.0] | held 0.75, fixed static U(0.25, 1.25) | U[0.5, 1.0] on peg and socket |
| Peg offset in hand | x ±3 mm, z [14, 20] mm | x, z ±3 mm | x, z ±3 mm (peg rigid during the episode) |
| Hand yaw at reset | ±45° | ±45° | ±45° (orientation then held) |
| Hole-pose noise σ | 2.5 mm | 1 mm | 1 mm (`pos_noise_std`) |
| Obs noise | pos 0.25 mm, force 1 N | + 0.1° rot., velocity by finite differences | as Isaac (velocity noise as Gaussian of the same size) |
| Rewards | coarse + fine + bonuses − β·force − \|p − y\| | + baseline K(5, 4), −0.1 ‖Δa‖, −0.001 ‖Δp‖/λ, prediction penalty delayed until 25% success | as Isaac (gate per env from a running success rate) |
| Critic | asymmetric | gains, λ, α, held/fixed poses, joint pos | + dead zone, friction, peg offset, success label |
| PPO | recurrent, asymmetric | LSTM 2x1024 + MLP 512-128-64 ELU, γ 0.995, lr 1e-4 KL-adaptive (0.008), 4 epochs, minibatch 512, horizon 128, value clip 0.2 | same settings available (`lr_schedule=adaptive`, `activation=elu`, `clip_range_vf`) |

Remaining differences: the contact force is the step-mean peg contact force (Isaac: wrist force with
EMA 0.25); roll/pitch rates stay in the observation (Isaac zeroes them); the held-part mass is not
randomized (exact gravity compensation would cancel it); the scripted expert cannot handle the dead
zone, so `view.py` / `eval.py policy=scripted` run with `NO_DR`.

Runs trained before these defaults changed (20 Hz, 7.5 s, critic layout, rewards) need their old
settings restored as overrides to evaluate; `eval.py` prints the differing settings.
