\# Spiral + RL yaw: spiral search, hole trigger, learned local x / y / yaw correction

State as of 2026-10-16. All numbers are from simulation: MuJoCo 3.10.0, the Franka test bed, an 8 mm peg (diameter,
side or across flats) and a 25 mm deep hole. "n/50" means successes out of 50 episodes; with 50 episodes the 95 %
interval is about ±10 points. Every comparison runs the methods on the same reset seeds (2000+): the same hole-estimate
errors, friction, tilt and yaw draws.

## 1. Idea

The scripted spiral search with the catch rule fails when the hole is yawed against a non-round peg. A cuboid with
0.5 mm radial clearance is blocked above about 7.7° of yaw error. The new pipeline:

1. **Spiral** (scripted, unchanged): approach at 1.5 N, Archimedean spiral (pitch 0.8 mm, 10 mm/s, out to 4.6 mm),
   2 mm drop detection.
2. **Hole trigger**: a classifier on the sensed contact state (wrench, spring deflection, lateral velocity, tilt,
   depth) decides that the tip is within about 1 mm of the hole and stops the spiral.
3. **Local phase (RL)**: a recurrent PPO policy moves the lateral anchor (±1.5 mm around the stop point) and the yaw
   target (±12°) for up to 6 s. A 2 mm drop switches to the scripted insert (soft gains, 5 N, compliant yaw). Without
   a drop the spiral resumes.

## 2. Implementation (branch `torque-rl-forge`)

| Piece | Where |
|---|---|
| Yaw target for the axis-compliance controller | `src/mujoco_rl_bed/control/axis_compliance.py` (`set_target(yaw=...)`) |
| Action term `spiral_local` (phases, trigger options `external` / `classifier` / `oracle`, `start_local`, compliant-yaw insert) | `projects/compliant_RL/terms.py` |
| Hole yaw error event `cr_hole_yaw_error` (shape symmetry, magnitude bins), `hole_yaw_error`, local-phase observations, `cr_yaw_align` reward, `cr_s3_start` (mixed starts), `cr_s3_local_timeout` | `projects/compliant_RL/terms.py` |
| Tasks `compliant_peg_s3` (full pipeline), `_s3_local` (synthetic local starts), `_s3_handoff` (spiral + in-env trigger), `_s3_mixed` (both) | `projects/compliant_RL/task.py` |
| Hole trigger `HoleTrigger` + model `models/hole_trigger_classifier.pkl` (scikit-learn 1.7.0) | `projects/compliant_RL/hole_trigger.py` |
| Privileged local corrector `LocalOracle` (ceiling) | `projects/compliant_RL/scripted.py` |
| End-to-end evaluation (`current`, `trigger_oracle`, `trigger_policy`, `perfect_policy`) | `projects/compliant_RL/eval_s3.py` |
| Benchmark method `spiral_rl_yaw` and hole-yaw condition `conditions.yaw_deg` | `projects/benchmark/compare.py` (not committed yet) |

**Policy.** Recurrent PPO with an asymmetric critic: LSTM 256, MLP 256-128, ELU.
- **Actor inputs (20):** wrench (noise 1 N / 0.05 Nm), spring deflection, TCP lateral velocity, roll/pitch, depth,
  offset from the stop point, TCP yaw, time in the local phase, previous action.
- **Critic inputs:** the actor inputs plus the true tip offset, the true yaw error, friction, success and contact force.
- **Reward:** as `compliant_peg_s2`: keypoint kernels + 0.5 lateral kernel + place and success bonuses
  − 0.2 contact penalty − 0.1 action rate − 0.01 ‖a‖², plus 0.5 × a yaw-alignment kernel. The yaw term is needed
  because the keypoints lie on the peg axis and cannot see yaw.

## 3. Results so far

### 3.1 Does the design have leverage? (scripted, no learning)
Cuboid, 0.5 mm clearance, hole error U(0, 4 mm), no tilt, friction U(0.5, 1.0), yaw error exactly 0 / 3 / 6 / 9°.

| Method | 0° | 3° | 6° | 9° | All |
|---|---|---|---|---|---|
| Current pipeline (spiral + catch rule) | 50 | 47 | 33 | 0 | 130/200 |
| Catch-rule trigger → oracle local x / y / yaw | 49 | 46 | 35 | 7 | 137/200 |
| Catch-rule trigger → oracle, yaw only | 48 | 45 | 29 | 5 | 127/200 |
| Perfect trigger (tip < 1 mm) → oracle local x / y / yaw | 50 | 50 | 50 | 50 | 200/200 |

- The local correction has a large ceiling: +70 out of 200.
- Both x/y and yaw corrections are needed.
- The catch rule (|F_xy| > 3.5 N) is blind to yaw-blocked passes. At 9°, 33/50 episodes never fire, although the tip
  passes 0.11 mm from the axis.

(Earlier, separately: tuning the catch rule per condition gave no gain, 88.1 % vs 87.5 % for one global rule. Even an
oracle push timing gained only about 3 points.)

### 3.2 Hole trigger
- **What tells "near the hole"** (per-step ROC AUC at 0 / 3 / 6 / 9° yaw):
  - depth: 0.73 / 0.79 / 0.80 / 0.71;
  - lateral velocity over 4 steps (the spiral slows): 0.59 / 0.70 / 0.80 / 0.74;
  - **yaw torque: 0.50–0.54, no information**;
  - classifier on a 4-step window of all 21 inputs: 0.94 / 0.95 / 0.95 / 0.89.
- **Cuboid-trained classifier → oracle correction:** 50 / 50 / 46 / 40 = **186/200**, against 137 with the catch rule.
- **Zero-shot on other shapes** (classifier → oracle, against the current pipeline, out of 200):

  | Shape, clearance, yaw levels | Result |
  |---|---|
  | Hexagon 0.5 mm, 0 / 10 / 20 / 25° | 161 vs 83 |
  | Octagon 0.5 mm, 0 / 5 / 10 / 20° | 191 vs 172 |
  | Octagon 0.25 mm, 0 / 10 / 15 / 20° | 104 vs 41 |

- **Shipped model,** retrained on mixed data: 4 shapes, clearance N(0.3, 0.1) mm in [0.1, 0.5], friction U(0.3, 1.0),
  yaw up to half the symmetry angle, no tilt; 782 episodes.
  - Held-out-condition AUC 0.81. It fires within 1.5 mm of the hole in 59 % of episodes at 0.11 false fires per
    episode; the catch rule manages 34 %.
  - Strong: cylinder 94 %, octagon 74 %, small yaw 73 %, clearance 0.4–0.5 mm 81 %.
  - Weak: cuboid 24 %, hexagon 45 %, large yaw about 33 %, clearance 0.1–0.2 mm 34 %. These are an information limit;
    longer input windows did not help.
  - Fires when p > 0.9 for 1 step.

### 3.3 RL training runs (server: RTX 6000 Ada, 48 environments)
End-to-end evaluation as in 3.1, local phase 6 s, deterministic policy.

| Run (5M steps each) | Trigger → policy | Perfect trigger → policy |
|---|---|---|
| `s3_local`: synthetic starts only (training task: 70 %; zero action 13 %, oracle 93 %) | 146/200 (38, 42, 38, 28) | 180/200 |
| `s3_handoff`: fine-tuned on spiral + trigger hand-offs only | 135/200 (44, 45, 33, 13) | 171/200 |
| **`s3_mixed`: fine-tuned on 50 % synthetic + 50 % hand-off starts, yaw half in 6–11°, compliant-yaw insert** | **156/200 (45, 47, 43, 21)** | **199/200** |
| References: current pipeline 127/200, trigger → oracle 187/200 | | |

- **`s3_local`:** with a 3 s local phase it scored only 126/200 (the policy needs about 4 s), hence the 6 s default.
  After trigger hand-offs, only 55–64 % within 1.5 mm ended in a drop; after perfect-trigger hand-offs it was 72–82 %.
  The trigger's hand-off states (peg caught on the edge, moving) were not in training.
- **`s3_handoff`:** fixed the hand-offs at 0–6° (drop rate 70–79 %) but forgot the yaw correction at 9°. Blocked cases
  were rare, and the spiral steps diluted the signal.
- **`s3_mixed`:** keeps both skills. With a perfect trigger the local correction is essentially solved (199/200).
- **Yaw 1–6° in 1° steps (`s3_mixed`, out of 50):**

  | Method | 1° | 2° | 3° | 4° | 5° | 6° | All |
  |---|---|---|---|---|---|---|---|
  | Current pipeline | 50 | 49 | 47 | 44 | 39 | 30 | 259/300 |
  | Trigger → `s3_mixed` policy | 50 | 48 | 47 | 45 | 44 | 43 | **277/300** |
  | Trigger → oracle | 50 | 50 | 50 | 50 | 49 | 49 | 299/300 |
  | Perfect trigger → `s3_mixed` policy | 50 | 49 | 50 | 49 | 50 | 50 | 298/300 |

  It ties the current pipeline at 1–3° and wins from 4° up. It is slower: median 7–9.5 s against 4.5–7 s.
- **Compliant yaw in the insert:** small effect. At 7°, 23 vs 22 of 30 inserted; at 3–5° no change; no extra force
  spikes.

### 3.4 Validation of `spiral_RL_yaw` (= the `s3_mixed` model)
4 shapes × 100 episodes; clearance 0.5 mm; xy error U(0, 4 mm), tilt U(0, 5°), yaw U(0, 5°); friction U(0.5, 1.0);
20 s. Results in `projects/benchmark/results/spiral_RL_yaw_validation/`.

| Shape | Success | Median time | Median F_max | Not found / jammed |
|---|---|---|---|---|
| Cylinder | 68/100 | 4.4 s | 3.7 N | 28 / 4 |
| Cuboid | 72/100 | 6.1 s | 5.7 N | 27 / 1 |
| Hexagon | 67/100 | 4.8 s | 4.2 N | 28 / 5 |
| Octagon | 73/100 | 5.3 s | 4.1 N | 25 / 2 |
| **All** | **280/400 (70 %)** | 5.1 s | | |

- **By xy error:** 0–1 mm 100 %, 1–2 mm 94 %, 2–3 mm 52 %, 3–4 mm 21 %.
- **By tilt or yaw** (1° bins up to 5°): flat at 57–75 %.
- **Tilt is the cause.** The cuboid at yaw 1° gets 48/50 without tilt and 36/50 with tilt U(0, 5°), on the same seeds.

**Where tilt breaks it:** cuboid, 0.5 mm, xy U(0, 4 mm), tilt U(0, 5°).

| Method | Yaw 0° | Yaw 5° | All |
|---|---|---|---|
| Current pipeline | 50 | 48 | 98/100 |
| Hole trigger → oracle | 37 | 37 | 74/100 |
| Hole trigger → `spiral_RL_yaw` | 37 | 34 | 71/100 |
| Perfect trigger → `spiral_RL_yaw` | 49 | 49 | 98/100 |

- **The policy copes with tilt** (98/100 with a good hand-off).
- **The trigger does not.** It was trained without tilt and fires about 1.75 times per episode, only about 60 % of
  them near the hole. Each false fire costs up to 6 s of the 20 s budget, so large xy errors run out of time.
- **Tilt helps the current pipeline:** a tilted cuboid's leading corner catches the hole even when yawed (5°: 48/50
  with tilt, 39/50 without). So with tilt up to 5° the new pipeline is currently a net loss. Its yaw advantage shows
  without tilt (6°: 43 vs 30, 9°: 21 vs 0).

## 4. Models and files

| What | Where |
|---|---|
| Best model (server) | `projects/compliant_RL/runs/compliant_peg_s3_mixed_20261016-195117_s2_s3_mixed` |
| Best model (laptop copy) | `projects/compliant_RL/runs/spiral_RL_yaw` |
| Earlier runs (server) | `..._s3_local_20261016-173229_s0_s3_local`, `..._s3_handoff_20261016-184850_s1_s3_handoff` |
| Hole trigger model | `projects/compliant_RL/models/hole_trigger_classifier.pkl` |
| Commits | `5e14eb6` (stage 3), `8b3da60` (hand-off training), `4e6cdee` (mixed starts, yaw bins, compliant-yaw insert) |
| Not committed | `compare.py` (`spiral_rl_yaw` method, `conditions.yaw_deg`, fix: keep a task's own yaw event when the benchmark sets none), `benchmark_comparison.yaml` (new keys), `eval_s3.py` (`tilt` option), this file |
| Trigger data and training scripts | session scratchpad only (`s3_mixed_collect.py`, `s3_mixed_train.py`, …): temporary, to be moved into the repo |

## 5. Known issues

1. **The hole trigger fails under tilt:** false fires, see 3.4. This is the main gap.
2. **The hole trigger is weak for large yaw errors and tight clearances** on the cuboid and hexagon. That is an
   information limit of the passive spiral, not a training-data limit.
3. **Hand-offs are still worse than the oracle's:** the drop rate after trigger hand-offs is 69–75 % for the policy
   against 88–94 % for the oracle. At 9° it is 40 % against 88 %.
4. **The new pipeline is slower** (7–9.5 s against 4.5–7 s) and still loses 5/50 at 0° without tilt, where the current
   pipeline never fails.
5. **Training speed:** the classifier runs in every environment process. Start training with `OMP_NUM_THREADS=1`;
   without it, threads oversubscribe the CPU (716 steps/s instead of several thousand).
6. **The model is trained on the cuboid only** (0.5 mm, no tilt); other shapes are zero-shot.

## 6. Plan and next steps

In priority order. Each step is checked with `eval_s3.py` (same seeds) and the validation in 3.4.

1. **Fix the trigger under tilt.**
   - Retrain the hole-trigger classifier with tilt U(0, 5°) in the data (same pipeline: collect plain-spiral data,
     gradient boosting, held-out conditions). About 15 minutes.
   - Target: trigger → oracle under tilt from 74/100 toward the no-tilt level (93–100 %).
2. **Make false fires cheap.** End a local phase early when nothing is caught, for example no depth change and no edge
   contact within about 1 s, so a false fire costs 1 s instead of 6 s.
3. **Combine the triggers.** Start the local phase on the classifier or on the catch rule. The catch rule works well
   under tilt (current pipeline 98/100).
4. **Fine-tune the policy with tilt and all shapes** (`compliant_peg_s3_mixed` with tilt U(0, 5°) and a shape mix),
   from `spiral_RL_yaw`, about 5M steps.
5. **Close the hand-off gap at large yaw:** the `s3_mixed2` run (`p_local` 0.3, two thirds of yaw errors in 6–11°),
   from `spiral_RL_yaw`.
6. **Housekeeping:** commit the benchmark method and the `eval_s3` tilt option; move the trigger data-collection and
   training scripts into the repo; limit the classifier to one thread inside `HoleTrigger` so `OMP_NUM_THREADS` is not
   needed.
7. **Later:** active yaw dithering during the spiral, to make large yaw errors detectable (the passive spiral carries
   too little signal); a speed-up of the local phase (time penalty) to match the current pipeline's time.

**Decision point after steps 1–3:** if the trigger under tilt reaches about 90 % or more, run step 4 and re-validate.
If it does not, keep the catch-rule pipeline for tilted cases and use the RL yaw phase only when tilt is small.

## 7. Commands

From `mujoco_rl_bed/`, with the venv active. On the server, prefix training and evaluation with `OMP_NUM_THREADS=1`.

```bash
# Train (synthetic local starts), then fine-tune on mixed starts
python -u projects/compliant_RL/train.py task=compliant_peg_s3_local device=cuda:1 n_envs=48 \
  algo=recurrent_ppo asymmetric=true n_steps=128 batch_size=512 n_epochs=4 learning_rate=1e-4 lr_schedule=adaptive \
  kl_threshold=0.008 gamma=0.99 gae_lambda=0.95 clip_range=0.2 clip_range_vf=0.2 vf_coef=1.0 ent_coef=0.0 \
  log_std_init=-0.5 lstm_hidden_size=256 n_lstm_layers=1 net_arch="(256,128)" activation=elu \
  total_timesteps=5000000 checkpoint_every=500000 seed=0 run_name=s3_local
OMP_NUM_THREADS=1 python -u projects/compliant_RL/train.py task=compliant_peg_s3_mixed init_from=<s3_local_run> \
  device=cuda:1 n_envs=48 algo=recurrent_ppo asymmetric=true n_steps=128 batch_size=512 n_epochs=4 \
  learning_rate=1e-4 lr_schedule=adaptive kl_threshold=0.008 gamma=0.99 gae_lambda=0.95 clip_range=0.2 \
  clip_range_vf=0.2 vf_coef=1.0 ent_coef=0.0 lstm_hidden_size=256 n_lstm_layers=1 net_arch="(256,128)" activation=elu \
  total_timesteps=5000000 checkpoint_every=500000 seed=2 run_name=s3_mixed

# End-to-end evaluation (same seeds as above); tilt and yaw levels are options
OMP_NUM_THREADS=1 python projects/compliant_RL/eval_s3.py run=spiral_RL_yaw "yaws=[0,3,6,9]" workers=4
OMP_NUM_THREADS=1 python projects/compliant_RL/eval_s3.py run=spiral_RL_yaw tilt=0-5 "yaws=[0,5]" workers=4

# Validation across shapes (section 3.4)
OMP_NUM_THREADS=1 python projects/benchmark/compare.py config=projects/benchmark/benchmark_comparison.yaml \
  "methods=[spiral_rl_yaw]" episodes=100 workers=4 "conditions.shapes=[cylinder,cuboid,hexagon,octagon]" \
  "conditions.clearance_mm=[0.5]" "conditions.hole_error_mm=['0-4']" "conditions.tilt_deg=['0-5']" \
  "conditions.friction=['0.5-1.0']" "conditions.yaw_deg=['0-5']" out=projects/benchmark/results/spiral_RL_yaw_validation

# Viewer (close the window to stop; view.realtime=false runs faster than real time)
OMP_NUM_THREADS=1 python projects/benchmark/compare.py config=projects/benchmark/benchmark_comparison.yaml \
  view.enabled=true view.method=spiral_rl_yaw episodes=5 "conditions.shapes=[cuboid,hexagon]" \
  "conditions.clearance_mm=[0.5]" "conditions.hole_error_mm=['0-4']" "conditions.tilt_deg=['0-5']" \
  "conditions.friction=['0.5-1.0']" "conditions.yaw_deg=['0-5']"
```
