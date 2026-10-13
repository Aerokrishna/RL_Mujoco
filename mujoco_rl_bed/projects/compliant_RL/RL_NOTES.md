# Residual RL on the compliant spiral search: why it does not learn, and what could make it find the hole

Reference for future work on `compliant_RL` (state as of 2026-10-10). Numbers are from deterministic evaluations on
fixed seeds unless noted; "n/40" = successes out of 40 episodes, about ±3 episodes of noise.

## 1. Setup in one paragraph

Franka + 8 mm peg (cylinder; cuboid 8x8 mm also available) into an 8.5 mm hole (0.25 mm clearance per side), 25 mm
deep. Controller `AxisCompliance` (`src/mujoco_rl_bed/control/axis_compliance.py`): force control along the tool
axis, compliant lateral and roll/pitch, stiff yaw (external yaw torque masked). The start pose is above a hole
*estimate* that is off by a known magnitude in a random direction, with a random tilt. Base = scripted Archimedean
spiral of the lateral spring anchor in the start TCP frame (`spiral_residual` action term in
`projects/compliant_RL/terms.py`): approach at 1 N, touchdown detection, spiral with stiff search gains
(k_lat 2000, k_tilt 3), drop detection (0.5 mm along the push axis), then soft insert gains (k_lat 200,
k_tilt 0.1/0.4) and 5 N push. RL adds a residual on top. Tasks in `projects/compliant_RL/task.py`:
`compliant_peg_s2` (2 mm, ≤6°), `_s2b` (0-4 mm, ≤8°, speed action 0-3x), `_s2_box` (cuboid), `_s2c`/`_s2d`
(Δxy only + curriculum), `_s2e` (Δxy + speed 0-1x, dwell). Asymmetric PPO (actor sees wrench/spiral state/depth/
spring deflection/last action; critic sees the true hole offset).

## 2. What happened (all RL runs vs the scripted spiral)

| Run | Design | Result vs zero residual (same seeds) |
|---|---|---|
| A | s2, exact 2 mm, ≤6°, Δxy ±1 mm + speed 0-2x, MLP 64-64 | 28 → 33/40 on its own task (clearest gain, +12.5 pts); not faster |
| B | s2b 0-4 mm, init from A + ent_coef | 30 → 22/40 (worse) |
| C | s2b from scratch, log_std -1.0 | 30 → 25/40 (worse) |
| D | s2c curriculum, thresholds = base success | stages advanced by chance in 300k steps; no gain, stopped |
| E | s2d strict curriculum (exact 2/3/4 mm, thresholds 0.95/0.80/0.65), GPU server, ~5M | exact 2/3/4 mm: 33→37, 23→26, 17→21; U(0,4): 30→27 |
| F | s2e Δxy + speed 0-1x (zero = 5 mm/s), kp_xy doubled, ~1.4M | 29 → 29/40; speed near hole 0.490 vs far 0.494 (no dwelling) |

FORGE comparison (no tilt, 15 s, n=20): at 3-4 mm our scripted spiral beats FORGE on both shapes (cylinder 7/20 vs
2-4/20, cuboid 10-11/20 vs 0-4/20) with 3-5x lower forces (median 4-8 N vs 15-27 N). The advantage comes from
compliance + spiral, not from RL (Run A ≈ spiral within ±2 there).

The steering metric `res_search_toward_hole_mm` (mean residual component toward the true hole, logging only) stayed
≤ 0 in every run: no policy ever learned to push toward the hole.

## 3. Why RL cannot learn to find the hole (evidence)

1. **The observation does not say where the hole is.** Decoding the true hole direction from rollouts
   (gradient boosting, held-out episodes; score = mean cosine, chance 0): spiral position alone already gives
   +0.42 near the hole (geometry prior: the hole is inside the search disc). Adding the wrench with the training
   noise (1 N / 0.05 Nm) adds nothing (+0.38-0.41 near, any window: now, 3 steps, 1 s, 4 s). Noise-free adds only
   +0.05-0.09; it is half gone at 0.1 N / 0.005 Nm and gone at 0.3 N / 0.015 Nm. Physical reason: a flat peg face
   sliding on a flat top at 1 N push produces tiny torque changes (~0.002 Nm) when the support shifts, ~25x below the
   torque noise. A reward cannot teach steering toward something the actor cannot sense: PPO learns the average
   action over all hole positions consistent with the observation, which cancels to ~0 lateral bias.
2. **Proximity ("you are close") is only partially sensed.** Near-miss band (tip 0.25-1 mm from axis, excluding
   already-dropping samples) vs farther: ROC AUC 0.70 (wrench, 3 steps, training noise) vs 0.57 (position only);
   at 50 % recall only 13-15 % of alarms are true. The current 3-step wrench history is as good as a 1 s window, so
   memory (LSTM) does not help here. The first, inflated numbers (AUC 0.8-0.88) came from samples where the peg was
   already over the hole and dropping. Always exclude those.
3. **The failures are coverage misses, not capture or insertion failures.** Failure breakdown of the scripted
   spiral (40 episodes, ≤10° tilt): every time the tip came within 0.25 mm of the axis it dropped (0 capture
   failures); 28 % never got closer than 0.33-0.76 mm (coverage); false drops only above 6.5° tilt. With pitch
   0.8 mm the worst-case ring gap (0.4 mm) exceeds the 0.25 mm capture radius, and the tip also lags the anchor by
   ~1 mm and stick-slips, covering only 20-55 % of the commanded path.
4. **The action cannot fix coverage much.** A ±1-1.5 mm residual on a spiral that already lags ~1 mm cannot
   reshape the traced path; the spiral moves 10 mm/s, so the tip passes a near-miss spot in ~0.1 s. When a speed
   action was offered (A, B, C, F) the policy barely used it or slowed everywhere uniformly.
5. **The dwell incentive is drowned by noise.** Upper bound of "detect near, then local 1 mm fine spiral" with a
   perfect (oracle) trigger: 41 → 49/60. A real detector catches 25-50 % of near misses at 15-50 % precision, so
   pausing mostly wastes time on false alarms; the expected return gain is far below return variance. Run F
   (kp_xy doubled to pay for dwelling) learned no selective slowing.
6. **The base already solves most of what is learnable.** Tilt is handled by compliance (40/40 insertions at up to
   10° once the edge is in the hole, with k_tilt 0.1, d_tilt 0.4, hover 2 mm); insertion and capture are physical.
   What is left (coverage) is geometric and is solved better by scripting (see 4.1).
7. **Training-design pitfalls that made it worse.**
   - Curriculum thresholds equal to the base success → environments advance by chance (Run D). Thresholds must sit
     clearly above the base (Run E: +12-23 pts) and windows should be long (50 episodes per env).
   - Training only on exact offsets (Run E stages) specialises the policy; it then underperforms on U(0, 4 mm)
     because offsets < 2 mm were rarely seen.
   - `init_from` copies the old log-std (e.g. 0.13): exploration cannot be raised by `log_std_init`; `ent_coef`
     0.005 did not raise it either (Run B). Higher noise (log_std -1.0, Run C) breaks the base during rollouts.
   - 20 s episodes (500 steps) make credit assignment for early speed decisions hard.

## 4. What could motivate / enable the policy to find the hole

### 4.1 Fix the base first (largest, cheapest lever; no RL needed)
- **Finer spiral pitch.** Exact 4 mm, ≤8° tilt, scripted only: pitch 0.8 → 17/40 within 20 s; 0.6 → 28/40 (median
  12.9 s); 0.5 → 29/40 within 20 s, 31/40 within 30 s; 0.4 → 16/40 within 20 s (too slow), 29/40 within 30 s.
  Speeding up 0.5 mm to 15 mm/s hurts (21/40) because tracking degrades. Pitch ≤ 2 x capture radius (0.5 mm) is
  the geometric rule; 0.6 mm is the best time/coverage trade.
- Variable pitch (coarse inside, fine outside), ring-first search when the error magnitude is known, better
  tracking (stiff search gains: k_lat ~2000 with damping ratio ~1.5-4; k_tilt 3 during search so the tip slides
  instead of pivoting on a pinned tip).

### 4.2 Make the hole observable (prerequisite for any steering policy)
- Lower force/torque noise in training if the real sensor allows (the cue only appears below ~0.1 N / 0.005 Nm).
- Stronger contact signal: higher search force (more torque signal but more friction and stick-slip), chamfered
  holes (a chamfer turns the push into a lateral force toward the hole: compliance alone then centres the peg).
- Deliberate tilt-and-slide search (leading edge dips into the hole): with training noise and a 1 s window the
  direction score gain rises to +0.31-0.39 at 8° tilt (vs ~0 flat), but the current base fails with it (2/120)
  because the drop detector fires while the peg centre is a peg radius away. The user dropped this idea.
- Active probing (small deliberate motions) + memory (LSTM) is the only case where an LSTM is expected to help.

### 4.3 Only then shape the reward
- A dense close-to-hole term (`kp_xy`, or a potential-based progress term on the true tip-axis distance; privileged,
  training only) helps once the observation carries the cue; without the cue it only reshapes the search layout.
- Keep success worth more than proximity so dwelling never beats inserting.

### 4.4 Where RL is likely to pay off instead
- Asymmetric parts needing yaw: the controller holds yaw stiff and ignores external yaw torque; a yaw action or yaw
  compliance during insertion, symmetric keypoints in the reward (current keypoints lie on the peg axis and are
  yaw-blind), and a curriculum on yaw error. Cuboid assets exist (`BoxPeg`, `RectHole`, `cr_align_hole_yaw`).
- Jam/wedge recovery, high friction (tilted insertion drops to 29/40 at μ = 1.0), tight clearances.

## 5. Checklist before any new RL run
1. Measure zero-residual and an oracle/privileged policy with the same action space on fixed seeds (2000+i). If the
   oracle is not clearly above the base, fix the action space first.
2. Run an information probe: can the hole direction / proximity be decoded from exactly the actor's inputs (with
   the training noise) better than from spiral position alone? If not, RL cannot steer.
3. Curriculum thresholds > base success by a clear margin; include the final distribution early.
4. Evaluate deterministically on fixed seeds with `eval.py` and stage overrides, e.g.
   `"task.events.curriculum.params.stages=((0.0,0.004,8.0,0.0046),)" "task.events.curriculum.params.advance_at=()"`
   (otherwise the curriculum restarts at stage 0 during evaluation). `eval.py` prints a fixed key list; steering
   and speed metrics need a custom loop.
5. Judge training from 100k-step window averages, not snapshots, and only from the log itself. Do not call
   non-convergence before ~1-1.5M steps (stage 1 needed ~1M to take off).

## 6. Known gotchas
- `speed_near_hole` / `speed_far_hole` episode metrics produce `nan` (episodes without near or far samples) and SB3
  averages them to `nan`: use an offline nan-safe evaluation.
- Long runs on the laptop: start in `screen` (no tmux installed); suspend pauses training; memory is tight with
  Firefox open (no swap), causing multi-second stalls. On the GPU server use `n_envs` ≈ free physical cores
  (load average ~11 of 32 cores → `n_envs=16`, `batch_size=1024`); the GPU barely helps (MuJoCo is CPU-bound).
- After pulling new code on another machine, re-run `pip install -e .` (editable installs record the package list;
  `compliant_RL` was missing until reinstall).
- `pkill -f <pattern>` also matches the shell running it; use `ps | grep "[p]attern"` to find PIDs first.
- Diagnostics scripts used for sections 3-4 lived in a session scratchpad (`cue_probe.py`, `spiral_failures.py`,
  `hybrid.py`, `forge_compare.py`, `s2_checks.py`, `s2_eval.py`, `steer_eval.py`) and are not in the repo; the
  methods above are described precisely enough to recreate them.
