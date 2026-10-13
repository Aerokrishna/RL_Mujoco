# Peg-in-hole benchmark: Spiral+RL vs FORGE vs TacDiffusion

Config: [`benchmark_comparison.yaml`](benchmark_comparison.yaml). Raw episodes and the full 48-condition table (95 % CI,
time, F_max, failure split): `results/benchmark_comparison/` (`episodes.jsonl`, `summary.md`).

```
python projects/benchmark/compare.py config=projects/benchmark/benchmark_comparison.yaml workers=4
```

`workers` is capped at 4 because each worker takes about 1 GB of RAM. With 11 workers this 15 GB laptop ran out of
memory and froze.

To rerun it on another machine (conda env, pinned MuJoCo 3.10.0, model download, TacDiffusion submodule), follow the
"Quick start" in the [repository README](../../../README.md#quick-start-reproduce-the-benchmark).

## Setup

- **Grid:** 3 shapes (cylinder, cuboid, octagon) × clearance 0.25 / 0.5 mm per side × hole-position error 2 / 4 mm
  (random direction) × arm tilt 0 / 5° (random axis) × friction 0.4 / 0.7. That gives 48 conditions, with 20 episodes
  per method in each.
- **Common to all methods:** an 8 mm peg in a 25 mm deep hole and 20 s per episode. Every method runs the same reset
  seeds (2000–2019) in every condition, so all methods face identical hole errors, tilts and friction draws.
- **Models (the best kept so far):**

  | Method | Model | What it is |
  |---|---|---|
  | Spiral+RL | `compliant_RL/runs/compliant_peg_s2g_20261012-122716_s0_s2g_bc` | compliance spiral search plus an LSTM "try insert" policy, behaviour-cloned from the scripted catch rule |
  | FORGE | `forge/runs/jitter_but_works` | FORGE policy, actions sampled |
  | TacDiffusion | authors' `TacDiffusion_model_256.onnx` | 512 model: 45 ms per call on this CPU; 256: 14 ms |

- **Training conditions:** Spiral+RL was trained on the cylinder at 0.25 mm with error 0–4 mm, tilt 0–8° and
  friction 0.5–1.0. Cuboid, octagon, 0.5 mm and friction 0.4 are outside its training range.

## Results

| Method | Success (960 episodes) | Median time to insert | Median F_max | Failures: not found / jammed |
|---|---|---|---|---|
| **Spiral+RL** | **641 (67 %)** | 6.3 s | **5.0 N** | 314 / 5 |
| FORGE | 211 (22 %) | 6.2 s | 17.0 N | 468 / 281 |
| TacDiffusion (256) | *pending* | | | |

Success by factor (out of 480 per level):

| Factor | Level | Spiral+RL | FORGE |
|---|---|---|---|
| Clearance | 0.25 mm | 255 (53 %) | 85 (18 %) |
| | 0.5 mm | 386 (80 %) | 126 (26 %) |
| Hole error | 2 mm | 372 (78 %) | 193 (40 %) |
| | 4 mm | 269 (56 %) | 18 (4 %) |
| Tilt | 0° | 353 (74 %) | 211 (44 %) |
| | 5° | 288 (60 %) | **0 (0 %)** |
| Friction | 0.4 | 353 (74 %) | 109 (23 %) |
| | 0.7 | 288 (60 %) | 102 (21 %) |

Success by shape, clearance and tilt (out of 80: 2 errors × 2 frictions × 20 episodes):

| Shape | Clearance | Tilt 0°: Spiral+RL / FORGE | Tilt 5°: Spiral+RL / FORGE |
|---|---|---|---|
| Cylinder | 0.25 | 39 / 24 | 49 / 0 |
| Cylinder | 0.5 | 68 / 41 | 72 / 0 |
| Cuboid | 0.25 | 56 / 34 | 22 / 0 |
| Cuboid | 0.5 | 80 / 45 | 37 / 0 |
| Octagon | 0.25 | 40 / 27 | 49 / 0 |
| Octagon | 0.5 | 70 / 40 | 59 / 0 |

## Findings

1. **Spiral+RL wins in every one of the 12 shape × clearance × tilt cells:** 67 % overall against 22 % for FORGE,
   with about 3× lower contact force (median F_max 5 N against 17 N).
2. **FORGE fails completely under tilt:** 0 of 480 at 5°. At 0° tilt it works only with a 2 mm error (193 of 240);
   with a 4 mm error it manages 18 of 240. Its failures split between not finding the hole and jamming (281
   jams), which matches its high forces. Spiral+RL almost never jams (5 episodes).
3. **Spiral+RL holds up with a 4 mm error (56 %),** because the spiral covers a 4.6 mm radius. Its failures are
   almost all "not found": a spiral pass crosses the hole without the edge catching the peg, so the catch policy
   gets no signal.
4. **Tilt helps the round shapes at tight clearance but hurts the cuboid.** At 0.25 mm, 5° tilt raises Spiral+RL
   success for the cylinder (39 to 49) and the octagon (40 to 49): the tilted tip catches the edge more easily. For
   the cuboid it drops from 136 to 59 of 160: a tilted square peg hangs on the hole corner.
5. **Outside its training range, Spiral+RL generalizes:** 0.5 mm is easier (80 %), and friction 0.4 is better than
   0.7. The cuboid with tilt is the one weak spot.

TacDiffusion results will be added when its run finishes.
