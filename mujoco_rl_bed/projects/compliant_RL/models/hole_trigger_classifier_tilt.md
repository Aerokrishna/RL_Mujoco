Hole trigger `hole_trigger_classifier_tilt.pkl`: p > 0.85 for 3 steps; config {'n_cond': 400, 'eps': 4, 'seed': 4000, 'workers': 4, 'out': 'hole_trigger_classifier_tilt.pkl', 'clr_mean': 0.3, 'clr_std': 0.1, 'clr_lo': 0.1, 'clr_hi': 0.5, 'err': '0-4', 'friction': '0.3-1.0', 'tilt': '0-5', 'tilt_zero_share': 0.5, 'yaw_cap': 12.0, 'time_max': 20.0, 'max_false': 0.15}

| group | episodes | AUC (held-out conditions) | new: fires near / false per ep | previous model: near / false | catch rule: near / false |
|---|---|---|---|---|---|
| shape cylinder | 389 | 0.77 | 0.85 / 0.05 | 0.81 / 0.89 | 0.65 / 0.16 |
| shape cuboid | 394 | 0.84 | 0.55 / 0.28 | 0.66 / 1.11 | 0.45 / 0.27 |
| shape hexagon | 388 | 0.85 | 0.68 / 0.19 | 0.79 / 1.25 | 0.57 / 0.17 |
| shape octagon | 390 | 0.84 | 0.69 / 0.08 | 0.81 / 0.92 | 0.57 / 0.06 |
| clearance 0.1-0.2 mm | 248 | 0.81 | 0.46 / 0.22 | 0.60 / 1.44 | 0.41 / 0.24 |
| clearance 0.2-0.3 mm | 535 | 0.84 | 0.65 / 0.16 | 0.74 / 1.18 | 0.55 / 0.17 |
| clearance 0.3-0.4 mm | 513 | 0.84 | 0.76 / 0.11 | 0.82 / 0.80 | 0.59 / 0.13 |
| clearance 0.4-0.5 mm | 265 | 0.83 | 0.86 / 0.14 | 0.88 / 0.85 | 0.69 / 0.17 |
| tilt 0 | 814 | 0.80 | 0.51 / 0.01 | 0.74 / 0.16 | 0.42 / 0.07 |
| tilt 0-2.5 deg | 354 | 0.86 | 0.82 / 0.21 | 0.77 / 2.11 | 0.64 / 0.18 |
| tilt 2.5-max deg | 393 | 0.86 | 0.94 / 0.39 | 0.83 / 1.91 | 0.77 / 0.35 |
| yaw 0-4 deg (non-round) | 380 | 0.83 | 0.81 / 0.13 | 0.84 / 0.77 | 0.64 / 0.12 |
| yaw 4-8 deg (non-round) | 389 | 0.86 | 0.62 / 0.17 | 0.77 / 1.24 | 0.52 / 0.17 |
| yaw 8-max deg (non-round) | 403 | 0.84 | 0.49 / 0.24 | 0.65 / 1.26 | 0.44 / 0.22 |
| all | 1561 | 0.83 | 0.69 / 0.15 | 0.77 / 1.04 | 0.56 / 0.17 |
