"""Residual RL for peg insertion on top of an axis compliance controller (built on the FORGE project).

Importing `compliant_RL` registers its terms and the `compliant_peg` task (and FORGE's terms it reuses).
Entry points: `train.py`, `eval.py` in this folder (runs are written to `projects/compliant_RL/runs/`).
"""

import forge  # noqa: F401  (registers the FORGE terms reused here)
from compliant_RL import task, terms  # noqa: F401  (registration side effects)
