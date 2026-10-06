"""FORGE project (Noseworthy et al., arXiv:2408.04587) built on the `mujoco_rl_bed` core.

Importing `forge` registers its terms (`forge.terms`) and tasks (`forge.task`: `forge_peg`).
Entry points: `train.py`, `eval.py` in this folder (runs are written to `projects/forge/runs/`).
"""

from forge import task, terms  # noqa: F401  (registration side effects)
