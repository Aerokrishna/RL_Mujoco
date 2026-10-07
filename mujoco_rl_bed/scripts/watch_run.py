"""Print a compact line of training scalars every N environment steps (handy over SSH).

Reads the run's TensorBoard event files, so it works while training is running (from another
terminal / tmux pane) and on finished runs. Ctrl+C to stop.

Usage:
    python scripts/watch_run.py run=forge_peg_20261007-095722_s0_cpu_asym_n1_c025_dr
    python scripts/watch_run.py run=projects/forge/runs/<run_dir> every=100000 interval=60

`run` may be an absolute/relative path, or a folder name inside `run_root`
(default: projects/forge/runs). With `run=latest`, the newest run in `run_root` is used.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from mujoco_rl_bed.sim.scene import PROJECT_ROOT
from mujoco_rl_bed.utils.config import apply_overrides, parse_cli

# (label, TensorBoard tag, format, number of recent values averaged)
COLUMNS = [
    ("fps", "time/fps", "{:.0f}", 1),
    ("rew", "rollout/ep_rew_mean", "{:.1f}", 1),
    ("success", "episode/is_success", "{:.2f}", 3),
    ("ever_succ", "episode/ever_success", "{:.2f}", 3),
    ("placed", "episode/ever_placed", "{:.2f}", 3),
    ("Fmean", "episode/contact_force_mean", "{:.2f}", 3),
    ("Fmax", "episode/contact_force_max", "{:.1f}", 3),
    ("force_pen", "episode/reward_terms/contact_penalty", "{:.1f}", 3),
    ("act_rate", "episode/reward_terms/action_rate", "{:.2f}", 3),
    ("p_final", "episode/pred_success_final", "{:.2f}", 3),
    ("std", "train/std", "{:.3f}", 1),
    ("ev", "train/explained_variance", "{:.2f}", 1),
]


@dataclass
class WatchCfg:
    """Options.

    Attributes:
        run: Run directory (path or folder name in `run_root`), or "latest".
        run_root: Folder holding runs (relative paths are relative to the repository root).
        every: Print one line each time the step count crosses a multiple of this.
        interval: Seconds between checks of the event files.
    """

    run: str = "latest"
    run_root: str = "projects/forge/runs"
    every: int = 50_000
    interval: float = 30.0


def find_run(run: str, run_root: str) -> Path:
    """Resolve the run directory.

    Args:
        run: Path, folder name, or "latest".
        run_root: Folder holding runs.

    Returns:
        Existing run directory.
    """
    root = Path(run_root) if Path(run_root).is_absolute() else PROJECT_ROOT / run_root
    if run == "latest":
        runs = sorted((p for p in root.iterdir() if (p / "config.json").exists()), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise FileNotFoundError(f"No runs in {root}")
        return runs[-1]
    for cand in (Path(run), root / run, PROJECT_ROOT / run):
        if (cand / "config.json").exists():
            return cand
    raise FileNotFoundError(f"Run '{run}' not found (looked in cwd, {root}, {PROJECT_ROOT})")


def main(argv: list[str]) -> None:
    """Poll the run's scalars and print a line per `every` steps.

    Args:
        argv: key=value options.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = apply_overrides(WatchCfg(), parse_cli(argv))
    run_dir = find_run(cfg.run, cfg.run_root)
    print(f"[watch] {run_dir}  (every {cfg.every} steps; Ctrl+C to stop)", flush=True)
    last_bucket = -1
    while True:
        files = sorted(glob.glob(os.path.join(run_dir, "tb", "**", "events.*"), recursive=True))
        if files:
            ea = EventAccumulator(os.path.dirname(files[-1]), size_guidance={"scalars": 0})
            ea.Reload()
            tags = set(ea.Tags()["scalars"])
            if "rollout/ep_rew_mean" in tags:
                step = ea.Scalars("rollout/ep_rew_mean")[-1].step
                if step // cfg.every > last_bucket:
                    last_bucket = step // cfg.every
                    parts = [f"step={step / 1e3:.0f}k"]
                    for label, tag, fmt, n in COLUMNS:
                        if tag in tags:
                            vals = [e.value for e in ea.Scalars(tag)][-n:]
                            parts.append(f"{label}=" + fmt.format(sum(vals) / len(vals)))
                    print(" ".join(parts), flush=True)
        time.sleep(cfg.interval)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        pass
