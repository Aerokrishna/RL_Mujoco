"""Training entry-point tests: KL-adaptive learning rate (rl_games style), activation, end-to-end smoke run."""

from __future__ import annotations

import json

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mujoco_rl_bed.rl.train import AdaptiveLR, main


def test_adaptive_lr_rule() -> None:
    """lr / 1.5 above 2x the target KL, x 1.5 below half of it, unchanged in between, within the bounds."""
    s = AdaptiveLR(1e-4, kl_threshold=0.008)
    s.update(0.02)
    assert s.lr == pytest.approx(1e-4 / 1.5) and s(0.3) == s.lr
    s.update(0.008)
    assert s.lr == pytest.approx(1e-4 / 1.5)
    s.update(0.001)
    assert s.lr == pytest.approx(1e-4)
    for _ in range(100):
        s.update(1.0)
    assert s.lr == pytest.approx(1e-6)


def test_train_smoke_adaptive_elu(tmp_path) -> None:
    """A tiny asymmetric RecurrentPPO run with the adaptive schedule and ELU trains, saves, and adapts the rate."""
    run = main(["task=reach", "algo=recurrent_ppo", "asymmetric=true", "n_envs=2", "vec=dummy", "n_steps=32",
                "batch_size=32", "n_epochs=2", "total_timesteps=256", "lstm_hidden_size=16", "net_arch=(16,)",
                "activation=elu", "lr_schedule=adaptive", "kl_threshold=1e-6", "learning_rate=1e-3",
                f"run_root={tmp_path}", "verbose=0", "checkpoint_every=100000"])
    assert (run / "final_model.zip").exists()
    conf = json.loads((run / "config.json").read_text())["train"]
    assert conf["activation"] == "elu" and conf["lr_schedule"] == "adaptive"
    ea = EventAccumulator(str(next((run / "tb").iterdir())))
    ea.Reload()
    lrs = [e.value for e in ea.Scalars("train/learning_rate")]
    # any real update exceeds a KL target of 1e-6, so the rate is divided by 1.5 after each update
    assert len(lrs) >= 2 and lrs[-1] < lrs[0]
