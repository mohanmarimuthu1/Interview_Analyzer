import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from interviewlens.utils.seed import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_config_composes():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["train.epochs=3"])
    assert cfg.train.epochs == 3
    assert cfg.wandb.enabled is False


def test_seed_is_reproducible():
    seed_everything(0)
    a = (random.random(), np.random.rand())
    seed_everything(0)
    b = (random.random(), np.random.rand())
    assert a == b


def test_train_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "interviewlens.train", "--help"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "train" in out.stdout
