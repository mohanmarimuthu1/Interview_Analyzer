"""Tests for FER data + emotion model - skipped when torch/timm are not installed."""

import csv
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("torchvision")

from interviewlens.data.fer import (  # noqa: E402
    FERPLUS_CLASSES,
    FERDataset,
    build_transform,
    load_fer,
)
from interviewlens.vision.emotion import EmotionNet  # noqa: E402

USAGES = ["Training"] * 6 + ["PublicTest"] * 3 + ["PrivateTest"] * 3


def _write_fer(path, n_classes=7):
    rng = np.random.default_rng(0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "pixels", "Usage"])
        for i, usage in enumerate(USAGES):
            px = " ".join(map(str, rng.integers(0, 256, 48 * 48)))
            w.writerow([i % n_classes, px, usage])


def _write_ferplus(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Usage", "Image name", *FERPLUS_CLASSES, "unknown", "NF"])
        for i, usage in enumerate(USAGES):
            votes = [0] * 10
            votes[i % 10] = 10  # rows 8, 9 -> unknown / NF wins, dropped
            name = "" if i == 0 else f"fer{i:07d}.png"  # row 0 dropped by FER+
            w.writerow([usage, name, *votes])


def test_load_fer2013(tmp_path):
    _write_fer(tmp_path / "fer.csv")
    x, y = load_fer(tmp_path / "fer.csv", "train")
    assert x.shape == (6, 48, 48) and x.dtype == np.uint8
    assert y.tolist() == [0, 1, 2, 3, 4, 5]
    x, y = load_fer(tmp_path / "fer.csv", "test")
    assert y.tolist() == [9 % 7, 10 % 7, 11 % 7]


def test_load_ferplus_majority(tmp_path):
    _write_fer(tmp_path / "fer.csv")
    _write_ferplus(tmp_path / "plus.csv")
    _, y = load_fer(tmp_path / "fer.csv", "train", tmp_path / "plus.csv")
    assert y.tolist() == [1, 2, 3, 4, 5]  # row 0 has no image name
    _, y = load_fer(tmp_path / "fer.csv", "val", tmp_path / "plus.csv")
    assert y.tolist() == [6, 7]  # row 8 is unknown


def test_dataset_item_shape():
    tf = build_transform(64, True, hflip=True, rotate=10, jitter=0.3, randaugment=True)
    ds = FERDataset(np.zeros((2, 48, 48), np.uint8), np.array([0, 1]), tf)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert int(y) == 0


def test_model_shapes():
    model = EmotionNet("resnet18", 7, pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 7)
    assert model.embed_dim == 512
    assert model.embed(x).shape == (2, 512)


def test_tiny_batch_overfit():
    torch.manual_seed(0)
    model = EmotionNet("resnet18", 2, pretrained=False)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1] * 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert (model(x).argmax(1) == y).all()


def test_train_emotion_smoke(tmp_path):
    _write_fer(tmp_path / "fer.csv")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "interviewlens.train_emotion",
            f"data.csv={tmp_path / 'fer.csv'}",
            "data.img_size=32",
            "data.num_workers=0",
            "model.pretrained=false",
            "train.epochs=1",
            "train.batch_size=4",
            f"reports_dir={tmp_path / 'reports'}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    report = tmp_path / "reports" / "fer2013_resnet18_s42"
    assert (report / "metrics.json").exists()
    assert (report / "confusion.png").exists()
