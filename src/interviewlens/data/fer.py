"""FER2013 and FER+ facial expression data (48x48 grayscale)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

FER2013_CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
FERPLUS_CLASSES = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)
# official FER2013 Usage values; PublicTest is our val set, PrivateTest the test set
SPLITS = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        # icml_face_data.csv from Kaggle has " Usage", " pixels" headers
        header = [h.strip() for h in next(reader)]
        return [dict(zip(header, row, strict=True)) for row in reader]


def load_fer(
    csv_path: str | Path, split: str, ferplus_csv: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Images (N, 48, 48) uint8 and labels (N,) int64 for one split.

    With ``ferplus_csv`` the labels come from the FER+ majority vote over the
    8 emotion columns. Rows where unknown/NF gets the most votes, or that FER+
    dropped (empty image name), are removed.
    """
    rows = _read_csv(csv_path)
    keep = [i for i, r in enumerate(rows) if r["Usage"] == SPLITS[split]]

    if ferplus_csv is None:
        labels = [int(rows[i]["emotion"]) for i in keep]
    else:
        plus = _read_csv(ferplus_csv)
        if len(plus) != len(rows):
            raise ValueError(f"FER+ has {len(plus)} rows, FER2013 has {len(rows)}")
        cols = (*FERPLUS_CLASSES, "unknown", "NF")
        kept, labels = [], []
        for i in keep:
            votes = [int(plus[i][c]) for c in cols]
            best = int(np.argmax(votes))
            if plus[i]["Image name"].strip() and best < len(FERPLUS_CLASSES):
                kept.append(i)
                labels.append(best)
        keep = kept

    images = np.stack(
        [np.array(rows[i]["pixels"].split(), dtype=np.uint8).reshape(48, 48) for i in keep]
    )
    return images, np.array(labels, dtype=np.int64)


def build_transform(
    img_size: int, train: bool, hflip: bool, rotate: float, jitter: float, randaugment: bool
) -> v2.Compose:
    t: list[v2.Transform] = [
        v2.Resize(img_size, antialias=True),
        v2.Grayscale(num_output_channels=3),
    ]
    if train:
        if hflip:
            t.append(v2.RandomHorizontalFlip())
        t.append(v2.RandomRotation(rotate))
        t.append(v2.ColorJitter(brightness=jitter, contrast=jitter))
        if randaugment:
            t.append(v2.RandAugment())
    t += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(t)


class FERDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: v2.Compose) -> None:
        self.images = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 48, 48) uint8
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[i]), self.labels[i]
