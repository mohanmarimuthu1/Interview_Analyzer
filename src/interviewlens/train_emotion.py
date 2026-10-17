"""M2 emotion training: train on FER2013 Training, pick the best epoch on
PublicTest, report PrivateTest accuracy / macro-F1 and a confusion matrix."""

import json
import logging
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from interviewlens.data.fer import (
    FER2013_CLASSES,
    FERPLUS_CLASSES,
    FERDataset,
    build_transform,
    load_fer,
)
from interviewlens.eval.metrics import accuracy, confusion_matrix, macro_f1
from interviewlens.utils.git import git_hash
from interviewlens.utils.seed import seed_everything
from interviewlens.vision.emotion import EmotionNet

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _loader(cfg: DictConfig, split: str, ferplus_csv: str | None) -> DataLoader:
    train = split == "train"
    images, labels = load_fer(cfg.data.csv, split, ferplus_csv)
    tf = build_transform(cfg.data.img_size, train, **cfg.train.aug)
    return DataLoader(
        FERDataset(images, labels, tf),
        batch_size=cfg.train.batch_size,
        shuffle=train,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device, enabled=device == "cuda"):
            ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def _save_confusion(cm: np.ndarray, classes: tuple[str, ...], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="emotion")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    commit = git_hash()
    log.info("git %s\n%s", commit, OmegaConf.to_yaml(cfg))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ferplus_csv = cfg.data.ferplus_csv if cfg.data.labels == "ferplus" else None
    classes = FERPLUS_CLASSES if ferplus_csv else FER2013_CLASSES
    loaders = {s: _loader(cfg, s, ferplus_csv) for s in ("train", "val", "test")}
    log.info("sizes %s", {s: len(dl.dataset) for s, dl in loaders.items()})  # type: ignore[arg-type]

    model = EmotionNet(cfg.model.arch, len(classes), cfg.model.pretrained, cfg.model.drop_rate)
    model.to(device)

    weight = None
    if cfg.train.class_balance:
        counts = torch.bincount(loaders["train"].dataset.labels, minlength=len(classes))  # type: ignore[attr-defined]
        weight = (counts.sum() / (len(classes) * counts.clamp(min=1))).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.train.label_smoothing)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=cfg.train.epochs * len(loaders["train"])
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    run = None
    if cfg.wandb.enabled:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=cfg.run_name,
            config={**OmegaConf.to_container(cfg, resolve=True), "git_hash": commit},  # type: ignore[dict-item]
        )

    best_acc, best_path = -1.0, out_dir / "best.pt"
    for epoch in range(cfg.train.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)

        val_acc = accuracy(confusion_matrix(*_predict(model, loaders["val"], device), len(classes)))
        log.info("epoch %d loss %.4f val_acc %.4f", epoch, total / n, val_acc)
        if run:
            run.log({"epoch": epoch, "train_loss": total / n, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    cm = confusion_matrix(*_predict(model, loaders["test"], device), len(classes))
    metrics = {
        "run": cfg.run_name,
        "labels": cfg.data.labels,
        "arch": cfg.model.arch,
        "seed": cfg.seed,
        "git_hash": commit,
        "best_val_acc": best_acc,
        "test_acc": accuracy(cm),
        "test_macro_f1": macro_f1(cm),
        "classes": list(classes),
        "confusion": cm.tolist(),
    }
    log.info("test acc %.4f macro-F1 %.4f", metrics["test_acc"], metrics["test_macro_f1"])
    if run:
        run.summary.update({k: metrics[k] for k in ("test_acc", "test_macro_f1", "best_val_acc")})
        run.finish()

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    title = f"{cfg.run_name}  test acc {metrics['test_acc']:.3f}"
    _save_confusion(cm, classes, title, out_dir / "confusion.png")

    report = Path(cfg.reports_dir) / cfg.run_name
    report.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "confusion.png"):
        shutil.copy(out_dir / name, report / name)
    log.info("wrote %s", report)


if __name__ == "__main__":
    main()
