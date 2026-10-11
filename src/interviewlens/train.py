"""Training entry point. Real training loops land per milestone (M2+)."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from interviewlens.utils.git import git_hash
from interviewlens.utils.seed import seed_everything

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    commit = git_hash()
    log.info("git %s\n%s", commit, OmegaConf.to_yaml(cfg))

    if cfg.wandb.enabled:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config={**OmegaConf.to_container(cfg, resolve=True), "git_hash": commit},  # type: ignore[dict-item]
        )
        wandb.finish()


if __name__ == "__main__":
    main()
