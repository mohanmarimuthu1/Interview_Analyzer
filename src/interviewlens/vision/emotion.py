"""Facial expression classifier on top of a timm backbone."""

from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn


class EmotionNet(nn.Module):
    """timm backbone + linear head. ``embed`` gives the penultimate features
    (512-d for resnet18, 768-d for convnext_tiny) used later by fusion."""

    def __init__(
        self, arch: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0
    ) -> None:
        super().__init__()
        # timm models expose forward_features / forward_head / num_features
        self.backbone: Any = timm.create_model(
            arch, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
        )

    @property
    def embed_dim(self) -> int:
        return int(self.backbone.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        return self.backbone.forward_head(feats, pre_logits=True)
