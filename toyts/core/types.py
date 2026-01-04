from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LatentState:
    """Latent process state produced by a Process module."""

    centers: torch.Tensor  # [B, N]
    latent: torch.Tensor | None  # [B, K, L]
    y: dict[str, torch.Tensor]  # label tensors, e.g. {"shape": [B]}
    meta: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    """Observed signal produced by a View module."""

    x: torch.Tensor  # [B, C, L]
    y: dict[str, torch.Tensor]  # label tensors, e.g. {"shape": [B]}
    meta: dict[str, Any]
