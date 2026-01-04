from __future__ import annotations

import torch
from torch import nn

from ..core.types import LatentState


class Process(nn.Module):
    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        raise NotImplementedError
