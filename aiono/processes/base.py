from __future__ import annotations

import torch
from torch import nn

from ..core.types import LatentState


class Process(nn.Module):
    """Abstract base class for a generative process.

    A `Process` is a callable module that returns a `LatentState`, which
    encapsulates the underlying "ground truth" of a synthetic signal.
    """

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        """Generate a batch of latent states.

        Args:
            batch_size: The number of samples to generate.
            device: The torch device to use for generation.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            A `LatentState` object containing the generated latent process
            and its associated metadata and labels.
        """
        raise NotImplementedError
