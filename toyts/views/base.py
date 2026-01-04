from __future__ import annotations

import torch
from torch import nn

from ..core.types import LatentState, Observation


class View(nn.Module):
    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        raise NotImplementedError


class ViewChain(View):
    def __init__(self, *views: View) -> None:
        super().__init__()
        if not views:
            raise ValueError("ViewChain requires at least one view.")
        self.views = nn.ModuleList(views)

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        output: LatentState | Observation = input_state
        for view in self.views:
            output = view(output, rng=rng)
        if not isinstance(output, Observation):
            raise TypeError("ViewChain must end with an Observation.")
        return output
