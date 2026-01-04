from __future__ import annotations

import torch
from torch import nn

from ..core.rng import rng_make_generator, rng_split
from ..core.types import Observation
from ..processes.base import Process
from ..views.base import View, ViewChain


def _wrap_view(view: nn.Module) -> View:
    if isinstance(view, View):
        return view
    if isinstance(view, nn.Sequential):
        return ViewChain(*list(view))
    raise TypeError(
        "Each view must be a View or nn.Sequential of View modules. "
        f"Got {type(view).__name__}."
    )


class SynthPipeline(nn.Module):
    def __init__(self, process: Process, views: dict[str, nn.Module]):
        super().__init__()

        if not isinstance(views, dict) or not views:
            raise ValueError("views must be a non-empty dict[str, nn.Module].")

        self.process = process
        wrapped = {name: _wrap_view(view) for name, view in views.items()}
        self.views = nn.ModuleDict(wrapped)

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        rng: torch.Generator | None = None,
    ) -> dict[str, Observation]:
        generator, seed, _ = rng_make_generator(rng=rng, device=device)
        child_generators = rng_split(
            rng=generator,
            num_children=1 + len(self.views),
            device=device,
        )

        process_rng = child_generators[0]
        view_rngs = child_generators[1:]

        latent = self.process(batch_size, device, rng=process_rng)

        batch: dict[str, Observation] = {}
        for (name, view), view_rng in zip(self.views.items(), view_rngs, strict=True):
            observation = view(latent, rng=view_rng)
            meta = {**observation.meta, "pipeline_seed": seed}
            batch[name] = Observation(x=observation.x, y=observation.y, meta=meta)

        return batch
