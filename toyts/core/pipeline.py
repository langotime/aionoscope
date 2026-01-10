from __future__ import annotations

import torch
from torch import nn

from ..core.rng import rng_make_generator, rng_split
from ..core.types import Observation
from ..processes.base import Process
from ..views.base import View, ViewChain


def _wrap_view(view: nn.Module) -> View:
    """Normalize view modules to a View instance."""
    if isinstance(view, ViewChain):
        return view
    if isinstance(view, View):
        return ViewChain(view)
    if isinstance(view, nn.Sequential):
        return ViewChain(*list(view))
    raise TypeError(
        "Each view must be a View or nn.Sequential of View modules. "
        f"Got {type(view).__name__}."
    )


class SynthPipeline(nn.Module):
    """A pipeline that synthesizes observations from a latent process and a set of views.

    This module orchestrates the generation of synthetic data by first invoking a
    `Process` module to create a latent state, and then passing that latent state
    through one or more `View` modules to generate observations. It handles RNG
    splitting and metadata propagation across views.

    Args:
        process: A `Process` module that generates the latent state.
        views: A dictionary of `View` or `nn.Sequential` modules. Each entry
            represents a named "view" of the data (e.g., "clean", "noisy").
    """

    def __init__(self, process: Process, views: dict[str, nn.Module]):
        """Initialize the pipeline with a process and view dictionary."""
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
        """Generate a batch of synthetic observations.

        Args:
            batch_size: The number of samples to generate.
            device: The torch device to use for generation.
            rng: An optional `torch.Generator` for reproducibility. If `None`, a
                new generator is created with a time-based seed.

        Returns:
            A dictionary where keys are view names and values are `Observation`
            objects. Each observation contains the synthesized signal `x` of
            shape `[B, C, L]`, along with labels and metadata.
        """
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
