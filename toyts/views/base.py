from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..core.rng import rng_make_generator, rng_split
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta


class View(nn.Module):
    """Abstract base class for a view transformation.

    A `View` is a callable module that transforms a `LatentState` or an
    `Observation` into a new `Observation`. Views are used to create different
    "versions" of the underlying data, such as by adding noise, dropping
    channels, or changing the sampling rate.
    """

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply the view transformation.

        Args:
            input_state: The input `LatentState` or `Observation` to transform.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` object representing the transformed view.
        """
        raise NotImplementedError


class ViewChain(View):
    """A sequential container for `View` modules.

    Modules will be added to it in the order they are passed in the constructor.
    The `forward` method of this class will call the `forward` of each contained
    view in order. View metadata is accumulated into `Observation.meta["views"]`.

    Args:
        *views: A sequence of `View` modules to chain together.
    """

    def __init__(self, *views: View) -> None:
        """Initialize the chain with one or more views."""
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
        """Pass the input through the chain of views.

        Args:
            input_state: The initial `LatentState` or `Observation`.
            rng: An optional `torch.Generator` for reproducibility. The same
                generator is passed to each view in the chain.

        Returns:
            The final `Observation` after applying all views in sequence, with
            per-view metadata stored in `meta["views"]`.
        """
        device = input_state.x.device if isinstance(input_state, Observation) else input_state.centers.device
        generator, _, _ = rng_make_generator(rng=rng, device=device)
        child_generators = rng_split(
            rng=generator,
            num_children=len(self.views),
            device=device,
        )

        output: LatentState | Observation = input_state
        views_trace: list[dict[str, Any]] = []
        if isinstance(input_state, Observation):
            existing_views = input_state.meta.get("views")
            if existing_views is not None:
                if not isinstance(existing_views, list):
                    raise ValueError(
                        "Observation meta 'views' must be a list. "
                        f"Got {type(existing_views).__name__}."
                    )
                for entry in existing_views:
                    if not isinstance(entry, dict):
                        raise ValueError(
                            "Observation meta 'views' entries must be dicts. "
                            f"Got {type(entry).__name__}."
                        )
                views_trace = list(existing_views)
        for view, child_rng in zip(self.views, child_generators, strict=True):
            output = view(output, rng=child_rng)
            if not isinstance(output, Observation):
                raise TypeError("ViewChain must end with an Observation.")
            process_meta = utils_extract_process_meta(output.meta)
            view_meta = dict(output.meta)
            view_meta.pop("process", None)
            view_meta.pop("views", None)
            views_trace.append(view_meta)
            output = Observation(
                x=output.x,
                y=output.y,
                meta={**output.meta, "views": views_trace},
            )
        if not isinstance(output, Observation):
            raise TypeError("ViewChain must end with an Observation.")
        process_meta = utils_extract_process_meta(output.meta)
        meta = {
            "process": process_meta,
            "views": views_trace,
        }
        return Observation(x=output.x, y=output.y, meta=meta)
