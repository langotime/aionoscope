from __future__ import annotations

import torch
from torch import nn

from ..core.types import LatentState, Observation


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
    view in order.

    Args:
        *views: A sequence of `View` modules to chain together.
    """

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
        """Pass the input through the chain of views.

        Args:
            input_state: The initial `LatentState` or `Observation`.
            rng: An optional `torch.Generator` for reproducibility. The same
                generator is passed to each view in the chain.

        Returns:
            The final `Observation` after applying all views in sequence.
        """
        output: LatentState | Observation = input_state
        for view in self.views:
            output = view(output, rng=rng)
        if not isinstance(output, Observation):
            raise TypeError("ViewChain must end with an Observation.")
        return output
