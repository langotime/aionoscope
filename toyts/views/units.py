from __future__ import annotations

import torch

from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta, utils_sum_latent
from .base import View


def _extract_signal(
    input_state: LatentState | Observation,
    name: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
    """Extract a single-channel signal and metadata from a LatentState or Observation.

    - If the input is a `LatentState`, it sums the latent components to produce
      a single-channel signal.
    - If the input is an `Observation`, it passes the signal through directly.

    This helper is used by views that can operate on either latent or observed
    signals but produce a single, interpretable output channel.

    Args:
        input_state: The input `LatentState` or `Observation`.
        name: The name of the calling view, for error messages.

    Returns:
        A tuple containing:
        - `signal`: The extracted signal `[B, C, L]`. If from a `LatentState`,
          `C` will be 1.
        - `labels`: The labels from the input state.
        - `process_meta`: The metadata dictionary from the original process.
    """
    if isinstance(input_state, LatentState):
        if input_state.latent is None:
            raise ValueError(f"{name} requires LatentState.latent to be present.")
        signal = utils_sum_latent(input_state.latent)  # [B, 1, L]
        return signal, input_state.y, input_state.meta
    if isinstance(input_state, Observation):
        process_meta = utils_extract_process_meta(input_state.meta)
        return input_state.x, input_state.y, process_meta
    raise TypeError(
        f"{name} expects LatentState or Observation, got {type(input_state).__name__}."
    )


class UnitsAbsoluteView(View):
    """A view that represents the signal in its original, absolute units.

    This view acts as a passthrough, using `_extract_signal` to get a concrete
    signal (either by summing a latent state or using an existing observation)
    and wrapping it in a new `Observation` with appropriate metadata. It's a
    baseline for comparing against other unit transformations.
    """

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Extract the signal and return it in an `Observation`.

        Args:
            input_state: The `LatentState` or `Observation` to process.
            rng: This parameter is ignored.

        Returns:
            An `Observation` with the signal in absolute units. `x` has shape
            `[B, C, L]`.
        """
        signal, labels, process_meta = _extract_signal(
            input_state,
            name="UnitsAbsoluteView",
        )
        observed_signal = signal  # [B, C, L]

        meta = {
            "view": "UnitsAbsoluteView",
            "units": "absolute",
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class UnitsPercentOfCapacityView(View):
    """A view that scales the signal to be a percentage of a random capacity.

    This module simulates a common scenario in monitoring systems where a metric
    is represented not in absolute terms, but as a percentage of a total
    capacity (e.g., CPU usage, disk space).

    For each sample in the batch, a `capacity` value is sampled uniformly from
    `[capacity_min, capacity_max]`. The signal is then transformed by
    `x_percent = 100 * x_absolute / capacity`.

    Args:
        capacity_min: The minimum possible capacity value.
        capacity_max: The maximum possible capacity value.
    """

    def __init__(self, *, capacity_min: float, capacity_max: float) -> None:
        super().__init__()
        if capacity_min <= 0 or capacity_max <= 0:
            raise ValueError("capacity_min/max must be positive.")
        if capacity_max <= capacity_min:
            raise ValueError("capacity_max must be greater than capacity_min.")
        self.capacity_min = capacity_min
        self.capacity_max = capacity_max

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply the percentage-of-capacity transformation.

        Args:
            input_state: The `LatentState` or `Observation` to process.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` where `x` is the signal scaled to a percentage,
            with shape `[B, C, L]`.
        """
        signal, labels, process_meta = _extract_signal(
            input_state,
            name="UnitsPercentOfCapacityView",
        )
        device = signal.device

        generator, seed, _ = rng_make_generator(rng=rng, device=device)

        # Sample a capacity for each item in the batch
        capacity = torch.rand(
            (signal.shape[0], 1, 1),
            generator=generator,
            device=device,
        )  # [B, 1, 1]
        capacity = self.capacity_min + (self.capacity_max - self.capacity_min) * capacity  # [B, 1, 1]

        observed_signal = 100.0 * signal / capacity  # [B, C, L]

        meta = {
            "view": "UnitsPercentOfCapacityView",
            "units": "percent",
            "seed": seed,
            "capacity": capacity,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class ClippingView(View):
    """A view that clips signal values to a specified min/max range.

    This module simulates sensor saturation by enforcing hard limits on the
    signal's amplitude. Any values below `min_value` are set to `min_value`,
    and any values above `max_value` are set to `max_value`.

    Args:
        min_value: The minimum value for the clipping range.
        max_value: The maximum value for the clipping range.
    """

    def __init__(self, *, min_value: float, max_value: float) -> None:
        super().__init__()
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value.")
        self.min_value = min_value
        self.max_value = max_value

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply clipping to the signal.

        This method expects an `Observation` as input.

        Args:
            input_state: An `Observation` object.
            rng: This parameter is ignored.

        Returns:
            An `Observation` where `x` is the clipped signal, with the same
            shape as the input `[B, C, L]`.
        """
        if isinstance(input_state, LatentState):
            raise TypeError("ClippingView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        observed_signal = torch.clamp(
            input_state.x,
            min=self.min_value,
            max=self.max_value,
        )  # [B, C, L]
        meta = {
            "view": "ClippingView",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)
