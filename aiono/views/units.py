from __future__ import annotations

import torch

from ..core.samplers import ConstantSampler, SamplerLike, sampler_from_value, sampler_sample
from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View
from ._signal import views_extract_signal


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
        signal, labels, process_meta = views_extract_signal(input_state, name="UnitsAbsoluteView")
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
        capacity_min: Sampler for the minimum capacity value.
        capacity_max: Sampler for the maximum capacity value.
    """

    def __init__(
        self,
        *,
        capacity_min: SamplerLike[float],
        capacity_max: SamplerLike[float],
    ) -> None:
        """Initialize the capacity range for percent scaling."""
        super().__init__()
        capacity_min_sampler = sampler_from_value(capacity_min, name="capacity_min")
        capacity_max_sampler = sampler_from_value(capacity_max, name="capacity_max")
        if isinstance(capacity_min_sampler, ConstantSampler) and capacity_min_sampler.value <= 0:
            raise ValueError(
                f"capacity_min must be positive, got {capacity_min_sampler.value}."
            )
        if isinstance(capacity_max_sampler, ConstantSampler) and capacity_max_sampler.value <= 0:
            raise ValueError(
                f"capacity_max must be positive, got {capacity_max_sampler.value}."
            )
        if (
            isinstance(capacity_min_sampler, ConstantSampler)
            and isinstance(capacity_max_sampler, ConstantSampler)
            and capacity_max_sampler.value <= capacity_min_sampler.value
        ):
            raise ValueError("capacity_max must be greater than capacity_min.")
        self.capacity_min = capacity_min
        self.capacity_max = capacity_max
        self.capacity_min_sampler = capacity_min_sampler
        self.capacity_max_sampler = capacity_max_sampler

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
        signal, labels, process_meta = views_extract_signal(
            input_state, name="UnitsPercentOfCapacityView"
        )
        device = signal.device

        generator, seed, _ = rng_make_generator(rng=rng, device=device)

        capacity_min = sampler_sample(
            sampler=self.capacity_min_sampler,
            shape=(signal.shape[0],),
            rng=generator,
            device=device,
            dtype=torch.float32,
            name="capacity_min",
        )  # [B]
        capacity_max = sampler_sample(
            sampler=self.capacity_max_sampler,
            shape=(signal.shape[0],),
            rng=generator,
            device=device,
            dtype=torch.float32,
            name="capacity_max",
        )  # [B]
        if torch.any(capacity_min <= 0) or torch.any(capacity_max <= 0):
            raise ValueError("capacity_min/max must be positive for all samples.")
        if torch.any(capacity_max <= capacity_min):
            raise ValueError("capacity_max must be greater than capacity_min for all samples.")

        # Sample a capacity for each item in the batch
        capacity = torch.rand(
            (signal.shape[0], 1, 1),
            generator=generator,
            device=device,
        )  # [B, 1, 1]
        capacity = capacity_min[:, None, None] + (
            capacity_max - capacity_min
        )[:, None, None] * capacity  # [B, 1, 1]

        observed_signal = 100.0 * signal / capacity  # [B, C, L]

        samples = {
            "capacity_min": capacity_min,
            "capacity_max": capacity_max,
        }
        spec = {
            "capacity_min": self.capacity_min_sampler.spec(),
            "capacity_max": self.capacity_max_sampler.spec(),
        }
        meta = {
            "view": "UnitsPercentOfCapacityView",
            "units": "percent",
            "seed": seed,
            "capacity_min": capacity_min,
            "capacity_max": capacity_max,
            "capacity": capacity,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class ClippingView(View):
    """A view that clips signal values to a specified min/max range.

    This module simulates sensor saturation by enforcing hard limits on the
    signal's amplitude. Any values below `min_value` are set to `min_value`,
    and any values above `max_value` are set to `max_value`.

    Args:
        min_value: Sampler for the clipping minimum.
        max_value: Sampler for the clipping maximum.
    """

    def __init__(
        self,
        *,
        min_value: SamplerLike[float],
        max_value: SamplerLike[float],
    ) -> None:
        """Initialize clipping bounds."""
        super().__init__()
        min_value_sampler = sampler_from_value(min_value, name="min_value")
        max_value_sampler = sampler_from_value(max_value, name="max_value")
        if (
            isinstance(min_value_sampler, ConstantSampler)
            and isinstance(max_value_sampler, ConstantSampler)
            and max_value_sampler.value <= min_value_sampler.value
        ):
            raise ValueError("max_value must be greater than min_value.")
        self.min_value = min_value
        self.max_value = max_value
        self.min_value_sampler = min_value_sampler
        self.max_value_sampler = max_value_sampler

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
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` where `x` is the clipped signal, with the same
            shape as the input `[B, C, L]`.
        """
        if isinstance(input_state, LatentState):
            raise TypeError("ClippingView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        generator, seed, _ = rng_make_generator(rng=rng, device=input_state.x.device)
        batch_size = input_state.x.shape[0]
        min_value = sampler_sample(
            sampler=self.min_value_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="min_value",
        )  # [B]
        max_value = sampler_sample(
            sampler=self.max_value_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="max_value",
        )  # [B]
        if torch.any(max_value <= min_value):
            raise ValueError("max_value must be greater than min_value for all samples.")
        observed_signal = torch.clamp(
            input_state.x,
            min=min_value[:, None, None],
            max=max_value[:, None, None],
        )  # [B, C, L]
        samples = {
            "min_value": min_value,
            "max_value": max_value,
        }
        spec = {
            "min_value": self.min_value_sampler.spec(),
            "max_value": self.max_value_sampler.spec(),
        }
        meta = {
            "view": "ClippingView",
            "min_value": min_value,
            "max_value": max_value,
            "seed": seed,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)


class NormalizeView(View):
    """A view that normalizes the signal to have zero mean and unit variance.

    This module performs channel-wise instance normalization. For each channel
    in each sample of the batch, it subtracts the mean of that channel and
    divides by its standard deviation.
    """

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply channel-wise normalization.

        This method expects an `Observation` as input.

        Args:
            input_state: An `Observation` object.
            rng: This parameter is ignored.

        Returns:
            An `Observation` object where `x` is the normalized signal, with
            the same shape as the input `[B, C, L]`.
        """
        if isinstance(input_state, LatentState):
            raise TypeError("NormalizeView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        mean = input_state.x.mean(dim=-1, keepdim=True)  # [B, C, 1]
        std = input_state.x.std(dim=-1, keepdim=True, unbiased=False)  # [B, C, 1]
        if torch.any(std <= 0):
            # Add a small epsilon for numerical stability if needed, or raise.
            # For synthetic data, std should ideally not be zero.
            raise ValueError("NormalizeView requires positive std for normalization.")
        observed_signal = (input_state.x - mean) / (std + 1e-8)  # [B, C, L]

        meta = {
            "view": "NormalizeView",
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)
