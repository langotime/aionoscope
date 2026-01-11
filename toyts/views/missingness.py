from __future__ import annotations

from typing import Any

import torch

from ..core.samplers import (
    ConstantSampler,
    SamplerLike,
    sampler_from_value,
    sampler_sample,
    sampler_sample_scalar,
)
from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View


class MissingnessView(View):
    """A view that introduces various forms of missing data into an observation.

    This module simulates common data corruption issues by applying one or more
    of the following transformations to an `Observation`:
    1.  **Dropout**: Randomly sets individual data points to zero.
    2.  **Gaps**: Sets a continuous segment of data points in a channel to zero.
    3.  **Hold**: Replaces a data point with the value from the previous time step.

    These operations are applied independently and in sequence.

    Args:
        dropout_prob: Sampler for point dropout probability.
        gap_prob: Sampler for gap activation probability.
        gap_length: Sampler for the gap length (per batch).
        hold_prob: Sampler for sample-and-hold probability.
    """

    def __init__(
        self,
        *,
        dropout_prob: SamplerLike[float],
        gap_prob: SamplerLike[float],
        gap_length: SamplerLike[int],
        hold_prob: SamplerLike[float],
    ) -> None:
        """Initialize missingness parameters."""
        super().__init__()
        dropout_prob_sampler = sampler_from_value(dropout_prob, name="dropout_prob")
        gap_prob_sampler = sampler_from_value(gap_prob, name="gap_prob")
        gap_length_sampler = sampler_from_value(gap_length, name="gap_length")
        hold_prob_sampler = sampler_from_value(hold_prob, name="hold_prob")

        if isinstance(dropout_prob_sampler, ConstantSampler) and not (
            0.0 <= float(dropout_prob_sampler.value) <= 1.0
        ):
            raise ValueError("dropout_prob must be in [0, 1].")
        if isinstance(gap_prob_sampler, ConstantSampler) and not (
            0.0 <= float(gap_prob_sampler.value) <= 1.0
        ):
            raise ValueError("gap_prob must be in [0, 1].")
        if isinstance(hold_prob_sampler, ConstantSampler) and not (
            0.0 <= float(hold_prob_sampler.value) <= 1.0
        ):
            raise ValueError("hold_prob must be in [0, 1].")
        if isinstance(gap_length_sampler, ConstantSampler):
            if isinstance(gap_length_sampler.value, bool) or not isinstance(
                gap_length_sampler.value, int
            ):
                raise ValueError("gap_length must be an integer.")
            if gap_length_sampler.value < 0:
                raise ValueError("gap_length must be non-negative.")

        self.dropout_prob = dropout_prob
        self.gap_prob = gap_prob
        self.gap_length = gap_length
        self.hold_prob = hold_prob
        self.dropout_prob_sampler = dropout_prob_sampler
        self.gap_prob_sampler = gap_prob_sampler
        self.gap_length_sampler = gap_length_sampler
        self.hold_prob_sampler = hold_prob_sampler

    @staticmethod
    def sample_masks(
        meta: dict[str, Any],
        *,
        shape: tuple[int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Recreate missingness masks from view metadata."""
        if "mask_seed" not in meta:
            raise ValueError("MissingnessView meta is missing 'mask_seed'.")
        if "mask_version" not in meta:
            raise ValueError("MissingnessView meta is missing 'mask_version'.")
        if meta["mask_version"] != 1:
            raise ValueError(f"Unsupported mask_version {meta['mask_version']}.")
        if "dropout_prob" not in meta:
            raise ValueError("MissingnessView meta is missing 'dropout_prob'.")
        if "gap_prob" not in meta:
            raise ValueError("MissingnessView meta is missing 'gap_prob'.")
        if "gap_length" not in meta:
            raise ValueError("MissingnessView meta is missing 'gap_length'.")
        if "hold_prob" not in meta:
            raise ValueError("MissingnessView meta is missing 'hold_prob'.")

        if len(shape) != 3:
            raise ValueError(f"shape must be [B, C, L], got {shape}.")
        batch_size, channels, seq_len = shape

        def _resolve_prob(name: str) -> torch.Tensor:
            value = meta[name]
            if isinstance(value, torch.Tensor):
                if value.ndim != 1:
                    raise ValueError(f"{name} must have shape [B], got {value.shape}.")
                if value.shape[0] == 1:
                    value = value.expand(batch_size)
                elif value.shape[0] != batch_size:
                    raise ValueError(
                        f"{name} must have shape [B], got {value.shape}."
                    )
                return value.to(device=device, dtype=torch.float32)
            if isinstance(value, (float, int)):
                return torch.full(
                    (batch_size,),
                    float(value),
                    device=device,
                    dtype=torch.float32,
                )  # [B]
            raise ValueError(f"{name} must be a float or tensor, got {type(value).__name__}.")

        dropout_prob = _resolve_prob("dropout_prob")  # [B]
        gap_prob = _resolve_prob("gap_prob")  # [B]
        hold_prob = _resolve_prob("hold_prob")  # [B]

        if torch.any((dropout_prob < 0.0) | (dropout_prob > 1.0)):
            raise ValueError("dropout_prob must be in [0, 1] for all samples.")
        if torch.any((gap_prob < 0.0) | (gap_prob > 1.0)):
            raise ValueError("gap_prob must be in [0, 1] for all samples.")
        if torch.any((hold_prob < 0.0) | (hold_prob > 1.0)):
            raise ValueError("hold_prob must be in [0, 1] for all samples.")

        gap_length_value = meta["gap_length"]
        if isinstance(gap_length_value, torch.Tensor):
            if gap_length_value.numel() != 1:
                raise ValueError("gap_length must be a scalar when stored as a tensor.")
            if gap_length_value.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError("gap_length tensor must have integer dtype.")
            gap_length = int(gap_length_value.item())
        elif isinstance(gap_length_value, int) and not isinstance(gap_length_value, bool):
            gap_length = gap_length_value
        else:
            raise ValueError(
                f"gap_length must be an int, got {type(gap_length_value).__name__}."
            )
        if gap_length < 0:
            raise ValueError("gap_length must be non-negative.")

        generator = torch.Generator(device=device)
        generator.manual_seed(int(meta["mask_seed"]))

        dropout_mask = torch.ones(
            (batch_size, channels, seq_len),
            device=device,
            dtype=torch.bool,
        )  # [B, C, L]
        if torch.any(dropout_prob > 0):
            dropout_mask = torch.rand(
                (batch_size, channels, seq_len),
                generator=generator,
                device=device,
            )  # [B, C, L]
            dropout_mask = dropout_mask > dropout_prob[:, None, None]  # [B, C, L]

        apply_gap = torch.zeros(
            (batch_size, channels),
            device=device,
            dtype=torch.bool,
        )  # [B, C]
        gap_start = torch.zeros(
            (batch_size, channels),
            device=device,
            dtype=torch.int64,
        )  # [B, C]
        gap_mask = torch.ones(
            (batch_size, channels, seq_len),
            device=device,
            dtype=torch.bool,
        )  # [B, C, L]
        if torch.any(gap_prob > 0) and gap_length > 0:
            apply_gap = torch.rand(
                (batch_size, channels),
                generator=generator,
                device=device,
            )  # [B, C]
            apply_gap = apply_gap < gap_prob[:, None]  # [B, C]

            max_start = seq_len - gap_length
            if max_start < 0:
                raise ValueError(
                    f"gap_length {gap_length} exceeds seq_len {seq_len}."
                )
            gap_start = torch.randint(
                0,
                max_start + 1,
                (batch_size, channels),
                generator=generator,
                device=device,
            )  # [B, C]

            time_idx = torch.arange(seq_len, device=device)  # [L]
            in_gap = (time_idx[None, None, :] >= gap_start[:, :, None]) & (
                time_idx[None, None, :] < gap_start[:, :, None] + gap_length
            )  # [B, C, L]
            gap_mask = ~(apply_gap[:, :, None] & in_gap)  # [B, C, L]

        hold_mask = torch.zeros(
            (batch_size, channels, seq_len),
            device=device,
            dtype=torch.bool,
        )  # [B, C, L]
        if torch.any(hold_prob > 0):
            hold_mask = torch.rand(
                (batch_size, channels, seq_len),
                generator=generator,
                device=device,
            )  # [B, C, L]
            hold_mask = hold_mask < hold_prob[:, None, None]  # [B, C, L]

        return {
            "dropout_mask": dropout_mask,
            "gap_mask": gap_mask,
            "hold_mask": hold_mask,
            "gap_start": gap_start,
            "apply_gap": apply_gap,
        }

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply missing data transformations.

        This method expects an `Observation` as input.

        Args:
            input_state: An `Observation` object.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` object where `x` is the signal with missing data,
            of the same shape as the input `[B, C, L]`.
        """
        if isinstance(input_state, LatentState):
            raise TypeError("MissingnessView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        generator, seed, _ = rng_make_generator(rng=rng, device=input_state.x.device)
        observed_signal = input_state.x  # [B, C, L]
        batch_size, channels, seq_len = observed_signal.shape

        max_seed = 2**63 - 1
        mask_seed = torch.randint(
            0,
            max_seed,
            (1,),
            generator=generator,
            device=input_state.x.device,
        )  # [1]
        mask_seed = int(mask_seed.item())

        dropout_prob = sampler_sample(
            sampler=self.dropout_prob_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="dropout_prob",
        )  # [B]
        gap_prob = sampler_sample(
            sampler=self.gap_prob_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="gap_prob",
        )  # [B]
        hold_prob = sampler_sample(
            sampler=self.hold_prob_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="hold_prob",
        )  # [B]
        gap_length_samples, gap_length_value = sampler_sample_scalar(
            sampler=self.gap_length_sampler,
            rng=generator,
            device=input_state.x.device,
            dtype=torch.int64,
            name="gap_length",
        )  # [1]
        if torch.any((dropout_prob < 0.0) | (dropout_prob > 1.0)):
            raise ValueError("dropout_prob must be in [0, 1] for all samples.")
        if torch.any((gap_prob < 0.0) | (gap_prob > 1.0)):
            raise ValueError("gap_prob must be in [0, 1] for all samples.")
        if torch.any((hold_prob < 0.0) | (hold_prob > 1.0)):
            raise ValueError("hold_prob must be in [0, 1] for all samples.")
        if isinstance(gap_length_value, bool):
            raise ValueError("gap_length must be an integer.")
        gap_length = int(gap_length_value)
        if gap_length < 0:
            raise ValueError("gap_length must be non-negative.")

        samples = {
            "dropout_prob": dropout_prob,
            "gap_prob": gap_prob,
            "gap_length": gap_length_samples,
            "hold_prob": hold_prob,
        }
        spec = {
            "dropout_prob": self.dropout_prob_sampler.spec(),
            "gap_prob": self.gap_prob_sampler.spec(),
            "gap_length": self.gap_length_sampler.spec(),
            "hold_prob": self.hold_prob_sampler.spec(),
        }
        meta = {
            "view": "MissingnessView",
            "seed": seed,
            "mask_seed": mask_seed,
            "mask_version": 1,
            "dropout_prob": dropout_prob,
            "gap_prob": gap_prob,
            "gap_length": gap_length_samples,
            "hold_prob": hold_prob,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }

        masks = self.sample_masks(
            meta,
            shape=(batch_size, channels, seq_len),
            device=input_state.x.device,
        )

        observed_signal = observed_signal * masks["dropout_mask"]  # [B, C, L]
        observed_signal = observed_signal * masks["gap_mask"]  # [B, C, L]
        previous = torch.cat(
            [observed_signal[:, :, :1], observed_signal[:, :, :-1]],
            dim=2,
        )  # [B, C, L]
        observed_signal = torch.where(
            masks["hold_mask"],
            previous,
            observed_signal,
        )  # [B, C, L]

        return Observation(x=observed_signal, y=input_state.y, meta=meta)
