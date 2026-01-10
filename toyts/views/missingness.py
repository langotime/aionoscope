from __future__ import annotations

from typing import Any

import torch

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
        dropout_prob: The probability of setting a single data point to zero.
        gap_prob: The probability of introducing a zero-gap in a given channel.
        gap_length: The length of the zero-gap, if one is introduced.
        hold_prob: The probability of replacing a data point with its previous
            value (sample-and-hold).
    """

    def __init__(
        self,
        *,
        dropout_prob: float,
        gap_prob: float,
        gap_length: int,
        hold_prob: float,
    ) -> None:
        """Initialize missingness parameters."""
        super().__init__()
        if not (0.0 <= dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1].")
        if not (0.0 <= gap_prob <= 1.0):
            raise ValueError("gap_prob must be in [0, 1].")
        if gap_length < 0:
            raise ValueError("gap_length must be non-negative.")
        if not (0.0 <= hold_prob <= 1.0):
            raise ValueError("hold_prob must be in [0, 1].")

        self.dropout_prob = dropout_prob
        self.gap_prob = gap_prob
        self.gap_length = gap_length
        self.hold_prob = hold_prob

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

        dropout_prob = meta["dropout_prob"]
        gap_prob = meta["gap_prob"]
        gap_length = meta["gap_length"]
        hold_prob = meta["hold_prob"]
        if not isinstance(dropout_prob, (float, int)):
            raise ValueError(
                f"dropout_prob must be a float, got {type(dropout_prob).__name__}."
            )
        if not isinstance(gap_prob, (float, int)):
            raise ValueError(f"gap_prob must be a float, got {type(gap_prob).__name__}.")
        if not isinstance(gap_length, int):
            raise ValueError(f"gap_length must be an int, got {type(gap_length).__name__}.")
        if not isinstance(hold_prob, (float, int)):
            raise ValueError(f"hold_prob must be a float, got {type(hold_prob).__name__}.")

        generator = torch.Generator(device=device)
        generator.manual_seed(int(meta["mask_seed"]))

        dropout_mask = torch.ones(
            (batch_size, channels, seq_len),
            device=device,
            dtype=torch.bool,
        )  # [B, C, L]
        if dropout_prob > 0:
            dropout_mask = torch.rand(
                (batch_size, channels, seq_len),
                generator=generator,
                device=device,
            )  # [B, C, L]
            dropout_mask = dropout_mask > float(dropout_prob)  # [B, C, L]

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
        if gap_prob > 0 and gap_length > 0:
            apply_gap = torch.rand(
                (batch_size, channels),
                generator=generator,
                device=device,
            )  # [B, C]
            apply_gap = apply_gap < float(gap_prob)  # [B, C]

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
        if hold_prob > 0:
            hold_mask = torch.rand(
                (batch_size, channels, seq_len),
                generator=generator,
                device=device,
            )  # [B, C, L]
            hold_mask = hold_mask < float(hold_prob)  # [B, C, L]

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

        meta = {
            "view": "MissingnessView",
            "seed": seed,
            "mask_seed": mask_seed,
            "mask_version": 1,
            "dropout_prob": self.dropout_prob,
            "gap_prob": self.gap_prob,
            "gap_length": self.gap_length,
            "hold_prob": self.hold_prob,
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
