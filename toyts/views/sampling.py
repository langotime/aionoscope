from __future__ import annotations

import torch

from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View


class SamplingAggregationView(View):
    """A view that downsamples or aggregates the signal over time.

    This module changes the temporal resolution of the signal. It supports three modes:
    1.  **"downsample"**: Selects every `stride`-th data point.
    2.  **"mean"**: Computes the mean over non-overlapping windows of size `window`.
    3.  **"max"**: Computes the max over non-overlapping windows of size `window`.

    Args:
        mode: The operation to perform. One of "downsample", "mean", or "max".
        stride: The step size for "downsample" mode. Must be a positive integer.
        window: The window size for "mean" and "max" modes. The sequence length
            must be divisible by the window size.
    """

    def __init__(
        self,
        *,
        mode: str,
        stride: int | None = None,
        window: int | None = None,
    ) -> None:
        super().__init__()
        valid_modes = {"downsample", "mean", "max"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode}.")
        self.mode = mode
        self.stride = stride
        self.window = window

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply the sampling or aggregation transformation.

        This method expects an `Observation` as input.

        Args:
            input_state: An `Observation` object.
            rng: This parameter is ignored.

        Returns:
            An `Observation` object with the transformed signal `x`. The shape
            will be `[B, C, L']`, where `L'` is the new sequence length.
        """
        if isinstance(input_state, LatentState):
            raise TypeError("SamplingAggregationView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        batch_size, channels, seq_len = input_state.x.shape

        if self.mode == "downsample":
            if self.stride is None or self.stride <= 0:
                raise ValueError("stride must be a positive integer for downsample mode.")
            observed_signal = input_state.x[:, :, :: self.stride]  # [B, C, L']
            meta = {
                "view": "SamplingAggregationView",
                "mode": self.mode,
                "stride": self.stride,
                "process": process_meta,
            }
            return Observation(x=observed_signal, y=input_state.y, meta=meta)

        # --- Aggregation Modes ("mean", "max") ---
        if self.window is None or self.window <= 0:
            raise ValueError("window must be a positive integer for aggregation modes.")
        if seq_len % self.window != 0:
            raise ValueError(
                f"seq_len {seq_len} must be divisible by window {self.window} for aggregation."
            )

        # Reshape to create non-overlapping windows
        reshaped_signal = input_state.x.view(
            batch_size,
            channels,
            seq_len // self.window,
            self.window,
        )  # [B, C, L', W]

        if self.mode == "mean":
            observed_signal = reshaped_signal.mean(dim=-1)  # [B, C, L']
        else:  # self.mode == "max"
            observed_signal = reshaped_signal.max(dim=-1).values  # [B, C, L']

        meta = {
            "view": "SamplingAggregationView",
            "mode": self.mode,
            "window": self.window,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)
