from __future__ import annotations

import torch

from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View


class SamplingAggregationView(View):
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

        if self.window is None or self.window <= 0:
            raise ValueError("window must be a positive integer for aggregation modes.")
        if seq_len % self.window != 0:
            raise ValueError(
                f"seq_len {seq_len} must be divisible by window {self.window} for aggregation."
            )

        reshaped_signal = input_state.x.view(
            batch_size,
            channels,
            seq_len // self.window,
            self.window,
        )  # [B, C, L', W]

        if self.mode == "mean":
            observed_signal = reshaped_signal.mean(dim=-1)  # [B, C, L']
        else:
            observed_signal = reshaped_signal.max(dim=-1).values  # [B, C, L']

        meta = {
            "view": "SamplingAggregationView",
            "mode": self.mode,
            "window": self.window,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)
