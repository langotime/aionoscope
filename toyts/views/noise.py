from __future__ import annotations

import math

import torch

from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View


class NoiseView(View):
    def __init__(self, *, noise_std: float) -> None:
        super().__init__()
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}.")
        self.noise_std = noise_std

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        if isinstance(input_state, LatentState):
            raise TypeError("NoiseView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        generator, seed, _ = rng_make_generator(rng=rng, device=input_state.x.device)
        noise = torch.randn(
            input_state.x.shape,
            generator=generator,
            device=input_state.x.device,
        )  # [B, C, L]
        observed_signal = input_state.x + noise * self.noise_std  # [B, C, L]

        meta = {
            "view": "NoiseView",
            "seed": seed,
            "noise_std": self.noise_std,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)


class BaselineWanderView(View):
    def __init__(
        self,
        *,
        amplitude_std: float,
        freq_min: float,
        freq_max: float,
    ) -> None:
        super().__init__()
        if amplitude_std < 0:
            raise ValueError(f"amplitude_std must be non-negative, got {amplitude_std}.")
        if freq_min <= 0 or freq_max <= 0:
            raise ValueError("freq_min/max must be positive.")
        if freq_max < freq_min:
            raise ValueError("freq_max must be >= freq_min.")

        self.amplitude_std = amplitude_std
        self.freq_min = freq_min
        self.freq_max = freq_max

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        if isinstance(input_state, LatentState):
            raise TypeError("BaselineWanderView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        generator, seed, _ = rng_make_generator(rng=rng, device=input_state.x.device)

        batch_size, channels, seq_len = input_state.x.shape
        time_grid = torch.linspace(0, 1, steps=seq_len, device=input_state.x.device)  # [L]
        time_grid = time_grid[None, None, :]  # [1, 1, L]

        freq = torch.rand(
            (batch_size, channels, 1),
            generator=generator,
            device=input_state.x.device,
        )  # [B, C, 1]
        freq = self.freq_min + (self.freq_max - self.freq_min) * freq  # [B, C, 1]

        phase = torch.rand(
            (batch_size, channels, 1),
            generator=generator,
            device=input_state.x.device,
        )  # [B, C, 1]
        phase = phase * (2.0 * math.pi)  # [B, C, 1]

        amplitude = torch.randn(
            (batch_size, channels, 1),
            generator=generator,
            device=input_state.x.device,
        )  # [B, C, 1]
        amplitude = amplitude * self.amplitude_std  # [B, C, 1]

        baseline = amplitude * torch.sin(2.0 * math.pi * freq * time_grid + phase)  # [B, C, L]
        observed_signal = input_state.x + baseline  # [B, C, L]

        meta = {
            "view": "BaselineWanderView",
            "seed": seed,
            "amplitude_std": self.amplitude_std,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)


class NormalizeView(View):
    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        if isinstance(input_state, LatentState):
            raise TypeError("NormalizeView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        mean = input_state.x.mean(dim=-1, keepdim=True)  # [B, C, 1]
        std = input_state.x.std(dim=-1, keepdim=True, unbiased=False)  # [B, C, 1]
        if torch.any(std <= 0):
            raise ValueError("NormalizeView requires positive std for normalization.")
        observed_signal = (input_state.x - mean) / std  # [B, C, L]

        meta = {
            "view": "NormalizeView",
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)
