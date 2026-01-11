from __future__ import annotations

import math

import torch

from ..core.samplers import ConstantSampler, SamplerLike, sampler_from_value, sampler_sample
from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View


class NoiseView(View):
    """A view that adds Gaussian noise to an observation.

    This module applies simple additive white Gaussian noise (AWGN) to the
    input signal.

    Args:
        noise_std: Sampler for the Gaussian noise standard deviation.
    """

    def __init__(self, *, noise_std: SamplerLike[float]) -> None:
        """Initialize additive noise parameters."""
        super().__init__()
        noise_std_sampler = sampler_from_value(noise_std, name="noise_std")
        if isinstance(noise_std_sampler, ConstantSampler) and noise_std_sampler.value < 0:
            raise ValueError(
                f"noise_std must be non-negative, got {noise_std_sampler.value}."
            )
        self.noise_std = noise_std
        self.noise_std_sampler = noise_std_sampler

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply additive Gaussian noise.

        This method expects an `Observation` as input.

        Args:
            input_state: An `Observation` object.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` object where `x` is the noisy signal, with the
            same shape as the input `[B, C, L]`.
        """
        if isinstance(input_state, LatentState):
            raise TypeError("NoiseView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        generator, seed, _ = rng_make_generator(rng=rng, device=input_state.x.device)
        batch_size = input_state.x.shape[0]
        noise_std = sampler_sample(
            sampler=self.noise_std_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="noise_std",
        )  # [B]
        if torch.any(noise_std < 0):
            raise ValueError("noise_std must be non-negative for all samples.")

        noise = torch.randn(
            input_state.x.shape,
            generator=generator,
            device=input_state.x.device,
        )  # [B, C, L]
        observed_signal = input_state.x + noise * noise_std[:, None, None]  # [B, C, L]

        samples = {"noise_std": noise_std}
        spec = {"noise_std": self.noise_std_sampler.spec()}
        meta = {
            "view": "NoiseView",
            "seed": seed,
            "noise_std": noise_std,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)


class BaselineWanderView(View):
    """A view that adds low-frequency sinusoidal noise (baseline wander).

    This module simulates baseline wander by adding a separate sine wave to each
    channel of the input signal. The amplitude, frequency, and phase of the
    sine wave are randomized for each channel.

    Args:
        amplitude_std: Sampler for the sine-wave amplitude scale.
        freq_min: Sampler for the minimum sine frequency.
        freq_max: Sampler for the maximum sine frequency.
    """

    def __init__(
        self,
        *,
        amplitude_std: SamplerLike[float],
        freq_min: SamplerLike[float],
        freq_max: SamplerLike[float],
    ) -> None:
        """Initialize baseline wander parameters."""
        super().__init__()
        amplitude_std_sampler = sampler_from_value(amplitude_std, name="amplitude_std")
        freq_min_sampler = sampler_from_value(freq_min, name="freq_min")
        freq_max_sampler = sampler_from_value(freq_max, name="freq_max")
        if isinstance(amplitude_std_sampler, ConstantSampler) and amplitude_std_sampler.value < 0:
            raise ValueError(
                f"amplitude_std must be non-negative, got {amplitude_std_sampler.value}."
            )
        if isinstance(freq_min_sampler, ConstantSampler) and freq_min_sampler.value <= 0:
            raise ValueError(f"freq_min must be positive, got {freq_min_sampler.value}.")
        if isinstance(freq_max_sampler, ConstantSampler) and freq_max_sampler.value <= 0:
            raise ValueError(f"freq_max must be positive, got {freq_max_sampler.value}.")
        if (
            isinstance(freq_min_sampler, ConstantSampler)
            and isinstance(freq_max_sampler, ConstantSampler)
            and freq_max_sampler.value < freq_min_sampler.value
        ):
            raise ValueError("freq_max must be >= freq_min.")

        self.amplitude_std = amplitude_std
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.amplitude_std_sampler = amplitude_std_sampler
        self.freq_min_sampler = freq_min_sampler
        self.freq_max_sampler = freq_max_sampler

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply baseline wander.

        This method expects an `Observation` as input.

        Args:
            input_state: An `Observation` object.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` object where `x` is the signal with baseline
            wander, with the same shape as the input `[B, C, L]`.
        """
        if isinstance(input_state, LatentState):
            raise TypeError("BaselineWanderView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        generator, seed, _ = rng_make_generator(rng=rng, device=input_state.x.device)

        batch_size, channels, seq_len = input_state.x.shape
        time_grid = torch.linspace(0, 1, steps=seq_len, device=input_state.x.device)  # [L]
        time_grid = time_grid[None, None, :]  # [1, 1, L]

        amplitude_std = sampler_sample(
            sampler=self.amplitude_std_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="amplitude_std",
        )  # [B]
        if torch.any(amplitude_std < 0):
            raise ValueError("amplitude_std must be non-negative for all samples.")

        freq_min = sampler_sample(
            sampler=self.freq_min_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="freq_min",
        )  # [B]
        freq_max = sampler_sample(
            sampler=self.freq_max_sampler,
            shape=(batch_size,),
            rng=generator,
            device=input_state.x.device,
            dtype=torch.float32,
            name="freq_max",
        )  # [B]
        if torch.any(freq_min <= 0) or torch.any(freq_max <= 0):
            raise ValueError("freq_min/max must be positive for all samples.")
        if torch.any(freq_max < freq_min):
            raise ValueError("freq_max must be >= freq_min for all samples.")

        # Sample frequency, phase, and amplitude for each channel
        freq = torch.rand(
            (batch_size, channels, 1),
            generator=generator,
            device=input_state.x.device,
        )  # [B, C, 1]
        freq = freq_min[:, None, None] + (freq_max - freq_min)[:, None, None] * freq  # [B, C, 1]

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
        amplitude = amplitude * amplitude_std[:, None, None]  # [B, C, 1]

        # Create and add the baseline wander
        baseline = amplitude * torch.sin(2.0 * math.pi * freq * time_grid + phase)  # [B, C, L]
        observed_signal = input_state.x + baseline  # [B, C, L]

        samples = {
            "amplitude_std": amplitude_std,
            "freq_min": freq_min,
            "freq_max": freq_max,
        }
        spec = {
            "amplitude_std": self.amplitude_std_sampler.spec(),
            "freq_min": self.freq_min_sampler.spec(),
            "freq_max": self.freq_max_sampler.spec(),
        }
        meta = {
            "view": "BaselineWanderView",
            "seed": seed,
            "amplitude_std": amplitude_std,
            "freq_min": freq_min,
            "freq_max": freq_max,
            "freq": freq,
            "phase": phase,
            "amplitude": amplitude,
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
