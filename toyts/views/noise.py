from __future__ import annotations

import math

import torch

from ..core.samplers import ConstantSampler, SamplerLike, sampler_from_value, sampler_sample
from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from .base import View
from ._enabled import views_resolve_enabled_mask
from ._signal import views_extract_signal


def _enabled_any(mask: torch.Tensor) -> bool:
    return bool(torch.any(mask).item())


class GaussianNoiseView(View):
    """A view that adds Gaussian noise to a signal.

    This module applies simple additive white Gaussian noise (AWGN) to the
    input signal.

    Args:
        noise_std: Sampler for the Gaussian noise standard deviation.
        enabled_key: Optional per-sample gating key (bool [B]) in process meta.
    """

    def __init__(
        self,
        *,
        noise_std: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        """Initialize additive noise parameters."""
        super().__init__()
        noise_std_sampler = sampler_from_value(noise_std, name="noise_std")
        if isinstance(noise_std_sampler, ConstantSampler) and noise_std_sampler.value < 0:
            raise ValueError(
                f"noise_std must be non-negative, got {noise_std_sampler.value}."
            )
        self.noise_std = noise_std
        self.noise_std_sampler = noise_std_sampler
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply additive Gaussian noise.

        Args:
            input_state: A `LatentState` or `Observation`.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` object where `x` is the noisy signal, with the
            same shape as the input `[B, C, L]`.
        """
        signal, labels, process_meta = views_extract_signal(input_state, name="GaussianNoiseView")

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="GaussianNoiseView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "GaussianNoiseView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        noise_std = sampler_sample(
            sampler=self.noise_std_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="noise_std",
        )  # [B]
        if torch.any(noise_std < 0):
            raise ValueError("noise_std must be non-negative for all samples.")
        if self.enabled_key is not None:
            noise_std = noise_std * enabled_mask.to(dtype=noise_std.dtype)  # [B]

        noise = torch.randn(
            signal.shape,
            generator=generator,
            device=signal.device,
        )  # [B, C, L]
        observed_signal = signal + noise * noise_std[:, None, None]  # [B, C, L]

        samples = {"noise_std": noise_std}
        spec = {"noise_std": self.noise_std_sampler.spec()}
        meta = {
            "view": "GaussianNoiseView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "noise_std": noise_std,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class UniformNoiseView(View):
    """A view that adds uniform noise in [-amplitude, amplitude]."""

    def __init__(
        self,
        *,
        amplitude: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        if isinstance(amplitude_sampler, ConstantSampler) and amplitude_sampler.value < 0:
            raise ValueError(f"amplitude must be non-negative, got {amplitude_sampler.value}.")
        self.amplitude = amplitude
        self.amplitude_sampler = amplitude_sampler
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="UniformNoiseView")

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="UniformNoiseView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "UniformNoiseView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        if torch.any(amplitude < 0):
            raise ValueError("amplitude must be non-negative for all samples.")
        if self.enabled_key is not None:
            amplitude = amplitude * enabled_mask.to(dtype=amplitude.dtype)  # [B]

        base = torch.rand(
            signal.shape,
            generator=generator,
            device=signal.device,
            dtype=torch.float32,
        )  # [B, C, L]
        noise = (base * 2.0 - 1.0) * amplitude[:, None, None]  # [B, C, L]
        observed_signal = signal + noise  # [B, C, L]

        samples = {"amplitude": amplitude}
        spec = {"amplitude": self.amplitude_sampler.spec()}
        meta = {
            "view": "UniformNoiseView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "amplitude": amplitude,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class LaplaceNoiseView(View):
    """A view that adds Laplace (double-exponential) noise with scale `scale`."""

    def __init__(
        self,
        *,
        scale: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        scale_sampler = sampler_from_value(scale, name="scale")
        if isinstance(scale_sampler, ConstantSampler) and scale_sampler.value < 0:
            raise ValueError(f"scale must be non-negative, got {scale_sampler.value}.")
        self.scale = scale
        self.scale_sampler = scale_sampler
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="LaplaceNoiseView")

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="LaplaceNoiseView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "LaplaceNoiseView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        scale = sampler_sample(
            sampler=self.scale_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="scale",
        )  # [B]
        if torch.any(scale < 0):
            raise ValueError("scale must be non-negative for all samples.")
        if self.enabled_key is not None:
            scale = scale * enabled_mask.to(dtype=scale.dtype)  # [B]

        e1 = torch.empty(signal.shape, device=signal.device, dtype=torch.float32)  # [B, C, L]
        e1 = e1.exponential_(1.0, generator=generator)  # [B, C, L]
        e2 = torch.empty(signal.shape, device=signal.device, dtype=torch.float32)  # [B, C, L]
        e2 = e2.exponential_(1.0, generator=generator)  # [B, C, L]
        noise = (e1 - e2) * scale[:, None, None]  # [B, C, L]
        observed_signal = signal + noise  # [B, C, L]

        samples = {"scale": scale}
        spec = {"scale": self.scale_sampler.spec()}
        meta = {
            "view": "LaplaceNoiseView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "scale": scale,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


def _random_walk_noise(
    *,
    generator: torch.Generator,
    shape: tuple[int, int, int],
    step_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate a random-walk noise tensor with shape [B, C, L]."""
    steps = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )  # [B, C, L]
    steps = steps * step_std[:, None, None]  # [B, C, L]
    walk = torch.cumsum(steps, dim=2)  # [B, C, L]
    walk = walk - walk[:, :, :1]  # [B, C, L]
    return walk


class RandomWalkNoiseView(View):
    """A view that adds random-walk (integrated white) noise."""

    def __init__(
        self,
        *,
        step_std: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        step_std_sampler = sampler_from_value(step_std, name="step_std")
        if isinstance(step_std_sampler, ConstantSampler) and step_std_sampler.value < 0:
            raise ValueError(f"step_std must be non-negative, got {step_std_sampler.value}.")
        self.step_std = step_std
        self.step_std_sampler = step_std_sampler
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(
            input_state, name="RandomWalkNoiseView"
        )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="RandomWalkNoiseView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "RandomWalkNoiseView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        step_std = sampler_sample(
            sampler=self.step_std_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="step_std",
        )  # [B]
        if torch.any(step_std < 0):
            raise ValueError("step_std must be non-negative for all samples.")
        if self.enabled_key is not None:
            step_std = step_std * enabled_mask.to(dtype=step_std.dtype)  # [B]

        walk = _random_walk_noise(
            generator=generator,
            shape=signal.shape,
            step_std=step_std,
            device=signal.device,
        )  # [B, C, L]
        observed_signal = signal + walk  # [B, C, L]

        samples = {"step_std": step_std}
        spec = {"step_std": self.step_std_sampler.spec()}
        meta = {
            "view": "RandomWalkNoiseView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "step_std": step_std,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class BrownNoiseView(View):
    """A view that adds Brown noise (canonical: random-walk noise)."""

    def __init__(
        self,
        *,
        step_std: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        self._impl = RandomWalkNoiseView(step_std=step_std, enabled_key=enabled_key)

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        obs = self._impl(input_state, rng=rng)
        meta = dict(obs.meta)
        meta["view"] = "BrownNoiseView"
        return Observation(x=obs.x, y=obs.y, meta=meta)


class ColoredNoiseView(View):
    """A view that adds colored noise with power spectrum ~ 1/f^alpha."""

    def __init__(
        self,
        *,
        alpha: SamplerLike[float],
        noise_std: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        alpha_sampler = sampler_from_value(alpha, name="alpha")
        noise_std_sampler = sampler_from_value(noise_std, name="noise_std")
        if isinstance(alpha_sampler, ConstantSampler) and alpha_sampler.value < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha_sampler.value}.")
        if isinstance(noise_std_sampler, ConstantSampler) and noise_std_sampler.value < 0:
            raise ValueError(
                f"noise_std must be non-negative, got {noise_std_sampler.value}."
            )
        self.alpha = alpha
        self.noise_std = noise_std
        self.alpha_sampler = alpha_sampler
        self.noise_std_sampler = noise_std_sampler
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="ColoredNoiseView")

        batch_size, channels, seq_len = signal.shape
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="ColoredNoiseView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "ColoredNoiseView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        alpha = sampler_sample(
            sampler=self.alpha_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="alpha",
        )  # [B]
        if torch.any(alpha < 0):
            raise ValueError("alpha must be non-negative for all samples.")

        noise_std = sampler_sample(
            sampler=self.noise_std_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="noise_std",
        )  # [B]
        if torch.any(noise_std < 0):
            raise ValueError("noise_std must be non-negative for all samples.")
        if self.enabled_key is not None:
            noise_std = noise_std * enabled_mask.to(dtype=noise_std.dtype)  # [B]

        white = torch.randn(
            (batch_size, channels, seq_len),
            generator=generator,
            device=signal.device,
            dtype=torch.float32,
        )  # [B, C, L]
        white_fft = torch.fft.rfft(white, dim=2)  # [B, C, F]
        freqs = torch.arange(white_fft.shape[2], device=signal.device, dtype=torch.float32)  # [F]
        freqs = torch.clamp(freqs, min=1.0)  # [F]

        scale = freqs[None, None, :].pow(-alpha[:, None, None] / 2.0)  # [B, 1, F]
        scale = scale.clone()
        scale[:, :, 0] = 0.0
        colored_fft = white_fft * scale  # [B, C, F]
        colored = torch.fft.irfft(colored_fft, n=seq_len, dim=2)  # [B, C, L]

        std = colored.std(dim=2, keepdim=True, unbiased=False)  # [B, C, 1]
        if torch.any(std <= 0):
            raise ValueError("ColoredNoiseView requires positive std for normalization.")
        colored = colored / std  # [B, C, L]
        colored = colored * noise_std[:, None, None]  # [B, C, L]

        observed_signal = signal + colored  # [B, C, L]

        samples = {"alpha": alpha, "noise_std": noise_std}
        spec = {
            "alpha": self.alpha_sampler.spec(),
            "noise_std": self.noise_std_sampler.spec(),
        }
        meta = {
            "view": "ColoredNoiseView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "alpha": alpha,
            "noise_std": noise_std,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class BaselineWanderView(View):
    """A view that adds low-frequency sinusoidal noise (baseline wander).

    This module simulates baseline wander by adding a separate sine wave to each
    channel of the input signal. The amplitude, frequency, and phase of the
    sine wave are randomized for each channel.

    Args:
        amplitude_std: Sampler for the sine-wave amplitude scale.
        freq_min: Sampler for the minimum sine frequency (Hz).
        freq_max: Sampler for the maximum sine frequency (Hz).
        enabled_key: Optional per-sample gating key (bool [B]) in process meta.
    """

    def __init__(
        self,
        *,
        amplitude_std: SamplerLike[float],
        freq_min: SamplerLike[float],
        freq_max: SamplerLike[float],
        enabled_key: str | None = None,
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
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply baseline wander.

        Args:
            input_state: A `LatentState` or `Observation`.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` object where `x` is the signal with baseline
            wander, with the same shape as the input `[B, C, L]`.
        """
        signal, labels, process_meta = views_extract_signal(
            input_state, name="BaselineWanderView"
        )

        batch_size, channels, seq_len = signal.shape
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="BaselineWanderView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "BaselineWanderView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        if "sample_rate_hz" not in process_meta:
            raise ValueError(
                "BaselineWanderView requires process_meta['sample_rate_hz'] to interpret frequencies."
            )
        sample_rate_hz = process_meta["sample_rate_hz"]
        if not isinstance(sample_rate_hz, (int, float)):
            raise ValueError(
                "BaselineWanderView requires sample_rate_hz to be a float. "
                f"Got {type(sample_rate_hz).__name__}."
            )
        if float(sample_rate_hz) <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        time_grid = torch.arange(seq_len, device=signal.device, dtype=torch.float32)  # [L]
        time_grid = (time_grid / float(sample_rate_hz))[None, None, :]  # [1, 1, L]

        amplitude_std = sampler_sample(
            sampler=self.amplitude_std_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude_std",
        )  # [B]
        if torch.any(amplitude_std < 0):
            raise ValueError("amplitude_std must be non-negative for all samples.")

        freq_min = sampler_sample(
            sampler=self.freq_min_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="freq_min",
        )  # [B]
        freq_max = sampler_sample(
            sampler=self.freq_max_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
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
            device=signal.device,
        )  # [B, C, 1]
        freq = freq_min[:, None, None] + (freq_max - freq_min)[:, None, None] * freq  # [B, C, 1]

        phase = torch.rand(
            (batch_size, channels, 1),
            generator=generator,
            device=signal.device,
        )  # [B, C, 1]
        phase = phase * (2.0 * math.pi)  # [B, C, 1]

        amplitude = torch.randn(
            (batch_size, channels, 1),
            generator=generator,
            device=signal.device,
        )  # [B, C, 1]
        amplitude = amplitude * amplitude_std[:, None, None]  # [B, C, 1]
        if self.enabled_key is not None:
            amplitude = amplitude * enabled_mask[:, None, None].to(dtype=amplitude.dtype)  # [B, C, 1]

        # Create and add the baseline wander
        baseline = amplitude * torch.sin(2.0 * math.pi * freq * time_grid + phase)  # [B, C, L]
        observed_signal = signal + baseline  # [B, C, L]

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
            "enabled_key": self.enabled_key,
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
        return Observation(x=observed_signal, y=labels, meta=meta)
