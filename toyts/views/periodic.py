from __future__ import annotations

import math
from typing import Any

import torch

from ..core.samplers import ConstantSampler, SamplerLike, sampler_from_value, sampler_sample
from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ._enabled import views_resolve_enabled_mask
from ._signal import views_extract_signal
from .base import View


def _enabled_any(mask: torch.Tensor) -> bool:
    return bool(torch.any(mask).item())


def _make_sample_index(seq_len: int) -> torch.Tensor:
    idx = torch.arange(seq_len, dtype=torch.float32)  # [L]
    return idx[None, None, :]  # [1, 1, L]


def _require_sample_rate_hz(process_meta: dict[str, Any], *, name: str) -> float:
    if "sample_rate_hz" not in process_meta:
        raise ValueError(f"{name} requires process_meta['sample_rate_hz'] to be present.")
    sample_rate_hz = process_meta["sample_rate_hz"]
    if not isinstance(sample_rate_hz, (int, float)):
        raise ValueError(
            f"{name} requires sample_rate_hz to be a float. Got {type(sample_rate_hz).__name__}."
        )
    if float(sample_rate_hz) <= 0:
        raise ValueError(f"{name} requires sample_rate_hz to be positive, got {sample_rate_hz}.")
    return float(sample_rate_hz)


class SineWaveView(View):
    """Add a sine wave component with frequency in Hz."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        frequency_hz: SamplerLike[float],
        phase: SamplerLike[float],
        offset: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("sample_idx", _make_sample_index(seq_len))
        self.amplitude = amplitude
        self.frequency_hz = frequency_hz
        self.phase = phase
        self.offset = offset
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.frequency_hz_sampler = sampler_from_value(frequency_hz, name="frequency_hz")
        self.phase_sampler = sampler_from_value(phase, name="phase")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

        for sampler_name, sampler in (("amplitude", self.amplitude_sampler), ("phase", self.phase_sampler), ("offset", self.offset_sampler)):
            if isinstance(sampler, ConstantSampler):
                scalar = float(sampler.value)
                if not torch.isfinite(torch.tensor(scalar)):
                    raise ValueError(f"{sampler_name} must be finite, got {scalar}.")

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="SineWaveView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "SineWaveView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="SineWaveView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {"view": "SineWaveView", "enabled_key": self.enabled_key, "process": process_meta}
            return Observation(x=signal, y=labels, meta=meta)

        sample_rate_hz = _require_sample_rate_hz(process_meta, name="SineWaveView")
        t_sec = self.sample_idx / sample_rate_hz  # [1, 1, L]

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        frequency_hz = sampler_sample(
            sampler=self.frequency_hz_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="frequency_hz",
        )  # [B]
        phase = sampler_sample(
            sampler=self.phase_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="phase",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]

        if torch.any(frequency_hz <= 0):
            raise ValueError("SineWaveView frequency_hz must be positive for all samples.")
        if not torch.all(torch.isfinite(amplitude)) or not torch.all(torch.isfinite(phase)) or not torch.all(torch.isfinite(offset)):
            raise ValueError("SineWaveView parameters must be finite for all samples.")

        arg = 2.0 * math.pi * frequency_hz[:, None, None] * t_sec + phase[:, None, None]  # [B, 1, L]
        component = amplitude[:, None, None] * torch.sin(arg) + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "phase": phase,
            "offset": offset,
        }
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "frequency_hz": self.frequency_hz_sampler.spec(),
            "phase": self.phase_sampler.spec(),
            "offset": self.offset_sampler.spec(),
        }
        meta = {
            "view": "SineWaveView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "sample_rate_hz": sample_rate_hz,
            **samples,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class TriangleWaveView(View):
    """Add a triangle wave component with frequency in Hz."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        frequency_hz: SamplerLike[float],
        phase: SamplerLike[float],
        offset: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("sample_idx", _make_sample_index(seq_len))
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.frequency_hz_sampler = sampler_from_value(frequency_hz, name="frequency_hz")
        self.phase_sampler = sampler_from_value(phase, name="phase")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="TriangleWaveView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "TriangleWaveView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="TriangleWaveView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {"view": "TriangleWaveView", "enabled_key": self.enabled_key, "process": process_meta}
            return Observation(x=signal, y=labels, meta=meta)

        sample_rate_hz = _require_sample_rate_hz(process_meta, name="TriangleWaveView")
        t_sec = self.sample_idx / sample_rate_hz  # [1, 1, L]

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        frequency_hz = sampler_sample(
            sampler=self.frequency_hz_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="frequency_hz",
        )  # [B]
        phase = sampler_sample(
            sampler=self.phase_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="phase",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]

        if torch.any(frequency_hz <= 0):
            raise ValueError("TriangleWaveView frequency_hz must be positive for all samples.")
        cycles = frequency_hz[:, None, None] * t_sec + phase[:, None, None] / (2.0 * math.pi)  # [B, 1, L]
        frac = cycles - torch.floor(cycles)  # [B, 1, L]
        tri = 1.0 - 4.0 * torch.abs(frac - 0.5)  # [-1, 1]  [B, 1, L]
        component = amplitude[:, None, None] * tri + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "phase": phase,
            "offset": offset,
        }
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "frequency_hz": self.frequency_hz_sampler.spec(),
            "phase": self.phase_sampler.spec(),
            "offset": self.offset_sampler.spec(),
        }
        meta = {
            "view": "TriangleWaveView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "sample_rate_hz": sample_rate_hz,
            **samples,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class SawtoothWaveView(View):
    """Add a sawtooth wave component with frequency in Hz."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        frequency_hz: SamplerLike[float],
        phase: SamplerLike[float],
        offset: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("sample_idx", _make_sample_index(seq_len))
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.frequency_hz_sampler = sampler_from_value(frequency_hz, name="frequency_hz")
        self.phase_sampler = sampler_from_value(phase, name="phase")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="SawtoothWaveView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "SawtoothWaveView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="SawtoothWaveView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {"view": "SawtoothWaveView", "enabled_key": self.enabled_key, "process": process_meta}
            return Observation(x=signal, y=labels, meta=meta)

        sample_rate_hz = _require_sample_rate_hz(process_meta, name="SawtoothWaveView")
        t_sec = self.sample_idx / sample_rate_hz  # [1, 1, L]

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        frequency_hz = sampler_sample(
            sampler=self.frequency_hz_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="frequency_hz",
        )  # [B]
        phase = sampler_sample(
            sampler=self.phase_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="phase",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]

        if torch.any(frequency_hz <= 0):
            raise ValueError("SawtoothWaveView frequency_hz must be positive for all samples.")
        cycles = frequency_hz[:, None, None] * t_sec + phase[:, None, None] / (2.0 * math.pi)  # [B, 1, L]
        frac = cycles - torch.floor(cycles)  # [B, 1, L]
        saw = 2.0 * frac - 1.0  # [-1, 1)  [B, 1, L]
        component = amplitude[:, None, None] * saw + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "phase": phase,
            "offset": offset,
        }
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "frequency_hz": self.frequency_hz_sampler.spec(),
            "phase": self.phase_sampler.spec(),
            "offset": self.offset_sampler.spec(),
        }
        meta = {
            "view": "SawtoothWaveView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "sample_rate_hz": sample_rate_hz,
            **samples,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class SquareWaveView(View):
    """Add a square wave component with frequency in Hz."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        frequency_hz: SamplerLike[float],
        phase: SamplerLike[float],
        offset: SamplerLike[float],
        duty_cycle: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("sample_idx", _make_sample_index(seq_len))
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.frequency_hz_sampler = sampler_from_value(frequency_hz, name="frequency_hz")
        self.phase_sampler = sampler_from_value(phase, name="phase")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.duty_cycle_sampler = sampler_from_value(duty_cycle, name="duty_cycle")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="SquareWaveView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "SquareWaveView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="SquareWaveView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {"view": "SquareWaveView", "enabled_key": self.enabled_key, "process": process_meta}
            return Observation(x=signal, y=labels, meta=meta)

        sample_rate_hz = _require_sample_rate_hz(process_meta, name="SquareWaveView")
        t_sec = self.sample_idx / sample_rate_hz  # [1, 1, L]

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        frequency_hz = sampler_sample(
            sampler=self.frequency_hz_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="frequency_hz",
        )  # [B]
        phase = sampler_sample(
            sampler=self.phase_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="phase",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]
        duty_cycle = sampler_sample(
            sampler=self.duty_cycle_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="duty_cycle",
        )  # [B]

        if torch.any(frequency_hz <= 0):
            raise ValueError("SquareWaveView frequency_hz must be positive for all samples.")
        if torch.any((duty_cycle <= 0) | (duty_cycle >= 1)):
            raise ValueError("SquareWaveView duty_cycle must be in (0, 1) for all samples.")

        cycles = frequency_hz[:, None, None] * t_sec + phase[:, None, None] / (2.0 * math.pi)  # [B, 1, L]
        frac = cycles - torch.floor(cycles)  # [B, 1, L]
        square = (frac < duty_cycle[:, None, None]).to(dtype=torch.float32) * 2.0 - 1.0  # [B, 1, L]
        component = amplitude[:, None, None] * square + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "phase": phase,
            "offset": offset,
            "duty_cycle": duty_cycle,
        }
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "frequency_hz": self.frequency_hz_sampler.spec(),
            "phase": self.phase_sampler.spec(),
            "offset": self.offset_sampler.spec(),
            "duty_cycle": self.duty_cycle_sampler.spec(),
        }
        meta = {
            "view": "SquareWaveView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "sample_rate_hz": sample_rate_hz,
            **samples,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class ChirpView(View):
    """Add a linear chirp component with frequency in Hz."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        f0_hz: SamplerLike[float],
        f1_hz: SamplerLike[float],
        phase: SamplerLike[float],
        offset: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 1:
            raise ValueError(f"seq_len must be > 1 for ChirpView, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("sample_idx", _make_sample_index(seq_len))
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.f0_hz_sampler = sampler_from_value(f0_hz, name="f0_hz")
        self.f1_hz_sampler = sampler_from_value(f1_hz, name="f1_hz")
        self.phase_sampler = sampler_from_value(phase, name="phase")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="ChirpView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "ChirpView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="ChirpView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {"view": "ChirpView", "enabled_key": self.enabled_key, "process": process_meta}
            return Observation(x=signal, y=labels, meta=meta)

        sample_rate_hz = _require_sample_rate_hz(process_meta, name="ChirpView")
        t_sec = self.sample_idx / sample_rate_hz  # [1, 1, L]
        duration_sec = (self.seq_len - 1) / sample_rate_hz

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        f0_hz = sampler_sample(
            sampler=self.f0_hz_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="f0_hz",
        )  # [B]
        f1_hz = sampler_sample(
            sampler=self.f1_hz_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="f1_hz",
        )  # [B]
        phase = sampler_sample(
            sampler=self.phase_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="phase",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]

        if torch.any(f0_hz <= 0) or torch.any(f1_hz <= 0):
            raise ValueError("ChirpView f0_hz/f1_hz must be positive for all samples.")
        if duration_sec <= 0:
            raise ValueError(f"ChirpView duration_sec must be positive, got {duration_sec}.")

        df = (f1_hz - f0_hz)[:, None, None]  # [B, 1, 1]
        phi = 2.0 * math.pi * (
            f0_hz[:, None, None] * t_sec + 0.5 * df * t_sec.pow(2) / duration_sec
        ) + phase[:, None, None]  # [B, 1, L]
        component = amplitude[:, None, None] * torch.sin(phi) + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {
            "amplitude": amplitude,
            "f0_hz": f0_hz,
            "f1_hz": f1_hz,
            "phase": phase,
            "offset": offset,
        }
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "f0_hz": self.f0_hz_sampler.spec(),
            "f1_hz": self.f1_hz_sampler.spec(),
            "phase": self.phase_sampler.spec(),
            "offset": self.offset_sampler.spec(),
        }
        meta = {
            "view": "ChirpView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "sample_rate_hz": sample_rate_hz,
            "duration_sec": duration_sec,
            **samples,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class DampedSineWaveView(View):
    """Add a damped sine wave component with frequency in Hz."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        frequency_hz: SamplerLike[float],
        tau_sec: SamplerLike[float],
        phase: SamplerLike[float],
        offset: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("sample_idx", _make_sample_index(seq_len))
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.frequency_hz_sampler = sampler_from_value(frequency_hz, name="frequency_hz")
        self.tau_sec_sampler = sampler_from_value(tau_sec, name="tau_sec")
        self.phase_sampler = sampler_from_value(phase, name="phase")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(
            input_state, name="DampedSineWaveView"
        )
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "DampedSineWaveView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="DampedSineWaveView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {"view": "DampedSineWaveView", "enabled_key": self.enabled_key, "process": process_meta}
            return Observation(x=signal, y=labels, meta=meta)

        sample_rate_hz = _require_sample_rate_hz(process_meta, name="DampedSineWaveView")
        t_sec = self.sample_idx / sample_rate_hz  # [1, 1, L]

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        frequency_hz = sampler_sample(
            sampler=self.frequency_hz_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="frequency_hz",
        )  # [B]
        tau_sec = sampler_sample(
            sampler=self.tau_sec_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="tau_sec",
        )  # [B]
        phase = sampler_sample(
            sampler=self.phase_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="phase",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]

        if torch.any(frequency_hz <= 0):
            raise ValueError("DampedSineWaveView frequency_hz must be positive for all samples.")
        if torch.any(tau_sec <= 0):
            raise ValueError("DampedSineWaveView tau_sec must be positive for all samples.")

        envelope = torch.exp(-t_sec * (1.0 / tau_sec[:, None, None]))  # [B, 1, L]
        arg = 2.0 * math.pi * frequency_hz[:, None, None] * t_sec + phase[:, None, None]  # [B, 1, L]
        component = amplitude[:, None, None] * envelope * torch.sin(arg) + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "tau_sec": tau_sec,
            "phase": phase,
            "offset": offset,
        }
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "frequency_hz": self.frequency_hz_sampler.spec(),
            "tau_sec": self.tau_sec_sampler.spec(),
            "phase": self.phase_sampler.spec(),
            "offset": self.offset_sampler.spec(),
        }
        meta = {
            "view": "DampedSineWaveView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "sample_rate_hz": sample_rate_hz,
            **samples,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)

