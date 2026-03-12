from __future__ import annotations

import torch

from ..core.samplers import ConstantSampler, SamplerLike, sampler_from_value, sampler_sample
from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ._enabled import views_resolve_enabled_mask
from ._signal import views_extract_signal
from .base import View


def _enabled_any(mask: torch.Tensor) -> bool:
    return bool(torch.any(mask).item())


def _make_t_grid(seq_len: int) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, steps=seq_len, dtype=torch.float32)  # [L]
    return t[None, None, :]  # [1, 1, L]


class LinearTrendView(View):
    """Add a linear trend component."""

    def __init__(
        self,
        *,
        seq_len: int,
        slope: SamplerLike[float],
        intercept: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("t_grid", _make_t_grid(seq_len))
        self.slope = slope
        self.intercept = intercept
        self.slope_sampler = sampler_from_value(slope, name="slope")
        self.intercept_sampler = sampler_from_value(intercept, name="intercept")
        self.enabled_key = enabled_key

        for sampler_name, sampler in (("slope", self.slope_sampler), ("intercept", self.intercept_sampler)):
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
        signal, labels, process_meta = views_extract_signal(input_state, name="LinearTrendView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "LinearTrendView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="LinearTrendView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "LinearTrendView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        slope = sampler_sample(
            sampler=self.slope_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="slope",
        )  # [B]
        intercept = sampler_sample(
            sampler=self.intercept_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="intercept",
        )  # [B]
        if not torch.all(torch.isfinite(slope)) or not torch.all(torch.isfinite(intercept)):
            raise ValueError("LinearTrendView parameters must be finite for all samples.")

        component = slope[:, None, None] * (self.t_grid - 0.5) + intercept[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {"slope": slope, "intercept": intercept}
        spec = {
            "slope": self.slope_sampler.spec(),
            "intercept": self.intercept_sampler.spec(),
        }
        meta = {
            "view": "LinearTrendView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "slope": slope,
            "intercept": intercept,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class QuadraticTrendView(View):
    """Add a quadratic trend component."""

    def __init__(
        self,
        *,
        seq_len: int,
        a: SamplerLike[float],
        b: SamplerLike[float],
        c: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("t_grid", _make_t_grid(seq_len))
        self.a = a
        self.b = b
        self.c = c
        self.a_sampler = sampler_from_value(a, name="a")
        self.b_sampler = sampler_from_value(b, name="b")
        self.c_sampler = sampler_from_value(c, name="c")
        self.enabled_key = enabled_key

        for sampler_name, sampler in (("a", self.a_sampler), ("b", self.b_sampler), ("c", self.c_sampler)):
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
        signal, labels, process_meta = views_extract_signal(input_state, name="QuadraticTrendView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "QuadraticTrendView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="QuadraticTrendView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "QuadraticTrendView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        a = sampler_sample(
            sampler=self.a_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="a",
        )  # [B]
        b = sampler_sample(
            sampler=self.b_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="b",
        )  # [B]
        c = sampler_sample(
            sampler=self.c_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="c",
        )  # [B]
        if not torch.all(torch.isfinite(a)) or not torch.all(torch.isfinite(b)) or not torch.all(torch.isfinite(c)):
            raise ValueError("QuadraticTrendView parameters must be finite for all samples.")

        t = self.t_grid - 0.5  # [1, 1, L]
        component = a[:, None, None] * t.pow(2) + b[:, None, None] * t + c[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {"a": a, "b": b, "c": c}
        spec = {
            "a": self.a_sampler.spec(),
            "b": self.b_sampler.spec(),
            "c": self.c_sampler.spec(),
        }
        meta = {
            "view": "QuadraticTrendView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "a": a,
            "b": b,
            "c": c,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class LogTrendView(View):
    """Add a logarithmic trend component."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        offset: SamplerLike[float],
        epsilon: float,
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("t_grid", _make_t_grid(seq_len))
        self.amplitude = amplitude
        self.offset = offset
        self.epsilon = float(epsilon)
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

        for sampler_name, sampler in (("amplitude", self.amplitude_sampler), ("offset", self.offset_sampler)):
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
        signal, labels, process_meta = views_extract_signal(input_state, name="LogTrendView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "LogTrendView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="LogTrendView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "LogTrendView",
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
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]
        if not torch.all(torch.isfinite(amplitude)) or not torch.all(torch.isfinite(offset)):
            raise ValueError("LogTrendView parameters must be finite for all samples.")

        component = amplitude[:, None, None] * torch.log(self.epsilon + self.t_grid) + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {"amplitude": amplitude, "offset": offset}
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "offset": self.offset_sampler.spec(),
        }
        meta = {
            "view": "LogTrendView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "epsilon": self.epsilon,
            "amplitude": amplitude,
            "offset": offset,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class ExponentialTrendView(View):
    """Add an exponential trend component using `expm1(rate * (t - 0.5))`."""

    def __init__(
        self,
        *,
        seq_len: int,
        rate: SamplerLike[float],
        offset: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("t_grid", _make_t_grid(seq_len))
        self.rate = rate
        self.offset = offset
        self.rate_sampler = sampler_from_value(rate, name="rate")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

        for sampler_name, sampler in (("rate", self.rate_sampler), ("offset", self.offset_sampler)):
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
        signal, labels, process_meta = views_extract_signal(
            input_state, name="ExponentialTrendView"
        )
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "ExponentialTrendView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="ExponentialTrendView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "ExponentialTrendView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        rate = sampler_sample(
            sampler=self.rate_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="rate",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]
        if not torch.all(torch.isfinite(rate)) or not torch.all(torch.isfinite(offset)):
            raise ValueError("ExponentialTrendView parameters must be finite for all samples.")

        component = torch.expm1(rate[:, None, None] * (self.t_grid - 0.5)) + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {"rate": rate, "offset": offset}
        spec = {"rate": self.rate_sampler.spec(), "offset": self.offset_sampler.spec()}
        meta = {
            "view": "ExponentialTrendView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "rate": rate,
            "offset": offset,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class SigmoidTrendView(View):
    """Add a sigmoid (logistic) trend component."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude: SamplerLike[float],
        center: SamplerLike[float],
        sharpness: SamplerLike[float],
        offset: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("t_grid", _make_t_grid(seq_len))
        self.amplitude = amplitude
        self.center = center
        self.sharpness = sharpness
        self.offset = offset
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.center_sampler = sampler_from_value(center, name="center")
        self.sharpness_sampler = sampler_from_value(sharpness, name="sharpness")
        self.offset_sampler = sampler_from_value(offset, name="offset")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(input_state, name="SigmoidTrendView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "SigmoidTrendView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="SigmoidTrendView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "SigmoidTrendView",
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
        center = sampler_sample(
            sampler=self.center_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="center",
        )  # [B]
        sharpness = sampler_sample(
            sampler=self.sharpness_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="sharpness",
        )  # [B]
        offset = sampler_sample(
            sampler=self.offset_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="offset",
        )  # [B]

        if not torch.all(torch.isfinite(amplitude)) or not torch.all(torch.isfinite(offset)):
            raise ValueError("SigmoidTrendView parameters must be finite for all samples.")
        if torch.any((center < 0) | (center > 1)):
            raise ValueError("SigmoidTrendView center must be in [0, 1] for all samples.")
        if torch.any(sharpness <= 0):
            raise ValueError("SigmoidTrendView sharpness must be positive for all samples.")

        z = (self.t_grid - center[:, None, None]) * sharpness[:, None, None]  # [B, 1, L]
        component = amplitude[:, None, None] * torch.sigmoid(z) + offset[:, None, None]  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {"amplitude": amplitude, "center": center, "sharpness": sharpness, "offset": offset}
        spec = {
            "amplitude": self.amplitude_sampler.spec(),
            "center": self.center_sampler.spec(),
            "sharpness": self.sharpness_sampler.spec(),
            "offset": self.offset_sampler.spec(),
        }
        meta = {
            "view": "SigmoidTrendView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "amplitude": amplitude,
            "center": center,
            "sharpness": sharpness,
            "offset": offset,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class PiecewiseLinearTrendView(View):
    """Add a piecewise-linear trend with one changepoint."""

    def __init__(
        self,
        *,
        seq_len: int,
        slope1: SamplerLike[float],
        slope2: SamplerLike[float],
        change_t: SamplerLike[float],
        intercept: SamplerLike[float],
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.register_buffer("t_grid", _make_t_grid(seq_len))
        self.slope1 = slope1
        self.slope2 = slope2
        self.change_t = change_t
        self.intercept = intercept
        self.slope1_sampler = sampler_from_value(slope1, name="slope1")
        self.slope2_sampler = sampler_from_value(slope2, name="slope2")
        self.change_t_sampler = sampler_from_value(change_t, name="change_t")
        self.intercept_sampler = sampler_from_value(intercept, name="intercept")
        self.enabled_key = enabled_key

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = views_extract_signal(
            input_state, name="PiecewiseLinearTrendView"
        )
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "PiecewiseLinearTrendView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="PiecewiseLinearTrendView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {
                "view": "PiecewiseLinearTrendView",
                "enabled_key": self.enabled_key,
                "process": process_meta,
            }
            return Observation(x=signal, y=labels, meta=meta)

        generator, seed, _ = rng_make_generator(rng=rng, device=signal.device)
        slope1 = sampler_sample(
            sampler=self.slope1_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="slope1",
        )  # [B]
        slope2 = sampler_sample(
            sampler=self.slope2_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="slope2",
        )  # [B]
        change_t = sampler_sample(
            sampler=self.change_t_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="change_t",
        )  # [B]
        intercept = sampler_sample(
            sampler=self.intercept_sampler,
            shape=(batch_size,),
            rng=generator,
            device=signal.device,
            dtype=torch.float32,
            name="intercept",
        )  # [B]
        if torch.any((change_t < 0) | (change_t > 1)):
            raise ValueError("PiecewiseLinearTrendView change_t must be in [0, 1] for all samples.")

        t = self.t_grid - 0.5  # [1, 1, L]
        t0 = change_t[:, None, None] - 0.5  # [B, 1, 1]
        y_pre = slope1[:, None, None] * t + intercept[:, None, None]  # [B, 1, L]
        y_post = (
            slope2[:, None, None] * t
            + intercept[:, None, None]
            + (slope1 - slope2)[:, None, None] * t0
        )  # [B, 1, L]
        component = torch.where(self.t_grid < change_t[:, None, None], y_pre, y_post)  # [B, 1, L]
        if self.enabled_key is not None:
            component = component * enabled_mask[:, None, None].to(dtype=component.dtype)  # [B, 1, L]
        observed_signal = signal + component  # [B, C, L]

        samples = {
            "slope1": slope1,
            "slope2": slope2,
            "change_t": change_t,
            "intercept": intercept,
        }
        spec = {
            "slope1": self.slope1_sampler.spec(),
            "slope2": self.slope2_sampler.spec(),
            "change_t": self.change_t_sampler.spec(),
            "intercept": self.intercept_sampler.spec(),
        }
        meta = {
            "view": "PiecewiseLinearTrendView",
            "seed": seed,
            "enabled_key": self.enabled_key,
            "slope1": slope1,
            "slope2": slope2,
            "change_t": change_t,
            "intercept": intercept,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)

