from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math
import torch


class Sampler:
    """Base interface for parameter samplers."""

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a tensor of samples with the requested shape."""
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """Return a minimal description of this sampler."""
        raise NotImplementedError


SamplerLike = Sampler | float | int | bool | torch.Tensor


def sampler_from_value(value: SamplerLike, *, name: str) -> Sampler:
    """Normalize a sampler-like input into a Sampler instance."""
    if isinstance(value, Sampler):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must be a scalar or Sampler, got tensor with shape {value.shape}."
            )
        return ConstantSampler(value=value.item())
    if isinstance(value, (float, int, bool)):
        return ConstantSampler(value=value)
    raise TypeError(
        f"{name} must be a Sampler, scalar, or 0-dim tensor. Got {type(value).__name__}."
    )


def sampler_sample(
    *,
    sampler: Sampler,
    shape: tuple[int, ...],
    rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Sample a tensor and verify the expected shape."""
    samples = sampler.sample(shape=shape, rng=rng, device=device, dtype=dtype)  # [*shape]
    if samples.shape != shape:
        raise ValueError(
            f"{name} sampler returned shape {samples.shape}, expected {shape}."
        )
    return samples


def sampler_sample_scalar(
    *,
    sampler: Sampler,
    rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> tuple[torch.Tensor, float | int | bool]:
    """Sample a single scalar value and return (tensor[1], python scalar)."""
    samples = sampler_sample(
        sampler=sampler,
        shape=(1,),
        rng=rng,
        device=device,
        dtype=dtype,
        name=name,
    )  # [1]
    return samples, samples.item()


@dataclass(frozen=True)
class ConstantSampler(Sampler):
    """Return a constant scalar value."""

    value: float | int | bool

    def __post_init__(self) -> None:
        if not isinstance(self.value, (float, int, bool)):
            raise TypeError(
                f"ConstantSampler value must be a scalar, got {type(self.value).__name__}."
            )

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a tensor filled with the constant value."""
        samples = torch.full(
            shape,
            fill_value=self.value,
            device=device,
            dtype=dtype,
        )  # [*shape]
        return samples

    def spec(self) -> dict[str, Any]:
        return {"kind": "constant", "value": self.value}


@dataclass(frozen=True)
class UniformSampler(Sampler):
    """Sample from a uniform distribution U(low, high)."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"UniformSampler requires high > low, got {self.low}, {self.high}.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not dtype.is_floating_point:
            raise ValueError("UniformSampler requires a floating dtype.")
        base = torch.rand(
            shape,
            generator=rng,
            device=device,
            dtype=dtype,
        )  # [*shape]
        return self.low + (self.high - self.low) * base

    def spec(self) -> dict[str, Any]:
        return {"kind": "uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class LogUniformSampler(Sampler):
    """Sample from a log-uniform distribution."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogUniformSampler requires low/high > 0.")
        if self.high <= self.low:
            raise ValueError(f"LogUniformSampler requires high > low, got {self.low}, {self.high}.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not dtype.is_floating_point:
            raise ValueError("LogUniformSampler requires a floating dtype.")
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        base = torch.rand(
            shape,
            generator=rng,
            device=device,
            dtype=dtype,
        )  # [*shape]
        samples = torch.exp(base * (log_high - log_low) + log_low)  # [*shape]
        return samples

    def spec(self) -> dict[str, Any]:
        return {"kind": "log_uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class NormalSampler(Sampler):
    """Sample from a normal distribution N(mean, std)."""

    mean: float
    std: float
    clamp: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError(f"NormalSampler requires std > 0, got {self.std}.")
        if self.clamp is not None:
            low, high = self.clamp
            if high < low:
                raise ValueError("NormalSampler clamp must satisfy low <= high.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not dtype.is_floating_point:
            raise ValueError("NormalSampler requires a floating dtype.")
        base = torch.randn(
            shape,
            generator=rng,
            device=device,
            dtype=dtype,
        )  # [*shape]
        samples = base * self.std + self.mean  # [*shape]
        if self.clamp is not None:
            samples = torch.clamp(samples, min=self.clamp[0], max=self.clamp[1])
        return samples

    def spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"kind": "normal", "mean": self.mean, "std": self.std}
        if self.clamp is not None:
            spec["clamp"] = self.clamp
        return spec


@dataclass(frozen=True)
class RandIntSampler(Sampler):
    """Sample integers uniformly from [low, high)."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"RandIntSampler requires high > low, got {self.low}, {self.high}.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("RandIntSampler requires an integer dtype.")
        samples = torch.randint(
            low=self.low,
            high=self.high,
            size=shape,
            generator=rng,
            device=device,
            dtype=dtype,
        )  # [*shape]
        return samples

    def spec(self) -> dict[str, Any]:
        return {"kind": "randint", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class BernoulliSampler(Sampler):
    """Sample booleans from a Bernoulli distribution."""

    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"BernoulliSampler requires p in [0, 1], got {self.p}.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is not torch.bool:
            raise ValueError("BernoulliSampler requires dtype=torch.bool.")
        base = torch.rand(
            shape,
            generator=rng,
            device=device,
        )  # [*shape]
        samples = base < self.p  # [*shape]
        return samples

    def spec(self) -> dict[str, Any]:
        return {"kind": "bernoulli", "p": self.p}


@dataclass(frozen=True)
class CategoricalSampler(Sampler):
    """Sample categorical indices from a probability vector."""

    probs: Sequence[float]

    def __post_init__(self) -> None:
        if not self.probs:
            raise ValueError("CategoricalSampler requires non-empty probs.")
        if any(prob < 0 for prob in self.probs):
            raise ValueError("CategoricalSampler requires non-negative probs.")
        if sum(self.probs) <= 0:
            raise ValueError("CategoricalSampler requires probs sum > 0.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is not torch.int64:
            raise ValueError("CategoricalSampler requires dtype=torch.int64.")
        probs = torch.tensor(self.probs, device=device, dtype=torch.float32)  # [K]
        numel = math.prod(shape) if shape else 1
        indices = torch.multinomial(
            probs,
            num_samples=numel,
            replacement=True,
            generator=rng,
        )  # [numel]
        samples = indices.reshape(shape)  # [*shape]
        return samples

    def spec(self) -> dict[str, Any]:
        return {"kind": "categorical", "probs": list(self.probs)}


@dataclass(frozen=True)
class ChoiceSampler(Sampler):
    """Sample indices from a list of choices."""

    choices: Sequence[Any]
    probs: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("ChoiceSampler requires non-empty choices.")
        if self.probs is not None and len(self.probs) != len(self.choices):
            raise ValueError("ChoiceSampler probs length must match choices length.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        probs = self.probs if self.probs is not None else [1.0] * len(self.choices)
        sampler = CategoricalSampler(probs=probs)
        return sampler.sample(
            shape=shape,
            rng=rng,
            device=device,
            dtype=dtype,
        )

    def spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"kind": "choice", "choices": list(self.choices)}
        if self.probs is not None:
            spec["probs"] = list(self.probs)
        return spec
