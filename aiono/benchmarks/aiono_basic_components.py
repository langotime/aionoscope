from __future__ import annotations

"""Versioned benchmark semantics for the Aiono basic-components family.

The current public contract is `aiono_basic_components/v1`.
Downstream consumers should treat this module as the source of truth for:

- the baseline sampling frequency;
- the waveform-specific recoverability rules used by `frequency_hz: auto`;
- the manifest fields required to make resolved periodic semantics explicit.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.samplers import UniformSampler


AIONO_BASIC_COMPONENTS_BENCHMARK_FAMILY = "aiono_basic_components"
AIONO_BASIC_COMPONENTS_BENCHMARK_VERSION = "v1"
AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ = 500

_EXPECTED_RECOVERABILITY_POLICY = {
    "sine": "nyquist",
    "sawtooth": "min_points_per_period",
    "square": "min_points_in_shorter_plateau",
}
_PERIODIC_SIGNAL_ORDER = ("sine", "sawtooth", "square")


@dataclass(frozen=True)
class UniformRange:
    """Closed-open numeric bounds consumed by UniformSampler."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(
                f"UniformRange requires high > low, got low={self.low} high={self.high}."
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, name: str) -> UniformRange:
        low = mapping.get("low")
        high = mapping.get("high")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"{name} must define numeric low/high bounds, got {mapping!r}.")
        return cls(low=float(low), high=float(high))

    def as_uniform_sampler(self) -> UniformSampler:
        return UniformSampler(self.low, self.high)

    def spec(self) -> dict[str, float | str]:
        return {"kind": "uniform", "low": float(self.low), "high": float(self.high)}


@dataclass(frozen=True)
class ResolvedPeriodicSignalConfig:
    """Resolved sampler bounds for one periodic waveform family."""

    name: str
    recoverability_policy: str
    amplitude: UniformRange
    frequency_hz: UniformRange
    phase: UniformRange
    offset: UniformRange
    duty_cycle: UniformRange | None = None

    def view_kwargs(self) -> dict[str, UniformSampler]:
        kwargs: dict[str, UniformSampler] = {
            "amplitude": self.amplitude.as_uniform_sampler(),
            "frequency_hz": self.frequency_hz.as_uniform_sampler(),
            "phase": self.phase.as_uniform_sampler(),
            "offset": self.offset.as_uniform_sampler(),
        }
        if self.duty_cycle is not None:
            kwargs["duty_cycle"] = self.duty_cycle.as_uniform_sampler()
        return kwargs

    def sampler_specs(self) -> dict[str, dict[str, float | str]]:
        specs = {
            "amplitude": self.amplitude.spec(),
            "frequency_hz": self.frequency_hz.spec(),
            "phase": self.phase.spec(),
            "offset": self.offset.spec(),
        }
        if self.duty_cycle is not None:
            specs["duty_cycle"] = self.duty_cycle.spec()
        return specs


@dataclass(frozen=True)
class AionoBasicComponentsPeriodicConfig:
    """Public config schema for Aiono basic-components periodic semantics."""

    frequency_hz: str | UniformRange
    min_full_periods: float
    nyquist_fraction: float
    recoverability_policy: dict[str, str]
    sawtooth_min_points_per_period: int
    square_min_points_in_shorter_plateau: int
    amplitude: UniformRange
    phase: UniformRange
    offset: UniformRange
    square_duty_cycle: UniformRange

    def __post_init__(self) -> None:
        if isinstance(self.frequency_hz, str):
            if self.frequency_hz != "auto":
                raise ValueError(
                    "frequency_hz must be 'auto' or an explicit UniformRange, "
                    f"got {self.frequency_hz!r}."
                )
        elif not isinstance(self.frequency_hz, UniformRange):
            raise TypeError(
                "frequency_hz must be 'auto' or UniformRange, "
                f"got {type(self.frequency_hz).__name__}."
            )
        if self.min_full_periods <= 0:
            raise ValueError(
                f"min_full_periods must be > 0, got {self.min_full_periods}."
            )
        if not 0 < self.nyquist_fraction <= 1:
            raise ValueError(
                f"nyquist_fraction must satisfy 0 < value <= 1, got {self.nyquist_fraction}."
            )
        if self.sawtooth_min_points_per_period < 1:
            raise ValueError(
                "sawtooth_min_points_per_period must be >= 1, "
                f"got {self.sawtooth_min_points_per_period}."
            )
        if self.square_min_points_in_shorter_plateau < 1:
            raise ValueError(
                "square_min_points_in_shorter_plateau must be >= 1, "
                f"got {self.square_min_points_in_shorter_plateau}."
            )
        if self.square_duty_cycle.low <= 0 or self.square_duty_cycle.high >= 1:
            raise ValueError(
                "square_duty_cycle must stay inside (0, 1), "
                f"got low={self.square_duty_cycle.low} high={self.square_duty_cycle.high}."
            )
        if self.recoverability_policy != _EXPECTED_RECOVERABILITY_POLICY:
            raise ValueError(
                "recoverability_policy must exactly match the current benchmark semantics: "
                f"{_EXPECTED_RECOVERABILITY_POLICY}. Got {self.recoverability_policy}."
            )

    @classmethod
    def v1(cls) -> AionoBasicComponentsPeriodicConfig:
        return cls(
            frequency_hz="auto",
            min_full_periods=1.0,
            nyquist_fraction=0.9,
            recoverability_policy=dict(_EXPECTED_RECOVERABILITY_POLICY),
            sawtooth_min_points_per_period=5,
            square_min_points_in_shorter_plateau=2,
            amplitude=UniformRange(0.2, 1.2),
            phase=UniformRange(0.0, 2.0 * 3.141592653589793),
            offset=UniformRange(-0.2, 0.2),
            square_duty_cycle=UniformRange(0.1, 0.9),
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> AionoBasicComponentsPeriodicConfig:
        frequency_hz_raw = mapping.get("frequency_hz")
        if frequency_hz_raw == "auto":
            frequency_hz: str | UniformRange = "auto"
        elif isinstance(frequency_hz_raw, Mapping):
            frequency_hz = UniformRange.from_mapping(frequency_hz_raw, name="frequency_hz")
        else:
            raise ValueError(
                "periodic.frequency_hz must be 'auto' or a {low, high} mapping. "
                f"Got {frequency_hz_raw!r}."
            )

        min_full_periods = mapping.get("min_full_periods")
        nyquist_fraction = mapping.get("nyquist_fraction")
        if not isinstance(min_full_periods, (int, float)):
            raise ValueError(
                "periodic.min_full_periods must be numeric, "
                f"got {min_full_periods!r}."
            )
        if not isinstance(nyquist_fraction, (int, float)):
            raise ValueError(
                "periodic.nyquist_fraction must be numeric, "
                f"got {nyquist_fraction!r}."
            )

        recoverability_policy_raw = mapping.get("recoverability_policy")
        if not isinstance(recoverability_policy_raw, Mapping):
            raise ValueError(
                "periodic.recoverability_policy must be a mapping, "
                f"got {recoverability_policy_raw!r}."
            )
        recoverability_policy = {
            str(key): str(value) for key, value in recoverability_policy_raw.items()
        }

        sawtooth_min_points_per_period = mapping.get("sawtooth_min_points_per_period")
        square_min_points_in_shorter_plateau = mapping.get(
            "square_min_points_in_shorter_plateau"
        )
        if not isinstance(sawtooth_min_points_per_period, int):
            raise ValueError(
                "periodic.sawtooth_min_points_per_period must be an int, "
                f"got {sawtooth_min_points_per_period!r}."
            )
        if not isinstance(square_min_points_in_shorter_plateau, int):
            raise ValueError(
                "periodic.square_min_points_in_shorter_plateau must be an int, "
                f"got {square_min_points_in_shorter_plateau!r}."
            )

        return cls(
            frequency_hz=frequency_hz,
            min_full_periods=float(min_full_periods),
            nyquist_fraction=float(nyquist_fraction),
            recoverability_policy=recoverability_policy,
            sawtooth_min_points_per_period=int(sawtooth_min_points_per_period),
            square_min_points_in_shorter_plateau=int(square_min_points_in_shorter_plateau),
            amplitude=UniformRange.from_mapping(mapping=_require_mapping(mapping, "amplitude"), name="amplitude"),
            phase=UniformRange.from_mapping(mapping=_require_mapping(mapping, "phase"), name="phase"),
            offset=UniformRange.from_mapping(mapping=_require_mapping(mapping, "offset"), name="offset"),
            square_duty_cycle=UniformRange.from_mapping(
                mapping=_require_mapping(mapping, "square_duty_cycle"),
                name="square_duty_cycle",
            ),
        )


@dataclass(frozen=True)
class ResolvedAionoBasicComponentsPeriodicContract:
    """Fully materialized benchmark semantics for one resolved sequence length."""

    benchmark_family: str
    benchmark_version: str
    baseline_sampling_frequency_hz: int
    sampling_frequency_hz: int
    seq_len: int
    duration_sec: float
    periodic_frequency_mode: str
    periodic_frequency_resolution_source: str
    periodic_frequency_min_full_periods: float
    periodic_frequency_nyquist_fraction: float
    sawtooth_min_points_per_period: int
    square_min_points_in_shorter_plateau: int
    square_duty_cycle_min: float
    square_duty_cycle_max: float
    square_shorter_plateau_fraction_min: float
    square_frequency_hz_recoverability_upper_bound: float
    signals: dict[str, ResolvedPeriodicSignalConfig]

    def signal(self, name: str) -> ResolvedPeriodicSignalConfig:
        try:
            return self.signals[name]
        except KeyError as error:
            raise KeyError(f"Unknown periodic signal {name!r}.") from error

    def manifest_fields(self) -> dict[str, object]:
        return {
            "benchmark_family": self.benchmark_family,
            "benchmark_version": self.benchmark_version,
            "baseline_sampling_frequency_hz": int(self.baseline_sampling_frequency_hz),
            "duration_sec": float(self.duration_sec),
            "periodic_frequency_mode": self.periodic_frequency_mode,
            "periodic_frequency_resolution_source": self.periodic_frequency_resolution_source,
            "periodic_frequency_min_full_periods": float(self.periodic_frequency_min_full_periods),
            "periodic_frequency_nyquist_fraction": float(self.periodic_frequency_nyquist_fraction),
            "sine_recoverability_policy": self.signal("sine").recoverability_policy,
            "sine_frequency_hz_resolved_low": float(self.signal("sine").frequency_hz.low),
            "sine_frequency_hz_resolved_high": float(self.signal("sine").frequency_hz.high),
            "sawtooth_recoverability_policy": self.signal("sawtooth").recoverability_policy,
            "sawtooth_frequency_hz_resolved_low": float(self.signal("sawtooth").frequency_hz.low),
            "sawtooth_frequency_hz_resolved_high": float(self.signal("sawtooth").frequency_hz.high),
            "square_recoverability_policy": self.signal("square").recoverability_policy,
            "square_frequency_hz_resolved_low": float(self.signal("square").frequency_hz.low),
            "square_frequency_hz_resolved_high": float(self.signal("square").frequency_hz.high),
            "sawtooth_min_points_per_period": int(self.sawtooth_min_points_per_period),
            "square_min_points_in_shorter_plateau": int(self.square_min_points_in_shorter_plateau),
            "square_duty_cycle_min": float(self.square_duty_cycle_min),
            "square_duty_cycle_max": float(self.square_duty_cycle_max),
            "square_shorter_plateau_fraction_min": float(self.square_shorter_plateau_fraction_min),
            "square_frequency_hz_recoverability_upper_bound": float(
                self.square_frequency_hz_recoverability_upper_bound
            ),
            "periodic_sampler_specs": {
                name: signal.sampler_specs() for name, signal in self.signals.items()
            },
        }


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping, got {value!r}.")
    return value


def _make_signal_config(
    *,
    name: str,
    recoverability_policy: str,
    amplitude: UniformRange,
    frequency_hz: UniformRange,
    phase: UniformRange,
    offset: UniformRange,
    duty_cycle: UniformRange | None = None,
) -> ResolvedPeriodicSignalConfig:
    return ResolvedPeriodicSignalConfig(
        name=name,
        recoverability_policy=recoverability_policy,
        amplitude=amplitude,
        frequency_hz=frequency_hz,
        phase=phase,
        offset=offset,
        duty_cycle=duty_cycle,
    )


def resolve_aiono_basic_components_periodic_contract(
    *,
    seq_len: int,
    sampling_frequency_hz: int,
    config: AionoBasicComponentsPeriodicConfig,
    benchmark_family: str = AIONO_BASIC_COMPONENTS_BENCHMARK_FAMILY,
    benchmark_version: str = AIONO_BASIC_COMPONENTS_BENCHMARK_VERSION,
) -> ResolvedAionoBasicComponentsPeriodicContract:
    """Resolve waveform-specific periodic bounds for `aiono_basic_components/v1`.

    `frequency_hz='auto'` expands to the widest recoverable range allowed by:

    - one-source-of-truth baseline `sampling_frequency_hz=500`;
    - `min_full_periods / duration_sec` for the lower bound;
    - Nyquist for `sine`;
    - points-per-period for `sawtooth`;
    - duty-cycle-aware shorter-plateau sampling for `square`.
    """

    if benchmark_family != AIONO_BASIC_COMPONENTS_BENCHMARK_FAMILY:
        raise ValueError(
            "Unexpected benchmark_family for Aiono basic-components resolver: "
            f"{benchmark_family!r}."
        )
    if benchmark_version != AIONO_BASIC_COMPONENTS_BENCHMARK_VERSION:
        raise ValueError(
            "Unexpected benchmark_version for Aiono basic-components resolver: "
            f"{benchmark_version!r}."
        )
    if int(sampling_frequency_hz) != AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ:
        raise ValueError(
            "Aiono basic-components v1 requires the baseline sampling frequency "
            f"{AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ} Hz, got "
            f"{int(sampling_frequency_hz)} Hz."
        )
    if seq_len <= 1:
        raise ValueError(f"seq_len must be > 1 to resolve periodic bounds, got {seq_len}.")

    duration_sec = float(seq_len - 1) / float(sampling_frequency_hz)
    f_min_full_period = float(config.min_full_periods) / duration_sec
    f_max_nyquist = float(config.nyquist_fraction) * float(sampling_frequency_hz) / 2.0
    square_shorter_plateau_fraction_min = min(
        float(config.square_duty_cycle.low),
        1.0 - float(config.square_duty_cycle.high),
    )
    if square_shorter_plateau_fraction_min <= 0:
        raise ValueError(
            "square_duty_cycle bounds make the shorter plateau non-positive: "
            f"low={config.square_duty_cycle.low} high={config.square_duty_cycle.high}."
        )

    resolved_high_auto = {
        "sine": f_max_nyquist,
        "sawtooth": min(
            f_max_nyquist,
            float(sampling_frequency_hz) / float(config.sawtooth_min_points_per_period),
        ),
        "square": min(
            f_max_nyquist,
            float(sampling_frequency_hz)
            * square_shorter_plateau_fraction_min
            / float(config.square_min_points_in_shorter_plateau),
        ),
    }
    for signal_name, resolved_high in resolved_high_auto.items():
        if resolved_high <= f_min_full_period:
            raise ValueError(
                "Resolved periodic range is empty for "
                f"{signal_name}: low={f_min_full_period} high={resolved_high} "
                f"(seq_len={seq_len}, sampling_frequency_hz={sampling_frequency_hz})."
            )

    if isinstance(config.frequency_hz, UniformRange):
        periodic_frequency_mode = "explicit"
        periodic_frequency_resolution_source = "config.explicit_bounds"
        explicit_bounds = config.frequency_hz
        violating_signals = [
            signal_name
            for signal_name, resolved_high in resolved_high_auto.items()
            if explicit_bounds.high > resolved_high or explicit_bounds.low < f_min_full_period
        ]
        if violating_signals:
            per_signal_high = {
                signal_name: float(resolved_high_auto[signal_name])
                for signal_name in violating_signals
            }
            raise ValueError(
                "Explicit periodic frequency bounds are not recoverable for all signals. "
                f"explicit_low={explicit_bounds.low} explicit_high={explicit_bounds.high} "
                f"minimum_low={f_min_full_period} per_signal_high={per_signal_high} "
                f"violating_signals={violating_signals}"
            )
        resolved_bounds = {
            signal_name: explicit_bounds for signal_name in _PERIODIC_SIGNAL_ORDER
        }
    else:
        periodic_frequency_mode = "auto"
        periodic_frequency_resolution_source = (
            "aiono.resolve_aiono_basic_components_periodic_contract"
        )
        resolved_bounds = {
            signal_name: UniformRange(low=f_min_full_period, high=resolved_high_auto[signal_name])
            for signal_name in _PERIODIC_SIGNAL_ORDER
        }

    signals = {
        "sine": _make_signal_config(
            name="sine",
            recoverability_policy=config.recoverability_policy["sine"],
            amplitude=config.amplitude,
            frequency_hz=resolved_bounds["sine"],
            phase=config.phase,
            offset=config.offset,
        ),
        "sawtooth": _make_signal_config(
            name="sawtooth",
            recoverability_policy=config.recoverability_policy["sawtooth"],
            amplitude=config.amplitude,
            frequency_hz=resolved_bounds["sawtooth"],
            phase=config.phase,
            offset=config.offset,
        ),
        "square": _make_signal_config(
            name="square",
            recoverability_policy=config.recoverability_policy["square"],
            amplitude=config.amplitude,
            frequency_hz=resolved_bounds["square"],
            phase=config.phase,
            offset=config.offset,
            duty_cycle=config.square_duty_cycle,
        ),
    }

    return ResolvedAionoBasicComponentsPeriodicContract(
        benchmark_family=benchmark_family,
        benchmark_version=benchmark_version,
        baseline_sampling_frequency_hz=AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ,
        sampling_frequency_hz=int(sampling_frequency_hz),
        seq_len=int(seq_len),
        duration_sec=duration_sec,
        periodic_frequency_mode=periodic_frequency_mode,
        periodic_frequency_resolution_source=periodic_frequency_resolution_source,
        periodic_frequency_min_full_periods=float(config.min_full_periods),
        periodic_frequency_nyquist_fraction=float(config.nyquist_fraction),
        sawtooth_min_points_per_period=int(config.sawtooth_min_points_per_period),
        square_min_points_in_shorter_plateau=int(config.square_min_points_in_shorter_plateau),
        square_duty_cycle_min=float(config.square_duty_cycle.low),
        square_duty_cycle_max=float(config.square_duty_cycle.high),
        square_shorter_plateau_fraction_min=float(square_shorter_plateau_fraction_min),
        square_frequency_hz_recoverability_upper_bound=float(resolved_high_auto["square"]),
        signals=signals,
    )
