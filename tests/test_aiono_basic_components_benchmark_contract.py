from __future__ import annotations

import math

import pytest

from aiono import (
    AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ,
    AionoBasicComponentsPeriodicConfig,
    UniformRange,
    resolve_aiono_basic_components_periodic_contract,
)


def test_periodic_contract_v1_auto_resolves_expected_bounds() -> None:
    contract = resolve_aiono_basic_components_periodic_contract(
        seq_len=5000,
        sampling_frequency_hz=AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ,
        config=AionoBasicComponentsPeriodicConfig.v1(),
    )

    expected_low = 500.0 / 4999.0
    assert contract.periodic_frequency_mode == "auto"
    assert contract.periodic_frequency_resolution_source.endswith(
        "resolve_aiono_basic_components_periodic_contract"
    )
    assert contract.sawtooth_min_points_per_period == 5
    assert contract.square_min_points_in_shorter_plateau == 2
    assert contract.square_shorter_plateau_fraction_min == pytest.approx(0.1)
    assert contract.signal("sine").frequency_hz.low == pytest.approx(expected_low)
    assert contract.signal("sine").frequency_hz.high == pytest.approx(225.0)
    assert contract.signal("sawtooth").frequency_hz.high == pytest.approx(100.0)
    assert contract.signal("square").frequency_hz.high == pytest.approx(25.0)
    assert contract.square_frequency_hz_recoverability_upper_bound == pytest.approx(25.0)
    assert contract.signal("square").sampler_specs()["duty_cycle"] == {
        "kind": "uniform",
        "low": 0.1,
        "high": 0.9,
    }


def test_periodic_contract_v1_rejects_wrong_sampling_frequency() -> None:
    with pytest.raises(ValueError, match="requires the baseline sampling frequency 500 Hz"):
        resolve_aiono_basic_components_periodic_contract(
            seq_len=5000,
            sampling_frequency_hz=1,
            config=AionoBasicComponentsPeriodicConfig.v1(),
        )


def test_periodic_contract_explicit_bounds_respect_square_duty_cycle() -> None:
    config = AionoBasicComponentsPeriodicConfig(
        frequency_hz=UniformRange(0.5, 26.0),
        min_full_periods=1.0,
        nyquist_fraction=0.9,
        recoverability_policy={
            "sine": "nyquist",
            "sawtooth": "min_points_per_period",
            "square": "min_points_in_shorter_plateau",
        },
        sawtooth_min_points_per_period=5,
        square_min_points_in_shorter_plateau=2,
        amplitude=UniformRange(0.2, 1.2),
        phase=UniformRange(0.0, 2.0 * math.pi),
        offset=UniformRange(-0.2, 0.2),
        square_duty_cycle=UniformRange(0.1, 0.9),
    )

    with pytest.raises(ValueError, match="violating_signals=\\['square'\\]"):
        resolve_aiono_basic_components_periodic_contract(
            seq_len=5000,
            sampling_frequency_hz=AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ,
            config=config,
        )
