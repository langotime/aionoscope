from __future__ import annotations

import torch

from toyts import TrendSeasonAnomalyProcess


def test_trend_season_samples_meta() -> None:
    device = torch.device("cpu")

    process = TrendSeasonAnomalyProcess(
        seq_len=64,
        components=4,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(321)

    batch_size = 3
    state = process(batch_size=batch_size, device=device, rng=generator)

    samples = state.meta["samples"]
    proc_samples = samples["TrendSeasonAnomalyProcess"]

    expected_keys = {
        "trend_slope",
        "trend_offset",
        "season_freq",
        "season_phase",
        "season_amp",
        "anomaly_amp",
        "anomaly_center",
        "anomaly_sigma",
    }
    assert set(proc_samples.keys()) == expected_keys

    for key in expected_keys:
        value = proc_samples[key]
        assert isinstance(value, torch.Tensor)
        assert value.shape == (batch_size,)
