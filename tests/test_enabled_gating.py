from __future__ import annotations

import torch

from toyts import SineWaveView, UniformSampler, ViewChain
from toyts.core.types import Observation
from toyts.views.noise import GaussianNoiseView
from toyts.views.trend import LinearTrendView


def test_component_views_respect_enabled_masks() -> None:
    device = torch.device("cpu")
    batch_size = 4
    seq_len = 32

    rng = torch.Generator(device=device).manual_seed(123)
    x = torch.randn((batch_size, 1, seq_len), generator=rng, device=device)  # [B, 1, L]

    enabled = torch.tensor([True, False, True, False], device=device)  # [B]
    process_meta = {
        "sample_rate_hz": 10.0,
        "enabled": {
            "trend": enabled,
            "noise": enabled,
            "sine": enabled,
        },
    }
    obs = Observation(x=x, y={}, meta={"process": process_meta})

    trend = LinearTrendView(seq_len=seq_len, slope=1.0, intercept=0.0, enabled_key="trend")
    out_trend = trend(obs, rng=torch.Generator(device=device).manual_seed(1))
    torch.testing.assert_close(out_trend.x[~enabled], x[~enabled])

    noise = GaussianNoiseView(noise_std=0.1, enabled_key="noise")
    out_noise = noise(obs, rng=torch.Generator(device=device).manual_seed(2))
    torch.testing.assert_close(out_noise.x[~enabled], x[~enabled])

    sine = SineWaveView(
        seq_len=seq_len,
        amplitude=1.0,
        frequency_hz=1.0,
        phase=0.0,
        offset=0.0,
        enabled_key="sine",
    )
    out_sine = sine(obs, rng=torch.Generator(device=device).manual_seed(3))
    torch.testing.assert_close(out_sine.x[~enabled], x[~enabled])


def test_viewchain_rng_splitting_is_stable_under_gating() -> None:
    device = torch.device("cpu")
    batch_size = 8
    seq_len = 16

    x = torch.zeros((batch_size, 1, seq_len), device=device)  # [B, 1, L]

    enabled_all = torch.ones((batch_size,), device=device, dtype=torch.bool)  # [B]
    enabled_none = torch.zeros((batch_size,), device=device, dtype=torch.bool)  # [B]

    obs_enabled = Observation(
        x=x,
        y={},
        meta={
            "process": {
                "enabled": {"n1": enabled_all, "n2": enabled_all},
            }
        },
    )
    obs_disabled = Observation(
        x=x,
        y={},
        meta={
            "process": {
                "enabled": {"n1": enabled_none, "n2": enabled_all},
            }
        },
    )

    chain = ViewChain(
        GaussianNoiseView(noise_std=UniformSampler(0.01, 0.1), enabled_key="n1"),
        GaussianNoiseView(noise_std=UniformSampler(0.01, 0.1), enabled_key="n2"),
    )

    g1 = torch.Generator(device=device).manual_seed(999)
    out1 = chain(obs_enabled, rng=g1)
    g2 = torch.Generator(device=device).manual_seed(999)
    out2 = chain(obs_disabled, rng=g2)

    noise_std_2_1 = out1.meta["views"][1]["noise_std"]  # [B]
    noise_std_2_2 = out2.meta["views"][1]["noise_std"]  # [B]
    torch.testing.assert_close(noise_std_2_1, noise_std_2_2)

