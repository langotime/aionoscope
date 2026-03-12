from __future__ import annotations

import torch

from aiono import ConstantProcess, SineWaveView, SynthPipeline, UniformSampler, ViewChain
from aiono.views.noise import GaussianNoiseView, UniformNoiseView
from aiono.views.trend import LinearTrendView


def test_basic_component_views_shape_and_determinism() -> None:
    device = torch.device("cpu")
    seq_len = 64

    process = ConstantProcess(seq_len=seq_len, sample_rate_hz=32.0, value=0.0)
    view = ViewChain(
        SineWaveView(
            seq_len=seq_len,
            amplitude=UniformSampler(0.5, 1.0),
            frequency_hz=UniformSampler(0.25, 2.0),
            phase=UniformSampler(0.0, 1.0),
            offset=0.0,
        ),
        LinearTrendView(seq_len=seq_len, slope=UniformSampler(-1.0, 1.0), intercept=0.0),
        GaussianNoiseView(noise_std=UniformSampler(0.01, 0.05)),
    )

    pipeline = SynthPipeline(process=process, views={"x": view})

    g1 = torch.Generator(device=device).manual_seed(123)
    g2 = torch.Generator(device=device).manual_seed(123)
    out1 = pipeline(batch_size=8, device=device, rng=g1)["x"]
    out2 = pipeline(batch_size=8, device=device, rng=g2)["x"]

    assert out1.x.shape == (8, 1, seq_len)
    assert out1.x.dtype == torch.float32
    torch.testing.assert_close(out1.x, out2.x)


def test_uniform_noise_is_bounded() -> None:
    device = torch.device("cpu")
    seq_len = 32
    amplitude = 0.5

    process = ConstantProcess(seq_len=seq_len, sample_rate_hz=1.0, value=0.0)
    view = ViewChain(UniformNoiseView(amplitude=amplitude))
    pipeline = SynthPipeline(process=process, views={"x": view})

    g = torch.Generator(device=device).manual_seed(999)
    obs = pipeline(batch_size=64, device=device, rng=g)["x"]

    assert float(obs.x.min().item()) >= -amplitude
    assert float(obs.x.max().item()) <= amplitude

