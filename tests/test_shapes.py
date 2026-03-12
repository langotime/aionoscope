from __future__ import annotations

import torch

from aiono import (
    ECGLeadsView,
    EventImpulseView,
    KernelConvView,
    PulseTrainProcess,
    SamplingAggregationView,
    SynthPipeline,
    TrendSeasonAnomalyProcess,
    UnitsAbsoluteView,
    ViewChain,
    make_pqrst_kernel_bank,
    pqrst_kernel_size,
)


def test_pulse_train_shapes() -> None:
    device = torch.device("cpu")

    process = PulseTrainProcess(
        seq_len=512,
        frequency_hz=6.0,
        sample_rate_hz=511.0,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
    )

    A0 = torch.tensor(
        [[1.0, 0.2, -0.1], [0.3, 0.9, 0.1]],
        dtype=torch.float32,
    )  # [C, K]

    spacing = (process.seq_len - 1) / (process.num_pulses + 1)
    kernel_size = pqrst_kernel_size(spacing=spacing, support_sigma=6.0)
    kernels = make_pqrst_kernel_bank(
        shape_names=process.shape_classes,
        spacing=spacing,
        kernel_size=kernel_size,
        device=device,
    )  # [K, T, W]
    padding = kernel_size // 2

    pipeline = SynthPipeline(
        process=process,
        views={
            "clean": ViewChain(
                EventImpulseView(
                    seq_len=process.seq_len,
                    amplitude_param="amplitude",
                    rounding="nearest",
                ),
                KernelConvView(kernels=kernels, padding=padding),
                ECGLeadsView(A0=A0, jitter_std=0.0, max_delay=0),
            )
        },
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(123)

    batch = pipeline(batch_size=4, device=device, rng=generator)
    obs = batch["clean"]

    assert obs.x.shape == (4, 2, 512)
    assert obs.x.dtype == torch.float32
    assert obs.y["shape"].shape == (4,)
    assert obs.y["rhythm"].shape == (4,)


def test_trend_shapes() -> None:
    device = torch.device("cpu")

    process = TrendSeasonAnomalyProcess(
        seq_len=300,
        components=4,
        regime_classes=["steady", "ramping", "spiky"],
        anomaly_classes=["none", "drop", "spike"],
    )

    pipeline = SynthPipeline(
        process=process,
        views={
            "abs": torch.nn.Sequential(
                UnitsAbsoluteView(),
                SamplingAggregationView(mode="mean", window=10),
            )
        },
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(456)

    batch = pipeline(batch_size=5, device=device, rng=generator)
    obs = batch["abs"]

    assert obs.x.shape == (5, 1, 30)
    assert obs.x.dtype == torch.float32
    assert obs.y["regime"].shape == (5,)
    assert obs.y["anomaly_type"].shape == (5,)
