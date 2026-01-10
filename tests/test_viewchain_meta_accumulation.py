from __future__ import annotations

import torch

from toyts import (
    BaselineWanderView,
    ECGLeadsView,
    EventImpulseView,
    KernelConvView,
    NormalizeView,
    PulseTrainProcess,
    SynthPipeline,
    ViewChain,
    make_pqrst_kernel_bank,
    pqrst_kernel_size,
)


def test_viewchain_accumulates_views_meta() -> None:
    device = torch.device("cpu")

    process = PulseTrainProcess(
        seq_len=128,
        frequency_hz=5.0,
        sample_rate_hz=127.0,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
    )

    A0 = torch.tensor(
        [[0.8, 0.1, 0.2], [0.2, 0.9, -0.1]],
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
                ECGLeadsView(A0=A0, jitter_std=0.01, max_delay=2),
                BaselineWanderView(
                    amplitude_std=0.05,
                    freq_min=0.1,
                    freq_max=0.5,
                ),
                NormalizeView(),
            )
        },
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(123)

    batch_size = 2
    batch = pipeline(batch_size=batch_size, device=device, rng=generator)
    obs = batch["clean"]

    assert set(obs.meta.keys()) == {"process", "views", "pipeline_seed"}
    assert "view" not in obs.meta

    views = obs.meta["views"]
    assert isinstance(views, list)
    assert len(views) == 5
    assert views[0]["view"] == "EventImpulseView"
    assert views[1]["view"] == "KernelConvView"
    assert views[2]["view"] == "ECGLeadsView"
    assert views[3]["view"] == "BaselineWanderView"
    assert views[4]["view"] == "NormalizeView"

    ecg_meta = obs.view_meta("ECGLeadsView")
    delays = ecg_meta["delays"]  # [B, C]
    assert delays.shape == (batch_size, obs.x.shape[1])

    wander_meta = obs.view_meta("BaselineWanderView")
    freq = wander_meta["freq"]  # [B, C, 1]
    phase = wander_meta["phase"]  # [B, C, 1]
    amplitude = wander_meta["amplitude"]  # [B, C, 1]
    assert freq.shape == (batch_size, obs.x.shape[1], 1)
    assert phase.shape == (batch_size, obs.x.shape[1], 1)
    assert amplitude.shape == (batch_size, obs.x.shape[1], 1)
