from __future__ import annotations

import torch

from toyts import (
    ECGLeadsView,
    NoiseView,
    PulseTrainProcess,
    SynthPipeline,
    ViewChain,
)


def test_determinism_pulse_train() -> None:
    device = torch.device("cpu")

    process = PulseTrainProcess(
        seq_len=256,
        frequency_hz=4.0,
        sample_rate_hz=255.0,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
    )

    A0 = torch.tensor(
        [[0.8, 0.1, 0.2], [0.2, 0.9, -0.1]],
        dtype=torch.float32,
    )  # [C, K]

    views = {
        "clean": ECGLeadsView(A0=A0, jitter_std=0.01, max_delay=2),
        "noisy": ViewChain(
            ECGLeadsView(A0=A0, jitter_std=0.01, max_delay=2),
            NoiseView(noise_std=0.1),
        ),
    }

    pipeline = SynthPipeline(process=process, views=views)

    g1 = torch.Generator(device=device)
    g1.manual_seed(999)
    g2 = torch.Generator(device=device)
    g2.manual_seed(999)

    batch1 = pipeline(batch_size=6, device=device, rng=g1)
    batch2 = pipeline(batch_size=6, device=device, rng=g2)

    torch.testing.assert_close(batch1["clean"].x, batch2["clean"].x)
    torch.testing.assert_close(batch1["noisy"].x, batch2["noisy"].x)
    torch.testing.assert_close(batch1["clean"].y["shape"], batch2["clean"].y["shape"])
    torch.testing.assert_close(batch1["clean"].y["rhythm"], batch2["clean"].y["rhythm"])
