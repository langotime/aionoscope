from __future__ import annotations

import pytest
import torch

from toyts import ECGLeadsView, PulseTrainProcess, SynthPipeline
from toyts.core.utils import utils_make_random_A0


def _make_pulse_process(seq_len: int) -> PulseTrainProcess:
    return PulseTrainProcess(
        seq_len=seq_len,
        frequency_hz=5.0,
        sample_rate_hz=float(seq_len - 1),
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
    )


def test_ecg_leads_accepts_batched_A0() -> None:
    device = torch.device("cpu")
    process = _make_pulse_process(seq_len=64)

    batch_size = 3
    num_leads = 4
    num_latent = 3

    A0 = torch.arange(
        batch_size * num_leads * num_latent,
        dtype=torch.float32,
    ).reshape(batch_size, num_leads, num_latent)  # [B, C, K]

    pipeline = SynthPipeline(
        process=process,
        views={"clean": ECGLeadsView(A0=A0, jitter_std=0.0, max_delay=0)},
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(123)

    batch = pipeline(batch_size=batch_size, device=device, rng=generator)
    obs = batch["clean"]

    assert obs.x.shape == (batch_size, num_leads, 64)
    assert obs.meta["A0"].shape == (batch_size, num_leads, num_latent)
    assert torch.allclose(obs.meta["A0"], A0)


def test_ecg_leads_rejects_batched_A0_mismatch() -> None:
    device = torch.device("cpu")
    process = _make_pulse_process(seq_len=32)

    A0 = torch.ones((2, 4, 3), dtype=torch.float32)  # [B, C, K]

    pipeline = SynthPipeline(
        process=process,
        views={"clean": ECGLeadsView(A0=A0, jitter_std=0.0, max_delay=0)},
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(321)

    with pytest.raises(ValueError, match="batch size"):
        pipeline(batch_size=3, device=device, rng=generator)


def test_ecg_leads_accepts_callable_A0() -> None:
    device = torch.device("cpu")
    process = _make_pulse_process(seq_len=48)

    batch_size = 2
    num_leads = 3
    num_latent = 3

    def sample_A0(
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, num_leads, num_latent), device=device)  # [B, C, K]

    pipeline = SynthPipeline(
        process=process,
        views={"clean": ECGLeadsView(A0=sample_A0, jitter_std=0.0, max_delay=0)},
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(999)

    batch = pipeline(batch_size=batch_size, device=device, rng=generator)
    obs = batch["clean"]

    zeros = torch.zeros_like(obs.x)  # [B, C, L]
    assert torch.allclose(obs.x, zeros)
    assert obs.meta["A0"].shape == (batch_size, num_leads, num_latent)


def test_utils_make_random_A0_shape() -> None:
    device = torch.device("cpu")
    rng = torch.Generator(device=device).manual_seed(111)

    A0 = utils_make_random_A0(
        num_leads=2,
        num_latent=3,
        rng=rng,
        device=device,
    )  # [C, K]

    assert A0.shape == (2, 3)
    assert A0.device == device
    assert A0.dtype == torch.float32
