from __future__ import annotations

import torch

from aiono import PulseTrainProcess


def test_event_train_samples_meta() -> None:
    device = torch.device("cpu")

    process = PulseTrainProcess(
        seq_len=128,
        frequency_hz=5.0,
        sample_rate_hz=127.0,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(123)

    batch_size = 4
    state = process(batch_size=batch_size, device=device, rng=generator)

    samples = state.meta["samples"]
    node_key = "EventTrainNode:events"
    assert node_key in samples

    node_samples = samples[node_key]
    intervals = node_samples["intervals"]  # [B, N+1]
    missed_indices = node_samples["missed_indices"]  # [B]
    phase_offset = node_samples["phase_offset"]  # [B, 1]

    assert intervals.shape == (batch_size, process.num_pulses + 1)
    assert missed_indices.shape == (batch_size,)
    assert phase_offset.shape == (batch_size, 1)

    assert intervals.dtype == torch.float32
    assert missed_indices.dtype == torch.int64
    assert phase_offset.dtype == torch.float32

    torch.testing.assert_close(
        intervals.sum(dim=1),
        torch.ones(batch_size, device=device),
    )

    valid_missed = (missed_indices == -1) | (
        (missed_indices >= 0) & (missed_indices <= process.num_pulses)
    )
    assert torch.all(valid_missed)
