from __future__ import annotations

import torch

from toyts import ECGLeadsView, PulseTrainProcess, SynthPipeline


def test_shortcut_baseline_on_rhythm() -> None:
    device = torch.device("cpu")

    process = PulseTrainProcess(
        seq_len=512,
        num_pulses=8,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
    )

    A0 = torch.tensor(
        [[1.0, 0.2, 0.0], [0.3, 0.7, -0.2]],
        dtype=torch.float32,
    )  # [C, K]

    pipeline = SynthPipeline(
        process=process,
        views={"clean": ECGLeadsView(A0=A0, jitter_std=0.01, max_delay=1)},
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(2024)

    batch = pipeline(batch_size=512, device=device, rng=generator)
    x = batch["clean"].x  # [B, C, L]
    y = batch["clean"].y["rhythm"]  # [B]

    mean = x.mean(dim=(1, 2))  # [B]
    std = x.std(dim=(1, 2), unbiased=False)  # [B]
    energy = x.pow(2).mean(dim=(1, 2))  # [B]
    threshold = x.abs().mean(dim=(1, 2), keepdim=True) + x.abs().std(dim=(1, 2), keepdim=True)  # [B, 1, 1]
    peak_count = (x.abs() > threshold).sum(dim=(1, 2))  # [B]

    features = torch.stack([mean, std, energy, peak_count], dim=1)  # [B, F]
    num_classes = int(y.max().item()) + 1

    class_means = []
    for class_idx in range(num_classes):
        class_mask = y == class_idx  # [B]
        if torch.count_nonzero(class_mask) == 0:
            raise ValueError(f"No samples for class {class_idx}.")
        class_means.append(features[class_mask].mean(dim=0))

    centers = torch.stack(class_means, dim=0)  # [C, F]
    distances = (features[:, None, :] - centers[None, :, :]).pow(2).sum(dim=2)  # [B, C]
    preds = distances.argmin(dim=1)  # [B]

    accuracy = (preds == y).float().mean()  # []
    assert accuracy < 0.6
