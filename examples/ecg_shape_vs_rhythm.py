from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from toyts import (
    BaselineWanderView,
    ECGLeadsView,
    EventImpulseView,
    GaussianNoiseView,
    KernelConvView,
    NormalizeView,
    PulseTrainProcess,
    SynthPipeline,
    make_pqrst_kernel_bank,
    pqrst_kernel_size,
)
from toyts.core.utils import utils_make_canonical_A0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process = PulseTrainProcess(
        seq_len=2048,
        frequency_hz=1.95,
        sample_rate_hz=500.0,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
        amplitude=1.0,
    )

    spacing = (process.seq_len - 1) / (process.num_pulses + 1)
    kernel_size = pqrst_kernel_size(spacing=spacing, support_sigma=6.0)
    kernels = make_pqrst_kernel_bank(
        shape_names=process.shape_classes,
        spacing=spacing,
        kernel_size=kernel_size,
        device=device,
    )  # [K, T, W]
    padding = kernel_size // 2

    A0 = utils_make_canonical_A0(num_leads=12, num_latent=3).to(device)  # [C, K]
    lead_names = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    def event_head() -> list[nn.Module]:
        return [
            EventImpulseView(
                seq_len=process.seq_len,
                amplitude_param="amplitude",
                rounding="nearest",
            ),
            KernelConvView(kernels=kernels, padding=padding),
        ]

    views = {
        "clean": nn.Sequential(
            *event_head(),
            ECGLeadsView(A0=A0, jitter_std=0.03, max_delay=3),
        ),
        "noisy": nn.Sequential(
            *event_head(),
            ECGLeadsView(A0=A0, jitter_std=0.03, max_delay=3),
            BaselineWanderView(amplitude_std=0.2, freq_min=0.1, freq_max=0.5),
            GaussianNoiseView(noise_std=0.15),
            NormalizeView(),
        ),
    }

    pipeline = SynthPipeline(process=process, views=views)
    batch = pipeline(batch_size=64, device=device)

    clean = batch["clean"].x  # [B, C, L]
    noisy = batch["noisy"].x  # [B, C, L]

    shape_labels = batch["clean"].y["shape"]  # [B]
    rhythm_labels = batch["clean"].y["rhythm"]  # [B]

    shape_names = batch["clean"].meta["process"]["shape_names"]
    rhythm_names = batch["clean"].meta["process"]["rhythm_names"]
    clean_ecg_meta = batch["clean"].view_meta("ECGLeadsView")
    print("clean ECG delays", tuple(clean_ecg_meta["delays"].shape))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ECG: clean vs noisy views and labels")

    sample_idx = 0
    clean_first = clean[sample_idx]  # [C, L]
    noisy_first = noisy[sample_idx]  # [C, L]
    shape_label_idx = int(shape_labels[sample_idx].item())
    rhythm_label_idx = int(rhythm_labels[sample_idx].item())
    shape_label = shape_names[shape_label_idx]
    rhythm_label = rhythm_names[rhythm_label_idx]
    sample_label = f"shape={shape_label}, rhythm={rhythm_label}"

    _plot_multilead(
        axes[0, 0],
        clean_first,
        f"clean view ({sample_label})",
        lead_names,
    )
    _plot_multilead(
        axes[0, 1],
        noisy_first,
        f"noisy view ({sample_label})",
        lead_names,
    )
    _plot_label_hist(axes[1, 0], shape_labels, shape_names, "shape labels")
    _plot_label_hist(axes[1, 1], rhythm_labels, rhythm_names, "rhythm labels")

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ecg_shape_vs_rhythm.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("clean", clean.shape, clean.dtype)
    print("noisy", noisy.shape, noisy.dtype)
    print("shape labels", shape_labels[:5])
    print("rhythm labels", rhythm_labels[:5])
    print("saved figure", output_path)


def _plot_multilead(
    ax: plt.Axes,
    signal: torch.Tensor,
    title: str,
    lead_names: list[str],
) -> None:
    signal_cpu = signal.detach().cpu()  # [C, L]
    num_leads, seq_len = signal_cpu.shape
    if len(lead_names) != num_leads:
        raise ValueError(
            "lead_names must match the number of leads. "
            f"Expected {num_leads}, got {len(lead_names)}."
        )

    time_axis = torch.arange(seq_len)  # [L]
    time_values = time_axis.tolist()

    offset_scale = signal_cpu.abs().max().item()
    offsets = torch.arange(num_leads) * (offset_scale * 2.5)  # [C]

    for lead_idx in range(num_leads):
        lead_values = (signal_cpu[lead_idx] + offsets[lead_idx]).tolist()  # [L]
        ax.plot(time_values, lead_values, linewidth=0.8)

    ax.set_yticks(offsets.tolist(), lead_names)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("lead + offset")
    ax.grid(True, alpha=0.3)


def _plot_label_hist(
    ax: plt.Axes,
    labels: torch.Tensor,
    names: list[str],
    title: str,
) -> None:
    labels_cpu = labels.detach().cpu()  # [B]
    counts = torch.bincount(labels_cpu, minlength=len(names))  # [K]

    ax.bar(range(len(names)), counts.tolist())
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("count")


if __name__ == "__main__":
    main()
