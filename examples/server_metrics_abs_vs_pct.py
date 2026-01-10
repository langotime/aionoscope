from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from toyts import (
    ClippingView,
    MissingnessView,
    NoiseView,
    SamplingAggregationView,
    SynthPipeline,
    TrendSeasonAnomalyProcess,
    UnitsAbsoluteView,
    UnitsPercentOfCapacityView,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process = TrendSeasonAnomalyProcess(
        seq_len=1440,
        components=4,
        regime_classes=["steady", "ramping", "spiky"],
        anomaly_classes=["none", "drop", "spike"],
    )

    views = {
        "abs": nn.Sequential(
            UnitsAbsoluteView(),
            NoiseView(noise_std=0.05),
            MissingnessView(dropout_prob=0.02, gap_prob=0.1, gap_length=12, hold_prob=0.05),
            SamplingAggregationView(mode="mean", window=5),
        ),
        "pct": nn.Sequential(
            UnitsPercentOfCapacityView(capacity_min=50.0, capacity_max=200.0),
            ClippingView(min_value=0.0, max_value=100.0),
            NoiseView(noise_std=0.03),
        ),
    }

    pipeline = SynthPipeline(process=process, views=views)
    batch = pipeline(batch_size=128, device=device)

    abs_x = batch["abs"].x  # [B, C, L]
    pct_x = batch["pct"].x  # [B, C, L]

    regime_labels = batch["abs"].y["regime"]  # [B]
    anomaly_labels = batch["abs"].y["anomaly_type"]  # [B]

    regime_names = batch["abs"].meta["process"]["regime_names"]
    anomaly_names = batch["abs"].meta["process"]["anomaly_names"]
    pct_capacity_meta = batch["pct"].view_meta("UnitsPercentOfCapacityView")
    print("pct capacity", tuple(pct_capacity_meta["capacity"].shape))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Server metrics: abs vs pct views and labels")

    _plot_multilead(axes[0, 0], abs_x[0], "abs view")  # [C, L]
    _plot_multilead(axes[0, 1], pct_x[0], "pct view")  # [C, L]
    _plot_label_hist(axes[1, 0], regime_labels, regime_names, "regime labels")
    _plot_label_hist(axes[1, 1], anomaly_labels, anomaly_names, "anomaly labels")

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "server_metrics_abs_vs_pct.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("abs", abs_x.shape, abs_x.dtype)
    print("pct", pct_x.shape, pct_x.dtype)
    print("regime labels", regime_labels[:5])
    print("anomaly labels", anomaly_labels[:5])
    print("saved figure", output_path)


def _plot_multilead(ax: plt.Axes, signal: torch.Tensor, title: str) -> None:
    signal_cpu = signal.detach().cpu()  # [C, L]
    num_channels, seq_len = signal_cpu.shape

    time_axis = torch.arange(seq_len)  # [L]
    time_values = time_axis.tolist()

    offset_scale = signal_cpu.abs().max().item()
    offsets = torch.arange(num_channels) * (offset_scale * 1.5)  # [C]

    for channel_idx in range(num_channels):
        channel_values = (signal_cpu[channel_idx] + offsets[channel_idx]).tolist()  # [L]
        ax.plot(time_values, channel_values, linewidth=0.9)

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("channel + offset")
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
