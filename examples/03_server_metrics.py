from __future__ import annotations

import torch

from toyts.core.pipeline import SynthPipeline
from toyts.processes.trend_season import TrendSeasonAnomalyProcess
from toyts.views.noise import NormalizeView
from toyts.views.units import (
    ClippingView,
    UnitsAbsoluteView,
    UnitsPercentOfCapacityView,
)


def main() -> None:
    """Generate a signal simulating a server metric like CPU usage."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(42)

    # 1. Define the Process
    # This process generates a signal with trend, seasonality, and anomalies,
    # perfect for simulating server metrics.
    process = TrendSeasonAnomalyProcess(
        seq_len=720,  # e.g., 12 hours of data at 1-minute resolution
        components=3,
        regime_classes=["steady", "ramping", "spiky"],
        anomaly_classes=["none", "drop", "spike"],
        slope_max=0.5,
        season_amp=15.0,
        spiky_boost=2.5,
        season_freq_min=1.0,  # 1 cycle per 12 hours
        season_freq_max=3.0,  # 3 cycles per 12 hours
        anomaly_scale=40.0,
    )

    # 2. Define the Views
    # We can model different ways the same underlying process might be observed.
    views = {
        # Raw value, e.g., raw request count
        "absolute": UnitsAbsoluteView(),
        # As a percentage of a variable daily capacity
        "percent_of_capacity": UnitsPercentOfCapacityView(
            capacity_min=80.0, capacity_max=120.0
        ),
        # As a percentage, but clipped at 0% and 100% (e.g., CPU %)
        "cpu_percent": torch.nn.Sequential(
            UnitsPercentOfCapacityView(capacity_min=90.0, capacity_max=110.0),
            ClippingView(min_value=0.0, max_value=100.0),
        ),
        # A normalized version for ML models
        "normalized": torch.nn.Sequential(
            UnitsPercentOfCapacityView(capacity_min=90.0, capacity_max=110.0),
            NormalizeView(),
        ),
    }

    # 3. Create and run the pipeline
    pipeline = SynthPipeline(process=process, views=views)
    pipeline.to(device)
    batch = pipeline(batch_size=4, device=device, rng=rng)

    # 4. Inspect the output
    print("Generated batch keys:", batch.keys())
    for name, obs in batch.items():
        print(f"--- View: {name} ---")
        print("  Signal shape:", obs.x.shape)
        print("  Regime labels:", obs.y["regime"])
        print("  Anomaly labels:", obs.y["anomaly_type"])
        print(f"  Signal range: [{obs.x.min():.2f}, {obs.x.max():.2f}]")

    capacity_meta = batch["percent_of_capacity"].view_meta("UnitsPercentOfCapacityView")
    print("\nView metadata:")
    print("  percent_of_capacity capacity:", tuple(capacity_meta["capacity"].shape))

    # Example of how to plot the different unit views
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            len(views), 1, figsize=(12, 10), sharex=True, sharey=False
        )
        for i, (name, obs) in enumerate(batch.items()):
            axes[i].plot(obs.x[0, 0, :].cpu().numpy())
            axes[i].set_title(f"View: '{name}'")
            axes[i].grid(True)
        plt.tight_layout()
        print("\nPlotting example output to '03_server_metrics.png'")
        plt.savefig("03_server_metrics.png")

    except ImportError:
        print(
            "\nMatplotlib not found. Skipping plot. "
            "Install it with: pip install matplotlib"
        )


if __name__ == "__main__":
    main()
