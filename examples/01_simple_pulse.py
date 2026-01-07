from __future__ import annotations

import torch

from toyts.core.pipeline import SynthPipeline
from toyts.core.utils import utils_make_canonical_A0
from toyts.processes.pulse_train import PulseTrainProcess
from toyts.views.ecg_leads import ECGLeadsView
from toyts.views.noise import BaselineWanderView, NoiseView, NormalizeView
from toyts.views.sampling import SamplingAggregationView


def main() -> None:
    """Generate and plot a simple pulse train signal."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(1234)

    # 1. Define the Process
    # This process generates a latent PQRST-style signal.
    process = PulseTrainProcess(
        seq_len=1024,
        num_pulses=10,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
        amplitude=2.0,
    )

    # 2. Define the Views
    # Views transform the latent signal into observed signals.
    # We create a simple pipeline that simulates 8 ECG leads.
    A0 = utils_make_canonical_A0(num_leads=8, num_latent=3)
    views = {
        "clean": ECGLeadsView(A0=A0, jitter_std=0.0, max_delay=0),
        "noisy": torch.nn.Sequential(
            ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=3),
            NoiseView(noise_std=0.1),
            BaselineWanderView(amplitude_std=0.3, freq_min=0.05, freq_max=0.2),
        ),
        "normalized_and_resampled": torch.nn.Sequential(
            ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=3),
            NoiseView(noise_std=0.1),
            BaselineWanderView(amplitude_std=0.3, freq_min=0.05, freq_max=0.2),
            NormalizeView(),
            SamplingAggregationView(mode="mean", window=4),
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
        print("  Shape labels:", obs.y["shape"])
        print("  Rhythm labels:", obs.y["rhythm"])

    # Example of how to plot one of the signals
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(batch["clean"].x[0, 0, :].cpu().numpy())
        ax1.set_title("View: 'clean'")
        ax1.grid(True)
        ax2.plot(batch["noisy"].x[0, 0, :].cpu().numpy())
        ax2.set_title("View: 'noisy'")
        ax2.grid(True)
        plt.tight_layout()
        print("\nPlotting example output to '01_simple_pulse.png'")
        plt.savefig("01_simple_pulse.png")

    except ImportError:
        print(
            "\nMatplotlib not found. Skipping plot. "
            "Install it with: pip install matplotlib"
        )


if __name__ == "__main__":
    main()
