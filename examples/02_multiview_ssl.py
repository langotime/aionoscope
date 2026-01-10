from __future__ import annotations

import torch

from toyts.core.pipeline import SynthPipeline
from toyts.core.utils import utils_make_canonical_A0
from toyts.kernels.pqrst import make_pqrst_kernel_bank, pqrst_kernel_size
from toyts.datasets.iterable import SynthBatchIterableDataset
from toyts.processes.pulse_train import PulseTrainProcess
from toyts.views.ecg_leads import ECGLeadsView
from toyts.views.events import EventImpulseView, KernelConvView
from toyts.views.missingness import MissingnessView
from toyts.views.noise import BaselineWanderView, NoiseView, NormalizeView
from toyts.views.sampling import SamplingAggregationView


def main() -> None:
    """Demonstrate creating a multi-view dataset for self-supervised learning."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Define the Process
    process = PulseTrainProcess(
        seq_len=1024,
        frequency_hz=1.2,
        sample_rate_hz=250.0,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
        amplitude=2.0,
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

    # 2. Define two different augmentations (views) for SSL
    A0 = utils_make_canonical_A0(num_leads=1, num_latent=3)

    def event_head() -> list[torch.nn.Module]:
        return [
            EventImpulseView(
                seq_len=process.seq_len,
                amplitude_param="amplitude",
                rounding="nearest",
            ),
            KernelConvView(kernels=kernels, padding=padding),
        ]

    # View 1: A moderately noisy, resampled version
    view1_transform = torch.nn.Sequential(
        *event_head(),
        ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=2),
        NoiseView(noise_std=0.1),
        BaselineWanderView(amplitude_std=0.2, freq_min=0.1, freq_max=0.3),
        SamplingAggregationView(mode="downsample", stride=2),
        NormalizeView(),
    )

    # View 2: A version with more noise and some missing data
    view2_transform = torch.nn.Sequential(
        *event_head(),
        ECGLeadsView(A0=A0, jitter_std=0.1, max_delay=4),
        NoiseView(noise_std=0.15),
        BaselineWanderView(amplitude_std=0.4, freq_min=0.05, freq_max=0.2),
        MissingnessView(dropout_prob=0.05, gap_prob=0.1, gap_length=50, hold_prob=0.01),
        NormalizeView(),
    )

    views = {
        "view1": view1_transform,
        "view2": view2_transform,
    }

    # 3. Create the pipeline and wrap it in an Iterable Dataset
    pipeline = SynthPipeline(process=process, views=views)
    pipeline.to(device)
    dataset = SynthBatchIterableDataset(
        pipeline=pipeline,
        batch_size=4,
        device=device,
        seed=42,
        max_batches=25,  # samples_per_epoch / batch_size
    )

    # 4. Use with a DataLoader
    # The dataloader will simply yield the batches from our dataset.
    # batch_size must be None because the dataset already produces batches.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
    )

    # 5. Inspect a batch
    first_batch = next(iter(dataloader))

    print("Dataloader yields batches with keys:", first_batch.keys())
    view1_obs = first_batch["view1"]
    view2_obs = first_batch["view2"]

    print("\n--- View 1 ---")
    print("  Signal shape:", view1_obs.x.shape)
    print("  Shape labels:", view1_obs.y["shape"])

    print("\n--- View 2 ---")
    print("  Signal shape:", view2_obs.x.shape)
    print("  Shape labels (should match View 1):", view2_obs.y["shape"])

    view1_ecg_meta = view1_obs.view_meta("ECGLeadsView")
    print("  View 1 ECG delays:", tuple(view1_ecg_meta["delays"].shape))

    # Verify that the ground truth labels are the same for both views
    assert torch.all(view1_obs.y["shape"] == view2_obs.y["shape"])
    assert torch.all(view1_obs.y["rhythm"] == view2_obs.y["rhythm"])
    print("\nSuccessfully verified that paired views share the same labels.")

    # Example of how to plot one of the pairs
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        ax1.plot(view1_obs.x[0, 0, :].cpu().numpy())
        ax1.set_title("View 1 (resampled)")
        ax1.grid(True)
        ax2.plot(view2_obs.x[0, 0, :].cpu().numpy())
        ax2.set_title("View 2 (full length, with missingness)")
        ax2.grid(True)
        plt.tight_layout()
        print("\nPlotting example output to '02_multiview_ssl.png'")
        plt.savefig("02_multiview_ssl.png")

    except ImportError:
        print(
            "\nMatplotlib not found. Skipping plot. "
            "Install it with: pip install matplotlib"
        )


if __name__ == "__main__":
    main()
