from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from toyts import (
    ECGMorphologyParams,
    ECGProcess,
    ECGRhythmParams,
    EventImpulseView,
    GaussianNoiseView,
    KernelConvView,
    NormalizeView,
    SynthPipeline,
    make_ptbxl_kernel_bank,
    ptbxl_kernel_size,
)
from toyts.ptbxl import PTBXLLabelSetSampler, ptbxl_all_codes, ptbxl_codes_by_group


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scp_codes = ptbxl_all_codes()
    rhythm_codes = ptbxl_codes_by_group("rhythm")
    sampler = PTBXLLabelSetSampler(scp_codes=scp_codes, normal_prob=0.0)
    process = ECGProcess(
        seq_len=2048,
        sample_rate_hz=500.0,
        scp_codes=scp_codes,
        scp_sampler=sampler,
        rhythm_params=ECGRhythmParams.ptbxl_defaults(),
        morphology_params=ECGMorphologyParams.ptbxl_defaults(),
    )

    kernel_size = ptbxl_kernel_size(sample_rate_hz=process.sample_rate_hz, support_ms=400.0)
    kernels = make_ptbxl_kernel_bank(
        sample_rate_hz=process.sample_rate_hz,
        kernel_size=kernel_size,
        device=device,
    )  # [K=12, T, W]
    padding = kernel_size // 2

    views = {
        "clean": torch.nn.Sequential(
            EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
            KernelConvView(kernels=kernels, padding=padding),
        ),
        "noisy": torch.nn.Sequential(
            EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
            KernelConvView(kernels=kernels, padding=padding),
            GaussianNoiseView(noise_std=0.15),
            NormalizeView(),
        ),
    }

    rng = torch.Generator(device=device)
    rng.manual_seed(2026)
    pipeline = SynthPipeline(process=process, views=views).to(device)
    batch = pipeline(batch_size=512, device=device, rng=rng)

    x = batch["clean"].x.detach().cpu()  # [B, C, L]
    y = batch["clean"].y["scp"].detach().cpu()  # [B, S]
    group_indices = batch["clean"].meta["process"]["label_groups"]["rhythm"]
    rhythm_names = rhythm_codes
    rhythm_mask = y[:, group_indices]  # [B, R]

    lead_idx = 1  # lead II-ish
    num_rows, num_cols = 3, 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle("ECGProcess: PTB-XL rhythm labels (lead II)")

    for class_idx, name in enumerate(rhythm_names):
        matches = torch.nonzero(rhythm_mask[:, class_idx], as_tuple=False).flatten()
        if matches.numel() == 0:
            raise ValueError(f"No samples for rhythm '{name}'.")
        sample_idx = int(matches[0].item())

        ax = axes[class_idx // num_cols][class_idx % num_cols]
        ax.plot(x[sample_idx, lead_idx].tolist(), linewidth=0.8)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ptbxl_rhythm_12_lead2.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
