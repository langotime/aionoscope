from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from toyts import (
    ECGMorphologyParams,
    ECGProcess,
    ECGRhythmParams,
    EventImpulseView,
    KernelConvView,
    SynthPipeline,
    make_ptbxl_kernel_bank,
    ptbxl_kernel_size,
)
from toyts.core.samplers import Sampler
from toyts.ptbxl import ptbxl_all_codes, ptbxl_codes_by_group


class GroupCycleSampler(Sampler):
    def __init__(
        self,
        *,
        scp_codes: list[str],
        target_codes: list[str],
        rhythm_codes: list[str],
    ) -> None:
        self.scp_codes = scp_codes
        self.target_codes = target_codes
        self.rhythm_codes = rhythm_codes
        self.code_to_index = {code: idx for idx, code in enumerate(self.scp_codes)}
        self.target_indices = [self.code_to_index[code] for code in target_codes]
        self.rhythm_indices = [self.code_to_index[code] for code in rhythm_codes]
        if not self.target_indices:
            raise ValueError("GroupCycleSampler requires target_codes.")
        if not self.rhythm_indices:
            raise ValueError("GroupCycleSampler requires rhythm_codes.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is not torch.bool:
            raise ValueError("GroupCycleSampler requires dtype=torch.bool.")
        if len(shape) != 2:
            raise ValueError(f"GroupCycleSampler expects shape [B, S], got {shape}.")
        batch_size, num_codes = shape
        if num_codes != len(self.scp_codes):
            raise ValueError("GroupCycleSampler shape does not match scp_codes.")

        labels = torch.zeros((batch_size, num_codes), device=device, dtype=torch.bool)  # [B, S]
        batch_idx = torch.arange(batch_size, device=device)  # [B]

        target_idx = batch_idx % len(self.target_indices)  # [B]
        target_indices = torch.tensor(self.target_indices, device=device, dtype=torch.int64)  # [T]
        labels[batch_idx, target_indices[target_idx]] = True

        rhythm_idx = batch_idx % len(self.rhythm_indices)  # [B]
        rhythm_indices = torch.tensor(self.rhythm_indices, device=device, dtype=torch.int64)  # [R]
        labels[batch_idx, rhythm_indices[rhythm_idx]] = True

        return labels

    def spec(self) -> dict[str, object]:
        return {"kind": "cycle", "targets": list(self.target_codes)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scp_codes = ptbxl_all_codes()
    rhythm_codes = ptbxl_codes_by_group("rhythm")
    form_codes = ptbxl_codes_by_group("form")
    sampler = GroupCycleSampler(
        scp_codes=scp_codes,
        target_codes=form_codes,
        rhythm_codes=rhythm_codes,
    )

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
        )
    }

    rng = torch.Generator(device=device)
    rng.manual_seed(2027)
    pipeline = SynthPipeline(process=process, views=views).to(device)
    batch = pipeline(batch_size=len(form_codes), device=device, rng=rng)

    x = batch["clean"].x.detach().cpu()  # [B, C, L]
    y = batch["clean"].y["scp"].detach().cpu()  # [B, S]
    form_indices = batch["clean"].meta["process"]["label_groups"]["form"]

    lead_indices = [1, 7, 10]  # II, V2, V5
    num_rows, num_cols = 4, 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle("ECGProcess: PTB-XL form codes (lead II + V2 + V5)")

    for code_idx, code in enumerate(form_codes):
        col = code_idx % num_cols
        row = code_idx // num_cols
        ax = axes[row][col]

        matches = torch.nonzero(y[:, form_indices[code_idx]], as_tuple=False).flatten()
        if matches.numel() == 0:
            raise ValueError(f"No samples for form code '{code}'.")
        sample_idx = int(matches[0].item())

        for lead_idx in lead_indices:
            ax.plot(x[sample_idx, lead_idx].tolist(), linewidth=0.8)
        ax.set_title(code)
        ax.grid(True, alpha=0.3)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ptbxl_form_19.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
