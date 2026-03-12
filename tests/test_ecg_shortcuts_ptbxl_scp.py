from __future__ import annotations

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


class CycleLabelSampler(Sampler):
    def __init__(self, *, scp_codes: list[str], rhythm_codes: list[str]) -> None:
        self.scp_codes = scp_codes
        self.rhythm_codes = rhythm_codes
        self.code_to_index = {code: idx for idx, code in enumerate(self.scp_codes)}
        self.rhythm_indices = [self.code_to_index[code] for code in rhythm_codes]
        self.nonrhythm_indices = [
            idx for idx, code in enumerate(self.scp_codes) if code not in set(rhythm_codes)
        ]
        if not self.rhythm_indices or not self.nonrhythm_indices:
            raise ValueError("CycleLabelSampler requires rhythm and non-rhythm codes.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is not torch.bool:
            raise ValueError("CycleLabelSampler requires dtype=torch.bool.")
        if len(shape) != 2:
            raise ValueError(f"CycleLabelSampler expects shape [B, S], got {shape}.")
        batch_size, num_codes = shape
        if num_codes != len(self.scp_codes):
            raise ValueError("CycleLabelSampler shape does not match scp_codes.")

        labels = torch.zeros((batch_size, num_codes), device=device, dtype=torch.bool)  # [B, S]
        batch_idx = torch.arange(batch_size, device=device)  # [B]

        rhythm_choice = batch_idx % len(self.rhythm_indices)  # [B]
        rhythm_indices = torch.tensor(self.rhythm_indices, device=device, dtype=torch.int64)  # [R]
        labels[batch_idx, rhythm_indices[rhythm_choice]] = True

        nonrhythm_choice = batch_idx % len(self.nonrhythm_indices)  # [B]
        nonrhythm_indices = torch.tensor(self.nonrhythm_indices, device=device, dtype=torch.int64)  # [NR]
        labels[batch_idx, nonrhythm_indices[nonrhythm_choice]] = True
        return labels

    def spec(self) -> dict[str, object]:
        return {"kind": "cycle", "codes": len(self.scp_codes)}


def test_shortcut_baseline_on_ptbxl_scp() -> None:
    device = torch.device("cpu")

    scp_codes = ptbxl_all_codes()
    rhythm_codes = ptbxl_codes_by_group("rhythm")
    sampler = CycleLabelSampler(scp_codes=scp_codes, rhythm_codes=rhythm_codes)

    process = ECGProcess(
        seq_len=1000,
        sample_rate_hz=250.0,
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

    pipeline = SynthPipeline(
        process=process,
        views={
            "clean": torch.nn.Sequential(
                EventImpulseView(
                    seq_len=process.seq_len,
                    amplitude_param="amplitude",
                    rounding="nearest",
                ),
                KernelConvView(kernels=kernels, padding=padding),
            )
        },
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(2025)
    batch = pipeline(batch_size=1180, device=device, rng=generator)

    x = batch["clean"].x  # [B, C, L]
    y = batch["clean"].y["scp"]  # [B, S]

    mean = x.mean(dim=(1, 2))  # [B]
    std = x.std(dim=(1, 2), unbiased=False)  # [B]
    energy = x.pow(2).mean(dim=(1, 2))  # [B]
    threshold = x.abs().mean(dim=(1, 2), keepdim=True) + x.abs().std(dim=(1, 2), keepdim=True)  # [B, 1, 1]
    peak_count = (x.abs() > threshold).sum(dim=(1, 2))  # [B]

    features = torch.stack([mean, std, energy, peak_count], dim=1)  # [B, F]

    diagnostic = batch["clean"].meta["process"]["label_groups"]["diagnostic"]  # [D]
    form = batch["clean"].meta["process"]["label_groups"]["form"]  # [F]
    nonrhythm_indices = sorted(set(diagnostic + form))

    y_nonrhythm = y[:, nonrhythm_indices]  # [B, L]

    train_size = features.shape[0] // 2
    feat_train = features[:train_size]  # [B_train, F]
    feat_test = features[train_size:]  # [B_test, F]
    y_train = y_nonrhythm[:train_size]  # [B_train, L]
    y_test = y_nonrhythm[train_size:]  # [B_test, L]

    balanced_acc = []
    for label_idx in range(y_train.shape[1]):
        pos_mask = y_train[:, label_idx]  # [B_train]
        neg_mask = ~pos_mask  # [B_train]
        if torch.count_nonzero(pos_mask) == 0 or torch.count_nonzero(neg_mask) == 0:
            raise ValueError("Shortcut test requires positives and negatives for every label.")

        pos_center = feat_train[pos_mask].mean(dim=0)  # [F]
        neg_center = feat_train[neg_mask].mean(dim=0)  # [F]

        dist_pos = (feat_test - pos_center).pow(2).sum(dim=1)  # [B_test]
        dist_neg = (feat_test - neg_center).pow(2).sum(dim=1)  # [B_test]
        preds = dist_pos < dist_neg  # [B_test]
        y_true = y_test[:, label_idx]  # [B_test]

        tp = (preds & y_true).sum().float()
        tn = (~preds & ~y_true).sum().float()
        pos = y_true.sum().float()
        neg = (~y_true).sum().float()
        tpr = tp / pos
        tnr = tn / neg
        balanced_acc.append(0.5 * (tpr + tnr))

    balanced_acc = torch.stack(balanced_acc).mean().item()
    assert balanced_acc < 0.75

    perm_gen = torch.Generator(device=device)
    perm_gen.manual_seed(1234)
    perm = torch.randperm(train_size, generator=perm_gen, device=device)  # [B_train]
    y_train_shuffled = y_train[perm]  # [B_train, L]

    balanced_acc_shuffled = []
    for label_idx in range(y_train.shape[1]):
        pos_mask = y_train_shuffled[:, label_idx]  # [B_train]
        neg_mask = ~pos_mask  # [B_train]
        if torch.count_nonzero(pos_mask) == 0 or torch.count_nonzero(neg_mask) == 0:
            raise ValueError("Shortcut test requires positives and negatives for every label.")

        pos_center = feat_train[pos_mask].mean(dim=0)  # [F]
        neg_center = feat_train[neg_mask].mean(dim=0)  # [F]

        dist_pos = (feat_test - pos_center).pow(2).sum(dim=1)  # [B_test]
        dist_neg = (feat_test - neg_center).pow(2).sum(dim=1)  # [B_test]
        preds = dist_pos < dist_neg  # [B_test]
        y_true = y_test[:, label_idx]  # [B_test]

        tp = (preds & y_true).sum().float()
        tn = (~preds & ~y_true).sum().float()
        pos = y_true.sum().float()
        neg = (~y_true).sum().float()
        tpr = tp / pos
        tnr = tn / neg
        balanced_acc_shuffled.append(0.5 * (tpr + tnr))

    balanced_acc_shuffled = torch.stack(balanced_acc_shuffled).mean().item()
    assert balanced_acc_shuffled < 0.6
