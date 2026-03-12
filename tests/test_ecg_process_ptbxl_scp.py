from __future__ import annotations

import torch

from aiono import ECGMorphologyParams, ECGProcess, ECGRhythmParams
from aiono.ptbxl import PTBXL_EVENT_TYPE_NAMES, PTBXLLabelSetSampler, ptbxl_all_codes


def _make_process(*, seq_len: int, sample_rate_hz: float) -> ECGProcess:
    scp_codes = ptbxl_all_codes()
    sampler = PTBXLLabelSetSampler(scp_codes=scp_codes, normal_prob=0.2)
    return ECGProcess(
        seq_len=seq_len,
        sample_rate_hz=sample_rate_hz,
        scp_codes=scp_codes,
        scp_sampler=sampler,
        rhythm_params=ECGRhythmParams.ptbxl_defaults(),
        morphology_params=ECGMorphologyParams.ptbxl_defaults(),
    )


def test_ecg_process_ptbxl_scp_shapes_and_dtypes() -> None:
    device = torch.device("cpu")
    process = _make_process(seq_len=1000, sample_rate_hz=250.0)

    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    latent = process(batch_size=32, device=device, rng=generator)

    assert latent.events is not None
    events = latent.events

    assert events.times.ndim == 2
    assert events.type_ids.shape == events.times.shape
    assert events.mask.shape == events.times.shape
    assert events.params.ndim == 3
    assert events.params.shape[:2] == events.times.shape
    assert events.params.shape[2] == 1

    assert events.times.dtype == torch.float32
    assert events.type_ids.dtype == torch.int64
    assert events.mask.dtype == torch.bool
    assert events.params.dtype == torch.float32

    assert events.schema.time_unit == "samples"
    assert events.schema.param_names == ["amplitude"]
    assert events.schema.type_names == PTBXL_EVENT_TYPE_NAMES

    assert "scp" in latent.y
    assert latent.y["scp"].shape == (32, len(ptbxl_all_codes()))
    assert latent.y["scp"].dtype == torch.bool

    label_names = latent.meta["label_names"]["scp"]
    assert label_names == ptbxl_all_codes()

    label_groups = latent.meta["label_groups"]
    assert set(label_groups.keys()) == {"rhythm", "diagnostic", "form"}

    rhythm_mask = latent.y["scp"][:, label_groups["rhythm"]]  # [B, R]
    assert torch.all(rhythm_mask.sum(dim=1) == 1)


def test_ecg_process_ptbxl_scp_determinism() -> None:
    device = torch.device("cpu")
    process = _make_process(seq_len=1000, sample_rate_hz=250.0)

    g1 = torch.Generator(device=device)
    g1.manual_seed(999)
    g2 = torch.Generator(device=device)
    g2.manual_seed(999)

    out1 = process(batch_size=16, device=device, rng=g1)
    out2 = process(batch_size=16, device=device, rng=g2)

    torch.testing.assert_close(out1.y["scp"], out2.y["scp"])
    assert out1.events is not None and out2.events is not None
    torch.testing.assert_close(out1.events.times, out2.events.times)
    torch.testing.assert_close(out1.events.type_ids, out2.events.type_ids)
    torch.testing.assert_close(out1.events.params, out2.events.params)
    torch.testing.assert_close(out1.events.mask, out2.events.mask)
