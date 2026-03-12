from __future__ import annotations

import torch

from toyts import ECGMorphologyParams, ECGProcess, ECGRhythmParams
from toyts.core.samplers import Sampler
from toyts.ptbxl import ptbxl_all_codes


class FixedLabelSampler(Sampler):
    def __init__(self, *, scp_codes: list[str], rhythm_code: str, active_codes: list[str]) -> None:
        self.scp_codes = scp_codes
        self.rhythm_code = rhythm_code
        self.active_codes = active_codes
        self.code_to_index = {code: idx for idx, code in enumerate(self.scp_codes)}
        if rhythm_code not in self.code_to_index:
            raise ValueError(f"Unknown rhythm_code '{rhythm_code}'.")
        for code in active_codes:
            if code not in self.code_to_index:
                raise ValueError(f"Unknown active code '{code}'.")

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is not torch.bool:
            raise ValueError("FixedLabelSampler requires dtype=torch.bool.")
        if len(shape) != 2:
            raise ValueError(f"FixedLabelSampler expects shape [B, S], got {shape}.")
        batch_size, num_codes = shape
        if num_codes != len(self.scp_codes):
            raise ValueError("FixedLabelSampler shape does not match scp_codes.")
        labels = torch.zeros((batch_size, num_codes), device=device, dtype=torch.bool)  # [B, S]
        labels[:, self.code_to_index[self.rhythm_code]] = True
        for code in self.active_codes:
            labels[:, self.code_to_index[code]] = True
        return labels

    def spec(self) -> dict[str, object]:
        return {"kind": "fixed", "rhythm": self.rhythm_code, "active": list(self.active_codes)}


def _make_fixed_process(*, active_codes: list[str]) -> ECGProcess:
    scp_codes = ptbxl_all_codes()
    sampler = FixedLabelSampler(scp_codes=scp_codes, rhythm_code="SR", active_codes=active_codes)
    return ECGProcess(
        seq_len=1000,
        sample_rate_hz=250.0,
        scp_codes=scp_codes,
        scp_sampler=sampler,
        rhythm_params=ECGRhythmParams.ptbxl_defaults(),
        morphology_params=ECGMorphologyParams.ptbxl_defaults(),
    )


def test_pr_interval_increases_for_lpr() -> None:
    device = torch.device("cpu")
    base = _make_fixed_process(active_codes=["NORM"])
    lpr = _make_fixed_process(active_codes=["LPR"])

    rng = torch.Generator(device=device)
    rng.manual_seed(1234)
    base_out = base(batch_size=256, device=device, rng=rng)

    rng.manual_seed(1234)
    lpr_out = lpr(batch_size=256, device=device, rng=rng)

    base_pr = base_out.meta["samples"]["ECGComponentEventsNode"]["pr_ms"]  # [B]
    lpr_pr = lpr_out.meta["samples"]["ECGComponentEventsNode"]["pr_ms"]  # [B]
    assert lpr_pr.mean().item() > base_pr.mean().item()


def test_qt_interval_increases_for_lngqt() -> None:
    device = torch.device("cpu")
    base = _make_fixed_process(active_codes=["NORM"])
    lngqt = _make_fixed_process(active_codes=["LNGQT"])

    rng = torch.Generator(device=device)
    rng.manual_seed(2222)
    base_out = base(batch_size=256, device=device, rng=rng)

    rng.manual_seed(2222)
    lngqt_out = lngqt(batch_size=256, device=device, rng=rng)

    base_qt = base_out.meta["samples"]["ECGComponentEventsNode"]["qt_ms"]  # [B]
    lngqt_qt = lngqt_out.meta["samples"]["ECGComponentEventsNode"]["qt_ms"]  # [B]
    assert lngqt_qt.mean().item() > base_qt.mean().item()


def test_st_shift_sign_for_std_and_ste() -> None:
    device = torch.device("cpu")
    std = _make_fixed_process(active_codes=["STD_"])
    ste = _make_fixed_process(active_codes=["STE_"])

    rng = torch.Generator(device=device)
    rng.manual_seed(3333)
    std_out = std(batch_size=128, device=device, rng=rng)

    rng.manual_seed(3333)
    ste_out = ste(batch_size=128, device=device, rng=rng)

    std_events = std_out.events
    ste_events = ste_out.events
    assert std_events is not None and ste_events is not None

    std_type = std_events.schema.type_id("st_shift_global")
    ste_type = ste_events.schema.type_id("st_shift_global")

    std_mask = std_events.mask & (std_events.type_ids == std_type)  # [B, E]
    ste_mask = ste_events.mask & (ste_events.type_ids == ste_type)  # [B, E]

    std_amp = std_events.params[:, :, 0][std_mask]  # [N]
    ste_amp = ste_events.params[:, :, 0][ste_mask]  # [N]

    assert std_amp.mean().item() < 0.0
    assert ste_amp.mean().item() > 0.0


def test_qwave_negative_amplitude() -> None:
    device = torch.device("cpu")
    qwave = _make_fixed_process(active_codes=["QWAVE"])

    rng = torch.Generator(device=device)
    rng.manual_seed(4444)
    out = qwave(batch_size=128, device=device, rng=rng)

    events = out.events
    assert events is not None
    qwave_type = events.schema.type_id("qrs_qwave_global")
    qwave_mask = events.mask & (events.type_ids == qwave_type)  # [B, E]
    qwave_amp = events.params[:, :, 0][qwave_mask]  # [N]
    assert qwave_amp.mean().item() < 0.0
