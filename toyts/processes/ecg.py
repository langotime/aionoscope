from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ..core.events import EventBatch, EventSchema
from ..core.samplers import Sampler
from ..core.types import LatentState
from ..core.utils import SAMPLES_PREFIX
from ..ptbxl import ptbxl_all_codes, ptbxl_codes_by_group, ptbxl_group_indices, ptbxl_effect_groups
from ..ptbxl.phenotypes import (
    NORM_CODES,
    PAC_CODES,
    PR_LONG_CODES,
    PR_SHORT_CODES,
    PRC_CODES,
    PVC_CODES,
    PTBXL_EVENT_TYPE_NAMES,
    QRS_VOLT_HIGH_CODES,
    QRS_VOLT_LOW_CODES,
    QT_LONG_CODES,
    T_LOW_CODES,
    T_MILD_CODES,
)
from ..ptbxl.samplers import PTBXLLabelSetSampler
from .graph import ProcessGraph, ProcessNode, ProcessState
from .nodes import SampleMultiLabelNode, UnionEventsNode


def _validate_range(name: str, value: tuple[float, float]) -> None:
    low, high = value
    if high <= low:
        raise ValueError(f"{name} must have high > low, got {value}.")


def _validate_int_range(name: str, value: tuple[int, int]) -> None:
    low, high = value
    if high <= low:
        raise ValueError(f"{name} must have high > low, got {value}.")


@dataclass(frozen=True)
class ECGRhythmParams:
    """Parameter ranges controlling rhythm timing patterns."""

    hr_bpm_ranges: dict[str, tuple[float, float]]
    psvt_svt_hr_bpm_range: tuple[float, float]
    flutter_rate_hz_range: tuple[float, float]
    regular_rr_jitter_std_range: tuple[float, float]
    sarrh_mod_amp_range: tuple[float, float]
    sarrh_cycles_per_window_range: tuple[float, float]
    sv_premature_prob_range: tuple[float, float]
    sv_premature_short_range: tuple[float, float]
    sv_premature_long_range: tuple[float, float]
    bigu_short_range: tuple[float, float]
    bigu_long_range: tuple[float, float]
    trigu_short_range: tuple[float, float]
    trigu_long_range: tuple[float, float]
    psvt_episode_start_frac_range: tuple[float, float]
    psvt_episode_duration_frac_range: tuple[float, float]
    pace_spike_delay_samples_range: tuple[int, int]
    pace_spike_amplitude_scale_range: tuple[float, float]
    flutter_wave_amplitude_scale_range: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.hr_bpm_ranges:
            raise ValueError("hr_bpm_ranges must be non-empty.")
        for code, rng in self.hr_bpm_ranges.items():
            if not code:
                raise ValueError("hr_bpm_ranges keys must be non-empty.")
            _validate_range(f"hr_bpm_ranges['{code}']", rng)

        _validate_range("psvt_svt_hr_bpm_range", self.psvt_svt_hr_bpm_range)
        _validate_range("flutter_rate_hz_range", self.flutter_rate_hz_range)
        _validate_range("regular_rr_jitter_std_range", self.regular_rr_jitter_std_range)
        _validate_range("sarrh_mod_amp_range", self.sarrh_mod_amp_range)
        _validate_range("sarrh_cycles_per_window_range", self.sarrh_cycles_per_window_range)
        _validate_range("sv_premature_prob_range", self.sv_premature_prob_range)
        _validate_range("sv_premature_short_range", self.sv_premature_short_range)
        _validate_range("sv_premature_long_range", self.sv_premature_long_range)
        _validate_range("bigu_short_range", self.bigu_short_range)
        _validate_range("bigu_long_range", self.bigu_long_range)
        _validate_range("trigu_short_range", self.trigu_short_range)
        _validate_range("trigu_long_range", self.trigu_long_range)
        _validate_range("psvt_episode_start_frac_range", self.psvt_episode_start_frac_range)
        _validate_range("psvt_episode_duration_frac_range", self.psvt_episode_duration_frac_range)
        _validate_int_range("pace_spike_delay_samples_range", self.pace_spike_delay_samples_range)
        _validate_range("pace_spike_amplitude_scale_range", self.pace_spike_amplitude_scale_range)
        _validate_range("flutter_wave_amplitude_scale_range", self.flutter_wave_amplitude_scale_range)

    def to_meta(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def ptbxl_defaults(cls) -> ECGRhythmParams:
        hr_bpm_ranges: dict[str, tuple[float, float]] = {
            "SR": (55.0, 90.0),
            "AFIB": (60.0, 140.0),
            "STACH": (110.0, 160.0),
            "SARRH": (55.0, 85.0),
            "SBRAD": (35.0, 55.0),
            "PACE": (60.0, 95.0),
            "SVARR": (60.0, 120.0),
            "BIGU": (60.0, 110.0),
            "AFLT": (120.0, 170.0),
            "SVTAC": (140.0, 210.0),
            "PSVT": (60.0, 95.0),
            "TRIGU": (60.0, 110.0),
        }

        return cls(
            hr_bpm_ranges=hr_bpm_ranges,
            psvt_svt_hr_bpm_range=(160.0, 230.0),
            flutter_rate_hz_range=(4.0, 6.5),
            regular_rr_jitter_std_range=(0.01, 0.04),
            sarrh_mod_amp_range=(0.05, 0.2),
            sarrh_cycles_per_window_range=(0.7, 1.5),
            sv_premature_prob_range=(0.06, 0.18),
            sv_premature_short_range=(0.6, 0.8),
            sv_premature_long_range=(1.2, 1.5),
            bigu_short_range=(0.65, 0.8),
            bigu_long_range=(1.2, 1.45),
            trigu_short_range=(0.6, 0.75),
            trigu_long_range=(1.2, 1.45),
            psvt_episode_start_frac_range=(0.2, 0.6),
            psvt_episode_duration_frac_range=(0.2, 0.5),
            pace_spike_delay_samples_range=(1, 4),
            pace_spike_amplitude_scale_range=(0.2, 0.5),
            flutter_wave_amplitude_scale_range=(0.15, 0.35),
        )


@dataclass(frozen=True)
class ECGMorphologyParams:
    """Parameter ranges controlling morphology and conduction effects."""

    p_amp_range: tuple[float, float]
    qrs_amp_range: tuple[float, float]
    t_amp_range: tuple[float, float]
    pr_ms_range: tuple[float, float]
    pr_long_ms_range: tuple[float, float]
    pr_short_ms_range: tuple[float, float]
    qt_ms_range: tuple[float, float]
    qt_long_ms_range: tuple[float, float]
    qrs_voltage_low_scale_range: tuple[float, float]
    qrs_voltage_high_scale_range: tuple[float, float]
    t_low_scale_range: tuple[float, float]
    t_mild_scale_range: tuple[float, float]
    av_block2_drop_prob_range: tuple[float, float]
    av_block3_atrial_hr_bpm_range: tuple[float, float]
    av_block3_rr_jitter_std_range: tuple[float, float]
    ectopy_count_range: tuple[int, int]
    ectopy_qrs_scale_range: tuple[float, float]
    ectopy_p_scale_range: tuple[float, float]
    ectopy_time_margin_frac_range: tuple[float, float]

    def __post_init__(self) -> None:
        _validate_range("p_amp_range", self.p_amp_range)
        _validate_range("qrs_amp_range", self.qrs_amp_range)
        _validate_range("t_amp_range", self.t_amp_range)
        _validate_range("pr_ms_range", self.pr_ms_range)
        _validate_range("pr_long_ms_range", self.pr_long_ms_range)
        _validate_range("pr_short_ms_range", self.pr_short_ms_range)
        _validate_range("qt_ms_range", self.qt_ms_range)
        _validate_range("qt_long_ms_range", self.qt_long_ms_range)
        _validate_range("qrs_voltage_low_scale_range", self.qrs_voltage_low_scale_range)
        _validate_range("qrs_voltage_high_scale_range", self.qrs_voltage_high_scale_range)
        _validate_range("t_low_scale_range", self.t_low_scale_range)
        _validate_range("t_mild_scale_range", self.t_mild_scale_range)
        _validate_range("av_block2_drop_prob_range", self.av_block2_drop_prob_range)
        _validate_range("av_block3_atrial_hr_bpm_range", self.av_block3_atrial_hr_bpm_range)
        _validate_range("av_block3_rr_jitter_std_range", self.av_block3_rr_jitter_std_range)
        _validate_int_range("ectopy_count_range", self.ectopy_count_range)
        _validate_range("ectopy_qrs_scale_range", self.ectopy_qrs_scale_range)
        _validate_range("ectopy_p_scale_range", self.ectopy_p_scale_range)
        _validate_range("ectopy_time_margin_frac_range", self.ectopy_time_margin_frac_range)

    def to_meta(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def ptbxl_defaults(cls) -> ECGMorphologyParams:
        return cls(
            p_amp_range=(0.05, 0.2),
            qrs_amp_range=(0.7, 1.5),
            t_amp_range=(0.1, 0.4),
            pr_ms_range=(120.0, 200.0),
            pr_long_ms_range=(210.0, 320.0),
            pr_short_ms_range=(80.0, 120.0),
            qt_ms_range=(340.0, 440.0),
            qt_long_ms_range=(460.0, 560.0),
            qrs_voltage_low_scale_range=(0.45, 0.7),
            qrs_voltage_high_scale_range=(1.3, 1.9),
            t_low_scale_range=(0.3, 0.6),
            t_mild_scale_range=(0.6, 0.9),
            av_block2_drop_prob_range=(0.15, 0.35),
            av_block3_atrial_hr_bpm_range=(65.0, 110.0),
            av_block3_rr_jitter_std_range=(0.01, 0.05),
            ectopy_count_range=(1, 3),
            ectopy_qrs_scale_range=(0.6, 1.2),
            ectopy_p_scale_range=(0.6, 1.2),
            ectopy_time_margin_frac_range=(0.02, 0.08),
        )


def _sample_uniform(
    *,
    low: float,
    high: float,
    shape: tuple[int, ...],
    rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if high <= low:
        raise ValueError(f"Uniform range requires high > low, got low={low}, high={high}.")
    u = torch.rand(shape, generator=rng, device=device, dtype=dtype)  # [*shape]
    return low + (high - low) * u


def _sample_uniform_int(
    *,
    low: int,
    high: int,
    shape: tuple[int, ...],
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if high <= low:
        raise ValueError(f"Uniform int range requires high > low, got low={low}, high={high}.")
    return torch.randint(low, high + 1, shape, generator=rng, device=device, dtype=torch.int64)  # [*shape]


class _ECGRhythmTimingNode(ProcessNode):
    """Generate QRS times plus pacing/flutter event streams."""

    def __init__(
        self,
        *,
        seq_len: int,
        sample_rate_hz: float,
        rhythm_codes: list[str],
        rhythm_indices: list[int],
        rhythm_params: ECGRhythmParams,
        schema: EventSchema,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        if not rhythm_codes:
            raise ValueError("rhythm_codes must be non-empty.")
        if len(set(rhythm_codes)) != len(rhythm_codes):
            raise ValueError("rhythm_codes must be unique.")
        if not rhythm_indices:
            raise ValueError("rhythm_indices must be non-empty.")

        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.duration_sec = (seq_len - 1) / sample_rate_hz
        self.rhythm_codes = list(rhythm_codes)
        self.rhythm_indices = list(rhythm_indices)
        self.rhythm_params = rhythm_params
        self.schema = schema

        self.code_to_id = {code: idx for idx, code in enumerate(self.rhythm_codes)}

        hr_max = max(rhythm_params.hr_bpm_ranges[code][1] for code in self.rhythm_codes)
        if "PSVT" in self.code_to_id:
            hr_max = max(hr_max, rhythm_params.psvt_svt_hr_bpm_range[1])

        self.max_beats = int(math.ceil(self.duration_sec * hr_max / 60.0)) + 2
        if self.max_beats <= 0:
            raise ValueError("ECGProcess max_beats computed <= 0; check seq_len/sample_rate_hz.")

        flutter_rate_max = rhythm_params.flutter_rate_hz_range[1]
        self.max_flutter_waves = int(math.ceil(self.duration_sec * flutter_rate_max)) + 2
        if self.max_flutter_waves <= 0:
            raise ValueError(
                "ECGProcess max_flutter_waves computed <= 0; check flutter_rate_hz_range."
            )

    def _mask_for(self, label: torch.Tensor, code: str) -> torch.Tensor:
        if code not in self.code_to_id:
            return torch.zeros_like(label, dtype=torch.bool)  # [B]
        return label == self.code_to_id[code]  # [B]

    def _make_beats(
        self,
        *,
        hr_bpm: torch.Tensor,
        rr_multipliers: torch.Tensor,
        rng: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_beats = rr_multipliers.shape
        rr_base = (self.sample_rate_hz * 60.0) / hr_bpm  # [B]
        rr_samples = rr_multipliers * rr_base[:, None]  # [B, N]

        phase = torch.rand((batch_size, 1), generator=rng, device=device) * rr_base[:, None]  # [B, 1]
        intervals = torch.cat([phase, rr_samples], dim=1)  # [B, N+1]
        times = intervals.cumsum(dim=1)[:, :-1]  # [B, N]

        times_idx_unclamped = torch.round(times).to(torch.int64)  # [B, N]
        mask = (times_idx_unclamped >= 0) & (times_idx_unclamped < self.seq_len)  # [B, N]
        times_idx = times_idx_unclamped.clamp(0, self.seq_len - 1)  # [B, N]
        times_samples = times_idx.to(torch.float32)  # [B, N]
        return times_samples, mask

    def _sort_times(
        self,
        *,
        times: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sort_times = times.masked_fill(~mask, float("inf"))  # [B, E]
        indices = sort_times.argsort(dim=1)  # [B, E]
        times_sorted = torch.gather(times, 1, indices)  # [B, E]
        mask_sorted = torch.gather(mask, 1, indices)  # [B, E]
        return times_sorted, mask_sorted

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        self._record_seed(state, rng)
        if "scp" not in state.y:
            raise ValueError("ECGProcess requires label 'scp' in state.y.")

        labels = state.y["scp"]  # [B, S]
        batch_size = state.batch_size
        device = state.device

        rhythm_mask = labels[:, self.rhythm_indices]  # [B, R]
        rhythm_count = rhythm_mask.sum(dim=1)  # [B]
        if not torch.all(rhythm_count == 1):
            raise ValueError("ECGProcess requires exactly one rhythm code per sample.")

        rhythm_id = rhythm_mask.to(torch.int64).argmax(dim=1)  # [B]

        hr_min = torch.tensor(
            [self.rhythm_params.hr_bpm_ranges[code][0] for code in self.rhythm_codes],
            device=device,
            dtype=torch.float32,
        )  # [R]
        hr_max = torch.tensor(
            [self.rhythm_params.hr_bpm_ranges[code][1] for code in self.rhythm_codes],
            device=device,
            dtype=torch.float32,
        )  # [R]

        u_hr = torch.rand((batch_size,), generator=rng, device=device)  # [B]
        hr_bpm = hr_min[rhythm_id] + u_hr * (hr_max[rhythm_id] - hr_min[rhythm_id])  # [B]

        mask_psvt = self._mask_for(rhythm_id, "PSVT")  # [B]
        hr_svt = _sample_uniform(
            low=self.rhythm_params.psvt_svt_hr_bpm_range[0],
            high=self.rhythm_params.psvt_svt_hr_bpm_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        hr_svt = torch.where(mask_psvt, hr_svt, hr_bpm)  # [B]

        start_frac = _sample_uniform(
            low=self.rhythm_params.psvt_episode_start_frac_range[0],
            high=self.rhythm_params.psvt_episode_start_frac_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        dur_frac = _sample_uniform(
            low=self.rhythm_params.psvt_episode_duration_frac_range[0],
            high=self.rhythm_params.psvt_episode_duration_frac_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        end_frac = (start_frac + dur_frac).clamp_max(1.0)  # [B]

        episode_start = start_frac * (self.seq_len - 1)  # [B]
        episode_end = end_frac * (self.seq_len - 1)  # [B]
        episode_start = torch.where(mask_psvt, episode_start, torch.zeros_like(episode_start))  # [B]
        episode_end = torch.where(mask_psvt, episode_end, torch.zeros_like(episode_end))  # [B]

        mask_afib = self._mask_for(rhythm_id, "AFIB")  # [B]
        mask_sarrh = self._mask_for(rhythm_id, "SARRH")  # [B]
        mask_svarr = self._mask_for(rhythm_id, "SVARR")  # [B]
        mask_bigu = self._mask_for(rhythm_id, "BIGU")  # [B]
        mask_trigu = self._mask_for(rhythm_id, "TRIGU")  # [B]
        mask_pace = self._mask_for(rhythm_id, "PACE")  # [B]
        mask_aflt = self._mask_for(rhythm_id, "AFLT")  # [B]

        jitter_std = _sample_uniform(
            low=self.rhythm_params.regular_rr_jitter_std_range[0],
            high=self.rhythm_params.regular_rr_jitter_std_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        jitter = torch.randn(
            (batch_size, self.max_beats),
            generator=rng,
            device=device,
            dtype=torch.float32,
        )  # [B, N]
        rr_multipliers = 1.0 + jitter * jitter_std[:, None]  # [B, N]
        rr_multipliers = torch.clamp(rr_multipliers, min=0.1)  # [B, N]

        if torch.any(mask_sarrh):
            phase = torch.rand((batch_size, 1), generator=rng, device=device) * (2.0 * math.pi)  # [B, 1]
            amp = _sample_uniform(
                low=self.rhythm_params.sarrh_mod_amp_range[0],
                high=self.rhythm_params.sarrh_mod_amp_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            cycles = _sample_uniform(
                low=self.rhythm_params.sarrh_cycles_per_window_range[0],
                high=self.rhythm_params.sarrh_cycles_per_window_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            grid = torch.linspace(0.0, 2.0 * math.pi, steps=self.max_beats, device=device)  # [N]
            sarrh = 1.0 + amp * torch.sin(phase + grid[None, :] * cycles)  # [B, N]
            sarrh = torch.clamp(sarrh, min=0.1)  # [B, N]
            rr_multipliers = torch.where(mask_sarrh[:, None], sarrh, rr_multipliers)  # [B, N]

        if torch.any(mask_afib):
            u = torch.rand((batch_size, self.max_beats), generator=rng, device=device)  # [B, N]
            exp_intervals = -torch.log(u.clamp_min(1e-6))  # [B, N]
            exp_intervals = exp_intervals / exp_intervals.mean(dim=1, keepdim=True)  # [B, N]
            exp_intervals = torch.clamp(exp_intervals, min=0.1)  # [B, N]
            rr_multipliers = torch.where(mask_afib[:, None], exp_intervals, rr_multipliers)  # [B, N]

        if torch.any(mask_svarr):
            prob = _sample_uniform(
                low=self.rhythm_params.sv_premature_prob_range[0],
                high=self.rhythm_params.sv_premature_prob_range[1],
                shape=(batch_size,),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B]
            idx = torch.arange(self.max_beats, device=device)  # [N]
            non_last = (idx < (self.max_beats - 1))[None, :]  # [1, N]
            non_first = (idx > 0)[None, :]  # [1, N]

            u = torch.rand((batch_size, self.max_beats), generator=rng, device=device)  # [B, N]
            premature = (u < prob[:, None]) & non_last  # [B, N]
            long_after = torch.roll(premature, shifts=1, dims=1) & non_first  # [B, N]

            short_value = _sample_uniform(
                low=self.rhythm_params.sv_premature_short_range[0],
                high=self.rhythm_params.sv_premature_short_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            long_value = _sample_uniform(
                low=self.rhythm_params.sv_premature_long_range[0],
                high=self.rhythm_params.sv_premature_long_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            svarr = torch.ones((batch_size, self.max_beats), device=device, dtype=torch.float32)  # [B, N]
            svarr = torch.where(premature, short_value, svarr)  # [B, N]
            svarr = torch.where(long_after, long_value, svarr)  # [B, N]
            rr_multipliers = torch.where(mask_svarr[:, None], svarr, rr_multipliers)  # [B, N]

        if torch.any(mask_bigu):
            idx = torch.arange(self.max_beats, device=device)  # [N]
            short = _sample_uniform(
                low=self.rhythm_params.bigu_short_range[0],
                high=self.rhythm_params.bigu_short_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            long = _sample_uniform(
                low=self.rhythm_params.bigu_long_range[0],
                high=self.rhythm_params.bigu_long_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            pattern = torch.where((idx % 2) == 0, short, long).to(torch.float32)  # [B, N]
            rr_multipliers = torch.where(mask_bigu[:, None], pattern, rr_multipliers)  # [B, N]

        if torch.any(mask_trigu):
            idx = torch.arange(self.max_beats, device=device)  # [N]
            short = _sample_uniform(
                low=self.rhythm_params.trigu_short_range[0],
                high=self.rhythm_params.trigu_short_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            long = _sample_uniform(
                low=self.rhythm_params.trigu_long_range[0],
                high=self.rhythm_params.trigu_long_range[1],
                shape=(batch_size, 1),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B, 1]
            pattern = torch.where((idx % 3) == 0, short, long).to(torch.float32)  # [B, N]
            rr_multipliers = torch.where(mask_trigu[:, None], pattern, rr_multipliers)  # [B, N]

        beat_times, beat_mask = self._make_beats(
            hr_bpm=hr_bpm,
            rr_multipliers=rr_multipliers,
            rng=rng,
            device=device,
        )  # [B, N], [B, N]
        in_episode = (beat_times >= episode_start[:, None]) & (beat_times < episode_end[:, None])  # [B, N]
        beat_mask = beat_mask & ~(mask_psvt[:, None] & in_episode)  # [B, N]

        svt_jitter = torch.randn(
            (batch_size, self.max_beats),
            generator=rng,
            device=device,
            dtype=torch.float32,
        )  # [B, N]
        rr_svt = torch.clamp(1.0 + svt_jitter * jitter_std[:, None], min=0.1)  # [B, N]
        svt_times, svt_mask = self._make_beats(
            hr_bpm=hr_svt,
            rr_multipliers=rr_svt,
            rng=rng,
            device=device,
        )  # [B, N], [B, N]
        in_episode_svt = (svt_times >= episode_start[:, None]) & (svt_times < episode_end[:, None])  # [B, N]
        svt_mask = svt_mask & (mask_psvt[:, None] & in_episode_svt)  # [B, N]

        qrs_times = torch.cat([beat_times, svt_times], dim=1)  # [B, 2N]
        qrs_mask = torch.cat([beat_mask, svt_mask], dim=1)  # [B, 2N]
        qrs_times, qrs_mask = self._sort_times(times=qrs_times, mask=qrs_mask)  # [B, 2N]

        pace_delay = _sample_uniform_int(
            low=self.rhythm_params.pace_spike_delay_samples_range[0],
            high=self.rhythm_params.pace_spike_delay_samples_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
        )  # [B]
        pace_times_idx = torch.round(qrs_times).to(torch.int64) - pace_delay[:, None]  # [B, E]
        pace_valid = (pace_times_idx >= 0) & (pace_times_idx < self.seq_len)  # [B, E]
        pace_mask = qrs_mask & mask_pace[:, None] & pace_valid  # [B, E]
        pace_times_idx = pace_times_idx.clamp(0, self.seq_len - 1)  # [B, E]
        pace_times = pace_times_idx.to(torch.float32)  # [B, E]

        pace_amp_scale = _sample_uniform(
            low=self.rhythm_params.pace_spike_amplitude_scale_range[0],
            high=self.rhythm_params.pace_spike_amplitude_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]

        pace_type_id = self.schema.type_id("pace_spike")
        pace_type_ids = torch.full(
            qrs_times.shape,
            pace_type_id,
            device=device,
            dtype=torch.int64,
        )  # [B, E]
        pace_params = torch.zeros(
            (batch_size, qrs_times.shape[1], 1),
            device=device,
            dtype=torch.float32,
        )  # [B, E, 1]
        pace_params[:, :, 0] = pace_amp_scale[:, None].expand_as(qrs_times)  # [B, E]

        pace_spikes = EventBatch(
            times=pace_times,
            type_ids=pace_type_ids,
            params=pace_params,
            mask=pace_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        flutter_rate = _sample_uniform(
            low=self.rhythm_params.flutter_rate_hz_range[0],
            high=self.rhythm_params.flutter_rate_hz_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        flutter_interval = self.sample_rate_hz / flutter_rate  # [B]
        flutter_phase = torch.rand((batch_size, 1), generator=rng, device=device) * flutter_interval[:, None]  # [B, 1]
        flutter_idx = torch.arange(self.max_flutter_waves, device=device, dtype=torch.float32)  # [E]
        flutter_times = flutter_phase + flutter_idx[None, :] * flutter_interval[:, None]  # [B, E]

        flutter_times_idx = torch.round(flutter_times).to(torch.int64)  # [B, E]
        flutter_valid = (flutter_times_idx >= 0) & (flutter_times_idx < self.seq_len)  # [B, E]
        flutter_mask = flutter_valid & mask_aflt[:, None]  # [B, E]
        flutter_times_idx = flutter_times_idx.clamp(0, self.seq_len - 1)  # [B, E]
        flutter_times = flutter_times_idx.to(torch.float32)  # [B, E]

        flutter_amp_scale = _sample_uniform(
            low=self.rhythm_params.flutter_wave_amplitude_scale_range[0],
            high=self.rhythm_params.flutter_wave_amplitude_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]

        flutter_type_id = self.schema.type_id("flutter_wave")
        flutter_type_ids = torch.full(
            (batch_size, self.max_flutter_waves),
            flutter_type_id,
            device=device,
            dtype=torch.int64,
        )  # [B, E]
        flutter_params = torch.zeros(
            (batch_size, self.max_flutter_waves, 1),
            device=device,
            dtype=torch.float32,
        )  # [B, E, 1]
        flutter_params[:, :, 0] = flutter_amp_scale[:, None].expand(batch_size, self.max_flutter_waves)  # [B, E]

        flutter_waves = EventBatch(
            times=flutter_times,
            type_ids=flutter_type_ids,
            params=flutter_params,
            mask=flutter_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        state.data["qrs_times"] = qrs_times
        state.data["qrs_mask"] = qrs_mask
        state.data["pace_spikes"] = pace_spikes
        state.data["flutter_waves"] = flutter_waves

        samples_base = f"{SAMPLES_PREFIX}/ECGRhythmTimingNode"
        state.data[f"{samples_base}/hr_bpm"] = hr_bpm  # [B]
        state.data[f"{samples_base}/hr_svt_bpm"] = hr_svt  # [B]
        state.data[f"{samples_base}/psvt_episode_start"] = episode_start  # [B]
        state.data[f"{samples_base}/psvt_episode_end"] = episode_end  # [B]
        state.data[f"{samples_base}/flutter_rate_hz"] = flutter_rate  # [B]
        state.data[f"{samples_base}/pace_spike_delay"] = pace_delay  # [B]
        state.data[f"{samples_base}/pace_spike_scale"] = pace_amp_scale  # [B]
        state.data[f"{samples_base}/flutter_wave_scale"] = flutter_amp_scale  # [B]
        return state


class _ECGComponentEventsNode(ProcessNode):
    """Generate base P/QRS/T events with conduction/timing adjustments."""

    def __init__(
        self,
        *,
        seq_len: int,
        sample_rate_hz: float,
        morphology_params: ECGMorphologyParams,
        schema: EventSchema,
        pr_long_indices: list[int],
        pr_short_indices: list[int],
        qt_long_indices: list[int],
        volt_low_indices: list[int],
        volt_high_indices: list[int],
        t_low_indices: list[int],
        t_mild_indices: list[int],
        av_block2_indices: list[int],
        av_block3_indices: list[int],
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")

        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.morphology_params = morphology_params
        self.schema = schema
        self.pr_long_indices = pr_long_indices
        self.pr_short_indices = pr_short_indices
        self.qt_long_indices = qt_long_indices
        self.volt_low_indices = volt_low_indices
        self.volt_high_indices = volt_high_indices
        self.t_low_indices = t_low_indices
        self.t_mild_indices = t_mild_indices
        self.av_block2_indices = av_block2_indices
        self.av_block3_indices = av_block3_indices

        atrial_hr_max = morphology_params.av_block3_atrial_hr_bpm_range[1]
        self.max_p_events = int(
            math.ceil(((seq_len - 1) / sample_rate_hz) * atrial_hr_max / 60.0)
        ) + 2

    def _mask_for(self, labels: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if not indices:
            return torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)  # [B]
        return labels[:, indices].any(dim=1)  # [B]

    def _make_regular_times(
        self,
        *,
        hr_bpm: torch.Tensor,
        jitter_std: torch.Tensor,
        max_events: int,
        rng: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hr_bpm.shape[0]
        rr_base = (self.sample_rate_hz * 60.0) / hr_bpm  # [B]
        jitter = torch.randn(
            (batch_size, max_events),
            generator=rng,
            device=device,
            dtype=torch.float32,
        )  # [B, N]
        rr_multipliers = torch.clamp(1.0 + jitter * jitter_std[:, None], min=0.1)  # [B, N]
        rr_samples = rr_multipliers * rr_base[:, None]  # [B, N]
        phase = torch.rand((batch_size, 1), generator=rng, device=device) * rr_base[:, None]  # [B, 1]
        intervals = torch.cat([phase, rr_samples], dim=1)  # [B, N+1]
        times = intervals.cumsum(dim=1)[:, :-1]  # [B, N]
        times_idx = torch.round(times).to(torch.int64)  # [B, N]
        mask = (times_idx >= 0) & (times_idx < self.seq_len)  # [B, N]
        times_idx = times_idx.clamp(0, self.seq_len - 1)  # [B, N]
        times = times_idx.to(torch.float32)  # [B, N]
        return times, mask

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        self._record_seed(state, rng)
        if "scp" not in state.y:
            raise ValueError("ECGProcess requires label 'scp' in state.y.")
        if "qrs_times" not in state.data or "qrs_mask" not in state.data:
            raise ValueError("ECGProcess requires qrs_times and qrs_mask in state.data.")

        labels = state.y["scp"]  # [B, S]
        batch_size = state.batch_size
        device = state.device

        qrs_times = state.data["qrs_times"]  # [B, E]
        qrs_mask = state.data["qrs_mask"]  # [B, E]
        qrs_mask_base = qrs_mask.clone()  # [B, E]

        mask_pr_long = self._mask_for(labels, self.pr_long_indices)  # [B]
        mask_pr_short = self._mask_for(labels, self.pr_short_indices)  # [B]
        if torch.any(mask_pr_long & mask_pr_short):
            raise ValueError("PR long and PR short codes cannot co-occur.")

        pr_ms = _sample_uniform(
            low=self.morphology_params.pr_ms_range[0],
            high=self.morphology_params.pr_ms_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        pr_long = _sample_uniform(
            low=self.morphology_params.pr_long_ms_range[0],
            high=self.morphology_params.pr_long_ms_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        pr_short = _sample_uniform(
            low=self.morphology_params.pr_short_ms_range[0],
            high=self.morphology_params.pr_short_ms_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        pr_ms = torch.where(mask_pr_short, pr_short, pr_ms)  # [B]
        pr_ms = torch.where(mask_pr_long, pr_long, pr_ms)  # [B]
        pr_samples = pr_ms * (self.sample_rate_hz / 1000.0)  # [B]

        mask_qt_long = self._mask_for(labels, self.qt_long_indices)  # [B]
        qt_ms = _sample_uniform(
            low=self.morphology_params.qt_ms_range[0],
            high=self.morphology_params.qt_ms_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        qt_long = _sample_uniform(
            low=self.morphology_params.qt_long_ms_range[0],
            high=self.morphology_params.qt_long_ms_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        qt_ms = torch.where(mask_qt_long, qt_long, qt_ms)  # [B]
        qt_samples = qt_ms * (self.sample_rate_hz / 1000.0)  # [B]

        mask_volt_low = self._mask_for(labels, self.volt_low_indices)  # [B]
        mask_volt_high = self._mask_for(labels, self.volt_high_indices)  # [B]
        if torch.any(mask_volt_low & mask_volt_high):
            raise ValueError("LVOLT and HVOLT cannot co-occur.")

        qrs_amp = _sample_uniform(
            low=self.morphology_params.qrs_amp_range[0],
            high=self.morphology_params.qrs_amp_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        qrs_scale_low = _sample_uniform(
            low=self.morphology_params.qrs_voltage_low_scale_range[0],
            high=self.morphology_params.qrs_voltage_low_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        qrs_scale_high = _sample_uniform(
            low=self.morphology_params.qrs_voltage_high_scale_range[0],
            high=self.morphology_params.qrs_voltage_high_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        qrs_scale = torch.ones((batch_size,), device=device, dtype=torch.float32)  # [B]
        qrs_scale = torch.where(mask_volt_low, qrs_scale_low, qrs_scale)  # [B]
        qrs_scale = torch.where(mask_volt_high, qrs_scale_high, qrs_scale)  # [B]
        qrs_amp = qrs_amp * qrs_scale  # [B]

        p_amp = _sample_uniform(
            low=self.morphology_params.p_amp_range[0],
            high=self.morphology_params.p_amp_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        t_amp = _sample_uniform(
            low=self.morphology_params.t_amp_range[0],
            high=self.morphology_params.t_amp_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]

        mask_t_low = self._mask_for(labels, self.t_low_indices)  # [B]
        mask_t_mild = self._mask_for(labels, self.t_mild_indices)  # [B]
        if torch.any(mask_t_low & mask_t_mild):
            raise ValueError("LOWT/DIG cannot co-occur with mild T-abnormality codes.")

        t_scale_low = _sample_uniform(
            low=self.morphology_params.t_low_scale_range[0],
            high=self.morphology_params.t_low_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        t_scale_mild = _sample_uniform(
            low=self.morphology_params.t_mild_scale_range[0],
            high=self.morphology_params.t_mild_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        t_scale = torch.ones((batch_size,), device=device, dtype=torch.float32)  # [B]
        t_scale = torch.where(mask_t_low, t_scale_low, t_scale)  # [B]
        t_scale = torch.where(mask_t_mild, t_scale_mild, t_scale)  # [B]
        t_amp = t_amp * t_scale  # [B]

        mask_2avb = self._mask_for(labels, self.av_block2_indices)  # [B]
        mask_3avb = self._mask_for(labels, self.av_block3_indices)  # [B]
        if torch.any(mask_2avb & mask_3avb):
            raise ValueError("2AVB and 3AVB cannot co-occur.")

        drop_prob = _sample_uniform(
            low=self.morphology_params.av_block2_drop_prob_range[0],
            high=self.morphology_params.av_block2_drop_prob_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        drop_mask = torch.rand(qrs_times.shape, generator=rng, device=device) < drop_prob[:, None]  # [B, E]
        drop_mask = drop_mask & mask_2avb[:, None]  # [B, E]
        qrs_mask = qrs_mask & ~drop_mask  # [B, E]

        max_events = max(self.max_p_events, qrs_times.shape[1])
        p_times = torch.zeros((batch_size, max_events), device=device, dtype=torch.float32)  # [B, E_p]
        p_mask = torch.zeros((batch_size, max_events), device=device, dtype=torch.bool)  # [B, E_p]

        min_events = qrs_times.shape[1]
        base_p_times = qrs_times - pr_samples[:, None]  # [B, E_p]
        base_p_idx = torch.round(base_p_times).to(torch.int64)  # [B, E_p]
        base_p_valid = (base_p_idx >= 0) & (base_p_idx < self.seq_len)  # [B, E_p]
        base_p_idx = base_p_idx.clamp(0, self.seq_len - 1)  # [B, E_p]
        base_p_times = base_p_idx.to(torch.float32)  # [B, E_p]
        p_times[:, :min_events] = base_p_times  # [B, E_p]
        p_mask[:, :min_events] = qrs_mask_base & base_p_valid  # [B, E_p]

        atrial_hr = _sample_uniform(
            low=self.morphology_params.av_block3_atrial_hr_bpm_range[0],
            high=self.morphology_params.av_block3_atrial_hr_bpm_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        atrial_jitter = _sample_uniform(
            low=self.morphology_params.av_block3_rr_jitter_std_range[0],
            high=self.morphology_params.av_block3_rr_jitter_std_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        p_times_3avb, p_mask_3avb = self._make_regular_times(
            hr_bpm=atrial_hr,
            jitter_std=atrial_jitter,
            max_events=max_events,
            rng=rng,
            device=device,
        )  # [B, E_p], [B, E_p]
        if torch.any(mask_3avb):
            p_times = torch.where(mask_3avb[:, None], p_times_3avb, p_times)  # [B, E_p]
            p_mask = torch.where(mask_3avb[:, None], p_mask_3avb, p_mask)  # [B, E_p]

        t_times = qrs_times + qt_samples[:, None]  # [B, E]
        t_idx = torch.round(t_times).to(torch.int64)  # [B, E]
        t_valid = (t_idx >= 0) & (t_idx < self.seq_len)  # [B, E]
        t_idx = t_idx.clamp(0, self.seq_len - 1)  # [B, E]
        t_times = t_idx.to(torch.float32)  # [B, E]
        t_mask = qrs_mask & t_valid  # [B, E]

        p_type_id = self.schema.type_id("p")
        qrs_type_id = self.schema.type_id("qrs")
        t_type_id = self.schema.type_id("t")

        p_type_ids = torch.full(
            (batch_size, max_events),
            p_type_id,
            device=device,
            dtype=torch.int64,
        )  # [B, E_p]
        qrs_type_ids = torch.full(
            qrs_times.shape,
            qrs_type_id,
            device=device,
            dtype=torch.int64,
        )  # [B, E]
        t_type_ids = torch.full(
            qrs_times.shape,
            t_type_id,
            device=device,
            dtype=torch.int64,
        )  # [B, E]

        p_params = torch.zeros(
            (batch_size, max_events, 1),
            device=device,
            dtype=torch.float32,
        )  # [B, E_p, 1]
        p_params[:, :, 0] = p_amp[:, None].expand(batch_size, max_events)  # [B, E_p]

        qrs_params = torch.zeros(
            (batch_size, qrs_times.shape[1], 1),
            device=device,
            dtype=torch.float32,
        )  # [B, E, 1]
        qrs_params[:, :, 0] = qrs_amp[:, None].expand_as(qrs_times)  # [B, E]

        t_params = torch.zeros(
            (batch_size, qrs_times.shape[1], 1),
            device=device,
            dtype=torch.float32,
        )  # [B, E, 1]
        t_params[:, :, 0] = t_amp[:, None].expand_as(qrs_times)  # [B, E]

        p_events = EventBatch(
            times=p_times,
            type_ids=p_type_ids,
            params=p_params,
            mask=p_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )
        qrs_events = EventBatch(
            times=qrs_times,
            type_ids=qrs_type_ids,
            params=qrs_params,
            mask=qrs_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )
        t_events = EventBatch(
            times=t_times,
            type_ids=t_type_ids,
            params=t_params,
            mask=t_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        state.data["p_events"] = p_events
        state.data["qrs_events"] = qrs_events
        state.data["t_events"] = t_events
        state.data["p_times"] = p_times
        state.data["p_mask"] = p_mask
        state.data["t_times"] = t_times
        state.data["t_mask"] = t_mask
        state.data["p_amp"] = p_amp
        state.data["qrs_amp"] = qrs_amp
        state.data["t_amp"] = t_amp

        samples_base = f"{SAMPLES_PREFIX}/ECGComponentEventsNode"
        state.data[f"{samples_base}/pr_ms"] = pr_ms  # [B]
        state.data[f"{samples_base}/qt_ms"] = qt_ms  # [B]
        state.data[f"{samples_base}/qrs_scale"] = qrs_scale  # [B]
        state.data[f"{samples_base}/t_scale"] = t_scale  # [B]
        state.data[f"{samples_base}/av_block2_drop_prob"] = drop_prob  # [B]
        state.data[f"{samples_base}/atrial_hr_bpm"] = atrial_hr  # [B]
        state.data[f"{samples_base}/p_amp"] = p_amp  # [B]
        state.data[f"{samples_base}/qrs_amp"] = qrs_amp  # [B]
        state.data[f"{samples_base}/t_amp"] = t_amp  # [B]
        return state


class _ECGOverlayEventsNode(ProcessNode):
    """Generate overlay events driven by SCP code effects."""

    def __init__(
        self,
        *,
        seq_len: int,
        morphology_params: ECGMorphologyParams,
        schema: EventSchema,
        effect_groups: list[
            tuple[str, str, list[int], float, tuple[float, float], str, str]
        ],
        pac_indices: list[int],
        pvc_indices: list[int],
        prc_indices: list[int],
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.morphology_params = morphology_params
        self.schema = schema
        self.effect_groups = effect_groups
        self.pac_indices = pac_indices
        self.pvc_indices = pvc_indices
        self.prc_indices = prc_indices

    def _mask_for(self, labels: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if not indices:
            return torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)  # [B]
        return labels[:, indices].any(dim=1)  # [B]

    def _union_events(self, events_list: list[EventBatch]) -> EventBatch:
        schema = events_list[0].schema
        times = torch.cat([events.times for events in events_list], dim=1)  # [B, E]
        type_ids = torch.cat([events.type_ids for events in events_list], dim=1)  # [B, E]
        params = torch.cat([events.params for events in events_list], dim=1)  # [B, E, P]
        mask = torch.cat([events.mask for events in events_list], dim=1)  # [B, E]

        sort_times = times.masked_fill(~mask, float("inf"))  # [B, E]
        indices = sort_times.argsort(dim=1)  # [B, E]
        times_sorted = torch.gather(times, 1, indices)  # [B, E]
        type_sorted = torch.gather(type_ids, 1, indices)  # [B, E]
        mask_sorted = torch.gather(mask, 1, indices)  # [B, E]
        indices_params = indices[:, :, None].expand(-1, -1, params.shape[2])  # [B, E, P]
        params_sorted = torch.gather(params, 1, indices_params)  # [B, E, P]

        return EventBatch(
            times=times_sorted,
            type_ids=type_sorted,
            params=params_sorted,
            mask=mask_sorted,
            schema=schema,
            meta=events_list[0].meta,
        )

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        self._record_seed(state, rng)
        if "scp" not in state.y:
            raise ValueError("ECGProcess requires label 'scp' in state.y.")

        labels = state.y["scp"]  # [B, S]
        batch_size = state.batch_size
        device = state.device

        qrs_times = state.data["qrs_times"]  # [B, E]
        qrs_mask = state.data["qrs_mask"]  # [B, E]
        p_times = state.data["p_times"]  # [B, E_p]
        p_mask = state.data["p_mask"]  # [B, E_p]
        t_times = state.data["t_times"]  # [B, E]
        t_mask = state.data["t_mask"]  # [B, E]

        p_amp = state.data["p_amp"]  # [B]
        qrs_amp = state.data["qrs_amp"]  # [B]
        t_amp = state.data["t_amp"]  # [B]

        overlay_batches: list[EventBatch] = []

        for sample_name, event_type, indices, sign, scale_range, anchor, scale_from in self.effect_groups:
            mask = self._mask_for(labels, indices)  # [B]
            scale = _sample_uniform(
                low=scale_range[0],
                high=scale_range[1],
                shape=(batch_size,),
                rng=rng,
                device=device,
                dtype=torch.float32,
            )  # [B]
            if scale_from == "p":
                amplitude = p_amp * scale  # [B]
            elif scale_from == "t":
                amplitude = t_amp * scale  # [B]
            else:
                amplitude = qrs_amp * scale  # [B]
            amplitude = amplitude * sign  # [B]

            if anchor == "p":
                times = p_times  # [B, E_p]
                mask_events = p_mask & mask[:, None]  # [B, E_p]
                count = p_times.shape[1]
            elif anchor == "t":
                times = t_times  # [B, E]
                mask_events = t_mask & mask[:, None]  # [B, E]
                count = qrs_times.shape[1]
            else:
                times = qrs_times  # [B, E]
                mask_events = qrs_mask & mask[:, None]  # [B, E]
                count = qrs_times.shape[1]

            type_id = self.schema.type_id(event_type)
            type_ids = torch.full(
                (batch_size, count),
                type_id,
                device=device,
                dtype=torch.int64,
            )  # [B, E]
            params = torch.zeros(
                (batch_size, count, 1),
                device=device,
                dtype=torch.float32,
            )  # [B, E, 1]
            params[:, :, 0] = amplitude[:, None].expand(batch_size, count)  # [B, E]

            overlay_batches.append(
                EventBatch(
                    times=times,
                    type_ids=type_ids,
                    params=params,
                    mask=mask_events,
                    schema=self.schema,
                    meta={"seq_len": self.seq_len},
                )
            )

            samples_base = f"{SAMPLES_PREFIX}/ECGOverlayEventsNode"
            state.data[f"{samples_base}/{sample_name}"] = scale  # [B]

        max_ectopy = self.morphology_params.ectopy_count_range[1]
        margin_frac = _sample_uniform(
            low=self.morphology_params.ectopy_time_margin_frac_range[0],
            high=self.morphology_params.ectopy_time_margin_frac_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        time_min = margin_frac * (self.seq_len - 1)  # [B]
        time_max = (self.seq_len - 1) - time_min  # [B]
        base_times = torch.rand(
            (batch_size, max_ectopy),
            generator=rng,
            device=device,
            dtype=torch.float32,
        )  # [B, E]
        ectopy_times = time_min[:, None] + base_times * (time_max - time_min)[:, None]  # [B, E]
        ectopy_idx = torch.round(ectopy_times).to(torch.int64)  # [B, E]
        ectopy_idx = ectopy_idx.clamp(0, self.seq_len - 1)  # [B, E]
        ectopy_times = ectopy_idx.to(torch.float32)  # [B, E]

        ectopy_count = _sample_uniform_int(
            low=self.morphology_params.ectopy_count_range[0],
            high=self.morphology_params.ectopy_count_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
        )  # [B]
        ectopy_rank = torch.arange(max_ectopy, device=device)[None, :]  # [1, E]
        ectopy_mask = ectopy_rank < ectopy_count[:, None]  # [B, E]

        pac_mask = self._mask_for(labels, self.pac_indices)  # [B]
        pvc_mask = self._mask_for(labels, self.pvc_indices)  # [B]
        prc_mask = self._mask_for(labels, self.prc_indices)  # [B]

        pac_scale = _sample_uniform(
            low=self.morphology_params.ectopy_p_scale_range[0],
            high=self.morphology_params.ectopy_p_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        qrs_scale = _sample_uniform(
            low=self.morphology_params.ectopy_qrs_scale_range[0],
            high=self.morphology_params.ectopy_qrs_scale_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]

        if torch.any(pac_mask):
            pac_type_id = self.schema.type_id("p")
            pac_type_ids = torch.full(
                (batch_size, max_ectopy),
                pac_type_id,
                device=device,
                dtype=torch.int64,
            )  # [B, E]
            pac_params = torch.zeros(
                (batch_size, max_ectopy, 1),
                device=device,
                dtype=torch.float32,
            )  # [B, E, 1]
            pac_params[:, :, 0] = (p_amp * pac_scale)[:, None].expand(batch_size, max_ectopy)  # [B, E]
            pac_mask_events = ectopy_mask & pac_mask[:, None]  # [B, E]
            overlay_batches.append(
                EventBatch(
                    times=ectopy_times,
                    type_ids=pac_type_ids,
                    params=pac_params,
                    mask=pac_mask_events,
                    schema=self.schema,
                    meta={"seq_len": self.seq_len},
                )
            )

        if torch.any(pvc_mask):
            pvc_type_id = self.schema.type_id("qrs_wide")
            pvc_type_ids = torch.full(
                (batch_size, max_ectopy),
                pvc_type_id,
                device=device,
                dtype=torch.int64,
            )  # [B, E]
            pvc_params = torch.zeros(
                (batch_size, max_ectopy, 1),
                device=device,
                dtype=torch.float32,
            )  # [B, E, 1]
            pvc_params[:, :, 0] = (qrs_amp * qrs_scale)[:, None].expand(batch_size, max_ectopy)  # [B, E]
            pvc_mask_events = ectopy_mask & pvc_mask[:, None]  # [B, E]
            overlay_batches.append(
                EventBatch(
                    times=ectopy_times,
                    type_ids=pvc_type_ids,
                    params=pvc_params,
                    mask=pvc_mask_events,
                    schema=self.schema,
                    meta={"seq_len": self.seq_len},
                )
            )

        if torch.any(prc_mask):
            prc_type_id = self.schema.type_id("qrs")
            prc_type_ids = torch.full(
                (batch_size, max_ectopy),
                prc_type_id,
                device=device,
                dtype=torch.int64,
            )  # [B, E]
            prc_params = torch.zeros(
                (batch_size, max_ectopy, 1),
                device=device,
                dtype=torch.float32,
            )  # [B, E, 1]
            prc_params[:, :, 0] = (qrs_amp * qrs_scale)[:, None].expand(batch_size, max_ectopy)  # [B, E]
            prc_mask_events = ectopy_mask & prc_mask[:, None]  # [B, E]
            overlay_batches.append(
                EventBatch(
                    times=ectopy_times,
                    type_ids=prc_type_ids,
                    params=prc_params,
                    mask=prc_mask_events,
                    schema=self.schema,
                    meta={"seq_len": self.seq_len},
                )
            )

        if not overlay_batches:
            empty = EventBatch(
                times=torch.zeros((batch_size, 0), device=device, dtype=torch.float32),  # [B, 0]
                type_ids=torch.zeros((batch_size, 0), device=device, dtype=torch.int64),  # [B, 0]
                params=torch.zeros((batch_size, 0, 1), device=device, dtype=torch.float32),  # [B, 0, 1]
                mask=torch.zeros((batch_size, 0), device=device, dtype=torch.bool),  # [B, 0]
                schema=self.schema,
                meta={"seq_len": self.seq_len},
            )
            overlay_events = empty
        else:
            overlay_events = self._union_events(overlay_batches)

        state.data["overlay_events"] = overlay_events
        state.data[f"{SAMPLES_PREFIX}/ECGOverlayEventsNode/ectopy_margin_frac"] = margin_frac  # [B]
        state.data[f"{SAMPLES_PREFIX}/ECGOverlayEventsNode/ectopy_count"] = ectopy_count  # [B]
        state.data[f"{SAMPLES_PREFIX}/ECGOverlayEventsNode/ectopy_qrs_scale"] = qrs_scale  # [B]
        state.data[f"{SAMPLES_PREFIX}/ECGOverlayEventsNode/ectopy_p_scale"] = pac_scale  # [B]
        return state


class ECGProcess(nn.Module):
    """ECG-like event process with PTB-XL multi-label SCP support (71 codes)."""

    def __init__(
        self,
        *,
        seq_len: int,
        sample_rate_hz: float,
        scp_codes: list[str],
        scp_sampler: Sampler,
        rhythm_params: ECGRhythmParams,
        morphology_params: ECGMorphologyParams,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        if not scp_codes:
            raise ValueError("scp_codes must be non-empty.")
        if len(set(scp_codes)) != len(scp_codes):
            raise ValueError("scp_codes must be unique.")
        if scp_codes != ptbxl_all_codes():
            raise ValueError(
                "ECGProcess requires scp_codes in PTB-XL CSV order. "
                "Use ptbxl_all_codes()."
            )

        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.duration_sec = (seq_len - 1) / sample_rate_hz
        self.scp_codes = list(scp_codes)
        self.scp_sampler = scp_sampler
        self.rhythm_params = rhythm_params
        self.morphology_params = morphology_params

        self.schema = EventSchema(
            type_names=list(PTBXL_EVENT_TYPE_NAMES),
            param_names=["amplitude"],
            time_unit="samples",
        )

        code_to_index = {code: idx for idx, code in enumerate(self.scp_codes)}
        rhythm_codes = ptbxl_codes_by_group("rhythm")
        rhythm_indices = [code_to_index[code] for code in rhythm_codes]

        group_indices = ptbxl_group_indices(scp_codes=self.scp_codes)

        effect_specs = []
        for idx, group in enumerate(ptbxl_effect_groups()):
            indices = [code_to_index[code] for code in group.codes]
            suffix = "pos" if group.sign > 0 else "neg"
            sample_name = f"{group.event_type}_{suffix}_{idx}"
            effect_specs.append(
                (
                    sample_name,
                    group.event_type,
                    indices,
                    group.sign,
                    group.scale_range,
                    group.anchor,
                    group.scale_from,
                )
            )

        pr_long_indices = [code_to_index[code] for code in PR_LONG_CODES if code in code_to_index]
        pr_short_indices = [code_to_index[code] for code in PR_SHORT_CODES if code in code_to_index]
        qt_long_indices = [code_to_index[code] for code in QT_LONG_CODES if code in code_to_index]
        volt_low_indices = [code_to_index[code] for code in QRS_VOLT_LOW_CODES if code in code_to_index]
        volt_high_indices = [code_to_index[code] for code in QRS_VOLT_HIGH_CODES if code in code_to_index]
        t_low_indices = [code_to_index[code] for code in T_LOW_CODES if code in code_to_index]
        t_mild_indices = [code_to_index[code] for code in T_MILD_CODES if code in code_to_index]
        av_block2_indices = [code_to_index["2AVB"]] if "2AVB" in code_to_index else []
        av_block3_indices = [code_to_index["3AVB"]] if "3AVB" in code_to_index else []
        pac_indices = [code_to_index[code] for code in PAC_CODES if code in code_to_index]
        pvc_indices = [code_to_index[code] for code in PVC_CODES if code in code_to_index]
        prc_indices = [code_to_index[code] for code in PRC_CODES if code in code_to_index]

        if not rhythm_indices:
            raise ValueError("ECGProcess requires rhythm SCP codes in scp_codes.")
        if any(code not in code_to_index for code in NORM_CODES):
            raise ValueError("ECGProcess requires NORM in scp_codes.")

        base_meta: dict[str, object] = {
            "seq_len": self.seq_len,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_sec": self.duration_sec,
            "scp_codes": self.scp_codes,
            "event_type_names": list(self.schema.type_names),
            "rhythm_params": rhythm_params.to_meta(),
            "morphology_params": morphology_params.to_meta(),
        }

        self._graph = ProcessGraph(
            name="ECGProcess",
            outputs={"events"},
            base_meta=base_meta,
            graph=[
                SampleMultiLabelNode(
                    label_key="scp",
                    class_names=self.scp_codes,
                    sampler=self.scp_sampler,
                    label_groups=group_indices,
                ),
                _ECGRhythmTimingNode(
                    seq_len=self.seq_len,
                    sample_rate_hz=self.sample_rate_hz,
                    rhythm_codes=rhythm_codes,
                    rhythm_indices=rhythm_indices,
                    rhythm_params=self.rhythm_params,
                    schema=self.schema,
                ),
                _ECGComponentEventsNode(
                    seq_len=self.seq_len,
                    sample_rate_hz=self.sample_rate_hz,
                    morphology_params=self.morphology_params,
                    schema=self.schema,
                    pr_long_indices=pr_long_indices,
                    pr_short_indices=pr_short_indices,
                    qt_long_indices=qt_long_indices,
                    volt_low_indices=volt_low_indices,
                    volt_high_indices=volt_high_indices,
                    t_low_indices=t_low_indices,
                    t_mild_indices=t_mild_indices,
                    av_block2_indices=av_block2_indices,
                    av_block3_indices=av_block3_indices,
                ),
                _ECGOverlayEventsNode(
                    seq_len=self.seq_len,
                    morphology_params=self.morphology_params,
                    schema=self.schema,
                    effect_groups=effect_specs,
                    pac_indices=pac_indices,
                    pvc_indices=pvc_indices,
                    prc_indices=prc_indices,
                ),
                UnionEventsNode(
                    in_keys=[
                        "p_events",
                        "qrs_events",
                        "t_events",
                        "overlay_events",
                        "pace_spikes",
                        "flutter_waves",
                    ],
                    out_key="events",
                ),
            ],
        )

    @classmethod
    def ptbxl_defaults(
        cls,
        *,
        seq_len: int,
        sample_rate_hz: float,
        scp_sampler: Sampler | None = None,
    ) -> ECGProcess:
        scp_codes = ptbxl_all_codes()
        sampler = scp_sampler or PTBXLLabelSetSampler(
            scp_codes=scp_codes,
            normal_prob=0.25,
        )
        return cls(
            seq_len=seq_len,
            sample_rate_hz=sample_rate_hz,
            scp_codes=scp_codes,
            scp_sampler=sampler,
            rhythm_params=ECGRhythmParams.ptbxl_defaults(),
            morphology_params=ECGMorphologyParams.ptbxl_defaults(),
        )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        return self._graph(batch_size, device, rng=rng)
