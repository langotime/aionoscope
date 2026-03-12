from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..core.events import EventBatch, EventSchema
from ..core.samplers import ConstantSampler, Sampler, SamplerLike, sampler_from_value, sampler_sample
from ..core.types import LatentState
from ..core.utils import SAMPLES_PREFIX
from .graph import ProcessGraph, ProcessNode, ProcessState
from .nodes import SampleLabelNode, UnionEventsNode


@dataclass(frozen=True)
class ECGRhythmParams:
    """Parameters controlling PTB-XL-like rhythm simulation.

    All ranges and constants are explicit and are recorded in process meta.
    """

    hr_bpm_ranges: dict[str, tuple[float, float]]
    psvt_svt_hr_bpm_range: tuple[float, float]
    flutter_rate_hz_range: tuple[float, float]
    regular_rr_jitter_std: float
    sarrh_mod_amp_max: float
    sarrh_cycles_per_window: float
    sv_premature_prob: float
    sv_premature_short: float
    sv_premature_long: float
    bigu_short: float
    bigu_long: float
    trigu_short: float
    trigu_long: float
    psvt_episode_start_frac_range: tuple[float, float]
    psvt_episode_duration_frac_range: tuple[float, float]
    pace_spike_delay_samples: int
    pace_spike_amplitude_scale: float
    flutter_wave_amplitude_scale: float

    def __post_init__(self) -> None:
        if not self.hr_bpm_ranges:
            raise ValueError("hr_bpm_ranges must be non-empty.")
        for code, (low, high) in self.hr_bpm_ranges.items():
            if not code:
                raise ValueError("hr_bpm_ranges keys must be non-empty.")
            if low <= 0 or high <= 0 or high <= low:
                raise ValueError(
                    f"Invalid hr_bpm range for '{code}': low={low}, high={high}."
                )

        low_svt, high_svt = self.psvt_svt_hr_bpm_range
        if low_svt <= 0 or high_svt <= 0 or high_svt <= low_svt:
            raise ValueError(
                "Invalid psvt_svt_hr_bpm_range. "
                f"Got low={low_svt}, high={high_svt}."
            )

        flutter_low, flutter_high = self.flutter_rate_hz_range
        if flutter_low <= 0 or flutter_high <= 0 or flutter_high <= flutter_low:
            raise ValueError(
                "Invalid flutter_rate_hz_range. "
                f"Got low={flutter_low}, high={flutter_high}."
            )

        if self.regular_rr_jitter_std < 0:
            raise ValueError(
                "regular_rr_jitter_std must be non-negative, "
                f"got {self.regular_rr_jitter_std}."
            )
        if self.sarrh_mod_amp_max < 0:
            raise ValueError(
                "sarrh_mod_amp_max must be non-negative, "
                f"got {self.sarrh_mod_amp_max}."
            )
        if self.sarrh_cycles_per_window <= 0:
            raise ValueError(
                "sarrh_cycles_per_window must be positive, "
                f"got {self.sarrh_cycles_per_window}."
            )

        if not (0.0 <= self.sv_premature_prob <= 1.0):
            raise ValueError(
                "sv_premature_prob must be in [0, 1], "
                f"got {self.sv_premature_prob}."
            )
        if self.sv_premature_short <= 0 or self.sv_premature_long <= 0:
            raise ValueError("sv_premature_short/long must be positive.")

        if self.bigu_short <= 0 or self.bigu_long <= 0:
            raise ValueError("bigu_short/long must be positive.")
        if self.trigu_short <= 0 or self.trigu_long <= 0:
            raise ValueError("trigu_short/long must be positive.")

        start_low, start_high = self.psvt_episode_start_frac_range
        if start_low < 0 or start_high < 0 or start_high <= start_low:
            raise ValueError(
                "Invalid psvt_episode_start_frac_range. "
                f"Got low={start_low}, high={start_high}."
            )
        dur_low, dur_high = self.psvt_episode_duration_frac_range
        if dur_low <= 0 or dur_high <= 0 or dur_high <= dur_low:
            raise ValueError(
                "Invalid psvt_episode_duration_frac_range. "
                f"Got low={dur_low}, high={dur_high}."
            )

        if self.pace_spike_delay_samples < 0:
            raise ValueError(
                "pace_spike_delay_samples must be non-negative, "
                f"got {self.pace_spike_delay_samples}."
            )
        if self.pace_spike_amplitude_scale <= 0:
            raise ValueError(
                "pace_spike_amplitude_scale must be positive, "
                f"got {self.pace_spike_amplitude_scale}."
            )
        if self.flutter_wave_amplitude_scale <= 0:
            raise ValueError(
                "flutter_wave_amplitude_scale must be positive, "
                f"got {self.flutter_wave_amplitude_scale}."
            )

    def to_meta(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def ptbxl_defaults(cls) -> ECGRhythmParams:
        """Explicit defaults for the 12 PTB-XL rhythm SCP codes."""
        hr_bpm_ranges: dict[str, tuple[float, float]] = {
            "SR": (55.0, 85.0),
            "AFIB": (60.0, 130.0),
            "STACH": (110.0, 150.0),
            "SARRH": (55.0, 85.0),
            "SBRAD": (35.0, 50.0),
            "PACE": (60.0, 90.0),
            "SVARR": (60.0, 110.0),
            "BIGU": (60.0, 100.0),
            "AFLT": (120.0, 170.0),
            "SVTAC": (140.0, 200.0),
            "PSVT": (60.0, 90.0),
            "TRIGU": (60.0, 100.0),
        }

        return cls(
            hr_bpm_ranges=hr_bpm_ranges,
            psvt_svt_hr_bpm_range=(160.0, 220.0),
            flutter_rate_hz_range=(4.0, 6.0),
            regular_rr_jitter_std=0.02,
            sarrh_mod_amp_max=0.15,
            sarrh_cycles_per_window=1.0,
            sv_premature_prob=0.12,
            sv_premature_short=0.7,
            sv_premature_long=1.3,
            bigu_short=0.7,
            bigu_long=1.3,
            trigu_short=0.6,
            trigu_long=1.2,
            psvt_episode_start_frac_range=(0.2, 0.6),
            psvt_episode_duration_frac_range=(0.2, 0.5),
            pace_spike_delay_samples=2,
            pace_spike_amplitude_scale=0.4,
            flutter_wave_amplitude_scale=0.25,
        )


class _ECGRhythmEventsNode(ProcessNode):
    """Generate rhythm-specific EventBatch streams (beats + auxiliary events)."""

    def __init__(
        self,
        *,
        seq_len: int,
        sample_rate_hz: float,
        rhythm_codes: list[str],
        rhythm_params: ECGRhythmParams,
        schema: EventSchema,
        amplitude_sampler: Sampler,
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

        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.duration_sec = (seq_len - 1) / sample_rate_hz
        self.rhythm_codes = list(rhythm_codes)
        self.rhythm_params = rhythm_params
        self.schema = schema
        self.amplitude_sampler = amplitude_sampler

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

    def _sample_uniform(
        self,
        *,
        low: float,
        high: float,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if high <= low:
            raise ValueError(f"Uniform range requires high>low, got low={low}, high={high}.")
        u = torch.rand(shape, generator=rng, device=device, dtype=dtype)  # [*shape]
        return low + (high - low) * u

    def _make_regular_multipliers(
        self,
        *,
        batch_size: int,
        num_intervals: int,
        rng: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        jitter = torch.randn(
            (batch_size, num_intervals),
            generator=rng,
            device=device,
            dtype=torch.float32,
        )  # [B, N]
        multipliers = 1.0 + jitter * float(self.rhythm_params.regular_rr_jitter_std)  # [B, N]
        return torch.clamp(multipliers, min=0.1)

    def _make_beats(
        self,
        *,
        hr_bpm: torch.Tensor,
        rr_multipliers: torch.Tensor,
        amplitude: torch.Tensor,
        rng: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        type_ids = torch.zeros((batch_size, num_beats), device=device, dtype=torch.int64)  # [B, N]

        params = torch.zeros(
            (batch_size, num_beats, 1),
            device=device,
            dtype=torch.float32,
        )  # [B, N, P=1]
        params[:, :, 0] = amplitude[:, None].expand(batch_size, num_beats)  # [B, N]

        return times_samples, type_ids, params, mask

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        self._record_seed(state, rng)
        if "rhythm" not in state.y:
            raise ValueError("ECGProcess requires label 'rhythm' in state.y.")

        label = state.y["rhythm"].to(torch.int64)  # [B]
        batch_size = state.batch_size
        device = state.device

        hr_min = torch.tensor(
            [self.rhythm_params.hr_bpm_ranges[code][0] for code in self.rhythm_codes],
            device=device,
            dtype=torch.float32,
        )  # [C]
        hr_max = torch.tensor(
            [self.rhythm_params.hr_bpm_ranges[code][1] for code in self.rhythm_codes],
            device=device,
            dtype=torch.float32,
        )  # [C]

        u_hr = torch.rand((batch_size,), generator=rng, device=device)  # [B]
        hr_bpm = hr_min[label] + u_hr * (hr_max[label] - hr_min[label])  # [B]

        mask_psvt = self._mask_for(label, "PSVT")  # [B]
        hr_svt = self._sample_uniform(
            low=self.rhythm_params.psvt_svt_hr_bpm_range[0],
            high=self.rhythm_params.psvt_svt_hr_bpm_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        hr_svt = torch.where(mask_psvt, hr_svt, hr_bpm)  # [B]

        start_frac = self._sample_uniform(
            low=self.rhythm_params.psvt_episode_start_frac_range[0],
            high=self.rhythm_params.psvt_episode_start_frac_range[1],
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
        )  # [B]
        dur_frac = self._sample_uniform(
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

        mask_afib = self._mask_for(label, "AFIB")  # [B]
        mask_sarrh = self._mask_for(label, "SARRH")  # [B]
        mask_svarr = self._mask_for(label, "SVARR")  # [B]
        mask_bigu = self._mask_for(label, "BIGU")  # [B]
        mask_trigu = self._mask_for(label, "TRIGU")  # [B]
        mask_pace = self._mask_for(label, "PACE")  # [B]
        mask_aflt = self._mask_for(label, "AFLT")  # [B]

        rr_multipliers = self._make_regular_multipliers(
            batch_size=batch_size,
            num_intervals=self.max_beats,
            rng=rng,
            device=device,
        )  # [B, N]

        if torch.any(mask_sarrh):
            phase = torch.rand((batch_size, 1), generator=rng, device=device) * (2.0 * math.pi)  # [B, 1]
            amp = torch.rand((batch_size, 1), generator=rng, device=device) * float(
                self.rhythm_params.sarrh_mod_amp_max
            )  # [B, 1]
            grid = torch.linspace(
                0.0,
                2.0 * math.pi * float(self.rhythm_params.sarrh_cycles_per_window),
                steps=self.max_beats,
                device=device,
                dtype=torch.float32,
            )  # [N]
            sarrh = 1.0 + amp * torch.sin(phase + grid[None, :])  # [B, N]
            sarrh = torch.clamp(sarrh, min=0.1)  # [B, N]
            rr_multipliers = torch.where(mask_sarrh[:, None], sarrh, rr_multipliers)  # [B, N]

        if torch.any(mask_afib):
            u = torch.rand((batch_size, self.max_beats), generator=rng, device=device)  # [B, N]
            exp_intervals = -torch.log(u.clamp_min(1e-6))  # [B, N]
            exp_intervals = exp_intervals / exp_intervals.mean(dim=1, keepdim=True)  # [B, N]
            exp_intervals = torch.clamp(exp_intervals, min=0.1)  # [B, N]
            rr_multipliers = torch.where(mask_afib[:, None], exp_intervals, rr_multipliers)  # [B, N]

        if torch.any(mask_svarr):
            idx = torch.arange(self.max_beats, device=device)  # [N]
            non_last = (idx < (self.max_beats - 1))[None, :]  # [1, N]
            non_first = (idx > 0)[None, :]  # [1, N]

            u = torch.rand((batch_size, self.max_beats), generator=rng, device=device)  # [B, N]
            premature = (u < float(self.rhythm_params.sv_premature_prob)) & non_last  # [B, N]
            long_after = torch.roll(premature, shifts=1, dims=1) & non_first  # [B, N]

            short_value = torch.tensor(
                float(self.rhythm_params.sv_premature_short),
                device=device,
                dtype=torch.float32,
            )  # []
            long_value = torch.tensor(
                float(self.rhythm_params.sv_premature_long),
                device=device,
                dtype=torch.float32,
            )  # []
            svarr = torch.ones((batch_size, self.max_beats), device=device, dtype=torch.float32)  # [B, N]
            svarr = torch.where(premature, short_value, svarr)  # [B, N]
            svarr = torch.where(long_after, long_value, svarr)  # [B, N]
            rr_multipliers = torch.where(mask_svarr[:, None], svarr, rr_multipliers)  # [B, N]

        if torch.any(mask_bigu):
            idx = torch.arange(self.max_beats, device=device)  # [N]
            short = torch.tensor(float(self.rhythm_params.bigu_short), device=device)  # []
            long = torch.tensor(float(self.rhythm_params.bigu_long), device=device)  # []
            pattern = torch.where((idx % 2) == 0, short, long).to(torch.float32)  # [N]
            rr_multipliers = torch.where(mask_bigu[:, None], pattern[None, :], rr_multipliers)  # [B, N]

        if torch.any(mask_trigu):
            idx = torch.arange(self.max_beats, device=device)  # [N]
            short = torch.tensor(float(self.rhythm_params.trigu_short), device=device)  # []
            long = torch.tensor(float(self.rhythm_params.trigu_long), device=device)  # []
            pattern = torch.where((idx % 3) == 0, short, long).to(torch.float32)  # [N]
            rr_multipliers = torch.where(mask_trigu[:, None], pattern[None, :], rr_multipliers)  # [B, N]

        amplitude = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(batch_size,),
            rng=rng,
            device=device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]

        # Beats (main)
        beat_times, beat_type_ids, beat_params, beat_mask = self._make_beats(
            hr_bpm=hr_bpm,
            rr_multipliers=rr_multipliers,
            amplitude=amplitude,
            rng=rng,
            device=device,
        )
        in_episode = (beat_times >= episode_start[:, None]) & (beat_times < episode_end[:, None])  # [B, N]
        beat_mask = beat_mask & ~(mask_psvt[:, None] & in_episode)  # [B, N]

        beats = EventBatch(
            times=beat_times,
            type_ids=beat_type_ids,
            params=beat_params,
            mask=beat_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        # Beats (SVT episode), masked to PSVT window.
        rr_svt = self._make_regular_multipliers(
            batch_size=batch_size,
            num_intervals=self.max_beats,
            rng=rng,
            device=device,
        )  # [B, N]
        svt_times, svt_type_ids, svt_params, svt_mask = self._make_beats(
            hr_bpm=hr_svt,
            rr_multipliers=rr_svt,
            amplitude=amplitude,
            rng=rng,
            device=device,
        )
        in_episode_svt = (svt_times >= episode_start[:, None]) & (svt_times < episode_end[:, None])  # [B, N]
        svt_mask = svt_mask & (mask_psvt[:, None] & in_episode_svt)  # [B, N]

        beats_svt = EventBatch(
            times=svt_times,
            type_ids=svt_type_ids,
            params=svt_params,
            mask=svt_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        # Pace spikes: one per beat (PACE only), just before the beat.
        pace_delay = int(self.rhythm_params.pace_spike_delay_samples)
        beat_times_idx = beat_times.to(torch.int64)  # [B, N]
        pace_times_idx_unclamped = beat_times_idx - pace_delay  # [B, N]
        pace_valid = (pace_times_idx_unclamped >= 0) & (pace_times_idx_unclamped < self.seq_len)  # [B, N]
        pace_mask = beat_mask & mask_pace[:, None] & pace_valid  # [B, N]
        pace_times_idx = pace_times_idx_unclamped.clamp(0, self.seq_len - 1)  # [B, N]
        pace_times = pace_times_idx.to(torch.float32)  # [B, N]
        pace_type_ids = torch.full((batch_size, self.max_beats), 1, device=device, dtype=torch.int64)  # [B, N]
        pace_params = torch.zeros((batch_size, self.max_beats, 1), device=device, dtype=torch.float32)  # [B, N, 1]
        pace_params[:, :, 0] = (
            amplitude[:, None].expand(batch_size, self.max_beats) * float(self.rhythm_params.pace_spike_amplitude_scale)
        )  # [B, N]

        pace_spikes = EventBatch(
            times=pace_times,
            type_ids=pace_type_ids,
            params=pace_params,
            mask=pace_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        # Flutter waves: regular high-rate atrial activity (AFLT only).
        flutter_rate = self._sample_uniform(
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

        flutter_times_idx_unclamped = torch.round(flutter_times).to(torch.int64)  # [B, E]
        flutter_valid = (flutter_times_idx_unclamped >= 0) & (flutter_times_idx_unclamped < self.seq_len)  # [B, E]
        flutter_mask = flutter_valid & mask_aflt[:, None]  # [B, E]
        flutter_times_idx = flutter_times_idx_unclamped.clamp(0, self.seq_len - 1)  # [B, E]
        flutter_times = flutter_times_idx.to(torch.float32)  # [B, E]

        flutter_type_ids = torch.full(
            (batch_size, self.max_flutter_waves),
            2,
            device=device,
            dtype=torch.int64,
        )  # [B, E]
        flutter_params = torch.zeros(
            (batch_size, self.max_flutter_waves, 1),
            device=device,
            dtype=torch.float32,
        )  # [B, E, 1]
        flutter_params[:, :, 0] = (
            amplitude[:, None].expand(batch_size, self.max_flutter_waves)
            * float(self.rhythm_params.flutter_wave_amplitude_scale)
        )  # [B, E]

        flutter_waves = EventBatch(
            times=flutter_times,
            type_ids=flutter_type_ids,
            params=flutter_params,
            mask=flutter_mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )

        state.data["beats"] = beats
        state.data["beats_svt"] = beats_svt
        state.data["pace_spikes"] = pace_spikes
        state.data["flutter_waves"] = flutter_waves

        samples_base = f"{SAMPLES_PREFIX}/ECGRhythmEventsNode"
        state.data[f"{samples_base}/hr_bpm"] = hr_bpm  # [B]
        state.data[f"{samples_base}/hr_svt_bpm"] = hr_svt  # [B]
        state.data[f"{samples_base}/psvt_episode_start"] = episode_start  # [B]
        state.data[f"{samples_base}/psvt_episode_end"] = episode_end  # [B]
        state.data[f"{samples_base}/flutter_rate_hz"] = flutter_rate  # [B]
        return state


class ECGProcess(nn.Module):
    """ECG-like event process with PTB-XL rhythm label support (12 codes).

    This process emits an EventBatch with event types:
    - beat
    - pace_spike
    - flutter_wave

    The caller is expected to render events into dense signals using:
    `EventImpulseView -> KernelConvView -> ECGLeadsView`.
    """

    SUPPORTED_RHYTHM_CODES = {
        "SR",
        "AFIB",
        "STACH",
        "SARRH",
        "SBRAD",
        "PACE",
        "SVARR",
        "BIGU",
        "AFLT",
        "SVTAC",
        "PSVT",
        "TRIGU",
    }

    def __init__(
        self,
        *,
        seq_len: int,
        sample_rate_hz: float,
        rhythm_codes: list[str],
        rhythm_params: ECGRhythmParams,
        rhythm_sampler: SamplerLike[int] | None = None,
        amplitude: SamplerLike[float] = 1.0,
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

        unknown = sorted(set(rhythm_codes) - self.SUPPORTED_RHYTHM_CODES)
        if unknown:
            raise ValueError(
                "ECGProcess only supports PTB-XL rhythm codes in v1. "
                f"Unknown: {unknown}."
            )

        missing_ranges = sorted(code for code in rhythm_codes if code not in rhythm_params.hr_bpm_ranges)
        if missing_ranges:
            raise ValueError(
                "rhythm_params.hr_bpm_ranges is missing codes required by rhythm_codes. "
                f"Missing: {missing_ranges}."
            )

        amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        if isinstance(amplitude_sampler, ConstantSampler):
            if amplitude_sampler.value <= 0:
                raise ValueError(f"amplitude must be positive, got {amplitude_sampler.value}.")

        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.duration_sec = (seq_len - 1) / sample_rate_hz
        self.rhythm_codes = list(rhythm_codes)
        self.rhythm_params = rhythm_params

        self.schema = EventSchema(
            type_names=["beat", "pace_spike", "flutter_wave"],
            param_names=["amplitude"],
            time_unit="samples",
        )

        base_meta: dict[str, object] = {
            "seq_len": self.seq_len,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_sec": self.duration_sec,
            "rhythm_codes": self.rhythm_codes,
            "rhythm_names": self.rhythm_codes,
            "event_type_names": list(self.schema.type_names),
            "rhythm_params": rhythm_params.to_meta(),
        }
        if isinstance(amplitude_sampler, ConstantSampler):
            base_meta["amplitude"] = float(amplitude_sampler.value)

        self._graph = ProcessGraph(
            name="ECGProcess",
            outputs={"events"},
            base_meta=base_meta,
            graph=[
                SampleLabelNode(
                    label_key="rhythm",
                    class_names=self.rhythm_codes,
                    sampler=rhythm_sampler,
                ),
                _ECGRhythmEventsNode(
                    seq_len=self.seq_len,
                    sample_rate_hz=self.sample_rate_hz,
                    rhythm_codes=self.rhythm_codes,
                    rhythm_params=self.rhythm_params,
                    schema=self.schema,
                    amplitude_sampler=amplitude_sampler,
                ),
                UnionEventsNode(
                    in_keys=["beats", "beats_svt", "pace_spikes", "flutter_waves"],
                    out_key="events",
                ),
            ],
        )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        return self._graph(batch_size, device, rng=rng)
