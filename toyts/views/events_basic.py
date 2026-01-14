from __future__ import annotations

import math
from typing import Any, Literal

import torch

from ..core.types import LatentState, Observation
from ._enabled import views_resolve_enabled_mask
from ._signal import views_extract_signal
from .base import View


def _enabled_any(mask: torch.Tensor) -> bool:
    return bool(torch.any(mask).item())


class EventRenderView(View):
    """Render an EventBatch into an additive single-channel signal.

    This view supports multiple events per sample by summing contributions over
    all valid events (`mask == True`).
    """

    _Supported = Literal["spike", "level_change", "gaussian", "rect_pulse", "exp_decay", "ringdown"]

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude_param: str,
        rounding: Literal["nearest", "floor", "ceil"],
        sigma_sec_param: str | None = None,
        duration_sec_param: str | None = None,
        tau_sec_param: str | None = None,
        frequency_hz_param: str | None = None,
        phase_param: str | None = None,
        enabled_key: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if not amplitude_param:
            raise ValueError("amplitude_param must be non-empty.")
        if rounding not in {"nearest", "floor", "ceil"}:
            raise ValueError(f"rounding must be nearest/floor/ceil, got {rounding}.")
        for name, param in (
            ("sigma_sec_param", sigma_sec_param),
            ("duration_sec_param", duration_sec_param),
            ("tau_sec_param", tau_sec_param),
            ("frequency_hz_param", frequency_hz_param),
            ("phase_param", phase_param),
        ):
            if param is not None and not param:
                raise ValueError(f"{name} must be non-empty when provided.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.amplitude_param = amplitude_param
        self.rounding = rounding
        self.sigma_sec_param = sigma_sec_param
        self.duration_sec_param = duration_sec_param
        self.tau_sec_param = tau_sec_param
        self.frequency_hz_param = frequency_hz_param
        self.phase_param = phase_param
        self.enabled_key = enabled_key

        time_samples = torch.arange(seq_len, dtype=torch.float32)  # [L]
        self.register_buffer("time_samples", time_samples)

    def _require_sample_rate_hz(self, process_meta: dict[str, Any]) -> float:
        if "sample_rate_hz" not in process_meta:
            raise ValueError(
                "EventRenderView requires process_meta['sample_rate_hz'] to interpret time parameters."
            )
        sample_rate_hz = process_meta["sample_rate_hz"]
        if not isinstance(sample_rate_hz, (int, float)):
            raise ValueError(
                "EventRenderView requires sample_rate_hz to be a float. "
                f"Got {type(sample_rate_hz).__name__}."
            )
        if float(sample_rate_hz) <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        return float(sample_rate_hz)

    def _round_times(self, times: torch.Tensor) -> torch.Tensor:
        if self.rounding == "nearest":
            return torch.round(times)
        if self.rounding == "floor":
            return torch.floor(times)
        return torch.ceil(times)

    def _require_param_index(
        self,
        *,
        schema_param_names: list[str],
        param_name: str | None,
        required_for: str,
    ) -> int:
        if param_name is None:
            raise ValueError(
                f"EventRenderView requires '{required_for}' but '{required_for}_param' was not provided."
            )
        if param_name not in schema_param_names:
            raise ValueError(
                f"EventRenderView schema is missing param '{param_name}' required for '{required_for}'. "
                f"Available: {schema_param_names}."
            )
        return schema_param_names.index(param_name)

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        if isinstance(input_state, Observation):
            raise TypeError("EventRenderView expects LatentState, got Observation.")
        if input_state.events is None:
            raise ValueError("EventRenderView requires LatentState.events to be present.")

        signal, labels, process_meta = views_extract_signal(input_state, name="EventRenderView")
        if signal.shape[2] != self.seq_len:
            raise ValueError(
                "EventRenderView seq_len mismatch. "
                f"Expected {self.seq_len}, got {signal.shape[2]}."
            )

        batch_size = signal.shape[0]
        enabled_mask = views_resolve_enabled_mask(
            process_meta,
            enabled_key=self.enabled_key,
            batch_size=batch_size,
            device=signal.device,
            name="EventRenderView",
        )  # [B]
        if self.enabled_key is not None and not _enabled_any(enabled_mask):
            meta = {"view": "EventRenderView", "enabled_key": self.enabled_key, "process": process_meta}
            return Observation(x=signal, y=labels, meta=meta)

        events = input_state.events
        if events.schema.time_unit != "samples":
            raise ValueError(
                "EventRenderView expects events.schema.time_unit='samples'. "
                f"Got '{events.schema.time_unit}'."
            )

        times = events.times  # [B, E]
        type_ids = events.type_ids  # [B, E]
        params = events.params  # [B, E, P]
        mask = events.mask  # [B, E]

        if times.device != signal.device:
            raise ValueError(
                "EventRenderView requires events and signal on the same device. "
                f"events.device={times.device}, signal.device={signal.device}."
            )

        used_type_ids = torch.unique(type_ids[mask])  # [U]
        supported_names: set[str] = {
            "spike",
            "level_change",
            "gaussian",
            "rect_pulse",
            "exp_decay",
            "ringdown",
        }
        unsupported: list[str] = []
        for raw_id in used_type_ids.tolist():
            if raw_id < 0 or raw_id >= len(events.schema.type_names):
                raise ValueError("EventRenderView found type_ids outside schema range.")
            name = events.schema.type_names[int(raw_id)]
            if name not in supported_names:
                unsupported.append(name)
        if unsupported:
            raise ValueError(
                "EventRenderView found unsupported event types: "
                f"{sorted(set(unsupported))}. Supported: {sorted(supported_names)}."
            )

        amplitude_index = events.schema.param_id(self.amplitude_param)
        amplitude = params[:, :, amplitude_index]  # [B, E]

        times_rounded = self._round_times(times)  # [B, E]
        times_idx = times_rounded.to(torch.int64)  # [B, E]
        if torch.any((times_idx < 0) & mask):
            raise ValueError("EventRenderView found negative event times.")
        if torch.any((times_idx >= self.seq_len) & mask):
            raise ValueError("EventRenderView found event times >= seq_len.")

        out = torch.zeros((batch_size, self.seq_len), device=signal.device, dtype=torch.float32)  # [B, L]

        schema_type_names = events.schema.type_names
        schema_param_names = events.schema.param_names

        def type_id(name: str) -> int | None:
            if name not in schema_type_names:
                return None
            return schema_type_names.index(name)

        # --- spike: scatter-add amplitude into sample index ---
        spike_id = type_id("spike")
        if spike_id is not None:
            spike_mask = mask & (type_ids == spike_id)  # [B, E]
            valid = torch.nonzero(spike_mask, as_tuple=False)  # [V, 2]
            if valid.numel() > 0:
                batch_sel = valid[:, 0]  # [V]
                event_sel = valid[:, 1]  # [V]
                t_sel = times_idx[batch_sel, event_sel]  # [V]
                amp_sel = amplitude[batch_sel, event_sel].to(torch.float32)  # [V]
                flat = torch.zeros(
                    (batch_size * self.seq_len,),
                    device=signal.device,
                    dtype=torch.float32,
                )  # [B*L]
                linear_idx = batch_sel * self.seq_len + t_sel  # [V]
                flat.index_add_(0, linear_idx, amp_sel)
                out = out + flat.view(batch_size, self.seq_len)  # [B, L]

        # --- level_change: impulse + cumsum ---
        level_id = type_id("level_change")
        if level_id is not None:
            level_mask = mask & (type_ids == level_id)  # [B, E]
            valid = torch.nonzero(level_mask, as_tuple=False)  # [V, 2]
            if valid.numel() > 0:
                batch_sel = valid[:, 0]  # [V]
                event_sel = valid[:, 1]  # [V]
                t_sel = times_idx[batch_sel, event_sel]  # [V]
                amp_sel = amplitude[batch_sel, event_sel].to(torch.float32)  # [V]
                diff_flat = torch.zeros(
                    (batch_size * self.seq_len,),
                    device=signal.device,
                    dtype=torch.float32,
                )  # [B*L]
                linear_idx = batch_sel * self.seq_len + t_sel  # [V]
                diff_flat.index_add_(0, linear_idx, amp_sel)
                diff = diff_flat.view(batch_size, self.seq_len)  # [B, L]
                out = out + torch.cumsum(diff, dim=1)  # [B, L]

        # --- rect_pulse: diff impulse + cumsum ---
        rect_id = type_id("rect_pulse")
        if rect_id is not None:
            duration_index = self._require_param_index(
                schema_param_names=schema_param_names,
                param_name=self.duration_sec_param,
                required_for="duration_sec",
            )
            sample_rate_hz = self._require_sample_rate_hz(process_meta)
            rect_mask = mask & (type_ids == rect_id)  # [B, E]
            valid = torch.nonzero(rect_mask, as_tuple=False)  # [V, 2]
            if valid.numel() > 0:
                batch_sel = valid[:, 0]  # [V]
                event_sel = valid[:, 1]  # [V]
                t0 = times_idx[batch_sel, event_sel]  # [V]
                amp_sel = amplitude[batch_sel, event_sel].to(torch.float32)  # [V]
                duration_sec = params[batch_sel, event_sel, duration_index].to(torch.float32)  # [V]
                if torch.any(duration_sec <= 0):
                    raise ValueError("EventRenderView rect_pulse duration_sec must be positive.")
                duration_samples = torch.round(duration_sec * sample_rate_hz).to(torch.int64)  # [V]
                if torch.any(duration_samples <= 0):
                    raise ValueError("EventRenderView rect_pulse duration_samples must be positive.")
                t1 = t0 + duration_samples  # [V]
                if torch.any(t1 > self.seq_len):
                    raise ValueError("EventRenderView rect_pulse end time exceeds seq_len.")

                diff_flat = torch.zeros(
                    (batch_size * (self.seq_len + 1),),
                    device=signal.device,
                    dtype=torch.float32,
                )  # [B*(L+1)]
                start_linear = batch_sel * (self.seq_len + 1) + t0  # [V]
                end_linear = batch_sel * (self.seq_len + 1) + t1  # [V]
                diff_flat.index_add_(0, start_linear, amp_sel)
                diff_flat.index_add_(0, end_linear, -amp_sel)
                diff = diff_flat.view(batch_size, self.seq_len + 1)  # [B, L+1]
                pulse = torch.cumsum(diff, dim=1)[:, : self.seq_len]  # [B, L]
                out = out + pulse

        # --- gaussian: sum over events (vectorized over valid events) ---
        gaussian_id = type_id("gaussian")
        if gaussian_id is not None:
            sigma_index = self._require_param_index(
                schema_param_names=schema_param_names,
                param_name=self.sigma_sec_param,
                required_for="sigma_sec",
            )
            sample_rate_hz = self._require_sample_rate_hz(process_meta)
            gaussian_mask = mask & (type_ids == gaussian_id)  # [B, E]
            valid = torch.nonzero(gaussian_mask, as_tuple=False)  # [V, 2]
            if valid.numel() > 0:
                batch_sel = valid[:, 0]  # [V]
                event_sel = valid[:, 1]  # [V]
                t0 = times[batch_sel, event_sel].to(torch.float32)  # [V]
                amp_sel = amplitude[batch_sel, event_sel].to(torch.float32)  # [V]
                sigma_sec = params[batch_sel, event_sel, sigma_index].to(torch.float32)  # [V]
                if torch.any(sigma_sec <= 0):
                    raise ValueError("EventRenderView gaussian sigma_sec must be positive.")
                sigma_samples = sigma_sec * sample_rate_hz  # [V]
                dt = self.time_samples[None, :] - t0[:, None]  # [V, L]
                gauss = amp_sel[:, None] * torch.exp(-0.5 * (dt / sigma_samples[:, None]).pow(2))  # [V, L]
                out.index_add_(0, batch_sel, gauss)

        # --- exp_decay ---
        decay_id = type_id("exp_decay")
        if decay_id is not None:
            tau_index = self._require_param_index(
                schema_param_names=schema_param_names,
                param_name=self.tau_sec_param,
                required_for="tau_sec",
            )
            sample_rate_hz = self._require_sample_rate_hz(process_meta)
            decay_mask = mask & (type_ids == decay_id)  # [B, E]
            valid = torch.nonzero(decay_mask, as_tuple=False)  # [V, 2]
            if valid.numel() > 0:
                batch_sel = valid[:, 0]  # [V]
                event_sel = valid[:, 1]  # [V]
                t0 = times[batch_sel, event_sel].to(torch.float32)  # [V]
                amp_sel = amplitude[batch_sel, event_sel].to(torch.float32)  # [V]
                tau_sec = params[batch_sel, event_sel, tau_index].to(torch.float32)  # [V]
                if torch.any(tau_sec <= 0):
                    raise ValueError("EventRenderView exp_decay tau_sec must be positive.")
                dt_samples = self.time_samples[None, :] - t0[:, None]  # [V, L]
                dt_sec = dt_samples / sample_rate_hz  # [V, L]
                active = dt_sec >= 0  # [V, L]
                dt_sec_pos = torch.clamp(dt_sec, min=0.0)  # [V, L]
                decay = (
                    amp_sel[:, None]
                    * torch.exp(-dt_sec_pos / tau_sec[:, None])
                    * active.to(torch.float32)
                )  # [V, L]
                out.index_add_(0, batch_sel, decay)

        # --- ringdown ---
        ring_id = type_id("ringdown")
        if ring_id is not None:
            tau_index = self._require_param_index(
                schema_param_names=schema_param_names,
                param_name=self.tau_sec_param,
                required_for="tau_sec",
            )
            freq_index = self._require_param_index(
                schema_param_names=schema_param_names,
                param_name=self.frequency_hz_param,
                required_for="frequency_hz",
            )
            phase_index = self._require_param_index(
                schema_param_names=schema_param_names,
                param_name=self.phase_param,
                required_for="phase",
            )
            sample_rate_hz = self._require_sample_rate_hz(process_meta)
            ring_mask = mask & (type_ids == ring_id)  # [B, E]
            valid = torch.nonzero(ring_mask, as_tuple=False)  # [V, 2]
            if valid.numel() > 0:
                batch_sel = valid[:, 0]  # [V]
                event_sel = valid[:, 1]  # [V]
                t0 = times[batch_sel, event_sel].to(torch.float32)  # [V]
                amp_sel = amplitude[batch_sel, event_sel].to(torch.float32)  # [V]
                tau_sec = params[batch_sel, event_sel, tau_index].to(torch.float32)  # [V]
                freq_hz = params[batch_sel, event_sel, freq_index].to(torch.float32)  # [V]
                phase = params[batch_sel, event_sel, phase_index].to(torch.float32)  # [V]
                if torch.any(tau_sec <= 0):
                    raise ValueError("EventRenderView ringdown tau_sec must be positive.")
                if torch.any(freq_hz <= 0):
                    raise ValueError("EventRenderView ringdown frequency_hz must be positive.")

                dt_samples = self.time_samples[None, :] - t0[:, None]  # [V, L]
                dt_sec = dt_samples / sample_rate_hz  # [V, L]
                active = dt_sec >= 0  # [V, L]
                dt_sec_pos = torch.clamp(dt_sec, min=0.0)  # [V, L]
                env = torch.exp(-dt_sec_pos / tau_sec[:, None]) * active.to(torch.float32)  # [V, L]
                arg = 2.0 * math.pi * freq_hz[:, None] * dt_sec_pos + phase[:, None]  # [V, L]
                ring = amp_sel[:, None] * env * torch.sin(arg)  # [V, L]
                out.index_add_(0, batch_sel, ring)

        if self.enabled_key is not None:
            out = out * enabled_mask[:, None].to(dtype=out.dtype)  # [B, L]

        rendered = out[:, None, :]  # [B, 1, L]
        observed_signal = signal + rendered  # [B, 1, L] (broadcast OK)

        counts: dict[str, torch.Tensor] = {}
        for name in supported_names:
            tid = type_id(name)
            if tid is None:
                continue
            counts[name] = (mask & (type_ids == tid)).sum(dim=1)  # [B]

        meta = {
            "view": "EventRenderView",
            "rounding": self.rounding,
            "enabled_key": self.enabled_key,
            "counts": counts,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)
