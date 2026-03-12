from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

from ..core.events import EventBatch
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View


class EventStreamView(View):
    """Expose event streams as an Observation tensor.

    Output format: x is [B, E, 2+P] with columns [time, type_id, params...].
    The event mask and schema are stored in meta.
    """

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Pack events into a tensor observation."""
        if isinstance(input_state, Observation):
            raise TypeError("EventStreamView expects LatentState, got Observation.")
        if input_state.events is None:
            raise ValueError("EventStreamView requires LatentState.events to be present.")

        events = input_state.events
        times = events.times  # [B, E]
        type_ids = events.type_ids.to(times.dtype)  # [B, E]
        params = events.params  # [B, E, P]

        times_col = times[:, :, None]  # [B, E, 1]
        type_col = type_ids[:, :, None]  # [B, E, 1]
        packed = torch.cat([times_col, type_col, params], dim=2)  # [B, E, 2+P]

        meta = {
            "view": "EventStreamView",
            "schema": events.schema,
            "mask": events.mask,
            "process": input_state.meta,
        }
        return Observation(x=packed, y=input_state.y, meta=meta)


class EventImpulseView(View):
    """Convert event streams into per-type impulse trains."""

    def __init__(
        self,
        *,
        seq_len: int,
        amplitude_param: str,
        rounding: Literal["nearest", "floor", "ceil"],
    ) -> None:
        """Configure impulse rendering."""
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if rounding not in {"nearest", "floor", "ceil"}:
            raise ValueError(f"rounding must be nearest/floor/ceil, got {rounding}.")
        self.seq_len = seq_len
        self.amplitude_param = amplitude_param
        self.rounding = rounding

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Render an EventBatch into per-type impulses."""
        if isinstance(input_state, Observation):
            raise TypeError("EventImpulseView expects LatentState, got Observation.")
        if input_state.events is None:
            raise ValueError("EventImpulseView requires LatentState.events to be present.")

        events = input_state.events
        if events.schema.time_unit != "samples":
            raise ValueError(
                "EventImpulseView expects time_unit='samples'. "
                f"Got '{events.schema.time_unit}'."
            )

        amplitude_index = events.schema.param_id(self.amplitude_param)

        times = events.times  # [B, E]
        if self.rounding == "nearest":
            times = torch.round(times)  # [B, E]
        elif self.rounding == "floor":
            times = torch.floor(times)  # [B, E]
        else:
            times = torch.ceil(times)  # [B, E]
        times_idx = times.to(torch.int64)  # [B, E]

        mask = events.mask  # [B, E]
        if torch.any((times_idx < 0) & mask):
            raise ValueError("EventImpulseView found negative event times.")
        if torch.any((times_idx >= self.seq_len) & mask):
            raise ValueError("EventImpulseView found event times >= seq_len.")

        type_ids = events.type_ids  # [B, E]
        num_types = len(events.schema.type_names)
        if torch.any((type_ids < 0) | (type_ids >= num_types)):
            raise ValueError("EventImpulseView found type_ids outside schema range.")

        amplitude = events.params[:, :, amplitude_index]  # [B, E]
        amplitude = amplitude.masked_fill(~mask, 0.0)  # [B, E]

        batch_size, num_events = times_idx.shape
        device = times_idx.device

        batch_idx = torch.arange(batch_size, device=device)  # [B]
        batch_idx = batch_idx.repeat_interleave(num_events)  # [B*E]

        time_flat = times_idx.reshape(-1)  # [B*E]
        type_flat = type_ids.reshape(-1)  # [B*E]
        amp_flat = amplitude.reshape(-1)  # [B*E]
        mask_flat = mask.reshape(-1)  # [B*E]

        valid = torch.nonzero(mask_flat, as_tuple=False).flatten()  # [V]
        if valid.numel() == 0:
            impulse = torch.zeros(
                (batch_size, num_types, self.seq_len),
                device=device,
                dtype=amplitude.dtype,
            )  # [B, T, L]
        else:
            batch_sel = batch_idx[valid]  # [V]
            time_sel = time_flat[valid]  # [V]
            type_sel = type_flat[valid]  # [V]
            amp_sel = amp_flat[valid]  # [V]

            flat_size = batch_size * num_types * self.seq_len
            flat = torch.zeros(
                (flat_size,),
                device=device,
                dtype=amplitude.dtype,
            )  # [B*T*L]
            linear_idx = (batch_sel * num_types + type_sel) * self.seq_len + time_sel  # [V]
            flat.index_add_(0, linear_idx, amp_sel)
            impulse = flat.view(batch_size, num_types, self.seq_len)  # [B, T, L]

        meta = {
            "view": "EventImpulseView",
            "rounding": self.rounding,
            "process": input_state.meta,
        }
        return Observation(x=impulse, y=input_state.y, meta=meta)


class KernelConvView(View):
    """Apply a kernel bank to an impulse observation via conv1d."""

    def __init__(
        self,
        *,
        kernels: torch.Tensor,
        padding: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Configure the kernel bank and convolution padding."""
        super().__init__()
        if kernels.ndim != 3:
            raise ValueError(
                "kernels must have shape [K, T, W]. "
                f"Got {kernels.shape}."
            )
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}.")
        if bias is not None and bias.ndim != 1:
            raise ValueError(f"bias must have shape [K], got {bias.shape}.")
        if bias is not None and bias.shape[0] != kernels.shape[0]:
            raise ValueError(
                "bias length must match kernels out_channels. "
                f"bias.shape={bias.shape}, kernels.shape={kernels.shape}."
            )

        self.register_buffer("kernels", kernels.float())
        if bias is not None:
            self.register_buffer("bias", bias.float())
        else:
            self.bias = None
        self.padding = padding

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Convolve impulse trains into dense components."""
        if isinstance(input_state, LatentState):
            raise TypeError("KernelConvView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        signal = input_state.x  # [B, T, L]
        if signal.ndim != 3:
            raise ValueError(
                "KernelConvView expects input x with shape [B, T, L]. "
                f"Got {signal.shape}."
            )
        if signal.shape[1] != self.kernels.shape[1]:
            raise ValueError(
                "KernelConvView input channels do not match kernels. "
                f"x.shape={signal.shape}, kernels.shape={self.kernels.shape}."
            )

        observed = F.conv1d(
            signal,
            self.kernels,
            bias=self.bias,
            stride=1,
            padding=self.padding,
            dilation=1,
            groups=1,
        )  # [B, K, L]

        meta = {
            "view": "KernelConvView",
            "padding": self.padding,
            "kernel_size": int(self.kernels.shape[2]),
            "process": process_meta,
        }
        return Observation(x=observed, y=input_state.y, meta=meta)
