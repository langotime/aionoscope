from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class EventSchema:
    """Schema describing event types and parameters."""

    type_names: list[str]
    param_names: list[str]
    time_unit: str

    def type_id(self, name: str) -> int:
        """Return the integer id for a named event type."""
        if name not in self.type_names:
            raise ValueError(
                f"EventSchema is missing type '{name}'. "
                f"Available: {self.type_names}."
            )
        return self.type_names.index(name)

    def param_id(self, name: str) -> int:
        """Return the integer index for a named event parameter."""
        if name not in self.param_names:
            raise ValueError(
                f"EventSchema is missing param '{name}'. "
                f"Available: {self.param_names}."
            )
        return self.param_names.index(name)


@dataclass(frozen=True)
class EventBatch:
    """A padded batch of events.

    All tensors are batch-major with a fixed padded length `E`.
    """

    times: torch.Tensor  # [B, E]
    type_ids: torch.Tensor  # [B, E]
    params: torch.Tensor  # [B, E, P]
    mask: torch.Tensor  # [B, E]
    schema: EventSchema
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate tensor shapes and dtypes."""
        if self.times.ndim != 2:
            raise ValueError(f"times must have shape [B, E], got {self.times.shape}.")
        if self.type_ids.shape != self.times.shape:
            raise ValueError(
                "type_ids must match times shape. "
                f"type_ids.shape={self.type_ids.shape}, times.shape={self.times.shape}."
            )
        if self.mask.shape != self.times.shape:
            raise ValueError(
                "mask must match times shape. "
                f"mask.shape={self.mask.shape}, times.shape={self.times.shape}."
            )
        if self.params.ndim != 3:
            raise ValueError(f"params must have shape [B, E, P], got {self.params.shape}.")
        if self.params.shape[:2] != self.times.shape:
            raise ValueError(
                "params must match times shape in first two dims. "
                f"params.shape={self.params.shape}, times.shape={self.times.shape}."
            )
        if self.times.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"times must be float32/float64, got {self.times.dtype}.")
        if self.type_ids.dtype != torch.int64:
            raise ValueError(f"type_ids must be int64, got {self.type_ids.dtype}.")
        if self.params.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"params must be float32/float64, got {self.params.dtype}.")
        if self.mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool, got {self.mask.dtype}.")

    def to(self, device: torch.device) -> EventBatch:
        """Move all tensors to a new device."""
        return EventBatch(
            times=self.times.to(device),
            type_ids=self.type_ids.to(device),
            params=self.params.to(device),
            mask=self.mask.to(device),
            schema=self.schema,
            meta=self.meta,
        )


def events_select(base: EventBatch, other: EventBatch, mask: torch.Tensor) -> EventBatch:
    """Select per-sample events from `base` or `other` based on a batch mask."""
    if base.schema != other.schema:
        raise ValueError("EventBatch schemas do not match for selection.")
    if base.times.shape != other.times.shape:
        raise ValueError(
            "EventBatch shapes do not match for selection. "
            f"base.times.shape={base.times.shape}, other.times.shape={other.times.shape}."
        )
    if mask.ndim != 1:
        raise ValueError(f"mask must have shape [B], got {mask.shape}.")
    if mask.shape[0] != base.times.shape[0]:
        raise ValueError(
            "mask batch size does not match EventBatch. "
            f"mask.shape={mask.shape}, batch_size={base.times.shape[0]}."
        )

    mask_events = mask[:, None]  # [B, 1]
    mask_params = mask[:, None, None]  # [B, 1, 1]

    times = torch.where(mask_events, other.times, base.times)  # [B, E]
    type_ids = torch.where(mask_events, other.type_ids, base.type_ids)  # [B, E]
    params = torch.where(mask_params, other.params, base.params)  # [B, E, P]
    valid_mask = torch.where(mask_events, other.mask, base.mask)  # [B, E]

    return EventBatch(
        times=times,
        type_ids=type_ids,
        params=params,
        mask=valid_mask,
        schema=base.schema,
        meta=base.meta,
    )
