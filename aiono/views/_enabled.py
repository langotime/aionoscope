from __future__ import annotations

from typing import Any

import torch


def views_resolve_enabled_mask(
    process_meta: dict[str, Any],
    *,
    enabled_key: str | None,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Resolve an enabled mask `bool[B]` from process meta.

    If `enabled_key is None`, returns all-True.
    """
    if not name:
        raise ValueError("name must be non-empty.")
    if enabled_key is None:
        return torch.ones((batch_size,), device=device, dtype=torch.bool)  # [B]
    if not enabled_key:
        raise ValueError(f"{name} enabled_key must be non-empty when provided.")

    enabled = process_meta.get("enabled")
    if not isinstance(enabled, dict):
        raise ValueError(
            f"{name} requires process_meta['enabled'] to be a dict when enabled_key is set. "
            f"Got {type(enabled).__name__}."
        )
    mask = enabled.get(enabled_key)
    if not isinstance(mask, torch.Tensor):
        raise ValueError(
            f"{name} enabled mask must be a torch.Tensor. Got {type(mask).__name__}."
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} enabled mask must be bool, got {mask.dtype}.")
    if mask.shape != (batch_size,):
        raise ValueError(
            f"{name} enabled mask must have shape [B]. Got {mask.shape}, batch_size={batch_size}."
        )
    if mask.device != device:
        raise ValueError(
            f"{name} enabled mask device mismatch. mask.device={mask.device}, expected={device}."
        )
    return mask

