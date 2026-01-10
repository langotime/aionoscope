from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .events import EventBatch

@dataclass(frozen=True)
class LatentState:
    """Latent process state produced by a Process module."""

    centers: torch.Tensor  # [B, N]
    latent: torch.Tensor | None  # [B, K, L]
    events: EventBatch | None
    y: dict[str, torch.Tensor]  # label tensors, e.g. {"shape": [B]}
    meta: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    """Observed signal produced by a View module."""

    x: torch.Tensor  # [B, C, L]
    y: dict[str, torch.Tensor]  # label tensors, e.g. {"shape": [B]}
    meta: dict[str, Any]

    def view_meta(self, view_name: str) -> dict[str, Any]:
        """Return metadata for a specific view by name."""
        if not view_name:
            raise ValueError("view_name must be non-empty.")
        if "views" not in self.meta:
            raise ValueError(
                "Observation meta is missing 'views'. "
                "Ensure the observation comes from a ViewChain."
            )
        views = self.meta["views"]
        if not isinstance(views, list):
            raise ValueError(
                "Observation meta 'views' must be a list. "
                f"Got {type(views).__name__}."
            )
        matches = []
        for entry in views:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Observation meta 'views' entries must be dicts. "
                    f"Got {type(entry).__name__}."
                )
            if "view" not in entry:
                raise ValueError(
                    "Observation meta 'views' entries must include 'view'."
                )
            if entry["view"] == view_name:
                matches.append(entry)
        if not matches:
            raise ValueError(f"No view meta found for view '{view_name}'.")
        if len(matches) > 1:
            raise ValueError(
                f"Found multiple view meta entries for view '{view_name}'. "
                "Use Observation.meta['views'] to disambiguate."
            )
        return matches[0]
