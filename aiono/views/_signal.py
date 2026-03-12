from __future__ import annotations

from typing import Any

import torch

from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta, utils_sum_latent


def views_extract_signal(
    input_state: LatentState | Observation,
    *,
    name: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    """Extract a signal `[B, C, L]`, labels, and process meta from an input state.

    - If input is `LatentState`, sums latent components into a single-channel signal.
    - If input is `Observation`, returns the observation signal and process meta.
    """
    if not name:
        raise ValueError("name must be non-empty.")

    if isinstance(input_state, LatentState):
        if input_state.latent is None:
            raise ValueError(f"{name} requires LatentState.latent to be present.")
        signal = utils_sum_latent(input_state.latent)  # [B, 1, L]
        return signal, input_state.y, input_state.meta

    if isinstance(input_state, Observation):
        process_meta = utils_extract_process_meta(input_state.meta)
        return input_state.x, input_state.y, process_meta

    raise TypeError(
        f"{name} expects LatentState or Observation, got {type(input_state).__name__}."
    )

