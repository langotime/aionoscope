from __future__ import annotations

from typing import Any

import torch

from .rng import rng_make_generator
from .types import LatentState, Observation

SAMPLES_PREFIX = "__samples__"


def utils_require_latent(z: LatentState | Observation, name: str) -> LatentState:
    """Return LatentState or raise a TypeError with context."""
    if isinstance(z, LatentState):
        return z
    raise TypeError(f"{name} expects LatentState, got {type(z).__name__}.")


def utils_require_observation(z: LatentState | Observation, name: str) -> Observation:
    """Return Observation or raise a TypeError with context."""
    if isinstance(z, Observation):
        return z
    raise TypeError(f"{name} expects Observation, got {type(z).__name__}.")


def utils_sum_latent(latent: torch.Tensor) -> torch.Tensor:
    """Sum latent components into a single-channel signal."""

    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [B, K, L], got {latent.shape}.")

    signal = latent.sum(dim=1, keepdim=True)  # [B, 1, L]
    return signal


def utils_make_canonical_A0(num_leads: int, num_latent: int) -> torch.Tensor:
    """Create a deterministic mixing matrix with shape [C, K]."""
    if num_leads <= 0 or num_latent <= 0:
        raise ValueError("num_leads and num_latent must be positive.")

    lead_grid = torch.linspace(0.1, 1.0, steps=num_leads)[:, None]  # [C, 1]
    latent_grid = torch.linspace(1.0, 0.5, steps=num_latent)[None, :]  # [1, K]
    mixing_matrix = torch.cos(lead_grid * latent_grid * torch.pi)  # [C, K]
    return mixing_matrix


def utils_make_random_A0(
    num_leads: int,
    num_latent: int,
    *,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample a random mixing matrix with shape [C, K]."""
    if num_leads <= 0 or num_latent <= 0:
        raise ValueError("num_leads and num_latent must be positive.")
    if rng is None:
        raise ValueError("rng must be provided to generate random A0.")

    generator, _, _ = rng_make_generator(rng=rng, device=device)
    mixing_matrix = torch.randn(
        (num_leads, num_latent),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )  # [C, K]
    return mixing_matrix


def utils_extract_process_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract the process metadata dict from an observation meta payload."""
    if "process" not in meta:
        raise ValueError(
            "Observation meta is missing 'process'. "
            "Ensure views preserve process metadata."
        )
    process_meta = meta["process"]
    if not isinstance(process_meta, dict):
        raise ValueError(
            "Observation meta 'process' must be a dict. "
            f"Got {type(process_meta).__name__}."
        )
    return process_meta
