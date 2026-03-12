from __future__ import annotations

import torch


def morph_gaussian(relative_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian kernel evaluated at relative_t."""

    scaled = relative_t / sigma  # [B, N, K, L]
    values = torch.exp(-0.5 * scaled.pow(2))  # [B, N, K, L]
    return values


def morph_laplace(relative_t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Laplace kernel evaluated at relative_t."""

    scaled = relative_t.abs() / scale  # [B, N, K, L]
    values = torch.exp(-scaled)  # [B, N, K, L]
    return values


def morph_dog(
    relative_t: torch.Tensor,
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Difference of Gaussians (DoG) kernel."""

    scaled1 = relative_t / sigma1  # [B, N, K, L]
    scaled2 = relative_t / sigma2  # [B, N, K, L]
    gauss1 = torch.exp(-0.5 * scaled1.pow(2))  # [B, N, K, L]
    gauss2 = torch.exp(-0.5 * scaled2.pow(2))  # [B, N, K, L]
    values = gauss1 - alpha * gauss2  # [B, N, K, L]
    return values
