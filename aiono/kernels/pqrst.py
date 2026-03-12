from __future__ import annotations

import math

import torch


def pqrst_kernel_size(*, spacing: float, support_sigma: float) -> int:
    """Compute an odd kernel size that covers PQRST support in samples."""
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}.")
    if support_sigma <= 0:
        raise ValueError(f"support_sigma must be positive, got {support_sigma}.")

    offsets = [-0.2, 0.0, 0.25]
    sigmas = [0.08, 0.04, 0.1]
    max_offset = max(abs(value) for value in offsets) * spacing
    max_sigma = max(sigmas) * spacing
    radius = int(math.ceil(max_offset + support_sigma * max_sigma))
    return int(radius * 2 + 1)


def make_pqrst_kernel_bank(
    *,
    shape_names: list[str],
    spacing: float,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a [K, T, W] kernel bank for PQRST components and shape types."""
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}.")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for centered kernels.")
    if not shape_names:
        raise ValueError("shape_names must be non-empty.")

    offsets = torch.tensor([-0.2, 0.0, 0.25], device=device, dtype=dtype)  # [K]
    sigmas = torch.tensor([0.08, 0.04, 0.1], device=device, dtype=dtype)  # [K]
    weights = torch.tensor([0.5, 1.0, 0.45], device=device, dtype=dtype)  # [K]

    offsets = offsets * spacing  # [K]
    sigmas = sigmas * spacing  # [K]

    center = kernel_size // 2
    time_grid = (
        torch.arange(kernel_size, device=device, dtype=dtype) - center
    )  # [W]
    relative = time_grid[None, :] - offsets[:, None]  # [K, W]

    scaled = relative / sigmas[:, None]  # [K, W]
    gaussian = torch.exp(-0.5 * scaled.pow(2))  # [K, W]

    qrs_relative = relative[1:2, :]  # [1, W]
    qrs_gaussian = gaussian[1:2, :]  # [1, W]

    laplace_scale = sigmas[1] * 0.6  # []
    qrs_laplace = torch.exp(-qrs_relative.abs() / laplace_scale)  # [1, W]

    dog_sigma1 = sigmas[1] * 0.6  # []
    dog_sigma2 = sigmas[1] * 1.4  # []
    dog_alpha = 0.8
    qrs_dog = torch.exp(-0.5 * (qrs_relative / dog_sigma1).pow(2))  # [1, W]
    qrs_dog = qrs_dog - dog_alpha * torch.exp(
        -0.5 * (qrs_relative / dog_sigma2).pow(2)
    )  # [1, W]

    num_types = len(shape_names)
    kernels = torch.zeros((3, num_types, kernel_size), device=device, dtype=dtype)  # [K, T, W]

    for type_idx, name in enumerate(shape_names):
        if name == "gaussian":
            qrs = qrs_gaussian
        elif name == "sharp_laplace":
            qrs = qrs_laplace
        elif name == "biphasic_dog":
            qrs = qrs_dog
        else:
            raise ValueError(
                "Unknown shape name for PQRST kernel bank. "
                f"Got '{name}', expected one of {shape_names}."
            )

        component = gaussian.clone()  # [K, W]
        component[1:2, :] = qrs
        component = component * weights[:, None]  # [K, W]
        kernels[:, type_idx, :] = component  # [K, W]

    return kernels
