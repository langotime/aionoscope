from __future__ import annotations

import torch

from .pqrst import make_pqrst_kernel_bank

ECG_EVENT_TYPE_NAMES = ["beat", "pace_spike", "flutter_wave"]


def make_ecg_kernel_bank(
    *,
    spacing: float,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a [K, T, W] kernel bank matching ECG_EVENT_TYPE_NAMES.

    - Event type `beat` uses a simple PQRST-like kernel (gaussian variant).
    - Event type `pace_spike` uses a narrow spike on the QRS component.
    - Event type `flutter_wave` uses a narrow atrial-like bump on the P component.
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}.")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for centered kernels.")

    beat = make_pqrst_kernel_bank(
        shape_names=["gaussian"],
        spacing=spacing,
        kernel_size=kernel_size,
        device=device,
        dtype=dtype,
    )  # [K=3, T=1, W]

    kernels = torch.zeros(
        (3, len(ECG_EVENT_TYPE_NAMES), kernel_size),
        device=device,
        dtype=dtype,
    )  # [K=3, T=3, W]
    kernels[:, 0, :] = beat[:, 0, :]  # [K=3, W]

    center = kernel_size // 2
    time_grid = (
        torch.arange(kernel_size, device=device, dtype=dtype) - center
    )  # [W]

    sigma_spike = torch.tensor(1.0, device=device, dtype=dtype)  # []
    pace_spike = torch.exp(-0.5 * (time_grid / sigma_spike).pow(2))  # [W]
    kernels[1, 1, :] = pace_spike  # QRS component, `pace_spike` type

    sigma_flutter = torch.tensor(0.05 * spacing, device=device, dtype=dtype)  # []
    flutter_wave = torch.exp(-0.5 * (time_grid / sigma_flutter).pow(2))  # [W]
    kernels[0, 2, :] = flutter_wave  # P component, `flutter_wave` type

    return kernels

