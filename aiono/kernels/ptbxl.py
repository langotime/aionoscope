from __future__ import annotations

import math

import torch

from ..core.utils import utils_make_canonical_A0
from ..ptbxl.phenotypes import PTBXL_EVENT_TYPE_NAMES, PTBXL_LEAD_NAMES, PTBXL_LOCATIONS


def ptbxl_kernel_size(*, sample_rate_hz: float, support_ms: float) -> int:
    """Compute an odd kernel size covering the requested support window."""
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
    if support_ms <= 0:
        raise ValueError(f"support_ms must be positive, got {support_ms}.")
    radius = int(math.ceil(sample_rate_hz * (support_ms / 1000.0) * 0.5))
    return int(radius * 2 + 1)


def _gaussian(time_grid: torch.Tensor, *, sigma: float) -> torch.Tensor:
    return torch.exp(-0.5 * (time_grid / sigma).pow(2))


def _smooth_box(
    time_grid: torch.Tensor,
    *,
    start: float,
    end: float,
    edge: float,
) -> torch.Tensor:
    return 0.5 * (torch.tanh((time_grid - start) / edge) - torch.tanh((time_grid - end) / edge))


def make_ptbxl_kernel_bank(
    *,
    sample_rate_hz: float,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a [K=12, T, W] kernel bank for PTB-XL ECG events."""
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for centered kernels.")

    num_leads = len(PTBXL_LEAD_NAMES)
    num_types = len(PTBXL_EVENT_TYPE_NAMES)

    center = kernel_size // 2
    time_grid = (
        torch.arange(kernel_size, device=device, dtype=dtype) - center
    ) / sample_rate_hz  # [W]

    p_wave = _gaussian(time_grid, sigma=0.04)  # [W]
    qrs_wave = _gaussian(time_grid, sigma=0.015)  # [W]
    t_wave = _gaussian(time_grid, sigma=0.08)  # [W]

    qrs_wide = _gaussian(time_grid, sigma=0.03)  # [W]
    qrs_delta = _gaussian(time_grid + 0.04, sigma=0.02)  # [W]
    qrs_qwave = _gaussian(time_grid + 0.03, sigma=0.02)  # [W]

    st_shift = _smooth_box(time_grid, start=0.06, end=0.18, edge=0.01)  # [W]
    pace_spike = _gaussian(time_grid, sigma=0.003)  # [W]
    flutter_wave = _gaussian(time_grid, sigma=0.02)  # [W]

    base_A0 = utils_make_canonical_A0(num_leads=num_leads, num_latent=3).to(device=device, dtype=dtype)  # [C, 3]
    p_weights = base_A0[:, 0]  # [C]
    qrs_weights = base_A0[:, 1]  # [C]
    t_weights = base_A0[:, 2]  # [C]

    axis_left = torch.tensor(
        [1.0, 0.3, -0.5, -0.2, 0.8, -0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        device=device,
        dtype=dtype,
    )  # [C]
    axis_right = torch.tensor(
        [-0.6, 0.5, 0.9, -0.1, -0.4, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        device=device,
        dtype=dtype,
    )  # [C]
    qrs_lvh = torch.tensor(
        [1.0, 0.6, 0.4, -0.2, 0.9, 0.3, 0.2, 0.2, 0.4, 0.7, 1.0, 1.0],
        device=device,
        dtype=dtype,
    )  # [C]
    qrs_rvh = torch.tensor(
        [0.2, 0.4, 0.6, -0.1, 0.2, 0.5, 1.0, 0.9, 0.4, 0.2, 0.1, 0.1],
        device=device,
        dtype=dtype,
    )  # [C]
    qrs_septal = torch.tensor(
        [0.2, 0.4, 0.5, -0.1, 0.2, 0.3, 1.0, 0.8, 0.4, 0.2, 0.1, 0.1],
        device=device,
        dtype=dtype,
    )  # [C]
    p_lae = torch.tensor(
        [0.7, 1.0, 0.4, -0.2, 0.6, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        device=device,
        dtype=dtype,
    )  # [C]
    p_rae = torch.tensor(
        [0.3, 0.8, 0.9, -0.1, 0.2, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        device=device,
        dtype=dtype,
    )  # [C]

    kernels = torch.zeros(
        (num_leads, num_types, kernel_size),
        device=device,
        dtype=dtype,
    )  # [K, T, W]

    def _set(event: str, weights: torch.Tensor, wave: torch.Tensor) -> None:
        idx = PTBXL_EVENT_TYPE_NAMES.index(event)
        kernels[:, idx, :] = weights[:, None] * wave[None, :]

    _set("p", p_weights, p_wave)
    _set("qrs", qrs_weights, qrs_wave)
    _set("t", t_weights, t_wave)
    _set("qrs_wide", qrs_weights, qrs_wide)
    _set("qrs_delta", qrs_weights, qrs_delta)
    _set("qrs_axis_left", axis_left, qrs_wave)
    _set("qrs_axis_right", axis_right, qrs_wave)
    _set("qrs_lvh", qrs_lvh, qrs_wave)
    _set("qrs_rvh", qrs_rvh, qrs_wave)
    _set("qrs_septal", qrs_septal, qrs_wave)
    _set("p_lae", p_lae, p_wave)
    _set("p_rae", p_rae, p_wave)

    for loc, leads in PTBXL_LOCATIONS.items():
        mask = torch.zeros((num_leads,), device=device, dtype=dtype)  # [C]
        mask[leads] = 1.0
        _set(f"qrs_qwave_{loc}", mask, qrs_qwave)
        _set(f"st_shift_{loc}", mask, st_shift)
        _set(f"t_invert_{loc}", mask, t_wave)

    _set("pace_spike", qrs_weights, pace_spike)
    _set("flutter_wave", p_weights, flutter_wave)

    return kernels
