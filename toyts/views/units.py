from __future__ import annotations

import torch

from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta, utils_sum_latent
from .base import View


def _extract_signal(
    input_state: LatentState | Observation,
    name: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
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


class UnitsAbsoluteView(View):
    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = _extract_signal(
            input_state,
            name="UnitsAbsoluteView",
        )
        observed_signal = signal  # [B, C, L]

        meta = {
            "view": "UnitsAbsoluteView",
            "units": "absolute",
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class UnitsPercentOfCapacityView(View):
    def __init__(self, *, capacity_min: float, capacity_max: float) -> None:
        super().__init__()
        if capacity_min <= 0 or capacity_max <= 0:
            raise ValueError("capacity_min/max must be positive.")
        if capacity_max <= capacity_min:
            raise ValueError("capacity_max must be greater than capacity_min.")
        self.capacity_min = capacity_min
        self.capacity_max = capacity_max

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        signal, labels, process_meta = _extract_signal(
            input_state,
            name="UnitsPercentOfCapacityView",
        )
        device = signal.device

        generator, seed, _ = rng_make_generator(rng=rng, device=device)
        capacity = torch.rand(
            (signal.shape[0], 1, 1),
            generator=generator,
            device=device,
        )  # [B, 1, 1]
        capacity = self.capacity_min + (self.capacity_max - self.capacity_min) * capacity  # [B, 1, 1]

        observed_signal = 100.0 * signal / capacity  # [B, C, L]

        meta = {
            "view": "UnitsPercentOfCapacityView",
            "units": "percent",
            "seed": seed,
            "capacity": capacity,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=labels, meta=meta)


class ClippingView(View):
    def __init__(self, *, min_value: float, max_value: float) -> None:
        super().__init__()
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value.")
        self.min_value = min_value
        self.max_value = max_value

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        if isinstance(input_state, LatentState):
            raise TypeError("ClippingView expects an Observation, got LatentState.")

        process_meta = utils_extract_process_meta(input_state.meta)
        observed_signal = torch.clamp(
            input_state.x,
            min=self.min_value,
            max=self.max_value,
        )  # [B, C, L]
        meta = {
            "view": "ClippingView",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "process": process_meta,
        }
        return Observation(x=observed_signal, y=input_state.y, meta=meta)
