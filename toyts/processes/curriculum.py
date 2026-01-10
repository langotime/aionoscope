from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..core.curriculum import CurriculumSchedule, curriculum_sample_stage_id
from ..core.rng import rng_make_generator, rng_split
from ..core.types import LatentState
from .base import Process


@dataclass(frozen=True)
class CurriculumChoice:
    """A sampled curriculum choice for a batch."""

    step: int
    stage_id: int
    stage_name: str
    stage_probs: list[float]
    stage_seed: int


class CurriculumProcess(Process):
    """Select among multiple processes according to a curriculum schedule.

    This process chooses a *single* stage per batch according to a
    `CurriculumSchedule` and forwards generation to the corresponding stage
    process. This is the simplest curriculum integration pattern and keeps the
    process GPU-friendly (no per-sample branching/scattering).

    The chosen stage id is injected into labels under `stage_label_key`.
    """

    def __init__(
        self,
        *,
        stages: dict[str, nn.Module],
        schedule: CurriculumSchedule,
        stage_label_key: str,
        initial_step: int,
    ) -> None:
        """Initialize curriculum selection over a set of stages."""
        super().__init__()
        if not stages:
            raise ValueError("stages must be non-empty.")
        if list(stages.keys()) != schedule.stage_names:
            raise ValueError(
                "stages keys must match schedule.stage_names exactly. "
                f"stages={list(stages.keys())}, schedule={schedule.stage_names}."
            )
        if not stage_label_key:
            raise ValueError("stage_label_key must be non-empty.")
        if initial_step < 0:
            raise ValueError(f"initial_step must be non-negative, got {initial_step}.")

        self.schedule = schedule
        self.stage_label_key = stage_label_key
        self.stage_names = list(stages.keys())
        self.stages = nn.ModuleList([stages[name] for name in self.stage_names])
        self._step = initial_step

    @property
    def step(self) -> int:
        """Current curriculum step used to query the schedule."""
        return self._step

    def set_step(self, step: int) -> None:
        """Update the curriculum step (e.g., from the training loop)."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}.")
        self._step = step

    def _choose_stage(
        self,
        *,
        rng: torch.Generator,
        device: torch.device,
    ) -> CurriculumChoice:
        """Sample a stage id for the current step and return selection metadata."""
        probs = self.schedule.probs(step=self._step, device=device)  # [S]
        stage_id = curriculum_sample_stage_id(probs=probs, rng=rng)
        stage_name = self.stage_names[stage_id]

        return CurriculumChoice(
            step=self._step,
            stage_id=stage_id,
            stage_name=stage_name,
            stage_probs=probs.detach().cpu().tolist(),
            stage_seed=int(rng.initial_seed()),
        )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        """Generate a batch from one of the stage processes."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        generator, seed, _ = rng_make_generator(rng=rng, device=device)
        stage_rng, process_rng = rng_split(rng=generator, num_children=2, device=device)

        choice = self._choose_stage(rng=stage_rng, device=device)
        stage = self.stages[choice.stage_id]
        stage_state = stage(batch_size, device, rng=process_rng)
        if not isinstance(stage_state, LatentState):
            raise TypeError(
                "CurriculumProcess stage must return LatentState. "
                f"Got {type(stage_state).__name__}."
            )

        if self.stage_label_key in stage_state.y:
            raise ValueError(
                f"Stage process already produced label '{self.stage_label_key}'. "
                "Pick a different stage_label_key."
            )

        stage_label = torch.full(
            (batch_size,),
            fill_value=choice.stage_id,
            device=device,
            dtype=torch.int64,
        )  # [B]

        y = dict(stage_state.y)
        y[self.stage_label_key] = stage_label

        meta: dict[str, Any] = dict(stage_state.meta)
        label_names = meta.setdefault("label_names", {})
        label_names[self.stage_label_key] = self.stage_names

        meta = {
            "process": "CurriculumProcess",
            "seed": seed,
            "step": choice.step,
            "stage_id": choice.stage_id,
            "stage_name": choice.stage_name,
            "stage_probs": choice.stage_probs,
            "stage_seed": choice.stage_seed,
            "stage_process": meta,
            "schedule": {
                "stage_names": self.schedule.stage_names,
                "breakpoints": self.schedule.breakpoints,
            },
        }

        return LatentState(
            centers=stage_state.centers,
            latent=stage_state.latent,
            events=stage_state.events,
            y=y,
            meta=meta,
        )

