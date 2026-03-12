from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CurriculumSchedule:
    """Piecewise-linear schedule over a fixed set of stages.

    A schedule is defined by a set of breakpoints. Each breakpoint specifies a
    non-negative weight vector over stages. At an arbitrary `step`, weights are
    linearly interpolated between the surrounding breakpoints and normalized to
    probabilities.
    """

    stage_names: list[str]
    breakpoints: list[tuple[int, list[float]]]

    def __post_init__(self) -> None:
        """Validate schedule configuration."""
        if not self.stage_names:
            raise ValueError("CurriculumSchedule.stage_names must be non-empty.")
        if len(set(self.stage_names)) != len(self.stage_names):
            raise ValueError("CurriculumSchedule.stage_names must be unique.")
        if not self.breakpoints:
            raise ValueError("CurriculumSchedule.breakpoints must be non-empty.")

        expected_len = len(self.stage_names)
        last_step: int | None = None
        for step, weights in self.breakpoints:
            if step < 0:
                raise ValueError(f"Breakpoint step must be non-negative, got {step}.")
            if last_step is not None and step <= last_step:
                raise ValueError("Breakpoint steps must be strictly increasing.")
            last_step = step

            if len(weights) != expected_len:
                raise ValueError(
                    "Each breakpoint weight vector must match stage_names length. "
                    f"Got {len(weights)} vs {expected_len}."
                )
            if any(value < 0 for value in weights):
                raise ValueError(f"Breakpoint weights must be non-negative. Got {weights}.")
            if sum(weights) <= 0:
                raise ValueError(f"Breakpoint weights must have positive sum. Got {weights}.")

    def probs(self, *, step: int, device: torch.device) -> torch.Tensor:
        """Return normalized stage probabilities for the given step.

        Args:
            step: Curriculum step (e.g., global training step or batch index).
            device: Device for the returned probabilities.

        Returns:
            Probability vector `[S]` over `S=len(stage_names)` stages.
        """
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}.")

        steps = [value for value, _ in self.breakpoints]
        if step <= steps[0]:
            weights = self.breakpoints[0][1]
            probs = torch.tensor(weights, device=device, dtype=torch.float32)  # [S]
            return probs / probs.sum()
        if step >= steps[-1]:
            weights = self.breakpoints[-1][1]
            probs = torch.tensor(weights, device=device, dtype=torch.float32)  # [S]
            return probs / probs.sum()

        left_index = 0
        for index in range(len(steps) - 1):
            if steps[index] <= step < steps[index + 1]:
                left_index = index
                break

        left_step, left_weights = self.breakpoints[left_index]
        right_step, right_weights = self.breakpoints[left_index + 1]
        span = right_step - left_step
        if span <= 0:
            raise ValueError("Invalid schedule breakpoints: non-positive span.")

        t = float(step - left_step) / float(span)
        blended = [
            (1.0 - t) * float(left_weights[i]) + t * float(right_weights[i])
            for i in range(len(self.stage_names))
        ]
        probs = torch.tensor(blended, device=device, dtype=torch.float32)  # [S]
        if torch.any(probs < 0):
            raise ValueError(f"Interpolated weights must be non-negative. Got {blended}.")
        if probs.sum().item() <= 0:
            raise ValueError(f"Interpolated weights must have positive sum. Got {blended}.")
        return probs / probs.sum()


def curriculum_sample_stage_id(*, probs: torch.Tensor, rng: torch.Generator) -> int:
    """Sample a single stage id from probabilities.

    Args:
        probs: Probability vector `[S]` on the desired sampling device.
        rng: Torch generator used for sampling.

    Returns:
        Integer stage id in `[0, S-1]`.
    """
    if probs.ndim != 1:
        raise ValueError(f"probs must have shape [S], got {tuple(probs.shape)}.")
    if probs.numel() <= 0:
        raise ValueError("probs must be non-empty.")
    if torch.any(probs < 0):
        raise ValueError("probs must be non-negative.")
    if probs.sum().item() <= 0:
        raise ValueError("probs must have positive sum.")

    normalized = probs / probs.sum()  # [S]
    cdf = normalized.cumsum(dim=0)  # [S]
    u = torch.rand((), generator=rng, device=probs.device, dtype=torch.float32)  # []
    index = torch.searchsorted(cdf, u)  # []
    stage_id = int(index.item())
    if stage_id < 0 or stage_id >= int(probs.numel()):
        raise ValueError(
            "Sampled stage_id is out of range. "
            f"stage_id={stage_id}, num_stages={int(probs.numel())}."
        )
    return stage_id


def curriculum_stage_histogram(
    *,
    stage_ids: torch.Tensor,
    num_stages: int,
) -> torch.Tensor:
    """Compute a histogram over stage ids.

    Args:
        stage_ids: Stage indices `[N]`.
        num_stages: Total number of stages `S`.

    Returns:
        Counts `[S]` in int64.
    """
    if stage_ids.ndim != 1:
        raise ValueError(f"stage_ids must have shape [N], got {tuple(stage_ids.shape)}.")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}.")
    if torch.any((stage_ids < 0) | (stage_ids >= num_stages)):
        raise ValueError("stage_ids contain values outside [0, num_stages).")

    counts = torch.bincount(stage_ids.to(torch.int64), minlength=num_stages)  # [S]
    return counts
