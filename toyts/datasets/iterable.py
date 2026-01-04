from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from ..core.pipeline import SynthPipeline
from ..core.types import Observation


class SynthBatchIterableDataset(IterableDataset[dict[str, Observation]]):
    def __init__(
        self,
        *,
        pipeline: SynthPipeline,
        batch_size: int,
        device: torch.device,
        seed: int,
        max_batches: int | None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}.")
        if max_batches is not None and max_batches <= 0:
            raise ValueError(f"max_batches must be positive, got {max_batches}.")

        self.pipeline = pipeline
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[dict[str, Observation]]:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)

        batch_count = 0
        while self.max_batches is None or batch_count < self.max_batches:
            yield self.pipeline(self.batch_size, self.device, rng=generator)
            batch_count += 1
