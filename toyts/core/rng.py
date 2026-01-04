from __future__ import annotations

import time

import torch


def rng_make_generator(
    rng: torch.Generator | None,
    device: torch.device,
) -> tuple[torch.Generator, int, bool]:
    """Return a generator on the requested device and its seed.

    Returns (generator, seed, created).
    """

    if rng is None:
        seed = int(time.time_ns() % (2**63 - 1))
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator, seed, True

    if rng.device.type != device.type:
        raise ValueError(
            "Provided torch.Generator device type does not match requested device type. "
            f"rng.device={rng.device}, requested={device}."
        )

    if rng.device.index is not None and device.index is not None and rng.device.index != device.index:
        raise ValueError(
            "Provided torch.Generator device index does not match requested device index. "
            f"rng.device={rng.device}, requested={device}."
        )

    return rng, int(rng.initial_seed()), False


def rng_split(
    rng: torch.Generator,
    num_children: int,
    device: torch.device,
) -> list[torch.Generator]:
    """Split a generator into independent child generators."""

    if num_children <= 0:
        raise ValueError(f"num_children must be positive, got {num_children}.")

    max_seed = 2**63 - 1
    seeds = torch.randint(
        low=0,
        high=max_seed,
        size=(num_children,),
        generator=rng,
        device=device,
        dtype=torch.int64,
    )  # [num_children]

    child_generators: list[torch.Generator] = []
    for seed in seeds.tolist():
        child = torch.Generator(device=device)
        child.manual_seed(int(seed))
        child_generators.append(child)

    return child_generators
