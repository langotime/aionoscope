from __future__ import annotations

import argparse
import time
from typing import Sequence

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from toyts import (
    ECGLeadsView,
    NoiseView,
    PulseTrainProcess,
    SamplingAggregationView,
    SynthPipeline,
    TrendSeasonAnomalyProcess,
    UnitsAbsoluteView,
)
from toyts.core.utils import utils_make_canonical_A0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile ToyTS batch generation with torch.profiler.",
    )
    parser.add_argument("--process", choices=["pulse", "trend"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--num-pulses", type=int)
    parser.add_argument("--components", type=int)
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use --device cpu.")
        return torch.device("cuda")
    return torch.device("cpu")


def _build_pipeline(
    process_name: str,
    device: torch.device,
    seq_len: int,
    num_pulses: int | None,
    components: int | None,
) -> SynthPipeline:
    if process_name == "pulse":
        if num_pulses is None:
            raise ValueError("--num-pulses is required for process=pulse.")

        process = PulseTrainProcess(
            seq_len=seq_len,
            num_pulses=num_pulses,
            rhythm_classes=["regular", "irregular", "missed_beat"],
            shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
            latent_mode="pqrst3",
        )
        A0 = utils_make_canonical_A0(num_leads=12, num_latent=3).to(device)  # [C, K]
        views = {
            "ecg": ECGLeadsView(A0=A0, jitter_std=0.02, max_delay=2),
        }
        return SynthPipeline(process=process, views=views)

    if components is None:
        raise ValueError("--components is required for process=trend.")

    process = TrendSeasonAnomalyProcess(
        seq_len=seq_len,
        components=components,
        regime_classes=["steady", "ramping", "spiky"],
        anomaly_classes=["none", "drop", "spike"],
    )

    views = {
        "trend": nn.Sequential(
            UnitsAbsoluteView(),
            NoiseView(noise_std=0.05),
            SamplingAggregationView(mode="mean", window=5),
        )
    }
    return SynthPipeline(process=process, views=views)


def _profile(
    pipeline: SynthPipeline,
    device: torch.device,
    batch_size: int,
    steps: int,
    warmup: int,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(123)

    for _ in range(warmup):
        pipeline(batch_size=batch_size, device=device, rng=generator)

    activities: list[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(steps):
            pipeline(batch_size=batch_size, device=device, rng=generator)
            if device.type == "cuda":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_batches = float(steps)
    batches_per_sec = total_batches / elapsed if elapsed > 0 else 0.0

    sort_key = "self_cpu_time_total"
    if device.type == "cuda":
        sort_key = "self_cuda_time_total"

    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    print(f"batches_per_sec: {batches_per_sec:.3f}")
    print(f"elapsed_sec: {elapsed:.3f}")


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")

    device = _resolve_device(args.device)

    pipeline = _build_pipeline(
        process_name=args.process,
        device=device,
        seq_len=args.seq_len,
        num_pulses=args.num_pulses,
        components=args.components,
    )

    _profile(
        pipeline=pipeline,
        device=device,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
