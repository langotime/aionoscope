from __future__ import annotations

import argparse

import torch
from torch import nn

from aiono import (
    ECGLeadsView,
    EventImpulseView,
    GaussianNoiseView,
    KernelConvView,
    PulseTrainProcess,
    SamplingAggregationView,
    SynthPipeline,
    TrendSeasonAnomalyProcess,
    UnitsAbsoluteView,
    make_pqrst_kernel_bank,
    pqrst_kernel_size,
)
from aiono.core.utils import utils_make_canonical_A0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check torch.compile compatibility for Aionoscope pipelines.",
    )
    parser.add_argument("--process", choices=["pulse", "trend"], required=True)
    parser.add_argument(
        "--backend",
        choices=["eager", "aot_eager", "inductor"],
        default="eager",
        help="torch.compile backend to use for the compatibility check.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--frequency-hz", type=float)
    parser.add_argument("--sample-rate-hz", type=float)
    parser.add_argument("--components", type=int)
    parser.add_argument("--seed", type=int, required=True)
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
    frequency_hz: float | None,
    sample_rate_hz: float | None,
    components: int | None,
) -> SynthPipeline:
    if process_name == "pulse":
        if frequency_hz is None:
            raise ValueError("--frequency-hz is required for process=pulse.")
        if sample_rate_hz is None:
            raise ValueError("--sample-rate-hz is required for process=pulse.")

        process = PulseTrainProcess(
            seq_len=seq_len,
            frequency_hz=frequency_hz,
            sample_rate_hz=sample_rate_hz,
            rhythm_classes=["regular", "irregular", "missed_beat"],
            shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
            latent_mode="pqrst3",
        )
        spacing = (process.seq_len - 1) / (process.num_pulses + 1)
        kernel_size = pqrst_kernel_size(spacing=spacing, support_sigma=6.0)
        kernels = make_pqrst_kernel_bank(
            shape_names=process.shape_classes,
            spacing=spacing,
            kernel_size=kernel_size,
            device=device,
        )  # [K, T, W]
        padding = kernel_size // 2
        A0 = utils_make_canonical_A0(num_leads=12, num_latent=3).to(device)  # [C, K]
        views = {
            "ecg": nn.Sequential(
                EventImpulseView(
                    seq_len=process.seq_len,
                    amplitude_param="amplitude",
                    rounding="nearest",
                ),
                KernelConvView(kernels=kernels, padding=padding),
                ECGLeadsView(A0=A0, jitter_std=0.02, max_delay=2),
            ),
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
            GaussianNoiseView(noise_std=0.05),
            SamplingAggregationView(mode="mean", window=5),
        )
    }
    return SynthPipeline(process=process, views=views)


def _compile_and_run(
    pipeline: SynthPipeline,
    backend: str,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    compiled = torch.compile(pipeline, backend=backend)
    output = compiled(batch_size=batch_size, device=device, rng=generator)

    for name, observation in output.items():
        x = observation.x  # [B, C, L]
        print(f"{name}: backend={backend}, x={tuple(x.shape)}, dtype={x.dtype}")


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")

    device = _resolve_device(args.device)

    pipeline = _build_pipeline(
        process_name=args.process,
        device=device,
        seq_len=args.seq_len,
        frequency_hz=args.frequency_hz,
        sample_rate_hz=args.sample_rate_hz,
        components=args.components,
    )

    try:
        _compile_and_run(
            pipeline=pipeline,
            backend=args.backend,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "torch.compile failed for the requested pipeline. "
            "Provide the full traceback when reporting this failure."
        ) from exc


if __name__ == "__main__":
    main()
