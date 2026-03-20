from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from aiono import (
    AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ,
    AionoBasicComponentsPeriodicConfig,
    EnableComponentsNode,
    ECGMorphologyParams,
    ECGProcess,
    ECGRhythmParams,
    EventImpulseView,
    EventSchema,
    GateEventsByEnabledNode,
    GaussianNoiseView,
    KernelConvView,
    LinearTrendView,
    NormalizeView,
    PulseTrainProcess,
    ProcessGraph,
    SawtoothWaveView,
    SingleEventNode,
    SineWaveView,
    SquareWaveView,
    SynthPipeline,
    TrendSeasonAnomalyProcess,
    UnionEventsNode,
    UniformSampler,
    ViewChain,
    make_ptbxl_kernel_bank,
    make_pqrst_kernel_bank,
    ptbxl_kernel_size,
    pqrst_kernel_size,
    resolve_aiono_basic_components_periodic_contract,
)
from aiono.processes.constant import ConstantLatentNode
from aiono.ptbxl import PTBXLLabelSetSampler, ptbxl_all_codes
from aiono.core.utils import utils_make_canonical_A0
from aiono.views.ecg_leads import ECGLeadsView
from aiono.views.events import EventImpulseView as PulseEventImpulseView
from aiono.views.events import KernelConvView as PulseKernelConvView
from aiono.views.events_basic import EventRenderView
from aiono.views.noise import RandomWalkNoiseView, UniformNoiseView
from aiono.views.units import ClippingView, UnitsAbsoluteView, UnitsPercentOfCapacityView


@dataclass(frozen=True)
class SmokeResult:
    name: str
    shape: tuple[int, ...]
    labels: list[str]


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _simple_pulse_smoke(device: torch.device) -> SmokeResult:
    generator = torch.Generator(device=device).manual_seed(1234)
    process = PulseTrainProcess(
        seq_len=256,
        frequency_hz=1.2,
        sample_rate_hz=250.0,
        rhythm_classes=["regular", "irregular", "missed_beat"],
        shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
        latent_mode="pqrst3",
        amplitude=2.0,
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
    A0 = utils_make_canonical_A0(num_leads=4, num_latent=3).to(device)  # [C, K]

    pipeline = SynthPipeline(
        process=process,
        views={
            "clean": torch.nn.Sequential(
                PulseEventImpulseView(
                    seq_len=process.seq_len,
                    amplitude_param="amplitude",
                    rounding="nearest",
                ),
                PulseKernelConvView(kernels=kernels, padding=padding),
                ECGLeadsView(A0=A0, jitter_std=0.01, max_delay=1),
            )
        },
    ).to(device)

    batch = pipeline(batch_size=4, device=device, rng=generator)
    observation = batch["clean"]
    return SmokeResult(
        name="simple_pulse",
        shape=tuple(observation.x.shape),
        labels=sorted(observation.y.keys()),
    )


def _server_metrics_smoke(device: torch.device) -> SmokeResult:
    generator = torch.Generator(device=device).manual_seed(42)
    process = TrendSeasonAnomalyProcess(
        seq_len=128,
        components=3,
        regime_classes=["steady", "ramping", "spiky"],
        anomaly_classes=["none", "drop", "spike"],
    )
    pipeline = SynthPipeline(
        process=process,
        views={
            "cpu_percent": torch.nn.Sequential(
                UnitsAbsoluteView(),
                UnitsPercentOfCapacityView(capacity_min=90.0, capacity_max=110.0),
                ClippingView(min_value=0.0, max_value=100.0),
            )
        },
    ).to(device)

    batch = pipeline(batch_size=4, device=device, rng=generator)
    observation = batch["cpu_percent"]
    return SmokeResult(
        name="server_metrics",
        shape=tuple(observation.x.shape),
        labels=sorted(observation.y.keys()),
    )


def _build_basic_components_process(
    *,
    seq_len: int,
    sample_rate_hz: float,
    component_keys: list[str],
    num_enabled: int,
) -> ProcessGraph:
    schema = EventSchema(
        type_names=["spike", "level_change", "gaussian"],
        param_names=["amplitude", "sigma_sec"],
        time_unit="samples",
    )
    event_time_min = int(seq_len * 0.15)
    event_time_max = int(seq_len * 0.85)

    return ProcessGraph(
        name="SmokeBasicComponentsProcess",
        outputs={"latent", "events"},
        base_meta={
            "seq_len": seq_len,
            "sample_rate_hz": sample_rate_hz,
            "component_keys": list(component_keys),
            "num_enabled": num_enabled,
        },
        graph=[
            EnableComponentsNode(component_keys=component_keys, num_enabled=num_enabled),
            ConstantLatentNode(
                seq_len=seq_len,
                channels=1,
                value=0.0,
                out_key="latent",
            ),
            SingleEventNode(
                seq_len=seq_len,
                schema=schema,
                type_name="spike",
                time_min=event_time_min,
                time_max=event_time_max,
                amplitude=UniformSampler(0.8, 1.2),
                amplitude_param="amplitude",
                out_key="spike",
            ),
            GateEventsByEnabledNode(in_key="spike", enabled_key="spike", out_key="spike.gated"),
            SingleEventNode(
                seq_len=seq_len,
                schema=schema,
                type_name="level_change",
                time_min=event_time_min,
                time_max=event_time_max,
                amplitude=UniformSampler(-1.0, 1.0),
                amplitude_param="amplitude",
                out_key="level_change",
            ),
            GateEventsByEnabledNode(
                in_key="level_change",
                enabled_key="level_change",
                out_key="level_change.gated",
            ),
            SingleEventNode(
                seq_len=seq_len,
                schema=schema,
                type_name="gaussian",
                time_min=event_time_min,
                time_max=event_time_max,
                amplitude=UniformSampler(-1.0, 1.0),
                amplitude_param="amplitude",
                extra_params={"sigma_sec": UniformSampler(0.01, 0.06)},
                out_key="gaussian",
            ),
            GateEventsByEnabledNode(
                in_key="gaussian",
                enabled_key="gaussian",
                out_key="gaussian.gated",
            ),
            UnionEventsNode(
                in_keys=["spike.gated", "level_change.gated", "gaussian.gated"],
                out_key="events",
            ),
        ],
    )


def _basic_components_smoke(device: torch.device) -> SmokeResult:
    generator = torch.Generator(device=device).manual_seed(99)
    seq_len = 512
    sample_rate_hz = float(AIONO_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ)
    periodic_contract = resolve_aiono_basic_components_periodic_contract(
        seq_len=seq_len,
        sampling_frequency_hz=int(sample_rate_hz),
        config=AionoBasicComponentsPeriodicConfig.v1(),
    )
    component_keys = [
        "gaussian_noise",
        "uniform_noise",
        "random_walk_noise",
        "linear_trend",
        "sine",
        "sawtooth",
        "square",
        "spike",
        "level_change",
        "gaussian",
    ]
    process = _build_basic_components_process(
        seq_len=seq_len,
        sample_rate_hz=sample_rate_hz,
        component_keys=component_keys,
        num_enabled=2,
    )
    view = ViewChain(
        EventRenderView(
            seq_len=seq_len,
            amplitude_param="amplitude",
            rounding="nearest",
            sigma_sec_param="sigma_sec",
        ),
        GaussianNoiseView(noise_std=0.05, enabled_key="gaussian_noise"),
        UniformNoiseView(amplitude=0.1, enabled_key="uniform_noise"),
        RandomWalkNoiseView(step_std=0.02, enabled_key="random_walk_noise"),
        LinearTrendView(seq_len=seq_len, slope=0.5, intercept=0.0, enabled_key="linear_trend"),
        SineWaveView(
            seq_len=seq_len,
            amplitude=0.4,
            frequency_hz=periodic_contract.signal("sine").frequency_hz.low,
            phase=0.0,
            offset=0.0,
            enabled_key="sine",
        ),
        SawtoothWaveView(
            seq_len=seq_len,
            amplitude=0.2,
            frequency_hz=periodic_contract.signal("sawtooth").frequency_hz.low,
            phase=0.0,
            offset=0.0,
            enabled_key="sawtooth",
        ),
        SquareWaveView(
            seq_len=seq_len,
            amplitude=0.15,
            frequency_hz=periodic_contract.signal("square").frequency_hz.low,
            phase=0.0,
            offset=0.0,
            duty_cycle=periodic_contract.signal("square").duty_cycle.low,
            enabled_key="square",
        ),
    )
    pipeline = SynthPipeline(process=process, views={"signal": view}).to(device)
    batch = pipeline(batch_size=4, device=device, rng=generator)
    observation = batch["signal"]
    return SmokeResult(
        name="basic_components",
        shape=tuple(observation.x.shape),
        labels=sorted(observation.y.keys()),
    )


def _ptbxl_smoke(device: torch.device) -> SmokeResult:
    generator = torch.Generator(device=device).manual_seed(2026)
    scp_codes = ptbxl_all_codes()
    sampler = PTBXLLabelSetSampler(scp_codes=scp_codes, normal_prob=0.0)
    process = ECGProcess(
        seq_len=512,
        sample_rate_hz=500.0,
        scp_codes=scp_codes,
        scp_sampler=sampler,
        rhythm_params=ECGRhythmParams.ptbxl_defaults(),
        morphology_params=ECGMorphologyParams.ptbxl_defaults(),
    )
    kernel_size = ptbxl_kernel_size(sample_rate_hz=process.sample_rate_hz, support_ms=400.0)
    kernels = make_ptbxl_kernel_bank(
        sample_rate_hz=process.sample_rate_hz,
        kernel_size=kernel_size,
        device=device,
    )  # [K=12, T, W]
    padding = kernel_size // 2

    pipeline = SynthPipeline(
        process=process,
        views={
            "clean": torch.nn.Sequential(
                EventImpulseView(
                    seq_len=process.seq_len,
                    amplitude_param="amplitude",
                    rounding="nearest",
                ),
                KernelConvView(kernels=kernels, padding=padding),
                GaussianNoiseView(noise_std=0.05),
                NormalizeView(),
            )
        },
    ).to(device)

    batch = pipeline(batch_size=4, device=device, rng=generator)
    observation = batch["clean"]
    return SmokeResult(
        name="ptbxl",
        shape=tuple(observation.x.shape),
        labels=sorted(observation.y.keys()),
    )


def run_smoke_examples() -> list[SmokeResult]:
    device = _resolve_device()
    return [
        _simple_pulse_smoke(device),
        _server_metrics_smoke(device),
        _basic_components_smoke(device),
        _ptbxl_smoke(device),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run representative library smoke checks.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_smoke_examples()

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"results": [asdict(result) for result in results]}
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for result in results:
        print(f"{result.name}: shape={result.shape}, labels={result.labels}")


if __name__ == "__main__":
    main()
