from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from aiono import (
    EnableComponentsNode,
    EventRenderView,
    EventSchema,
    GateEventsByEnabledNode,
    LinearTrendView,
    LogTrendView,
    ProcessGraph,
    QuadraticTrendView,
    SigmoidTrendView,
    SingleEventNode,
    SynthPipeline,
    TOYTS_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ,
    ToyTSBasicComponentsPeriodicConfig,
    UnionEventsNode,
    ViewChain,
    resolve_toyts_basic_components_periodic_contract,
)
from aiono.datasets import SynthBatchIterableDataset
from aiono.processes.constant import ConstantLatentNode
from aiono.views.noise import (
    GaussianNoiseView,
    RandomWalkNoiseView,
    UniformNoiseView,
)
from aiono.views.periodic import (
    SawtoothWaveView,
    SineWaveView,
    SquareWaveView,
)
from aiono import UniformSampler


def build_basic_components_process(
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
        name="BasicComponentsProcess",
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
                value=UniformSampler(-1.0, 1.0),
                enabled_key="constant",
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


def plot_grid(
    *,
    x: torch.Tensor,
    enabled: dict[str, torch.Tensor],
    component_count: int | torch.Tensor,
    max_plots: int = 16,
) -> plt.Figure:
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape [B, C, L], got {tuple(x.shape)}.")
    if x.shape[1] != 1:
        raise ValueError(f"Expected x to have shape [B, 1, L], got {tuple(x.shape)}.")
    batch_size, _, seq_len = x.shape
    if isinstance(component_count, torch.Tensor):
        if component_count.shape != (batch_size,):
            raise ValueError(
                "Expected component_count to have shape [B]. "
                f"Got {tuple(component_count.shape)}, batch_size={batch_size}."
            )
        if component_count.dtype != torch.int64:
            raise ValueError(
                "Expected component_count to have dtype torch.int64. "
                f"Got {component_count.dtype}."
            )
        component_count_cpu = component_count.detach().cpu()  # [B]
    else:
        component_count_cpu = None
    num_plots = min(max_plots, batch_size)
    rows = int(math.floor(math.sqrt(num_plots)))
    cols = int(math.ceil(num_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5), sharex=True, sharey=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    x_cpu = x.detach().cpu()  # [B, 1, L]
    for plot_idx in range(num_plots):
        ax = axes_list[plot_idx]
        signal = x_cpu[plot_idx, 0]  # [L]
        ax.plot(torch.arange(seq_len), signal, linewidth=1.0)

        active = [name for name, mask in enabled.items() if bool(mask[plot_idx].item())]
        active_str = "+".join(active) if active else "none"
        if component_count_cpu is not None:
            k = int(component_count_cpu[plot_idx].item())
        else:
            k = int(component_count)
        ax.set_title(f"{active_str} (k={k})", fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

    for ax in axes_list[num_plots:]:
        ax.axis("off")

    fig.suptitle("Aionoscope: basic components (per-sample enabled masks)", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 5000
    sample_rate_hz = float(TOYTS_BASIC_COMPONENTS_BASELINE_SAMPLING_FREQUENCY_HZ)
    batch_size = 16
    seed = 1234

    periodic_contract = resolve_toyts_basic_components_periodic_contract(
        seq_len=seq_len,
        sampling_frequency_hz=int(sample_rate_hz),
        config=ToyTSBasicComponentsPeriodicConfig.v1(),
    )

    component_keys = [
        "constant",
        "gaussian_noise",
        "uniform_noise",
        "random_walk_noise",
        "linear_trend",
        "quadratic_trend",
        "log_trend",
        "sigmoid_trend",
        "sine",
        "sawtooth",
        "square",
        "spike",
        "level_change",
        "gaussian",
    ]

    num_enabled = 2
    if num_enabled < 1 or num_enabled > len(component_keys):
        raise ValueError(
            "num_enabled must satisfy 1 <= num_enabled <= len(component_keys). "
            f"Got num_enabled={num_enabled}, len(component_keys)={len(component_keys)}."
        )

    process = build_basic_components_process(
        seq_len=seq_len,
        sample_rate_hz=sample_rate_hz,
        component_keys=component_keys,
        num_enabled=num_enabled,
    )

    view = ViewChain(
        EventRenderView(
            seq_len=seq_len,
            amplitude_param="amplitude",
            rounding="nearest",
            sigma_sec_param="sigma_sec",
        ),
        GaussianNoiseView(noise_std=UniformSampler(0.02, 0.15), enabled_key="gaussian_noise"),
        UniformNoiseView(amplitude=UniformSampler(0.05, 0.25), enabled_key="uniform_noise"),
        RandomWalkNoiseView(step_std=UniformSampler(0.01, 0.08), enabled_key="random_walk_noise"),
        LinearTrendView(
            seq_len=seq_len,
            slope=UniformSampler(-2.0, 2.0),
            intercept=UniformSampler(-0.5, 0.5),
            enabled_key="linear_trend",
        ),
        QuadraticTrendView(
            seq_len=seq_len,
            a=UniformSampler(-4.0, 4.0),
            b=UniformSampler(-2.0, 2.0),
            c=UniformSampler(-0.5, 0.5),
            enabled_key="quadratic_trend",
        ),
        LogTrendView(
            seq_len=seq_len,
            amplitude=UniformSampler(-2.0, 2.0),
            offset=UniformSampler(-0.5, 0.5),
            epsilon=1e-3,
            enabled_key="log_trend",
        ),
        SigmoidTrendView(
            seq_len=seq_len,
            amplitude=UniformSampler(-2.0, 2.0),
            center=UniformSampler(0.2, 0.8),
            sharpness=UniformSampler(5.0, 20.0),
            offset=UniformSampler(-0.5, 0.5),
            enabled_key="sigmoid_trend",
        ),
        SineWaveView(
            seq_len=seq_len,
            **periodic_contract.signal("sine").view_kwargs(),
            enabled_key="sine",
        ),
        SawtoothWaveView(
            seq_len=seq_len,
            **periodic_contract.signal("sawtooth").view_kwargs(),
            enabled_key="sawtooth",
        ),
        SquareWaveView(
            seq_len=seq_len,
            **periodic_contract.signal("square").view_kwargs(),
            enabled_key="square",
        ),
    )

    pipeline = SynthPipeline(process=process, views={"mix": view})
    pipeline.to(device)

    dataset = SynthBatchIterableDataset(
        pipeline=pipeline,
        batch_size=batch_size,
        device=device,
        seed=seed,
        max_batches=1,
    )
    loader = DataLoader(dataset, batch_size=None)
    batch = next(iter(loader))

    obs = batch["mix"]
    print("x:", tuple(obs.x.shape))
    print("y keys:", sorted(obs.y.keys()))

    enabled = obs.meta["process"]["enabled"]
    for name in component_keys:
        count = int(enabled[name].sum().item())
        if count > 0:
            print(f"enabled[{name}]: {count}/{batch_size}")

    out_path = Path("examples/figures/06_basic_components.png")
    fig = plot_grid(
        x=obs.x,
        enabled=enabled,
        component_count=obs.y["component_count"],
        max_plots=batch_size,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
