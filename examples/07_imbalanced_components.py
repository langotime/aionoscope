from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from aiono import (
    CategoricalSampler,
    EnableComponentsNode,
    EventRenderView,
    EventSchema,
    GateEventsByEnabledNode,
    LinearTrendView,
    LogTrendView,
    PiecewiseLinearTrendView,
    ProcessGraph,
    QuadraticTrendView,
    SigmoidTrendView,
    SingleEventNode,
    SynthPipeline,
    UnionEventsNode,
    UniformSampler,
    ViewChain,
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


def build_imbalanced_components_process(
    *,
    seq_len: int,
    sample_rate_hz: float,
    component_keys: list[str],
    component_id_sampler: CategoricalSampler,
) -> ProcessGraph:
    schema = EventSchema(
        type_names=["spike", "level_change", "gaussian"],
        param_names=["amplitude", "sigma_sec"],
        time_unit="samples",
    )

    event_time_min = int(seq_len * 0.15)
    event_time_max = int(seq_len * 0.85)

    return ProcessGraph(
        name="ImbalancedComponentsProcess",
        outputs={"latent", "events"},
        base_meta={
            "seq_len": seq_len,
            "sample_rate_hz": sample_rate_hz,
            "component_keys": list(component_keys),
            "num_enabled": 1,
            "component_id_spec": component_id_sampler.spec(),
        },
        graph=[
            EnableComponentsNode(
                component_keys=component_keys,
                num_enabled=1,
                component_id=component_id_sampler,
            ),
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
    max_plots: int = 16,
) -> plt.Figure:
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape [B, C, L], got {tuple(x.shape)}.")
    if x.shape[1] != 1:
        raise ValueError(f"Expected x to have shape [B, 1, L], got {tuple(x.shape)}.")
    batch_size, _, seq_len = x.shape
    num_plots = min(max_plots, batch_size)
    rows = int(math.floor(math.sqrt(num_plots)))
    cols = int(math.ceil(num_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5), sharex=True, sharey=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    x_cpu = x.detach().cpu()  # [B, 1, L]
    t = torch.arange(seq_len)  # [L]
    for plot_idx in range(num_plots):
        ax = axes_list[plot_idx]
        signal = x_cpu[plot_idx, 0]  # [L]
        ax.plot(t, signal, linewidth=1.0)

        active = [name for name, mask in enabled.items() if bool(mask[plot_idx].item())]
        active_str = "+".join(active) if active else "none"
        ax.set_title(active_str, fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

    for ax in axes_list[num_plots:]:
        ax.axis("off")

    fig.suptitle("Aionoscope: imbalanced component sampling (per-sample enabled masks)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_component_histogram(*, component_keys: list[str], counts: torch.Tensor) -> plt.Figure:
    if counts.ndim != 1:
        raise ValueError(f"Expected counts to have shape [N], got {tuple(counts.shape)}.")
    if counts.dtype != torch.int64:
        raise ValueError(f"Expected counts to be int64, got {counts.dtype}.")
    if counts.shape[0] != len(component_keys):
        raise ValueError(
            "counts length must match component_keys length. "
            f"Got counts={counts.shape[0]}, component_keys={len(component_keys)}."
        )

    total = int(counts.sum().item())
    if total <= 0:
        raise ValueError("counts must have a positive sum.")

    pct = counts.to(torch.float32) / float(total) * 100.0  # [N]

    fig, ax = plt.subplots(figsize=(max(10.0, 0.55 * len(component_keys)), 4.0))
    ax.bar(list(range(len(component_keys))), pct.tolist())
    ax.set_xticks(list(range(len(component_keys))))
    ax.set_xticklabels(component_keys, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Percent (%)")
    ax.set_title(f"Empirical component frequencies (B={total})")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    return fig


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 512
    sample_rate_hz = 128.0
    batch_size = 16
    histogram_batch_size = 8192
    seed = 1234

    component_keys = [
        "constant",
        "gaussian_noise",
        "uniform_noise",
        "random_walk_noise",
        "linear_trend",
        "quadratic_trend",
        "log_trend",
        "sigmoid_trend",
        "piecewise_linear_trend",
        "sine",
        "sawtooth",
        "square",
        "spike",
        "level_change",
        "gaussian",
    ]

    weights_by_key = {key: 1.0 for key in component_keys}
    rare_keys = ["spike", "level_change", "gaussian"]
    rare_weight = 0.02
    for key in rare_keys:
        weights_by_key[key] = rare_weight

    if set(weights_by_key.keys()) != set(component_keys):
        raise ValueError(
            "weights_by_key keys must match component_keys exactly. "
            f"missing={sorted(set(component_keys) - set(weights_by_key.keys()))}, "
            f"extra={sorted(set(weights_by_key.keys()) - set(component_keys))}."
        )
    probs = [float(weights_by_key[key]) for key in component_keys]
    if any((not math.isfinite(prob)) or (prob < 0.0) for prob in probs):
        raise ValueError(f"weights_by_key must contain only finite, non-negative values. Got {probs}.")
    if sum(probs) <= 0.0:
        raise ValueError(f"weights_by_key must have positive sum. Got {probs}.")

    component_id_sampler = CategoricalSampler(probs=probs)

    process = build_imbalanced_components_process(
        seq_len=seq_len,
        sample_rate_hz=sample_rate_hz,
        component_keys=component_keys,
        component_id_sampler=component_id_sampler,
    )

    hist_rng = torch.Generator(device=device).manual_seed(seed)
    latent = process(batch_size=histogram_batch_size, device=device, rng=hist_rng)
    component_id = latent.y["component_id"].detach().cpu()  # [B]
    counts = torch.bincount(component_id, minlength=len(component_keys)).to(torch.int64)  # [N]

    print("component frequencies (empirical):")
    total = int(counts.sum().item())
    for key, count in zip(component_keys, counts.tolist(), strict=True):
        print(f"- {key}: {count}/{total} ({(100.0 * count / total):.3f}%)")

    figures_dir = Path("examples/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    hist_path = figures_dir / "07_imbalanced_components_hist.png"
    fig_hist = plot_component_histogram(component_keys=component_keys, counts=counts)
    fig_hist.savefig(hist_path, dpi=140)
    plt.close(fig_hist)
    print("saved:", hist_path)

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
        PiecewiseLinearTrendView(
            seq_len=seq_len,
            slope1=UniformSampler(-2.0, 2.0),
            slope2=UniformSampler(-2.0, 2.0),
            change_t=UniformSampler(0.2, 0.8),
            intercept=UniformSampler(-0.5, 0.5),
            enabled_key="piecewise_linear_trend",
        ),
        SineWaveView(
            seq_len=seq_len,
            amplitude=UniformSampler(0.2, 1.2),
            frequency_hz=UniformSampler(0.2, 6.0),
            phase=UniformSampler(0.0, 2.0 * math.pi),
            offset=UniformSampler(-0.2, 0.2),
            enabled_key="sine",
        ),
        SawtoothWaveView(
            seq_len=seq_len,
            amplitude=UniformSampler(0.2, 1.2),
            frequency_hz=UniformSampler(0.2, 6.0),
            phase=UniformSampler(0.0, 2.0 * math.pi),
            offset=UniformSampler(-0.2, 0.2),
            enabled_key="sawtooth",
        ),
        SquareWaveView(
            seq_len=seq_len,
            amplitude=UniformSampler(0.2, 1.2),
            frequency_hz=UniformSampler(0.2, 6.0),
            phase=UniformSampler(0.0, 2.0 * math.pi),
            offset=UniformSampler(-0.2, 0.2),
            duty_cycle=UniformSampler(0.1, 0.9),
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

    grid_path = figures_dir / "07_imbalanced_components_grid.png"
    fig_grid = plot_grid(x=obs.x, enabled=enabled, max_plots=batch_size)
    fig_grid.savefig(grid_path, dpi=140)
    plt.close(fig_grid)
    print("saved:", grid_path)


if __name__ == "__main__":
    main()
