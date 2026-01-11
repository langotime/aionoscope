from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from toyts import (
    BernoulliSampler,
    CategoricalSampler,
    ChoiceSampler,
    ClippingView,
    ConstantSampler,
    LogUniformSampler,
    MissingnessView,
    NoiseView,
    NormalSampler,
    RandIntSampler,
    TrendSeasonAnomalyProcess,
    UniformSampler,
    UnitsAbsoluteView,
    UnitsPercentOfCapacityView,
)


def sample_samplers(device: torch.device) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return samples from every sampler for visualization."""
    num_samples = 2048
    shape = (num_samples,)
    rng = torch.Generator(device=device).manual_seed(123)

    constant_samples = ConstantSampler(1.5).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.float32,
    )  # [N]
    uniform_samples = UniformSampler(0.1, 0.9).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.float32,
    )  # [N]
    log_uniform_samples = LogUniformSampler(0.01, 0.2).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.float32,
    )  # [N]
    normal_samples = NormalSampler(0.0, 0.8, clamp=(-1.5, 1.5)).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.float32,
    )  # [N]
    randint_samples = RandIntSampler(1, 5).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.int64,
    )  # [N]
    bernoulli_samples = BernoulliSampler(0.3).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.bool,
    )  # [N]
    categorical_samples = CategoricalSampler([0.1, 0.2, 0.7]).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.int64,
    )  # [N]
    choice_values = ["low", "mid", "high"]
    choice_samples = ChoiceSampler(choice_values, probs=[0.2, 0.5, 0.3]).sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=torch.int64,
    )  # [N]

    samples = {
        "constant": constant_samples,
        "uniform": uniform_samples,
        "log_uniform": log_uniform_samples,
        "normal": normal_samples,
        "randint": randint_samples,
        "bernoulli": bernoulli_samples,
        "categorical": categorical_samples,
        "choice": choice_samples,
    }
    return samples, choice_values


def plot_sampler_distributions(
    samples: dict[str, torch.Tensor],
    choice_labels: list[str],
    output_dir: Path,
) -> None:
    """Plot a distribution grid for all sampler outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    continuous_keys = ["constant", "uniform", "log_uniform", "normal"]
    discrete_keys = ["randint", "bernoulli", "categorical", "choice"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=False)
    for idx, key in enumerate(continuous_keys):
        ax = axes[0, idx]
        values = samples[key].detach().cpu().numpy()
        ax.hist(values, bins=40, color="steelblue", alpha=0.9)
        ax.set_title(key)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx, key in enumerate(discrete_keys):
        ax = axes[1, idx]
        values = samples[key]  # [N]
        if values.dtype == torch.bool:
            bins = torch.bincount(values.to(torch.int64), minlength=2)  # [2]
            labels = ["0", "1"]
        else:
            max_value = int(values.max().item())
            bins = torch.bincount(values, minlength=max_value + 1)  # [K]
            labels = [str(i) for i in range(max_value + 1)]

        if key == "choice":
            labels = choice_labels
            bins = bins[: len(choice_labels)]  # [K_choice]

        ax.bar(range(len(labels)), bins.detach().cpu().numpy(), color="darkorange", alpha=0.85)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(key)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Sampler distributions")
    fig.tight_layout()
    output_path = output_dir / "05_param_samplers_distributions.png"
    fig.savefig(output_path)
    print(f"saved figure {output_path}")


def build_pipeline_views() -> list[tuple[str, object]]:
    """Return named views used in the pipeline demo."""
    return [
        ("absolute", UnitsAbsoluteView()),
        (
            "percent_of_capacity",
            UnitsPercentOfCapacityView(
                capacity_min=ConstantSampler(1.0),
                capacity_max=UniformSampler(2.0, 3.0),
            ),
        ),
        ("noise", NoiseView(noise_std=LogUniformSampler(0.01, 0.1))),
        (
            "missingness",
            MissingnessView(
                dropout_prob=UniformSampler(0.0, 0.05),
                gap_prob=UniformSampler(0.0, 0.1),
                gap_length=RandIntSampler(0, 6),
                hold_prob=UniformSampler(0.0, 0.05),
            ),
        ),
        (
            "clipping",
            ClippingView(
                min_value=NormalSampler(-1.0, 0.2, clamp=(-1.5, -0.5)),
                max_value=NormalSampler(1.0, 0.2, clamp=(0.5, 1.5)),
            ),
        ),
    ]


def run_pipeline_demo(
    device: torch.device,
) -> list[tuple[str, object]]:
    """Generate a batch and apply each view sequentially."""
    process = TrendSeasonAnomalyProcess(
        seq_len=96,
        components=4,
        slope_max=UniformSampler(0.2, 0.6),
        season_amp=LogUniformSampler(0.5, 1.5),
        spiky_boost=ConstantSampler(2.0),
        season_freq_min=ConstantSampler(1.0),
        season_freq_max=ConstantSampler(3.0),
        anomaly_scale=LogUniformSampler(0.5, 1.5),
    )

    views = build_pipeline_views()

    rng = torch.Generator(device=device).manual_seed(999)
    state = process(batch_size=2, device=device, rng=rng)

    current = state
    outputs: list[tuple[str, object]] = []
    for name, view in views:
        current = view(current, rng=rng)
        outputs.append((name, current))

    return outputs


def plot_pipeline_steps(
    outputs: list[tuple[str, object]],
    output_dir: Path,
) -> None:
    """Plot the first sample after each view step."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(outputs), 1, figsize=(12, 8), sharex=True)
    for ax, (name, obs) in zip(axes, outputs):
        signal = obs.x[0, 0].detach().cpu()  # [L]
        time_idx = torch.arange(signal.shape[0])  # [L]
        ax.plot(time_idx.numpy(), signal.numpy(), linewidth=1.0)
        ax.set_title(name)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Pipeline steps (sample 0)")
    fig.tight_layout()
    output_path = output_dir / "05_param_samplers_pipeline.png"
    fig.savefig(output_path)
    print(f"saved figure {output_path}")


def plot_missingness_masks(
    output: tuple[str, object],
    output_dir: Path,
) -> None:
    """Plot missingness masks for the first sample."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _, obs = output
    masks = MissingnessView.sample_masks(
        obs.meta,
        shape=tuple(obs.x.shape),
        device=obs.x.device,
    )
    dropout_mask = masks["dropout_mask"][0, 0].detach().cpu()  # [L]
    gap_mask = masks["gap_mask"][0, 0].detach().cpu()  # [L]
    hold_mask = masks["hold_mask"][0, 0].detach().cpu()  # [L]
    time_idx = torch.arange(dropout_mask.shape[0])  # [L]

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    for ax, mask, title in [
        (axes[0], dropout_mask, "dropout"),
        (axes[1], gap_mask, "gap"),
        (axes[2], hold_mask, "hold"),
    ]:
        ax.step(time_idx.numpy(), mask.numpy(), where="mid")
        ax.set_title(title)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Missingness masks (sample 0)")
    fig.tight_layout()
    output_path = output_dir / "05_param_samplers_missingness_masks.png"
    fig.savefig(output_path)
    print(f"saved figure {output_path}")


def main() -> None:
    """Run sampler demos and save plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(__file__).resolve().parent / "figures"

    sampler_samples, choice_labels = sample_samplers(device)
    plot_sampler_distributions(sampler_samples, choice_labels, output_dir)

    outputs = run_pipeline_demo(device)
    plot_pipeline_steps(outputs, output_dir)

    missingness_output = next(
        output for output in outputs if output[0] == "missingness"
    )
    plot_missingness_masks(missingness_output, output_dir)


if __name__ == "__main__":
    main()
