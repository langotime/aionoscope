# ToyTS

**PyTorch-native synthetic time series dataset generator with explicit Process → View separation.**

ToyTS is designed to benchmark **Self-Supervised Learning (SSL)** and supervised approaches on time series data. It provides a flexible, GPU-friendly framework to generate multi-view datasets where the underlying latent dynamics ("Process") are strictly separated from the observation model ("View").

ToyTS is intended to support composable process graphs across domains. The goal is to build both very simple series (single event, repeated events of one type) and progressively richer combinations, enabling curriculum-style training where SSL models start from isolated events and scale to complex multi-component dynamics.

## Design Goals

*   **Composable process graphs**: Reusable blocks with branching/merging to build complex generators from simple parts.
*   **Latent-first processes**: Processes emit latent structures (continuous components and/or event streams); Views handle rendering to regular samples (e.g., impulse + conv1d) or keep event streams as observations.
*   **Curriculum-ready**: Easy progression from single events to multi-component mixtures.
*   **Reproducible**: Deterministic outputs given a fixed seed and graph.

## Core Concepts

The library is built around a unidirectional flow:

1.  **Process (Latent)**: Generates the "true" state of the system (events, hidden regimes, latent components). It defines the *content* and emits latent structures rather than sampled arrays.
    *   *Output:* `LatentState` (centers, latent curves, ground-truth labels).
2.  **View (Observation)**: Transforms the latent state into an observed representation. It defines the *style* and *distortions* (e.g., sensors, noise, sampling rate, missingness) and can render events into samples.
    *   *Output:* `Observation` (tensor `x`, labels `y`).
3.  **Pipeline**: Orchestrates the generation of one or more views from a single process execution, enabling perfect multi-view pairs for SSL.

## Features

*   **GPU-First**: All generation happens in PyTorch tensors. No Python loops over batch dimensions.
*   **SSL-Ready**: Trivial generation of multi-view batches (e.g., `{"clean": ..., "noisy": ...}`) sharing the same latent state.
*   **Anti-Shortcut**: Nuisance factors (noise, amplitude) are handled by Views, while labels are derived from the Process, preventing trivial shortcuts.
*   **Per-sample component mixing**: Processes can write `LatentState.meta["enabled"][key]` masks (`bool[B]`) and component Views can gate themselves with `enabled_key=...`, so a single static `ViewChain` can produce 1/2/3/... component mixtures without branching pipelines.
*   **Multi-event rendering**: `EventRenderView` materializes simple event types by summing over events, so mixtures like “spike + gaussian bump + trend” remain straightforward and reproducible.
*   **Modular**: Components are composable `nn.Module`s.

## Why These Features Exist

*   **Static pipelines**: `nn.Sequential` / `ViewChain` are fixed graphs, but research often needs per-sample “recipes” (some samples have 1 component, others have 2–3). `enabled_key` makes that possible without branching the pipeline.
*   **Stable randomness**: When `enabled_key` varies across samples, `ViewChain` splits the RNG per view so turning one component on/off does not change randomness in later components.
*   **Composable event mixtures**: Processes already merge multiple event streams; `EventRenderView` exists so those merged streams can be rendered additively (multiple events per sample) without writing a custom renderer.

See `examples/06_basic_components.py` for a minimal dataset that samples 1 (or k) enabled components per sample.

## Installation

```bash
pip install -e .
```

## Quick Start

### Simple process graph (two generators)

```python
import torch

from toyts.core.events import EventSchema
from toyts.core.pipeline import SynthPipeline
from toyts.processes.graph import ProcessGraph
from toyts import UniformSampler
from toyts.processes.nodes import EventTrainNode, SingleEventNode, UnionEventsNode
from toyts.views.events import EventStreamView

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

schema = EventSchema(
    type_names=["spike", "pulse"],
    param_names=["amplitude"],
    time_unit="samples",
)

process = ProcessGraph(
    name="simple_combo",
    outputs={"events"},
    base_meta={"seq_len": 512},
    graph=[
        SingleEventNode(
            seq_len=512,
            schema=schema,
            type_name="spike",
            time_min=64,
            time_max=448,
            amplitude=UniformSampler(0.8, 1.2),
            amplitude_param="amplitude",
            out_key="single",
        ),
        EventTrainNode(
            seq_len=512,
            num_events=6,
            schema=schema,
            mode="regular",
            type_label_key=None,
            type_id=schema.type_id("pulse"),
            amplitude=0.5,
            amplitude_param="amplitude",
            missed_gap_factor=2.0,
            out_key="train",
            centers_out_key="train_centers",
        ),
        UnionEventsNode(in_keys=["single", "train"], out_key="events"),
    ],
)

views = {"events": EventStreamView()}
pipe = SynthPipeline(process=process, views=views)
batch = pipe(batch_size=4, device=device)

events = batch["events"].x  # [B, E, 2+P]
```

### ECG wrapper example (complex process)

`PulseTrainProcess` is a reusable wrapper around a richer graph. This example renders events into dense samples and mixes them into ECG leads.

```python
import torch
from toyts.core import SynthPipeline
from toyts.core.utils import utils_make_canonical_A0
from toyts.kernels import make_pqrst_kernel_bank, pqrst_kernel_size
from toyts.processes import PulseTrainProcess
from toyts.views import ECGLeadsView, EventImpulseView, KernelConvView, GaussianNoiseView

# 1. Define the Process (Latent Dynamics)
# Generates an event stream with rhythm and shape labels
process = PulseTrainProcess(
    seq_len=2048,
    frequency_hz=1.95,
    sample_rate_hz=500.0,
    rhythm_classes=["regular", "irregular", "missed_beat"],
    shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
    latent_mode="pqrst3"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Build kernels and views (Observation Models)
spacing = (process.seq_len - 1) / (process.num_pulses + 1)
kernel_size = pqrst_kernel_size(spacing=spacing, support_sigma=6.0)
kernels = make_pqrst_kernel_bank(
    shape_names=process.shape_classes,
    spacing=spacing,
    kernel_size=kernel_size,
    device=device,
)
padding = kernel_size // 2

A0 = utils_make_canonical_A0(num_leads=12, num_latent=3)

# Create two views of the same process for SSL training
views = {
    "clean": torch.nn.Sequential(
        EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
        KernelConvView(kernels=kernels, padding=padding),
        ECGLeadsView(A0=A0, jitter_std=0.01, max_delay=0),
    ),
    "noisy": torch.nn.Sequential(
        EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
        KernelConvView(kernels=kernels, padding=padding),
        ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=2),
        GaussianNoiseView(noise_std=0.2)
    )
}

# 3. Build Pipeline
pipe = SynthPipeline(process, views)

# 4. Generate Batch on GPU
batch = pipe(batch_size=64, device=device)

x_clean = batch["clean"].x       # [64, 12, 2048]
x_noisy = batch["noisy"].x       # [64, 12, 2048]
labels = batch["clean"].y        # {"rhythm": [64], "shape": [64]}
```

## Architecture

*   `toyts.core`: Base types (`LatentState`, `Observation`) and pipeline logic.
*   `toyts.processes`: Latent generators (e.g., `PulseTrain`, `TrendSeasonAnomaly`).
*   `toyts.views`: Observation transforms (e.g., `ECGLeads`, `Units`, `Sampling`, `Missingness`).
*   `toyts.kernels`: Signal morphology kernels.
