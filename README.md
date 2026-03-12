# Aionoscope

**PyTorch-native synthetic time series dataset generator with explicit Process → View separation.**

Aionoscope is designed to benchmark **Self-Supervised Learning (SSL)** and supervised approaches on time series data. It provides a flexible, GPU-friendly framework to generate multi-view datasets where the underlying latent dynamics ("Process") are strictly separated from the observation model ("View").

Aionoscope is intended to support composable process graphs across domains. The goal is to build both very simple series (single event, repeated events of one type) and progressively richer combinations, enabling curriculum-style training where SSL models start from isolated events and scale to complex multi-component dynamics.

Use `ARCHITECTURE.md` for the stable design contracts and `DOCUMENTATION.md` for usage details, metadata conventions, and the longer example guide.

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
*   **Per-sample component mixing**: Processes can write `LatentState.meta["enabled"][key]` masks (`bool[B]`) and component Views can gate themselves with `enabled_key=...`, so a single static `ViewChain` can produce heterogeneous 1/2/3/... component mixtures without branching pipelines (including mixed component counts within one batch via `EnableComponentsNode(num_enabled=...)`).
*   **Multi-event rendering**: `EventRenderView` materializes simple event types by summing over events, so mixtures like “spike + gaussian bump + trend” remain straightforward and reproducible.
*   **Modular**: Components are composable `nn.Module`s.

## Why These Features Exist

*   **Static pipelines**: `nn.Sequential` / `ViewChain` are fixed graphs, but research often needs per-sample “recipes” (some samples have 1 component, others have 2–3). `enabled_key` makes that possible without branching the pipeline.
*   **Variable mixture complexity**: Some benchmarks need to mix “easy” (single-component) and “hard” (multi-component) samples in the same dataset stream; sampling `EnableComponentsNode(num_enabled=...)` per sample enables this without branching pipelines or dataloaders.
*   **Stable randomness**: When `enabled_key` varies across samples, `ViewChain` splits the RNG per view so turning one component on/off does not change randomness in later components.
*   **Composable event mixtures**: Processes already merge multiple event streams; `EventRenderView` exists so those merged streams can be rendered additively (multiple events per sample) without writing a custom renderer.

See `examples/06_basic_components.py` for balanced per-sample component selection, and
`examples/07_imbalanced_components.py` for imbalanced (rare) component sampling via
`CategoricalSampler` + `EnableComponentsNode(..., component_id=...)`.

For imbalanced **k-hot mixtures** (`num_enabled > 1`), see `examples/08_imbalanced_mixtures.py` and
`WeightedPermutationSampler` + `EnableComponentsNode(..., component_order=...)`.

## Public Patterns

*   Parameters can be fixed scalars or `SamplerLike` objects, so the same API works for deterministic and sampled generators.
*   Processes can stay event-first and let views render with `EventImpulseView`, `KernelConvView`, or `EventRenderView`.
*   One static `ViewChain` can serve balanced, imbalanced, and variable-complexity mixtures through `EnableComponentsNode` and `enabled_key`.
*   PTB-XL exposes one public target, `y["scp"]`; rhythm, diagnostic, and form tasks are slices over `label_groups`, not separate public label heads.

## Example Map

*   `examples/01_simple_pulse.py`: smallest end-to-end process -> view -> batch example.
*   `examples/02_multiview_ssl.py`: generate multiple observation views from one process execution.
*   `examples/06_basic_components.py`: balanced component mixtures with enabled masks.
*   `examples/07_imbalanced_components.py`: imbalanced single-component datasets via `component_id`.
*   `examples/08_imbalanced_mixtures.py`: imbalanced k-hot mixtures via `component_order`.
*   `examples/09_ptbxl_rhythm_12.py`, `examples/10_ptbxl_form_19.py`, `examples/11_ptbxl_diagnostic_44.py`: PTB-XL rhythm/form/diagnostic slices over the same `y["scp"]` tensor.

## Installation

```bash
uv sync
```

## Quick Start

### Simple process graph (two generators)

```python
import torch

from aiono.core.events import EventSchema
from aiono.core.pipeline import SynthPipeline
from aiono.processes.graph import ProcessGraph
from aiono import UniformSampler
from aiono.processes.nodes import EventTrainNode, SingleEventNode, UnionEventsNode
from aiono.views.events import EventStreamView

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
from aiono.core import SynthPipeline
from aiono.core.utils import utils_make_canonical_A0
from aiono.kernels import make_pqrst_kernel_bank, pqrst_kernel_size
from aiono.processes import PulseTrainProcess
from aiono.views import ECGLeadsView, EventImpulseView, KernelConvView, GaussianNoiseView

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

A0 = utils_make_canonical_A0(num_leads=12, num_latent=3).to(device)

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

### PTB-XL SCP multi-label example (ECGProcess)

`ECGProcess` generates ECG-like event streams labeled with the full PTB-XL SCP code set (71 codes). The public target is always `y["scp"]` (bool [B, 71]), and `meta["process"]["label_groups"]` provides rhythm/diagnostic/form slices over that same tensor. PTB-XL examples use the direct 12-lead kernel renderer (`EventImpulseView -> KernelConvView`), so there is no separate rhythm-only public label API.

```python
import torch
from aiono import (
    ECGMorphologyParams,
    ECGProcess,
    ECGRhythmParams,
    EventImpulseView,
    KernelConvView,
    SynthPipeline,
    make_ptbxl_kernel_bank,
    ptbxl_kernel_size,
)
from aiono.ptbxl import PTBXLLabelSetSampler, ptbxl_all_codes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scp_codes = ptbxl_all_codes()
sampler = PTBXLLabelSetSampler(scp_codes=scp_codes, normal_prob=0.25)
process = ECGProcess(
    seq_len=2048,
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
)
padding = kernel_size // 2

views = {
    "clean": torch.nn.Sequential(
        EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
        KernelConvView(kernels=kernels, padding=padding),
    )
}

pipe = SynthPipeline(process, views)
batch = pipe(batch_size=64, device=device)

scp = batch["clean"].y["scp"]  # [64, 71]
label_groups = batch["clean"].meta["process"]["label_groups"]
rhythm = scp[:, label_groups["rhythm"]]  # [64, 12]
```

## Architecture

See `ARCHITECTURE.md` for the design model and metadata contracts, and `DOCUMENTATION.md` for the longer usage guide. At a high level:

*   `aiono.core`: Base types (`LatentState`, `Observation`) and pipeline logic.
*   `aiono.processes`: Latent generators (e.g., `PulseTrainProcess`, `TrendSeasonAnomalyProcess`, `ECGProcess`).
*   `aiono.views`: Observation transforms (e.g., `ECGLeadsView`, `EventImpulseView`, `MissingnessView`).
*   `aiono.kernels`: Signal morphology kernels and PTB-XL lead renderers.
*   `aiono.ptbxl`: SCP taxonomy helpers, phenotype tables, and label samplers.
*   `aiono.datasets`: Iterable dataset wrappers.
