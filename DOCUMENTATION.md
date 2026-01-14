# ToyTS Documentation

## Introduction

ToyTS is a PyTorch-native synthetic time series dataset generator designed for researchers and developers working on machine learning for time series data. Its primary goal is to provide a flexible and powerful tool for benchmarking Self-Supervised Learning (SSL) and supervised models.

The core philosophy of ToyTS is the strict separation of the data-generating **Process** from the **View** or observation model. This design principle is crucial for developing and testing models that are robust to superficial variations in the data and can learn the underlying dynamics of the system.

ToyTS is intended to support composable processes across domains. The goal is to generate everything from the simplest time series (single event, repeated events of one type) to complex combinations, so SSL models can be trained with a curriculum that starts with isolated events and gradually increases complexity.

### Design Goals

*   **Composable process graphs**: Build complex generators from reusable blocks with branching and merging.
*   **Latent-first processes**: Processes emit latent structures (continuous components and/or event streams). Views are responsible for rendering into regular samples (e.g., impulse + conv1d) or keeping event streams as observations.
*   **Event streams as first-class**: Raw events are a valid representation for training and evaluation.
*   **Curriculum-ready**: Gradually scale complexity from single events to multi-component mixtures.
*   **Reproducible**: Deterministic outputs given a fixed seed and graph.

### Why ToyTS?

*   **Benchmark SSL**: ToyTS allows you to generate multiple, augmented "views" of the same underlying data, which is a key requirement for many SSL techniques, such as contrastive learning.
*   **Avoid Shortcuts**: By separating the latent process (which determines the labels) from the view (which introduces noise, augmentations, and other distortions), ToyTS helps in training models that learn meaningful features rather than exploiting superficial cues.
*   **GPU-Native**: The entire data generation pipeline is implemented in PyTorch, making it extremely fast and suitable for generating data on-the-fly during training, directly on the GPU.
*   **Modular and Extensible**: The library is designed to be easily extensible. You can create your own custom `Process` and `View` modules to simulate a wide variety of time series data.

## Avoiding Shortcuts (Anti-Cheating)

A *shortcut* (or *cheat*) happens when a model can predict labels from non-causal, easy-to-exploit artifacts of the generator (e.g., padding patterns, noise level, resampling rate) instead of learning the intended underlying dynamics. Synthetic datasets are especially prone to this because it is easy to accidentally couple labels to nuisance parameters.

Use the following rules when designing new `Process`/`View` components or composing datasets:

1.  **Define labels from the latent process only**: Labels must come from the `Process` latent state and its own RNG/parameters, not from any `View` parameters or from the final observation tensor `x`.
2.  **Keep views label-invariant by default**: Distributions of nuisance/view parameters (noise, quantization, clipping, missingness, sampling jitter, channel mixing, normalization, etc.) should be independent of `y` unless you are explicitly testing robustness to label-dependent shifts (and then document it).
3.  **Avoid structural cues**: Keep tensor shapes and preprocessing identical across classes. Beware of variable-length sequences, padding values, masks, NaN counts, and view-specific metadata that can reveal the class.
4.  **Avoid trivial global statistics**: If `y` becomes predictable from mean/DC offset, variance, energy, extrema, or similar cheap statistics, the task may be too easy or unintentionally leaking. Add nuisance variation so these statistics overlap across labels unless they are the intended signal.
5.  **Treat `meta` as debug-only**: `meta` exists for reproducibility and analysis. Do not feed it into models. Avoid storing labels (or near-label proxies) in `meta` unless strictly necessary for debugging.
6.  **Validate with shortcut checks**: Always run a sanity baseline (e.g., linear model on simple per-sample summary features) and a label-shuffle control. If performance is far above chance, inspect generator/view coupling and metadata for leakage.

## Core Concepts

The library is built around a unidirectional data flow, which can be visualized as follows:

```
+-----------+      +-----------------+      +-----------------+
|           |      |                 |      |      View 1     |
|  Process  +------>   LatentState   +------> (e.g., clean)   |
|           |      | (y, meta)       |      +-----------------+
+-----------+      |                 |
                   |                 |      +-----------------+
                   |                 +------>      View 2     |
                   |                 |      | (e.g., noisy)   |
                   +-----------------+      +-----------------+
```

1.  **Process**: This is a `torch.nn.Module` that generates the "ground truth" or latent state of the system. It is responsible for creating the underlying dynamics, such as events, regimes, or latent components. A process emits latent structures (not sampled arrays) along with labels and metadata.

2.  **LatentState**: A dataclass that holds the output of a `Process`. It contains:
    *   `centers`: The locations of events (empty if none).
    *   `latent`: The latent signal components (optional).
    *   `events`: Event streams or event parameters when needed.
    *   `y`: A dictionary of ground-truth labels.
    *   `meta`: A dictionary of metadata about the generation process.

3.  **View**: This is also a `torch.nn.Module` that transforms a `LatentState` or another `Observation` into a new `Observation`. Views are used to model how the latent state is observed in the real world. This can include rendering events into samples, adding noise, simulating sensors, introducing missing data, or changing the sampling rate.

4.  **Observation**: A dataclass that holds the output of a `View`. It contains:
    *   `x`: The observed representation (dense samples or event tensors).
    *   `y`: The ground-truth labels (passed through from the `LatentState`).
    *   `meta`: A dictionary of metadata. Process metadata is stored under `meta["process"]`. View metadata is stored under `meta["views"]` as an ordered list (one entry per view in the chain). `SynthPipeline` also adds `meta["pipeline_seed"]`. Use `Observation.view_meta("ViewName")` to fetch a specific view’s metadata.

5.  **SynthPipeline**: This module orchestrates the data generation. It takes a `Process` and a dictionary of `Views` and, when called, generates a batch of data containing all the requested views of the same underlying latent state.

## Sampled Parameters in Meta

Process-level sampled parameters that are not already present in outputs are stored under `LatentState.meta["samples"]`. This is a nested dictionary keyed by a process/node identifier (for example, `"TrendSeasonAnomalyProcess"` or `"EventTrainNode:events"`), with tensors for each sampled parameter.

For views, sampled parameters live in `Observation.meta["views"]`. Each view records sampler outputs under `samples` and sampler configuration under `spec` (legacy top-level keys remain). Large masks/noise are not stored; instead views expose helpers to regenerate them from seeds. For example, to re-create MissingnessView masks:

```python
miss_meta = observation.view_meta("MissingnessView")
masks = MissingnessView.sample_masks(
    miss_meta,
    shape=observation.x.shape,
    device=observation.x.device,
)
```

## Runtime Component Gating (Enabled Masks)

Many research datasets need **per-sample mixtures**: some samples contain 1 component, others contain 2–3 (trend + periodic + noise, etc.). A `ViewChain` / `nn.Sequential` is a static pipeline, so you cannot “remove” modules on a per-sample basis without building separate pipelines.

ToyTS supports runtime per-sample gating via **enabled masks**:

*   **Process side**: write `state.meta["enabled"][key] = bool[B]` for each component key.
*   **View side**: component views accept `enabled_key=key` and gate their contribution using the corresponding `bool[B]` mask from process metadata (`Observation.meta["process"]` after the first view).

Why this exists:

*   **Single pipeline, many recipes**: chain all components once, then select which ones are active per sample.
*   **Reproducibility under gating**: `ViewChain` splits the RNG per view, so turning one component on/off does not change the random stream used by later views.
*   **Variable mixture complexity**: sample `EnableComponentsNode(num_enabled=...)` per sample to mix different k-hot sizes (easy → hard) within the same batch/stream.

For convenient mask sampling, use `EnableComponentsNode(component_keys=[...], num_enabled=...)`:

*   `num_enabled=int` → fixed k-hot size for the whole batch
*   `num_enabled=SamplerLike[int]` (e.g. `RandIntSampler(1, N + 1)`) → per-sample k-hot sizes within the same batch

See `examples/06_basic_components.py` for a minimal dataset using enabled masks.

## Multi-Event Rendering (Summing Over Events)

Event streams are composable on the process side (generate multiple event sources, then merge them into one `EventBatch`). `EventRenderView` exists to make those merged streams easy to materialize into dense samples:

*   **Input**: `LatentState.events` (an `EventBatch`) + a latent baseline signal.
*   **Output**: an additive single-channel signal `[B, 1, L]` where **all valid events** (`mask == True`) contribute and are **summed**.

This is useful for mixtures like “spike + gaussian bump + trend” where you want to keep the process as events (for labels and reproducibility), but still render a dense training signal.

## Process Graphs and Branching Examples

Process graphs allow non-linear composition, which is hard to express as a simple chain. Typical branching/merging use cases:

*   **Conditional regimes**: route different subgraphs based on a sampled class (e.g., steady vs ramping vs spiky).
*   **Parallel components**: build events, trend, and seasonality in separate branches and merge them.
*   **Optional effects**: apply anomalies or perturbations only for specific classes.
*   **Curriculum generators**: switch between single events, event trains, and mixed dynamics with a shared interface.

## Installation

To install the library, you can clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd toyts
pip install -e .
```

## Simple ProcessGraph Example (Two Generators)

This example builds a small event process from two generator nodes and merges them into one stream.

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
pipeline = SynthPipeline(process=process, views=views)
batch = pipeline(batch_size=4, device=device)

events = batch["events"].x  # [B, E, 2+P]
```

## Quick Start: A Detailed Walk-through

Let's walk through the example from `toyts/examples/01_simple_pulse.py` to understand how to use the library.
`PulseTrainProcess` is a reusable wrapper around a richer process graph; this walkthrough uses it to demonstrate ECG-style rendering.

### Step 1: Define the Process

First, we define a `Process` that will generate our latent events. In this case, we use `PulseTrainProcess` to generate an ECG-like event stream.

```python
from toyts.processes.pulse_train import PulseTrainProcess

process = PulseTrainProcess(
    seq_len=1024,
    frequency_hz=1.2,
    sample_rate_hz=250.0,
    rhythm_classes=["regular", "irregular", "missed_beat"],
    shape_classes=["gaussian", "sharp_laplace", "biphasic_dog"],
    latent_mode="pqrst3",
    amplitude=2.0,
)
```

This process will generate events over a 1024-sample window at 250 Hz with a ~1.2 Hz pulse rate (~72 bpm). The pulses can have different rhythms and shapes, and the event types encode the QRS morphology.

### Step 2: Define the Views

Next, we define a set of `Views` to render events into latent components, then transform them into observed signals. We can create multiple views to simulate different observation conditions.

```python
import torch
from toyts.core.utils import utils_make_canonical_A0
from toyts.kernels.pqrst import make_pqrst_kernel_bank, pqrst_kernel_size
from toyts.views.ecg_leads import ECGLeadsView
from toyts.views.events import EventImpulseView, KernelConvView
from toyts.views.noise import GaussianNoiseView, BaselineWanderView
from toyts.views.sampling import SamplingAggregationView
from toyts.views.units import NormalizeView

spacing = (process.seq_len - 1) / (process.num_pulses + 1)
kernel_size = pqrst_kernel_size(spacing=spacing, support_sigma=6.0)
kernels = make_pqrst_kernel_bank(
    shape_names=process.shape_classes,
    spacing=spacing,
    kernel_size=kernel_size,
    device=torch.device("cpu"),
)
padding = kernel_size // 2

A0 = utils_make_canonical_A0(num_leads=8, num_latent=3)
views = {
    "clean": torch.nn.Sequential(
        EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
        KernelConvView(kernels=kernels, padding=padding),
        ECGLeadsView(A0=A0, jitter_std=0.0, max_delay=0),
    ),
    "noisy": torch.nn.Sequential(
        EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
        KernelConvView(kernels=kernels, padding=padding),
        ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=3),
        GaussianNoiseView(noise_std=0.1),
        BaselineWanderView(amplitude_std=0.3, freq_min=0.05, freq_max=0.2),
    ),
    "normalized_and_resampled": torch.nn.Sequential(
        EventImpulseView(seq_len=process.seq_len, amplitude_param="amplitude", rounding="nearest"),
        KernelConvView(kernels=kernels, padding=padding),
        ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=3),
        GaussianNoiseView(noise_std=0.1),
        BaselineWanderView(amplitude_std=0.3, freq_min=0.05, freq_max=0.2),
        NormalizeView(),
        SamplingAggregationView(mode="mean", window=4),
    ),
}
```

Here, we define three views:
*   `clean`: A clean view with 8 leads, no jitter, and no delay.
*   `noisy`: A noisy view with jitter, delay, additive noise, and baseline wander.
*   `normalized_and_resampled`: A view that is noisy, normalized, and resampled.

### Step 3: Create and Run the Pipeline

Now, we create a `SynthPipeline` with our process and views, move it to the desired device, and generate a batch of data.

```python
from toyts.core.pipeline import SynthPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = torch.Generator(device=device).manual_seed(1234)

pipeline = SynthPipeline(process=process, views=views)
pipeline.to(device)
batch = pipeline(batch_size=4, device=device, rng=rng)
```

### Step 4: Inspect the Output

The output `batch` is a dictionary where the keys are the names of the views. Each value is an `Observation` object.

```python
clean_signal = batch["clean"].x  # Shape: [4, 8, 1024]
noisy_signal = batch["noisy"].x  # Shape: [4, 8, 1024]
resampled_signal = batch["normalized_and_resampled"].x # Shape: [4, 8, 256]

# The labels are the same for all views
labels = batch["clean"].y
print(labels)
```

## Creating Custom Components

### Custom Process

To create a custom process, you need to inherit from `toyts.processes.base.Process` and implement the `forward` method. The `forward` method should return a `LatentState` object.

### Custom View

To create a custom view, you need to inherit from `toyts.views.base.View` and implement the `forward` method. The `forward` method takes a `LatentState` or `Observation` as input and should return an `Observation` object.

## Library Architecture

*   `toyts.core`: Base types (`LatentState`, `Observation`) and pipeline logic.
*   `toyts.processes`: Latent generators (e.g., `PulseTrainProcess`, `TrendSeasonAnomalyProcess`).
*   `toyts.views`: Observation transforms (e.g., `ECGLeadsView`, `UnitsAbsoluteView`, `SamplingAggregationView`, `MissingnessView`).
*   `toyts.kernels`: Signal morphology kernels used by some processes.
*   `toyts.datasets`: Iterable datasets for easy integration with `torch.utils.data.DataLoader`.
*   `toyts.examples`: Example scripts to get you started.

This documentation provides a starting point for using ToyTS. For more details on specific modules and their parameters, please refer to the in-code docstrings.
