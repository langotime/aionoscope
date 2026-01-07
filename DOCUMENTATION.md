# ToyTS Documentation

## Introduction

ToyTS is a PyTorch-native synthetic time series dataset generator designed for researchers and developers working on machine learning for time series data. Its primary goal is to provide a flexible and powerful tool for benchmarking Self-Supervised Learning (SSL) and supervised models.

The core philosophy of ToyTS is the strict separation of the data-generating **Process** from the **View** or observation model. This design principle is crucial for developing and testing models that are robust to superficial variations in the data and can learn the underlying dynamics of the system.

### Why ToyTS?

*   **Benchmark SSL**: ToyTS allows you to generate multiple, augmented "views" of the same underlying data, which is a key requirement for many SSL techniques, such as contrastive learning.
*   **Avoid Shortcuts**: By separating the latent process (which determines the labels) from the view (which introduces noise, augmentations, and other distortions), ToyTS helps in training models that learn meaningful features rather than exploiting superficial cues.
*   **GPU-Native**: The entire data generation pipeline is implemented in PyTorch, making it extremely fast and suitable for generating data on-the-fly during training, directly on the GPU.
*   **Modular and Extensible**: The library is designed to be easily extensible. You can create your own custom `Process` and `View` modules to simulate a wide variety of time series data.

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

1.  **Process**: This is a `torch.nn.Module` that generates the "ground truth" or latent state of the system. It is responsible for creating the underlying dynamics, such as events, regimes, or latent components. The output of a `Process` is a `LatentState` object, which contains the latent signal, the ground-truth labels (`y`), and metadata.

2.  **LatentState**: A dataclass that holds the output of a `Process`. It contains:
    *   `centers`: The locations of events.
    *   `latent`: The latent signal components.
    *   `y`: A dictionary of ground-truth labels.
    *   `meta`: A dictionary of metadata about the generation process.

3.  **View**: This is also a `torch.nn.Module` that transforms a `LatentState` or another `Observation` into a new `Observation`. Views are used to model how the latent state is observed in the real world. This can include adding noise, simulating sensor outputs, introducing missing data, or changing the sampling rate.

4.  **Observation**: A dataclass that holds the output of a `View`. It contains:
    *   `x`: The observed time series tensor.
    *   `y`: The ground-truth labels (passed through from the `LatentState`).
    *   `meta`: A dictionary of metadata from both the `Process` and the `View`.

5.  **SynthPipeline**: This module orchestrates the data generation. It takes a `Process` and a dictionary of `Views` and, when called, generates a batch of data containing all the requested views of the same underlying latent state.

## Installation

To install the library, you can clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd toyts
pip install -e .
```

## Quick Start: A Detailed Walk-through

Let's walk through the example from `toyts/examples/01_simple_pulse.py` to understand how to use the library.

### Step 1: Define the Process

First, we define a `Process` that will generate our latent signal. In this case, we use `PulseTrainProcess` to generate an ECG-like signal.

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

This process will generate a signal of length 1024 sampled at 250 Hz with a ~1.2 Hz pulse rate (~72 bpm). The pulses can have different rhythms and shapes, and the underlying latent signal will have 3 components (P, QRS, T).

### Step 2: Define the Views

Next, we define a set of `Views` to transform the latent signal into observed signals. We can create multiple views to simulate different observation conditions.

```python
import torch
from toyts.core.utils import utils_make_canonical_A0
from toyts.views.ecg_leads import ECGLeadsView
from toyts.views.noise import NoiseView, BaselineWanderView
from toyts.views.sampling import SamplingAggregationView
from toyts.views.noise import NormalizeView

A0 = utils_make_canonical_A0(num_leads=8, num_latent=3)
views = {
    "clean": ECGLeadsView(A0=A0, jitter_std=0.0, max_delay=0),
    "noisy": torch.nn.Sequential(
        ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=3),
        NoiseView(noise_std=0.1),
        BaselineWanderView(amplitude_std=0.3, freq_min=0.05, freq_max=0.2),
    ),
    "normalized_and_resampled": torch.nn.Sequential(
        ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=3),
        NoiseView(noise_std=0.1),
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
