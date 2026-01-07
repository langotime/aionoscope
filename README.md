# ToyTS

**PyTorch-native synthetic time series dataset generator with explicit Process → View separation.**

ToyTS is designed to benchmark **Self-Supervised Learning (SSL)** and supervised approaches on time series data. It provides a flexible, GPU-friendly framework to generate multi-view datasets where the underlying latent dynamics ("Process") are strictly separated from the observation model ("View").

## Core Concepts

The library is built around a unidirectional flow:

1.  **Process (Latent)**: Generates the "true" state of the system (events, hidden regimes, latent components). It defines the *content* (e.g., rhythm, morphology classes).
    *   *Output:* `LatentState` (centers, latent curves, ground-truth labels).
2.  **View (Observation)**: Transforms the latent state into an observed time series. It defines the *style* and *distortions* (e.g., sensors, noise, sampling rate, missingness).
    *   *Output:* `Observation` (tensor `x`, labels `y`).
3.  **Pipeline**: Orchestrates the generation of one or more views from a single process execution, enabling perfect multi-view pairs for SSL.

## Features

*   **GPU-First**: All generation happens in PyTorch tensors. No Python loops over batch dimensions.
*   **SSL-Ready**: Trivial generation of multi-view batches (e.g., `{"clean": ..., "noisy": ...}`) sharing the same latent state.
*   **Anti-Shortcut**: Nuisance factors (noise, amplitude) are handled by Views, while labels are derived from the Process, preventing trivial shortcuts.
*   **Modular**: Components are composable `nn.Module`s.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from toyts.processes import PulseTrainProcess
from toyts.views import ECGLeadsView, NoiseView
from toyts.core import SynthPipeline

# 1. Define the Process (Latent Dynamics)
# Generates ECG-like pulses with specific rhythm and shape classes
process = PulseTrainProcess(
    seq_len=2048,
    num_pulses=8,
    rhythm_classes=["regular", "irregular", "missed_beat"],
    shape_classes=["gaussian", "sharp", "biphasic"],
    latent_mode="pqrst3"  # Generates 3 latent components (P, QRS, T)
)

# 2. Define Views (Observation Models)
# Create two views of the same process for SSL training
views = {
    "clean": ECGLeadsView(num_leads=12, jitter_std=0.01),
    "noisy": torch.nn.Sequential(
        ECGLeadsView(num_leads=12, jitter_std=0.05),
        NoiseView(noise_std=0.2)
    )
}

# 3. Build Pipeline
pipe = SynthPipeline(process, views)

# 4. Generate Batch on GPU
batch = pipe(batch_size=64, device=torch.device("cuda"))

x_clean = batch["clean"].x       # [64, 12, 2048]
x_noisy = batch["noisy"].x       # [64, 12, 2048]
labels = batch["clean"].y        # {"rhythm": [64], "shape": [64]}
```

## Architecture

*   `toyts.core`: Base types (`LatentState`, `Observation`) and pipeline logic.
*   `toyts.processes`: Latent generators (e.g., `PulseTrain`, `TrendSeasonAnomaly`).
*   `toyts.views`: Observation transforms (e.g., `ECGLeads`, `Units`, `Sampling`, `Missingness`).
*   `toyts.kernels`: Signal morphology kernels.
