# Aionoscope Architecture

This document records the stable design choices that define the current library shape. Use [README.md](README.md) for onboarding and [DOCUMENTATION.md](DOCUMENTATION.md) for runnable usage patterns and examples.

## Goals

*   Benchmark SSL and supervised models on synthetic time series.
*   Support a progression from isolated events/components to richer mixtures and curricula.
*   Keep latent content separate from observation nuisances.
*   Stay reproducible, GPU-friendly, and vectorized.

## System Model

Aionoscope follows a one-way pipeline:

```text
Process -> LatentState -> ViewChain / named views -> Observation
```

Stable rules:

*   A `Process` owns latent truth, labels, and process metadata.
*   A `View` owns presentation: rendering, sensor mixing, units, sampling, missingness, and noise.
*   Labels are functions of process state, not of view parameters, unless a task explicitly documents label-dependent robustness testing.
*   Event streams and dense latent components are both first-class latent representations.

## Core Contracts

### `LatentState`

*   `centers`: event centers or an empty `[B, 0]` tensor when not used.
*   `latent`: optional dense latent components `[B, K, L]`.
*   `events`: optional `EventBatch`.
*   `y`: label dictionary.
*   `meta`: process metadata, seeds, and reproducibility payloads.

### `EventBatch`

`EventBatch` is the canonical event representation:

*   `times`: `[B, E]`
*   `type_ids`: `[B, E]`
*   `params`: `[B, E, P]`
*   `mask`: `[B, E]`
*   `schema`: stable type and parameter names

Event types are not observation channels. Views decide how event types map to impulse trains, kernel inputs, or observed channels.

### `Observation`

*   `x`: observed representation `[B, C, L]`, or a packed event tensor for `EventStreamView`.
*   `y`: passthrough labels from the process.
*   `meta`: always preserves `process`; `ViewChain` adds ordered `views`; `SynthPipeline` adds `pipeline_seed`.
*   `Observation.view_meta(view_name)` is the supported way to fetch one view's metadata entry.

## Execution Model

*   `SynthPipeline` runs the process once per batch and fans the same latent state out to named views.
*   `ProcessGraph` is the main latent execution model.
*   `ProcessChain` is linear sugar over `ProcessGraph`.
*   `Seq`, `Switch`, `Parallel`, and `Scope` are structural graph operators.
*   Graph nodes communicate through explicit `state.data` and `state.y` writes. There is no separate dependency DSL.

RNG behavior is structural and deterministic:

*   `SynthPipeline` splits one generator into one process RNG and one RNG per named view.
*   `ProcessGraph` container ops split RNGs for child ops in a fixed order.
*   `ViewChain` splits RNGs per view so enabling or disabling one component does not perturb later views.
*   Process execution records `trace_seeds` in process metadata for reproducibility/debugging.

## Rendering Architecture

### Generic paths

*   Dense-first: the process emits `latent`, and views transform it into observed signals.
*   Event-first: the process emits `events`, and views decide whether to expose events or render them.

### Event-first building blocks

*   `EventStreamView` exposes packed event tensors for event-native tasks.
*   `EventImpulseView` converts `EventBatch` into per-type impulse trains.
*   `KernelConvView` maps impulse trains to dense components or directly to observed channels.
*   `EventRenderView` directly materializes a small set of additive event families into `[B, 1, L]`.

### ECG-style rendering

`ECGLeadsView` is the generic lead-mixing view for latent component tensors. Its public `A0` forms are:

*   static `[C, K]`
*   batched `[B, C, K]`
*   callable `A0(batch_size, generator, device) -> [B, C, K]`

Mixing jitter and delays are view-side nuisance parameters and therefore stay out of label definition.

### PTB-XL rendering

PTB-XL uses an event-first process plus a direct 12-lead renderer:

```text
ECGProcess -> EventImpulseView -> KernelConvView(make_ptbxl_kernel_bank) -> Observation([B, 12, L])
```

This path does not require `ECGLeadsView`. That is intentional: localization and phenotype effects stay process/kernel-driven rather than becoming label-conditioned sensor noise.

## Physical Units and Sampling

*   When time has physical meaning, public APIs should prefer `frequency_hz` plus `sample_rate_hz` over raw count parameters.
*   `PulseTrainProcess` follows this rule by deriving pulse count from frequency and duration.
*   `EventRenderView` also depends on `sample_rate_hz` in process metadata when event parameters are expressed in seconds.
*   Versioned benchmark semantics that depend on physical units live in `aiono.benchmarks`, not in downstream repos.
*   For the current `aiono_basic_components/v1` benchmark family, `sampling_frequency=500 Hz` is a baseline contract and `frequency_hz: auto` resolves from sequence length plus waveform-specific recoverability rules.
*   `SamplerLike` is the public parameter convention:
    *   scalars and 0-d tensors normalize to `ConstantSampler`
    *   sampler-backed parameters always sample with an explicit `torch.Generator`
*   Shape-affecting sampled parameters must be handled explicitly and fail fast when invalid.

## Metadata and Reproducibility

*   Process metadata stays under `LatentState.meta` and is exposed to observations as `meta["process"]`.
*   View provenance accumulates under `Observation.meta["views"]` as an ordered list.
*   Process-side sampled values that are not already present in outputs live under `meta["samples"]`, keyed by stable node/process identifiers.
*   View-side sampled values live in the corresponding view metadata entry under `samples` and `spec`.
*   Do not store full `[B, C, L]` noise tensors, missingness masks, or other large regenerated artifacts in metadata. Store small sampled values and seeds instead.
*   `meta` is for reproducibility and debugging, not model input.

## Component Mixture Architecture

*   `EnableComponentsNode` writes per-component `enabled` masks (`bool[B]`) into process metadata.
*   Component views consume those masks through `enabled_key`.
*   One static `ViewChain` can therefore generate:
    *   balanced single-component samples
    *   imbalanced single-component datasets via `component_id`
    *   variable-k mixtures via `num_enabled=SamplerLike[int]`
    *   imbalanced k-hot mixtures via `component_order`

This pattern exists to vary mixture complexity without branching the pipeline graph or building separate datasets per recipe.

## PTB-XL Architecture

*   `aiono.ptbxl.scp` packages the SCP taxonomy data and is the source of truth for code ordering and group membership.
*   The public PTB-XL target is one multi-label tensor, `y["scp"]`.
*   `meta["process"]["label_names"]["scp"]` stores the ordered code list.
*   `meta["process"]["label_groups"]` stores rhythm, diagnostic, and form index slices into that same tensor.
*   Rhythm, diagnostic, and form tasks are slices over one target space, not separate public label heads.
*   Phenotype mapping, timing changes, and lead localization belong in the process and PTB-XL kernel design, not in view noise or sensor-specific label hacks.

## Package Map

*   `aiono.core`: base types, RNG helpers, pipeline orchestration, samplers.
*   `aiono.benchmarks`: versioned benchmark contracts and shared resolvers for downstream benchmark consumers.
*   `aiono.processes`: latent generators, graph runtime, and process nodes.
*   `aiono.views`: observation-space renderers and distortions.
*   `aiono.kernels`: kernel banks for event-to-signal rendering.
*   `aiono.ptbxl`: SCP taxonomy helpers, phenotype tables, and label samplers.
*   `aiono.datasets`: iterable dataset wrappers.

## Extension Rules

*   Add a new `Process` when you are changing latent content, event structure, labels, or process-owned randomness.
*   Add a new `View` when you are changing observation space, sensor mixing, units, sampling, missingness, clipping, or noise.
*   Add or change code in `aiono.benchmarks` when a downstream benchmark needs a versioned semantic contract that multiple repos must share.
*   Keep labels as functions of latent process state only unless a benchmark explicitly documents label-dependent augmentation.
*   Prefer vectorized tensor code over Python loops across batch or time.
*   Preserve `meta["process"]`; let `ViewChain` own view provenance accumulation.
*   Document expected invariances and add shortcut checks for new tasks.
