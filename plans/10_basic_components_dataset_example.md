# Plan: Basic components dataset example (single-component samples)

## Goal
Add a new example that builds a PyTorch dataset where **each sample** is **exactly one** basic time-series component, chosen with **equal probability** across all component types:
- constant value
- noise (uniform, brown, random-walk)
- trend (linear, quadratic, log)
- periodic (sine, sawtooth, square)
- single event (spike, level change, gaussian bump)

Constraints from request:
- **Uniformly sampled parameters** per component (explicit ranges; no hidden defaults).
- **Equal probability** between individual components.
- **Single events** must be generated as an **event stream** (process) and then **materialized by a view**, like ECG.
- **Noise/trend/periodic** must use a **constant-zero generator** (process) plus a **single relevant view**.
- Deterministic/reproducible with an explicit `torch.Generator` / seed; no global `torch.manual_seed`.

## Proposed API / composition (KISS + DRY)

### 1) Add a tiny ConstantProcess (library) with sample rate in meta
Create `toyts/processes/constant.py`:
- `ConstantProcess(seq_len: int, sample_rate_hz: float, value: SamplerLike[float], channels: int = 1)` → `LatentState(latent=constant[B, C, L], events=None, y={}, meta=...)`
  - If `value` is sampled: sample per-sample scalar `value[B]` and broadcast to `[B, C, L]`.
  - Store sampled `value[B]` in `meta["samples"]` (small tensor; no full-size duplication).
  - Store `seq_len` and `sample_rate_hz` in `meta` (so time-dependent views can interpret parameters in physical units).

Rationale: we still get the “constant baseline + single view” pattern by passing `value=0.0`, and we can also represent a pure constant component via a sampled constant baseline. Putting `sample_rate_hz` in **process meta** makes frequency-like parameters unambiguous and shared across views.

### 2) Add basic component Views (library) split by domain
Add additive component views, split into small focused modules:
- `toyts/views/trend.py` (trend components)
- `toyts/views/periodic.py` (periodic components)

Each view:
- Accepts `LatentState | Observation`
- Extracts the base signal (`x_base`) as `[B, C, L]` (sum latent when needed)
- Samples parameters via `SamplerLike` with **explicit** ranges provided by the caller
- Returns `Observation(x=x_base + component, y=input_state.y, meta={...})`
- Stores `seed` and sampled tensors in `meta["samples"]` + `meta["spec"]` (no full-size noise stored)
- Caches `t_grid` as a buffer when `seq_len` is fixed (`[1, 1, L]` float32)

Component set (one view per “individual component” so we can sample uniformly over them):
Trend (`toyts/views/trend.py`):
  - `LinearTrendView(slope: SamplerLike[float], intercept: SamplerLike[float])`
    - `component = slope * (t_grid - 0.5) + intercept`
  - `QuadraticTrendView(a: SamplerLike[float], b: SamplerLike[float], c: SamplerLike[float])`
    - `component = a*(t-0.5)^2 + b*(t-0.5) + c`
  - `LogTrendView(amplitude: SamplerLike[float], offset: SamplerLike[float], epsilon: float)`
    - `component = amplitude * log(epsilon + t_grid) + offset`
    - `epsilon` must be explicitly passed (fail-fast if `<=0`)
  - `ExponentialTrendView(rate: SamplerLike[float], offset: SamplerLike[float])`
    - `component = exp(rate * (t_grid - 0.5)) + offset` (or centered/scaled variant; keep it single-parameter)
  - `SigmoidTrendView(amplitude: SamplerLike[float], center: SamplerLike[float], sharpness: SamplerLike[float], offset: SamplerLike[float])`
    - Smooth “level change” as a trend using a logistic curve over `t_grid`.
  - `PiecewiseLinearTrendView(slope1: SamplerLike[float], slope2: SamplerLike[float], change_t: SamplerLike[float], intercept: SamplerLike[float])`
    - Two linear regimes with one changepoint at `change_t` (in `t_grid` units, 0..1).

Periodic (`toyts/views/periodic.py`):
  - `SineWaveView(amplitude, frequency_hz, phase, offset)`
  - `TriangleWaveView(amplitude, frequency_hz, phase, offset)`
  - `SawtoothWaveView(amplitude, frequency_hz, phase, offset)`
  - `SquareWaveView(amplitude, frequency_hz, phase, offset, duty_cycle)`
  - `ChirpView(amplitude, f0_hz, f1_hz, phase, offset)` (linear frequency sweep over the window)
  - `DampedSineWaveView(amplitude, frequency_hz, tau_sec, phase, offset)` (sine with exponential envelope)
  - `frequency_hz` is interpreted as **Hz** (cycles/second).
  - These views **require** `process_meta["sample_rate_hz"]` (fail-fast if missing) and use
    `t_sec = arange(L) / sample_rate_hz` when constructing the waveform.

Noise components live in `toyts/views/noise.py` (see step 3) so we don’t duplicate “noise” in two places.

### 3) Make existing “signal extraction” composable (small refactor)
Currently `_extract_signal` lives in `toyts/views/units.py`. To avoid duplicating this logic in `components.py`:
- Move/duplicate it into a small internal helper (e.g. `toyts/views/_signal.py`) and reuse in:
  - `toyts/views/units.py`
  - `toyts/views/trend.py`
  - `toyts/views/periodic.py`
  - `toyts/views/noise.py`

Update / extend `toyts/views/noise.py` (required for “ConstantProcess + single view” noise components):
- Rename existing `NoiseView` to `GaussianNoiseView` (clarity) and update all imports/exports/examples/tests/docs.
- Make `GaussianNoiseView` accept `LatentState | Observation` using the shared extraction helper.
- Add missing noise types as additional views in the same module (so the library has one canonical “noise” place):
  - `UniformNoiseView(amplitude: SamplerLike[float])`
  - `LaplaceNoiseView(scale: SamplerLike[float])` (heavy-tailed noise)
  - `RandomWalkNoiseView(step_std: SamplerLike[float])`
  - `BrownNoiseView(step_std: SamplerLike[float])` (either an alias of random-walk, or a normalized variant; pick one canonical definition and document it)
  - `ColoredNoiseView(alpha: SamplerLike[float])` (FFT-based; `alpha=1` pink, `alpha=2` brown)
- Update `BaselineWanderView` to interpret `freq_min/freq_max` as **Hz** and use `process_meta["sample_rate_hz"]` to build `t_sec` (fail-fast if missing).

### 4) Single-event components (event generator + materializing view)
We keep the same pattern as ECG: **process emits EventBatch**, then a **view renders** to `[B, 1, L]`.

Implementation (selected):
- Add `toyts/views/events_basic.py` with `SingleEventRenderView(seq_len: int, kind: Literal["spike","level_change","gaussian","rect_pulse","exp_decay","ringdown"], ...)`
- Expects exactly one valid event per sample (`E==1`, `mask==True`)
- Renders:
  - spike: scatter amplitude into nearest sample index
  - level_change: `x[..., t0:] += amplitude`
  - gaussian: `amplitude * exp(-0.5*((t-t0)/sigma)^2)` (`sigma_sec` sampled by process; uses `process_meta["sample_rate_hz"]`)
  - rect_pulse: `amplitude` applied on `[t0, t0 + duration]` (`duration_sec` sampled by process; uses `sample_rate_hz`)
  - exp_decay: `amplitude * exp(-(t-t0)/tau)` for `t>=t0` (`tau_sec` sampled by process; uses `sample_rate_hz`)
  - ringdown: `amplitude * exp(-(t-t0)/tau) * sin(2π f (t-t0) + phase)` for `t>=t0` (`frequency_hz`, `tau_sec`, `phase` sampled)

Processes for event components:
- Use `ProcessGraph` with `SingleEventNode` (or extend `SingleEventNode` to also write `__samples__` entries for time/amplitude).
- One pipeline per event type to keep “uniform over components” trivial and composable.

### 5) Mixing components with equal probability (dataset-level composition)
To get “equal probability per *sample*” while still yielding efficient **batches**:
- Add `toyts/datasets/mixture.py`:
  - `MixtureSynthBatchIterableDataset(pipelines: list[SynthPipeline], batch_size: int, device, seed, max_batches)`
  - Each batch:
    - sample `component_id ~ Uniform(0..M-1)` as `[B]`
    - group indices by `component_id`
    - run the corresponding pipeline with `batch_size=count`
    - scatter results back into a single batched `Observation` (single view key, e.g. `"x"`)
    - store `component_id` and `component_name` in `Observation.meta` (debug/repro only; not as training label)

Rationale: avoids per-sample branching inside views/processes; loops only over number of components `M` (small).

## Example deliverable
Add:
- `examples/06_basic_components.py`
- `examples/06_basic_components.ipynb` (same content, notebook-style cells; no `main()` in notebook)

Example structure:
1) Define `seq_len`, `device`, and dataset `seed`.
2) Build one pipeline per component:
   - noise: `ConstantProcess(seq_len, sample_rate_hz=..., value=0.0)` + a single noise view from `toyts/views/noise.py` (e.g., `GaussianNoiseView`, `UniformNoiseView`, ...).
   - trend: `ConstantProcess(seq_len, sample_rate_hz=..., value=0.0)` + the single corresponding view from `toyts/views/trend.py`.
   - periodic: `ConstantProcess(seq_len, sample_rate_hz=..., value=0.0)` + the single corresponding view from `toyts/views/periodic.py`.
   - constant: `ConstantProcess(seq_len, sample_rate_hz=..., value=UniformSampler(...))` + `UnitsAbsoluteView()` (single view).
   - events: `ProcessGraph` (single-event) + event materialization view.
3) Wrap pipelines with `MixtureSynthBatchIterableDataset`, then `DataLoader(batch_size=None)`.
4) Take one batch and:
   - print shapes and the per-component counts for sanity
   - plot a grid of signals (e.g. 4x4) with titles `component_name` and key parameters from `view_meta(...)`
   - save to `examples/figures/06_basic_components.png`

## Tests (pytest)
Add focused tests under `tests/`:
- `tests/test_basic_components_views.py`
  - shape/dtype checks (`[B, 1, L]`, float32)
  - determinism: fixed `torch.Generator` → identical outputs
  - range checks where applicable (e.g., uniform noise in [-A, A])
- `tests/test_mixture_dataset.py`
  - yields correct batch shapes
  - deterministic `component_id` sequence for a fixed dataset seed
  - smoke test that all components appear over enough samples (non-flaky: use deterministic ids rather than probabilistic thresholds)

Also extend `tests/test_viewchain_meta_accumulation.py` only if needed for new view meta keys.

## Validation
- `uv run pytest`
- `uv run python examples/06_basic_components.py` (and ensure the figure is generated under `examples/figures/`)

## Open questions for your review
1) Component list (RESOLVED): include base set + additional set above.
2) Parameter ranges: OK to hardcode explicit ranges in the example (RESOLVED: yes).
3) Event rendering: analytic `SingleEventRenderView` (RESOLVED: yes).
