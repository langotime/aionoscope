# Plan: PTB-XL ECG rhythm simulation (12 rhythm SCP labels)

## Goal
Extend ToyTS ECG simulation so it can generate ECG-like signals labeled with **all 12 PTB-XL “rhythm” SCP codes** from `scp_statements.csv`, while keeping the design **extendable to the full 71 SCP labels**.

Rhythm codes (from `scp_statements.csv`, `rhythm != ""`):
- `SR` (sinus rhythm)
- `AFIB` (atrial fibrillation)
- `STACH` (sinus tachycardia)
- `SARRH` (sinus arrhythmia)
- `SBRAD` (sinus bradycardia)
- `PACE` (paced rhythm)
- `SVARR` (supraventricular arrhythmia)
- `BIGU` (bigeminy)
- `AFLT` (atrial flutter)
- `SVTAC` (supraventricular tachycardia)
- `PSVT` (paroxysmal supraventricular tachycardia)
- `TRIGU` (trigeminy)

## Non-goals (for this first implementation)
- Perfect physiological fidelity / clinical-grade morphology.
- Supporting diagnostic/form SCP labels beyond the shared scaffolding.
- Training benchmarks; only add “cheap shortcut checks” in tests.

## Guiding constraints (repo rules)
- **Process defines labels** and latent generative choices; **views only distort/present**.
- Deterministic with a provided `torch.Generator`; no global seeding.
- No silent defaults: explicitly require key choices (label set, sampling assumptions).
- Avoid loops over batch/time; prefer vectorized tensor ops.

---

## Source of truth: `scp_statements.csv`
### What we will do
1) Treat `scp_statements.csv` as the authoritative mapping for:
   - which SCP codes belong to `diagnostic` / `form` / `rhythm`,
   - their human descriptions,
   - (later) the `diagnostic_class` / `diagnostic_subclass` hierarchy.
2) Move/copy this file into the installable package as data (so users don’t need repo-root files).

### Planned code addition
- `toyts/ptbxl/scp.py`
  - `load_scp_statements() -> dict[str, SCPStatement]` (parses packaged CSV)
  - `ptbxl_rhythm_codes() -> list[str]` (returns the 12 codes, CSV order)
  - `ptbxl_all_codes() -> list[str]` (all 71 SCP codes, CSV order)
  - `ptbxl_codes_by_group(group: Literal["diagnostic","form","rhythm"]) -> list[str]`

### Packaging
- Add `toyts/assets/ptbxl/scp_statements.csv` and include it via setuptools `package-data`.

Why: this keeps the future “support all 71 labels” path incremental and avoids hardcoding label lists in multiple places.

---

## Proposed public API (rhythm generation)
### New process
- `toyts.processes.ECGProcess`

Constructor (high-level):
- `seq_len: int`
- `sample_rate_hz: float`
- `rhythm_codes: list[str]` (explicit; v1 supports the 12 PTB-XL rhythm codes; order defines class index)
- `rhythm_sampler: Sampler` (samples `int64` class indices; supports imbalanced datasets)
- `amplitude: SamplerLike[float]` (beat amplitude; sampled independently of rhythm by default)
- `rhythm_params: ECGRhythmParams` (small dataclass with explicit per-class parameter ranges)

Outputs:
- `LatentState.events`: a single merged `EventBatch` containing all generated events (beats + rhythm-specific auxiliary events).
- `LatentState.y["rhythm"]`: `int64[B]` class index in `rhythm_codes`.
- `LatentState.meta["label_names"]["rhythm"] = rhythm_codes`.

Rationale:
- Keeps the process name generic while letting the caller choose a concrete taxonomy (PTB-XL today, other taxonomies later).
- Event-first output matches the existing ECG rendering pipeline (`EventImpulseView -> KernelConvView -> ECGLeadsView`).

### Kernel bank helper (for views)
Add a kernels helper that matches the process’ `EventSchema.type_names`:
- `toyts.kernels.make_ecg_kernel_bank(...) -> torch.Tensor` returning `[K, T, W]`

This will build:
- A single PQRST-like kernel for a canonical beat event type (`type_name="beat"`) by reusing `make_pqrst_kernel_bank(shape_names=["gaussian"], ...)`.
- Additional kernels for rhythm-specific auxiliary event types:
  - pacer spikes
  - flutter waves

We keep `K=3` latent components for now (P / QRS / T-ish) to preserve compatibility with `utils_make_canonical_A0(num_latent=3)` and existing examples.

---

## Process design (latent dynamics)
### A) Event schema (types and params)
We keep a single `EventSchema` across all samples:
- `time_unit="samples"`
- `param_names=["amplitude"]` (keep MVP simple)
- `type_names=["beat", "pace_spike", "flutter_wave"]` (always present; individual event masks decide which are used)

Extendability path:
- Future SCP labels (form/diagnostic) can add new event types and/or expand `param_names` (e.g., per-component scaling) without changing the view interfaces.

### B) Rhythm-to-latent mapping (what changes per rhythm)
We represent rhythm labels primarily via **beat timing patterns**, optionally augmented by auxiliary events:

1) `SR`: regular-ish RR with small jitter, normal rate range.
2) `SBRAD`: regular-ish RR, low rate range.
3) `STACH`: regular-ish RR, high rate range.
4) `SARRH`: smooth RR modulation (respiratory-like sinusoidal modulation), P waves preserved.
5) `AFIB`: irregularly-irregular RR (high variability); do not rely on view noise.
6) `AFLT`: regular ventricular response from a sampled AV conduction ratio; add high-rate `flutter_wave` events.
7) `SVARR`: mostly regular with occasional premature beats (short RR + compensatory pause); variability lower than AFIB.
8) `SVTAC`: sustained tachycardia (regular high rate).
9) `PSVT`: paroxysmal episode inside the window (SR baseline + SVT episode with abrupt onset/offset).
10) `BIGU`: alternating short/long RR pattern (premature every other beat).
11) `TRIGU`: premature every third beat (short/long pair repeating every 3 beats).
12) `PACE`: regular paced beats + `pace_spike` events before each beat.

Key rule: any rhythm-specific morphology we add (spikes/waves) is created in the **process**, not as a view effect, so labels remain a function of latent state.

### C) Vectorized beat-time generation (no Python loops over B)
Implement a dedicated process node (new file, ECG-specific) that generates:
- `times_beats: float32[B, E_beats]`
- `mask_beats: bool[B, E_beats]`
- `type_ids_beats: int64[B, E_beats]` (usually a single beat type, unless we add beat morphology types)

Core method (chosen for KISS + flexibility):
1) Sample per-sample target heart rate `hr_bpm[B]` conditional on `rhythm` (PTB-XL codes in v1).
2) Convert to per-sample target RR in samples: `rr_samples[B] = sample_rate_hz * 60 / hr_bpm`.
3) Choose a global maximum beat count `E_beats` from the highest supported HR range:
   - `E_beats = ceil(duration_sec * hr_bpm_max / 60) + margin`
4) Generate per-sample interval multipliers `m[B, E_beats+1]` based on rhythm:
   - regular-ish: `m=1 + small_noise`
   - AFIB: `m = -log(U)` (exponential-like variability)
   - SARRH: `m = 1 + a*sin(phase + step*arange)`
   - BIGU/TRIGU: fixed periodic multiplier patterns
   - SVARR: mostly 1, with scattered short/long adjustments
5) Convert to normalized intervals over the window and cumulative-sum to beat times:
   - create `intervals_raw = clamp(m, min=eps)` and normalize each sample so `intervals_raw.sum()==1`
   - `times_norm = cumsum(intervals, dim=1)[:, :-1]` then scale to samples
6) Compute a per-sample valid beat count mask based on `hr_bpm` and `duration_sec` (not based on label id directly):
   - `n_beats[B] = round(duration_sec * hr_bpm / 60)`
   - `mask_beats = arange(E_beats) < n_beats[:, None]`
7) For `PSVT`, generate two beat trains (baseline + tachy) and gate them by an episode time window, then `UnionEventsNode` + sort.

All sampling parameters used above are recorded into `meta["samples"]` via `__samples__/...` keys (store scalars/vectors like `hr_bpm`, episode window, conduction ratio; do not store full-size noise fields).

### D) Auxiliary events (pace spikes / flutter / fib)
Generate additional event batches with fixed maximum lengths and masks, then merge via `UnionEventsNode`:
- `pace_spike`: one per valid beat, at `beat_time - spike_delay_samples`
- `flutter_wave`: regular high-rate event train across the window (AFLT only)

This keeps a clean extension path: future SCP codes can add their own auxiliary event generators without rewriting the beat generator.

---

## View design (observation models)
We do not need new view types for this phase; we reuse the existing ECG rendering pipeline:
1) `EventImpulseView(seq_len=..., amplitude_param="amplitude", rounding="nearest")` → `[B, T, L]`
2) `KernelConvView(kernels=..., padding=...)` → `[B, K, L]`
3) `ECGLeadsView(A0=..., jitter_std=..., max_delay=...)` → `[B, C, L]`
4) Optional existing distortions: `BaselineWanderView`, `GaussianNoiseView`, `SamplingAggregationView`, `MissingnessView`, `NormalizeView`, etc.

Critical constraint: any *label-dependent* artifacts must remain in the process (events/kernels), not in the views.

---

## Extendability to all 71 SCP labels (design choices now)
### 1) One label taxonomy module, one stable ordering
The `toyts/ptbxl/scp.py` loader provides:
- stable list of 71 codes and group membership,
- descriptions and (later) diagnostic class/subclass.

All future label work uses this module, avoiding duplicated hardcoded lists.

### 2) “Effect modules” as ProcessGraph nodes
As we add more SCP labels, we implement small, composable `ProcessOp`s that:
- modify beat timing (rhythm-like codes),
- add auxiliary events (e.g., pacer spikes, ectopy markers),
- modify morphology via kernels/event params (e.g., ST/T changes),
- (later) apply lead-localized effects as process-level latent components (not view noise).

Each new SCP label should correspond to a single well-scoped node (or a tiny group), gated by a label mask derived from `state.y`.

### 3) Multi-label readiness without forcing it now
PTB-XL is fundamentally multi-label across 71 SCP codes.
For this phase, we keep `rhythm` as a clean 12-class label, but we will:
- keep the taxonomy API ready to later add `y["scp_multi_hot"]: bool[B, 71]` when we start supporting non-mutually-exclusive labels (PTB-XL code order from `ptbxl_all_codes()`).

---

## Tests (pytest)
Add tests under `tests/` that are cheap, deterministic, and target failure modes:

1) `test_ptbxl_scp_loader_rhythm_codes`
   - parses packaged CSV and asserts the rhythm code list equals the 12 codes above (order matters).

2) `test_ecg_process_ptbxl_rhythm_shapes_and_dtypes`
   - events tensors shapes/dtypes, `y["rhythm"]` shape `[B]`, meta contains label_names.

3) `test_ecg_process_ptbxl_rhythm_determinism`
   - fixed `torch.Generator` ⇒ identical outputs.

4) Per-rhythm property tests (small B)
   - `SBRAD` produces fewer beats than `SR` on average; `STACH` more.
   - `AFIB` has higher RR-variability metric than `SR`/`SARRH`.
   - `BIGU` / `TRIGU` show periodic interval patterns.
   - `PACE` produces `pace_spike` events aligned to beats.
   - `AFLT` produces many `flutter_wave` events.
   - `PSVT` shows an in-window rate shift (simple split-half HR estimate differs).

5) Shortcut checks (anti-cheating)
   - replicate the existing “simple stats nearest-center baseline” on `rhythm` and enforce a conservative ceiling (tune threshold so it catches blatant shortcuts but doesn’t overconstrain).
   - add a label-shuffle control where accuracy ~ chance.

All tests must run via `uv run pytest`.

---

## Documentation + examples
### README.md
- Add a short example: `ECGProcess` + kernel bank + ECG leads view, showing how to decode `rhythm` via `meta["process"]["label_names"]`.

### DOCUMENTATION.md
- Add a dedicated section “PTB-XL rhythm (12 labels)”:
  - the mapping from rhythm codes to generative patterns (high-level),
  - how to render events into 12-lead ECG,
  - how to reproduce a batch via stored seeds/meta.

### examples/
- Add `examples/ptbxl_rhythm_12_demo.py` + matching `examples/ptbxl_rhythm_12_demo.ipynb`:
  - grid visualization: one sample per rhythm code, same view chain,
  - histogram of sampled labels,
  - print key meta params (HR, episode window, conduction ratio).

---

## Implementation checklist (single phase, but ordered)
1) Add `toyts/ptbxl/scp.py` + package data wiring + loader tests.
2) Add kernel bank helper for `["beat","pace_spike","flutter_wave"]` (small refactor of `toyts/kernels/pqrst.py` if needed to avoid duplication) + unit tests for shape.
3) Add `ECGProcess` + ECG-specific process nodes for beat/aux events.
4) Add docs + example script/notebook.
5) Add shortcut tests + determinism tests for the new process.

## Acceptance criteria
- Pipeline can generate labeled batches for all 12 rhythm codes; labels are present and decodable.
- No view parameter is required to infer the label (labels are decided in the process).
- Deterministic with fixed `torch.Generator`.
- `uv run pytest` passes.
