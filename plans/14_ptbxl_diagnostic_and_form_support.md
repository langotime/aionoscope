# Plan: PTB-XL diagnostic + “shape” (form) labels in `ECGProcess`

## Goal
Extend the current PTB-XL ECG simulation so we can generate signals labeled with **all PTB-XL SCP codes (71)**, with a **single coherent multi-label target API** that works for:
- **rhythm** group (12 codes)
- **diagnostic** group (44 codes)
- **form** group (19 codes; “shape” in the dataset)

…in a way that is **extendable to all 71 SCP labels** without a combinatorial explosion of kernels or process branches.

Source of truth: `scp_statements.csv` (already packaged as `aiono/assets/ptbxl/scp_statements.csv` and loaded via `aiono.ptbxl.scp`).

## Scope / definition of “support”
For a label to be considered “supported”, the generated observation must contain a **process-level** morphological/temporal pattern that is **diagnostic for that code** (at least qualitatively), and the label must be fully reproducible from `meta` seeds + sampled parameters.

Non-goal: clinical-grade fidelity; we aim for **research-grade controllable variation** with strong anti-shortcut hygiene.

---

## Label inventory (from `scp_statements.csv`)
### Rhythm codes (12)
`SR, AFIB, STACH, SARRH, SBRAD, PACE, SVARR, BIGU, AFLT, SVTAC, PSVT, TRIGU`

### Diagnostic codes (44)
`NDT, NST_, DIG, LNGQT, NORM, IMI, ASMI, LVH, LAFB, ISC_, IRBBB, 1AVB, IVCD, ISCAL, CRBBB, CLBBB, ILMI, LAO/LAE, AMI, ALMI, ISCIN, INJAS, LMI, ISCIL, LPFB, ISCAS, INJAL, ISCLA, RVH, ANEUR, RAO/RAE, EL, WPW, ILBBB, IPLMI, ISCAN, IPMI, SEHYP, INJIN, INJLA, PMI, 3AVB, INJIL, 2AVB`

### Form (“shape”) codes (19)
`NDT, NST_, DIG, LNGQT, ABQRS, PVC, STD_, VCLVH, QWAVE, LOWT, NT_, PAC, LPR, INVT, LVOLT, HVOLT, TAB_, STE_, PRC(S)`

Notes:
- Some codes appear in both groups (e.g., `NDT`, `NST_`, `DIG`, `LNGQT`). We treat them as **one phenotype**, but expose them under whichever label head is requested.
- Many diagnostic codes are *location-specific* (e.g., `ISCAL`, `INJIN`, `ASMI`), so the simulation must support **lead-group localization** without pushing label-conditioned behavior into views.

---

## High-level design choice (avoid 71-way kernel blowup)
We avoid “one kernel per SCP code”. Instead we implement:
1) A **small, fixed set of primitive event types** (P/QRS/T + overlays).
2) A **table-driven mapping** from SCP code → (phenotype kind, location group, parameters).
3) Composition via **extra event streams** (overlay events), not by replacing everything.

This keeps:
- number of conv input channels `T` small and stable,
- supports multi-label combinations without kernel blowup,
- effects composable and testable.

---

## Process API changes (`ECGProcess`)
### Single PTB-XL label API: `y["scp"]`
PTB-XL tasks are naturally **multi-label**, and labels can span multiple groups. We expose **one** multi-hot target tensor over the full SCP universe.

Add to `ECGProcess.__init__`:
- `scp_codes: list[str]` (explicit ordered universe; for PTB-XL this is the **full 71** in CSV order)
- `scp_sampler: Sampler` (returns `torch.bool[B, len(scp_codes)]`)
Remove / supersede the current rhythm-only API:
- remove `rhythm_codes`, `rhythm_sampler` constructor args
- remove `y["rhythm"]` output head; rhythm is represented by the rhythm subset of `y["scp"]`

Labels emitted (when configured):
- `y["scp"]`: `bool[B, 71]`, multi-hot over the full PTB-XL SCP code list
- Group membership is provided via `meta`, not separate `y[...]` heads:
  - `meta["process"]["label_names"]["scp"] = scp_codes`
  - `meta["process"]["label_groups"]["rhythm"/"diagnostic"/"form"] = list[int]` indices into `scp_codes`

Rationale:
- Avoids inconsistencies for codes that belong to multiple groups (`NST_`, `DIG`, `LNGQT`, ...).
- Makes “add one more SCP label” = add one phenotype entry in one mapping table (see below).
- Supports rhythm + diagnostic + form with a single label tensor shape, enabling a single downstream training API.

### Sampling constraints (explicit, fail-fast)
Add explicit constraints in `ECGProcess` constructor:
- `scp_codes` and `scp_sampler` are required (no silent defaults).
- `scp_codes` must match `ptbxl_all_codes()` ordering for the PTB-XL task mode (no implicit reordering).
- Enforce **label-set validity** at the sampler level (no “auto-fixing” in the process):
  - exactly **one rhythm code** active per sample
  - `NORM` (diagnostic) is mutually exclusive with any other diagnostic code.
  - mutually-exclusive phenotype families (e.g., `STE_` vs `STD_` in the same location) must not co-occur.
  - incompatible conduction codes (e.g., `WPW` with `LPR`) must not co-occur.

---

## Event representation (key refactor for diagnostic/form fidelity)
Current rhythm implementation uses a “single beat event” approach. To support PR/QT timing and many morphology patterns, we move to **component events**:

### EventSchema (proposed)
`time_unit="samples"`, `param_names=["amplitude"]`, and `type_names` split into:

**Core components**
- `p` (P-wave)
- `qrs_normal`
- `t` (T-wave)

**QRS variants (small finite set)**
- `qrs_wide` (bundle branch / IVCD)
- `qrs_qwave` (Q-wave / infarct-like)
- `qrs_delta` (WPW-like pre-excitation / slur)

**ST/T overlays (location-grouped)**
- `st_shift_{loc}` (smoothed plateau after QRS; amplitude sign encodes elev/depr)
- `t_invert_{loc}` (localized T inversion overlay; amplitude sign fixed negative)

**Rhythm auxiliaries (already exist conceptually)**
- `pace_spike`
- `flutter_wave`

Where `{loc}` comes from a **small set of lead groups** (see below), not per-code.

This keeps total `T` roughly: `3 core + 3 qrs variants + (num_loc * overlay_kinds) + 2 auxiliaries`,
with `num_loc` ~ 6–8.

---

## Lead localization without label-conditioned views
To keep pathology in the process and avoid label-conditioned views, we standardize the rendering pipeline for PTB-XL tasks as:

`ECGProcess (events) -> EventImpulseView -> KernelConvView -> Observation([B, 12, L])`

and **do not use `ECGLeadsView`** for PTB-XL tasks.

### Kernel bank outputs 12 leads directly
Create `aiono.kernels.ptbxl` (new module) that builds a kernel bank with:
- output channels `K=12` (fixed PTB-XL lead order: I, II, III, aVR, aVL, aVF, V1–V6)
- input channels `T` matching the schema above
- each event type has a **lead-specific waveform** (weights/sign per lead)

Localization is achieved by:
- defining `{loc}` → lead masks (deterministic, non-random) and building `st_shift_{loc}` / `t_invert_{loc}` kernels with weights only on those lead subsets.

This keeps labels purely process-driven (which event types are emitted), and the view is a deterministic renderer.

---

## Phenotype mapping (table-driven)
Add a PTB-XL phenotype mapping module (data-only, explicit tables):
- `aiono/ptbxl/phenotypes.py`

It exposes:
- `PTBXLLoc` enum-like strings (e.g., `global`, `inferior`, `lateral`, `anterior`, `septal`, `anteroseptal`, `anterolateral`, `inferolateral`)
- `PTBXLPhenotype` dataclass describing which primitives to enable:
  - timing modifiers: `pr_shift_samples`, `qt_shift_samples`
  - qrs variant selection: `qrs_kind`
  - overlays: `st_shift_mv` (+ sign), `t_inversion` flag
  - ectopy: `pac_rate`, `pvc_rate` (future)
  - location: `loc`

Then map **every diagnostic+form SCP code** to one phenotype entry.
Multiple codes are allowed to map to the same phenotype initially (explicitly documented).

Multi-label composition rule:
- `y["scp"]` represents a **set of active SCP codes** per sample; the active phenotype is the **union** of the enabled primitives across those codes, subject to the sampler’s validity constraints (no conflicting unions).

This makes “add label support” = add/adjust one entry in a single mapping table.

---

## Process internals (nodes to add/refactor)
### 0) Multi-label sampling (new)
New node: `SampleMultiLabelNode`
- Inputs: `scp_codes`, `scp_sampler`
- Output: `state.y["scp"] = bool[B, S]`
- Writes sampled label masks into `meta["samples"][...]` for reproducibility.

PTB-XL-specific sampler (new, best single solution):
- `PTBXLLabelSetSampler` produces valid multi-label sets by sampling a small number of **compatible phenotype families** (conduction/timing, ST/T, MI/injury, hypertrophy, ectopy) and translating them into SCP code activations.
- This keeps co-occurrence plausible and avoids impossible label unions without per-sample “repair” logic.

### 1) Beat timing (reuse existing rhythm engine)
Keep the existing RR pattern generator (regular/irregular/bigu/trigu/psvt/etc.), but drive it from the **active rhythm SCP code** (from `y["scp"]` + rhythm-group indices) and output a **beat time grid** `t_qrs[B, N]` and `mask[B, N]`.

### 2) Component event generation
New node: `ECGComponentEventsNode`
- Inputs: `t_qrs` and the active phenotype mask derived from `y["scp"]` (including the active rhythm SCP code)
- Samples per-sample intervals:
  - PR (baseline + label modifiers; used for `LPR`, `WPW`)
  - QT (baseline + label modifiers; used for `LNGQT`)
- Emits separate streams:
  - P events at `t_p = t_qrs - pr_samples`
  - QRS events at `t_qrs` (variant chosen by phenotype)
  - T events at `t_t = t_qrs + qt_samples`

All tensors are vectorized; no Python loops over B or N.

### 3) Overlay event generation (ST/T, localized)
New node: `ECGOverlayEventsNode`
- Uses phenotype mapping to optionally emit:
  - `st_shift_{loc}` events at `t_qrs` (plateau kernel spans ST segment)
  - `t_invert_{loc}` events at `t_t` (or at `t_qrs` with delayed kernel)

Amplitude parameters:
- Use `params[..., amplitude]` as signed amplitude (negative for depression/inversion).
- Do not encode labels via obvious global scaling unless the label semantics demands it (e.g., `LVOLT/HVOLT`).

### 4) Union and output
Union streams via existing `UnionEventsNode` into `state.data["events"]`.

Meta / reproducibility:
- store sampled PR/QT per sample, overlay magnitudes, selected QRS kind, loc id in `__samples__/...`
- never store full-size per-sample noise fields

---

## Code-by-code coverage strategy (how each PTB-XL group is implemented)
We implement full label coverage by grouping codes into phenotype families:

### A) Timing / conduction family
- `LPR` (form), `1AVB/2AVB/3AVB` (diagnostic): PR modifications + dropped QRS (2AVB) + AV dissociation (3AVB as separate atrial P train + ventricular QRS/T train).
- `WPW`: short PR + delta-wave QRS variant.
- `IRBBB/CRBBB/CLBBB/ILBBB/IVCD`: QRS widening variants.

### B) ST/T abnormality family
- `NST_`, `NDT`, `ISC_`, `DIG`, `EL`, `ANEUR` (diagnostic) and `STD_`, `STE_`, `TAB_`, `NT_`, `INVT`, `LOWT` (form):
  - modeled via `st_shift_{loc}` and/or `t_invert_{loc}` overlays (loc may be `global` for non-local codes).
  - `LNGQT`: QT prolongation (timing), optionally plus mild T morphology change.

### C) MI / injury / ischemia localization family
- MI codes (`AMI/ASMI/ALMI/IMI/ILMI/IPMI/IPLMI/LMI/PMI`) and localized ischemia/injury codes (`ISCAN/ISCAS/ISCAL/ISCIN/ISCIL/ISCLA`, `INJAS/INJAL/INJIN/INJIL/INJLA`):
  - map to (kind ∈ {ischemia, injury, infarct}) × loc ∈ lead-groups
  - infarct includes `qrs_qwave` + ST/T overlays
  - ischemia/injury use ST/T overlays without Q-wave by default

### D) Hypertrophy / chamber enlargement family
- `LVH/RVH/SEHYP`, `LAO/LAE`, `RAO/RAE`:
  - lead-specific amplitude patterns (not uniform global scaling), implemented via per-lead weights in kernels for P and/or QRS.
  - `VCLVH`, `HVOLT`, `LVOLT` (form): controlled amplitude scaling but avoid trivially encoding labels by “overall energy only”; combine with compensatory scaling so simple global stats don’t saturate.

### E) Ectopy family (form)
- `PAC`, `PVC`, `PRC(S)`:
  - implement as occasional premature beats (timing) + QRS variant selection for PVC.
  - keep pacing/ectopy independent of rhythm label unless explicitly sampling correlated regimes (documented).

### F) Normal
- `NORM` (diagnostic): no additional overlays; uses baseline rhythm only.

---

## Tests (pytest)
Add tests focused on determinism + “effect exists” checks:

1) Taxonomy
- `test_ptbxl_scp_loader_all_codes_order`
- `test_ptbxl_scp_loader_diagnostic_codes_order`
- `test_ptbxl_scp_loader_form_codes_order`
  - `test_ptbxl_scp_loader_rhythm_codes_order`

2) Shape/dtype/determinism
- `test_ecg_process_ptbxl_scp_shapes_and_dtypes` (expects `y["scp"]` is `bool[B, 71]`)
- `test_ecg_process_ptbxl_scp_determinism`
- `test_ecg_process_ptbxl_scp_group_indices_consistency` (rhythm/diagnostic/form indices slice correctly)

3) Phenotype spot-checks (small, targeted)
- PR: `LPR` and `1AVB` produce larger sampled PR than `NORM`.
- QT: `LNGQT` produces larger sampled QT than `NORM`.
- ST elevation/depression: `STE_` vs `STD_` shows opposite-sign mean in ST window on affected leads.
- Q-wave: `QWAVE` / MI codes produce larger negative pre-QRS deflection metric on affected leads.
- Localization: one localized code (e.g., `ISCIN`) affects inferior lead mask more than aVL/V1.

4) Shortcut checks
- Extend the existing “cheap baseline on simple stats” to multi-label diagnostic/form:
  - evaluate macro-averaged per-label accuracy (or AUPRC) for a simple prototype/linear baseline.
  - label-shuffle controls should return near-prevalence performance (no suspicious lift).

---

## Documentation + examples
### Docs
- Update `DOCUMENTATION.md` with:
  - code list + how `y["scp"]` is configured (and how to obtain group slices via `meta["process"]["label_groups"]`)
  - “renderer for PTB-XL”: `EventImpulseView -> KernelConvView` producing `[B,12,L]`
  - reproducibility: which sampled parameters are stored in `meta["samples"]`

### Examples
Add new demo pairs (script + notebook) similar to the rhythm demo:
- Update `examples/09_ptbxl_rhythm_12.py` + `.ipynb` to use `y["scp"]` and slice rhythm indices.
- Add `examples/10_ptbxl_form_19.py` + `.ipynb` (one sample per form code; `y["scp"]` slice)
- Add `examples/11_ptbxl_diagnostic_44.py` + `.ipynb` (one sample per diagnostic code; `y["scp"]` slice)

Both should:
- render 12 leads directly (no `ECGLeadsView`)
- visualize lead II and a small subset of precordials
- print selected phenotype params (PR/QT/ST shift, loc)

---

## Acceptance criteria
- Can instantiate `ECGProcess` with `scp_codes=ptbxl_all_codes()` and generate batches with multi-hot `y["scp"]`.
- Group slices (rhythm/diagnostic/form) are available via `meta["process"]["label_groups"]` and match `scp_statements.csv`.
- For every code in each head, generated samples exhibit the intended qualitative phenotype (verified by targeted tests/metrics).
- Deterministic with fixed `torch.Generator`.
- `uv run pytest` passes (offline constraints respected in this environment).

---

## API consolidation / cleanup (remove superseded pieces)
After migrating PTB-XL examples/tests to the 12-lead kernel renderer, audit for unused ECG-specific rendering code:
- If `ECGLeadsView` is no longer used by any example/test (and not needed by the PTB-XL pipeline), remove:
  - `aiono/views/ecg_leads.py` (`ECGLeadsView`)
  - `aiono/core/utils.py:utils_make_canonical_A0` and `aiono/core/utils.py:utils_make_random_A0` (A0 helpers)
  - related tests and documentation snippets
- If `make_ecg_kernel_bank` (3-component + lead-mixing pipeline) becomes unused, supersede it with `aiono.kernels.ptbxl` and remove it to avoid two competing ECG rendering APIs.
