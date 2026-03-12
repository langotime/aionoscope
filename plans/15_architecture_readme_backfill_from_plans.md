# Plan: Promote stable plan decisions into `ARCHITECTURE.md` and `README.md`

## Goal
Review every historical plan in `plans/` and extract only the stable, repo-level decisions that should be promoted into:

- `ARCHITECTURE.md`: durable design choices, invariants, core data model, execution model, and major extension patterns.
- `README.md`: short human-facing overview of what Aionoscope does, which public APIs matter, and which examples/features a new user should notice first.

This plan is intentionally selective. It excludes implementation checklists, validation commands, open questions, rejected options, and migration-only details.

## Current state
- `README.md` already covers the high-level Process -> View idea, component gating, event rendering, and the current PTB-XL multi-label example.
- `DOCUMENTATION.md` already holds many advanced details (`meta["views"]`, `meta["samples"]`, example map, PTB-XL usage).
- `ARCHITECTURE.md` does not exist yet, so most stable design decisions still have no dedicated home.

## Inclusion rules
### Put into `ARCHITECTURE.md`
- Core invariants that shape the whole library.
- Canonical data representations and metadata contracts.
- Execution/runtime model (`ProcessGraph`, `SynthPipeline`, RNG splitting).
- Stable extension rules: what belongs in a Process vs a View, how events are represented, how labels are defined, how sampled parameters are exposed.
- Major domain-specific architecture that now defines the public library shape, especially PTB-XL support.

### Put into `README.md`
- One-screen mental model of the library.
- Public capabilities and the most important public APIs.
- Short explanations of why key features exist.
- Pointers to the best examples for common entry points.

### Do not put into either
- File-by-file implementation steps.
- Test inventories and validation commands.
- Historical alternatives that were rejected.
- Temporary migration notes unless the migration still affects current users.
- Fine-grained phenotype tables or exhaustive parameter lists that belong in dedicated docs.

## Per-plan review

| Plan | `ARCHITECTURE.md` | `README.md` | Disposition |
| --- | --- | --- | --- |
| `plans/0_initial.md` | Promote the foundational system model: synthetic time-series benchmark focus, Process -> View -> Observation, reproducibility, anti-shortcut rules, performance/testability expectations, and the conceptual package split. | Keep a condensed version of the goals and supported benchmark families. | Foundational. Much of this is already partially present in `README.md`, but it needs a stable architecture home. |
| `plans/1_documentation.md` | Do not copy the documentation-process rules themselves. At most, use the idea of a small architecture diagram and precise terminology. | Add or preserve one short onboarding diagram/flow if it improves readability. | Mostly authoring guidance, not library architecture. |
| `plans/2_gitignore.md` | No. | No. | Operational only. |
| `plans/3_ecg_leads_mask_variants.md` | Document `ECGLeadsView` as the generic lead-mixing renderer when it is used: static `A0`, batched `A0`, or callable `A0` are supported public forms. | Mention only briefly if the API remains public and recommended. | Low-priority architecture detail. Keep concise because PTB-XL may prefer a different rendering path. |
| `plans/4_frequency_pulse_train.md` | Record the physical-units decision: time-aware ECG/event processes should prefer `frequency_hz` plus `sample_rate_hz`, not a raw `num_pulses` user API. | Ensure examples and text use physical units consistently. | Stable public API rule. Already reflected in current examples; architecture doc should make it explicit. |
| `plans/5_composable_processes.md` | Promote heavily: canonical `EventBatch`, `LatentState.events`, `ProcessGraph` / `ProcessChain`, branching and merging at the latent/event level, rendering in Views, and deterministic RNG splitting across graph containers. | Keep a short feature-level explanation of composable process graphs and event-first rendering, plus a link to the simple graph example. | Highest-priority architecture content. |
| `plans/7_meta_accumulation_for_dense_targets.md` | Promote the metadata provenance contract: process metadata stays under `meta["process"]`, view metadata accumulates under ordered `meta["views"]`, and pipeline state lives separately. Also state that view keys are not merged into the top level. | Optional short mention that intermediate sampled/view parameters remain available for probing and debugging. | Architecture-level contract; README detail should stay minimal. |
| `plans/8_sampled_params_meta.md` | Promote the `meta["samples"]` contract and the storage rules: no duplication of outputs, no full-size noise or masks, store small sampled parameters and seeds only. | Add at most one sentence that sampled parameters are exported for reproducibility/probing. | Stable cross-cutting rule. Pair with the sampler API from plan 9. |
| `plans/9_param_samplers_api.md` | Promote the sampler architecture: `Sampler`, `SamplerLike`, `samples/spec` separation, and the MVP rule that shape-affecting sampled parameters are per-batch unless explicitly designed otherwise. | Add a short public-facing section or snippet showing that parameters can be constants or samplers, with a few common sampler names. | High-value public API and architecture content. |
| `plans/10_basic_components_dataset_example.md` | Promote the generic additive-components pattern: `ConstantProcess` baseline, additive Views, `enabled_key` gating, event components rendered from event streams, and `sample_rate_hz` as shared process metadata for time-aware Views. | Add or keep an examples index entry for the balanced basic-components dataset and mention that event components use `EventRenderView`. | Mostly already visible through examples, but the underlying pattern belongs in architecture. |
| `plans/11_variable_num_enabled_per_sample.md` | Promote the semantics of heterogeneous mixture complexity: `EnableComponentsNode(num_enabled=SamplerLike[int])`, `component_count` is always per-sample, and `component_id` exists only in the all-single-component case. | Mention the capability briefly, but not the full label-emission rule. | Stable behavior that matters for extension and downstream use. |
| `plans/11_imbalanced_components_dataset.md` | Promote the selection-policy split: uniform selection by default, `component_id` sampler for `k == 1`, and weighted ordering / `component_order` for imbalanced `k > 1` mixtures. | Keep the user-facing example map: balanced baseline, imbalanced single-label components, and imbalanced mixtures. | Publicly useful. Much of this is already in `README.md` and `DOCUMENTATION.md`; tighten wording rather than expanding heavily. |
| `plans/13_ptbxl_rhythm_12_labels.md` | Keep only the stable pieces that survive the later PTB-XL design: packaged `scp_statements.csv` as source of truth, rhythm encoded in process timing patterns, and rhythm-specific auxiliary events living in the Process. Do not document the older rhythm-only API as current architecture. | No separate README section for the superseded rhythm-only API. Fold any retained ideas into the current PTB-XL multi-label story. | Partially superseded by `plans/14_ptbxl_diagnostic_and_form_support.md`. |
| `plans/14_ptbxl_diagnostic_and_form_support.md` | Promote heavily: `y["scp"]` as the single PTB-XL multi-hot target, `label_groups` as group slices, table-driven phenotype mapping, small primitive event vocabulary, direct 12-lead kernel rendering for PTB-XL, and sampler-level label-validity constraints. Also document the boundary between the generic ECG rendering path and the PTB-XL-specific renderer. | Expand the PTB-XL README section with the current public API shape and point readers to rhythm/form/diagnostic example slices. | Highest-priority user-facing addition after plan 5 and plan 9. |

## What `ARCHITECTURE.md` should contain

### 1. Scope and design invariants
- Why Aionoscope exists: synthetic time-series benchmarks for SSL and supervised learning.
- Non-negotiable rules:
  - labels are functions of the Process, not the View;
  - Views own presentation, nuisance, and measurement distortion;
  - reproducibility flows through explicit `torch.Generator` usage;
  - large stochastic fields are not duplicated into metadata;
  - vectorized tensor execution is preferred over Python loops across batch elements.

### 2. Core data model
- `LatentState`
- `Observation`
- `EventSchema`
- `EventBatch`
- `meta` contract:
  - `meta["process"]`
  - `meta["views"]`
  - `meta["samples"]`
  - `meta["pipeline_seed"]`

### 3. Execution model
- `SynthPipeline` owns one Process execution and one-or-more Views.
- `ViewChain` preserves metadata provenance across chained Views.
- `ProcessGraph` / `ProcessChain` define latent generation, branching, parallel branches, and merge points.
- RNG splitting is structural and deterministic.

### 4. Rendering architecture
- Generic event-first path:
  - Process emits events and/or latent components.
  - Views render events into impulses or dense signals.
- Generic ECG-like path:
  - event/latent components -> kernel convolution -> `ECGLeadsView`
- PTB-XL path:
  - event process -> `EventImpulseView` -> PTB-XL kernel bank -> direct 12-lead output
- Explain why PTB-XL rendering can bypass `ECGLeadsView` without contradicting the generic architecture.

### 5. Parameter sampling and metadata
- `SamplerLike` is the public parameter convention.
- Sampled values are recorded for reproducibility and probing.
- Shape-affecting sampled parameters are constrained explicitly.
- Metadata stores low-dimensional sampled values and seeds, not full noise/mask tensors.

### 6. Component-mixture datasets
- `EnableComponentsNode`
- per-sample `enabled` masks
- balanced vs imbalanced selection
- variable `num_enabled`
- why this exists: curriculum and mixture-complexity experiments without branching the pipeline graph

### 7. PTB-XL architecture
- packaged taxonomy data as the single source of truth
- one `y["scp"]` target over the full ordered code set
- `label_groups` for rhythm/diagnostic/form slices
- phenotype mapping and lead localization live in the process/kernel design, not in label-conditioned view noise

### 8. Extension rules
- how to add a new Process
- how to add a new View
- when to add a sampler vs a hard-coded parameter
- what may and may not be written to metadata

## What `README.md` should add or tighten

### Add a short "Public patterns" section
- Parameters can be constants or samplers.
- Processes can emit events first and let Views render them.
- One static `ViewChain` can generate heterogeneous mixtures through `enabled` masks.

### Add a small example map
- balanced basic components: `examples/06_basic_components.py`
- imbalanced single-component dataset: `examples/07_imbalanced_components.py`
- imbalanced mixtures: `examples/08_imbalanced_mixtures.py`
- PTB-XL multi-label ECG: the PTB-XL example already in `README.md`

### Tighten the PTB-XL section
- State clearly that the public PTB-XL target is `y["scp"]` with `label_groups`.
- Keep rhythm/form/diagnostic as slices over the same label tensor.
- Avoid reintroducing the older rhythm-only API.

### Keep advanced details out of the README
- full metadata schema
- detailed phenotype mapping
- per-view provenance rules
- compatibility constraints for multi-label PTB-XL sampling

Those should be linked from `README.md` into `ARCHITECTURE.md` or `DOCUMENTATION.md`.

## Proposed implementation order
1. Create `ARCHITECTURE.md` with the section structure above.
2. Move the stable, cross-cutting decisions from the reviewed plans into `ARCHITECTURE.md`, preferring current API names over historical ones.
3. Update `README.md` only where it improves onboarding:
   - one short "Public patterns" section,
   - one short example map,
   - one tightened PTB-XL summary.
4. Add cross-links:
   - `README.md` -> `ARCHITECTURE.md`
   - `ARCHITECTURE.md` -> `DOCUMENTATION.md` for deep operational details
5. Avoid copying long tables or detailed rules into more than one document.

## Documentation impact
- New file: `ARCHITECTURE.md`
  - becomes the stable home for architecture decisions currently scattered across historical plans.
- Update `README.md`
  - keep it compact and onboarding-oriented;
  - link to `ARCHITECTURE.md` for design details and to `DOCUMENTATION.md` for operational usage.
- No large expansion of `DOCUMENTATION.md` is required for this pass, but wording should remain consistent with the new architecture terminology once the docs are updated.

## Acceptance criteria
- Every historical plan has an explicit disposition for `ARCHITECTURE.md` and `README.md`.
- `ARCHITECTURE.md` scope is clear enough to write without importing plan-local implementation noise.
- `README.md` changes stay short and user-facing.
- Superseded APIs from older plans are not promoted as current public architecture.
