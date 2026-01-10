# Plan: Accumulating meta across views (dense targets for probing)

## Goal
Make **all per-sample/per-channel parameters that are actually sampled during generation** available at the final `Observation.meta`, so we can use them later as regression/classification targets (e.g., for linear probing after representation learning).

Concretely, fix the current issue where intermediate view parameters (e.g. `ECGLeadsView.delays`) are **lost** when the next view returns a new `Observation(meta=...)`.

## Current behavior (why “just take meta” fails)
### What is already sampled (examples)
- **Process** (latent): `phase_offset_samples` and other continuous parameters are stored in the process meta (`LatentState.meta`).
- **Views** (observation): e.g. `ECGLeadsView` samples `delays: [B, C]`, `A: [B, C, K]`, and stores them in `Observation.meta`.
- **BaselineWanderView** samples `freq/phase/amplitude` per sample+channel but currently only stores the config (`freq_min/max`, `amplitude_std`) and drops the sampled values.

### Where it gets lost
- `SynthPipeline` wraps `nn.Sequential` into `ViewChain`.
- `ViewChain.forward()` currently just applies views sequentially and returns the last `Observation` **without accumulating meta**.
- Each view creates a fresh `meta = {..., "process": process_meta}` and thus **overwrites** any non-`process` fields produced earlier.

Result: only the *last* view’s non-process meta survives. That blocks using intermediate view parameters as dense targets.

## Options (analysis)
### Option A — Accumulate meta in `ViewChain` (recommended)
**Idea:** Make `ViewChain` responsible for *propagating and accumulating* meta across the chain.

How:
- After each view call, append the view’s metadata to a structured per-view list.
- Store that list at `Observation.meta["views"]` so collisions are avoided (e.g., multiple views all have `"seed"`).

Pros:
- Minimal changes to individual views (can remain “stateless” w.r.t. previous view meta).
- Keeps separation of responsibilities: `process` meta stays process-owned; view params stay view-owned.

Cons / design questions:
- Needs a clear collision policy for top-level keys (`"seed"`, `"view"`). If we keep a per-view trace, collisions become non-issues.
- Slightly larger meta payload (usually negligible vs `x: [B, C, L]`).

### Option B — Write view parameters into `process_meta`
**Idea:** Treat `meta["process"]` as a “global meta bus” and have views write their sampled params into it (namespaced).

Pros:
- No need to change `ViewChain`.
- Views already preserve `process_meta` via `meta["process"]`.

Cons:
- Breaks the intended separation: view distortions become stored under `"process"`.
- Encourages accidental coupling (“process meta” no longer means “process-only”).
- Requires touching *every* view that samples something worth exporting.

### Option C — Update every view to merge input meta into output meta
**Idea:** Change each view to start with `meta = dict(input_state.meta)` and then update it.

Pros:
- Meta accumulation works even when users call views manually without `ViewChain`.

Cons:
- Requires editing all built-in views (and sets expectations for user views).
- Still needs a collision policy; otherwise silent overwrites remain.

## Decision
Implement **Option A** with explicit constraints:
1) `ViewChain` accumulates view metadata into `meta["views"]` (ordered list).
2) `process` stays top-level and is **not** duplicated per view.
3) No top-level merge of view keys; any merging is caller-side.
4) Fix views that currently **sample but don’t export** useful parameters (starting with `BaselineWanderView`).

This is the smallest, most maintainable change that preserves process/view separation and enables dense target extraction.

## Proposed meta schema (explicit)
Final `Observation.meta` will contain:
- `"process"`: unchanged process meta dict
- `"views"`: `list[dict]` (ordered), one entry per view application in the chain
  - each entry stores the view’s returned meta **excluding** `"process"` (to avoid repetition)
- `"pipeline_seed"`: unchanged (added by `SynthPipeline`)

No view keys are merged to the top level; callers must choose how to merge or index `meta["views"]`.

## Implementation steps (single phase)
1) **Implement meta accumulation in `toyts/views/base.py:ViewChain.forward()`**
   - Carry forward an existing `views` list if the input is an `Observation` (supports nested chains).
   - After each view:
     - append a trace entry (view meta sans `"process"`)
     - rewrap the returned `Observation` with `meta = {**returned.meta, "views": trace}`
   - Final `Observation.meta` contains only `process`, `views`, and `pipeline_seed` (pipeline-only).
   - Fail fast with a clear error if the view returns meta without `"process"` (preserves current invariants used by `utils_extract_process_meta`).

2) **Export missing sampled parameters**
   - `toyts/views/noise.py:BaselineWanderView`:
     - store sampled `freq`, `phase`, `amplitude` in meta
       - `freq: [B, C, 1]`, `phase: [B, C, 1]`, `amplitude: [B, C, 1]`
     - keep existing config keys (`freq_min/max`, `amplitude_std`) as well.
   - (Optional after review) audit other views for “sampled but not exported” parameters.

3) **Tests (pytest)**
   - Add `tests/test_viewchain_meta_accumulation.py`
     - Build a small `ViewChain(ECGLeadsView(...), BaselineWanderView(...), NormalizeView())` and assert:
       - final `obs.meta["process"]` exists and is a dict
       - `obs.meta["views"]` exists and has length == number of views
       - ECG meta is present in the trace (e.g., `delays`, `A0`/`A`)
       - baseline wander sampled `freq/phase/amplitude` are present with correct shapes
   - Keep tests deterministic via a fixed `torch.Generator`.

4) **Docs / examples**
   - Update `DOCUMENTATION.md` (or `README.md`) with a short “How to use `meta` for probing targets” section:
     - where to find process params (`meta["process"]`)
     - where to find per-view params (`meta["views"][...]`)
   - Add a minimal snippet demonstrating extracting a dense target (e.g., delays regression).

## Acceptance criteria
- A chained view pipeline preserves intermediate view parameters in the final `Observation.meta` (via `views`, no top-level view merge).
- `BaselineWanderView` exposes sampled `freq/phase/amplitude` in meta.
- All tests pass: `uv run pytest`.
