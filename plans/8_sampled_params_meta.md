# Plan: Add sampled parameters to meta (no duplication, no full-size noise/masks)

## Goal
Expose all sampled parameters for processes/views in `meta` so they can be used as dense targets, **without** duplicating data that already exists in `LatentState`/`Observation` outputs and **without** storing full-size noise/masks.

## Rules (already agreed)
- Do **not** duplicate sampled tensors in `meta` if they already appear in `LatentState` or `Observation` outputs (e.g., `events`, `latent`, or `x`).
- Do **not** store full-size sampled noise/masks in `meta` (e.g., `[B,C,L]` noise or missingness masks).
- Store seeds + low‑dimensional sampled parameters needed for reproducibility and probing.
- View samples live in `Observation.meta["views"]` (already done). Process samples live in `LatentState.meta` (see schema below).

## Proposed meta schema for sampled parameters (process side)
Add a dedicated namespace in process meta to avoid collisions across nodes:
- `LatentState.meta["samples"]`: `dict[str, dict[str, Any]]`
  - keys are stable node/process identifiers, e.g.:
    - `"TrendSeasonAnomalyProcess"` for the monolithic process
    - `"EventTrainNode:<out_key>"` for graph nodes (prevents collisions)
    - `"TimeJitterNode:<out_key>"` similarly

This keeps process meta flat for existing keys while isolating sampled params.

## Audit: current gaps

### Processes
- `TrendSeasonAnomalyProcess` (aiono/processes/trend_season.py):
  - sampled but **not stored**: `slope`/`slope_noise`, `offset`, `season_freq`, `season_phase`, `season_amp` (after spiky boost), `anomaly_amp`, `anomaly_center`
  - `extra_noise` is full-size `[B, K-3, L]` → **do not store**; only store seed/std if needed
- `EventTrainNode` (aiono/processes/nodes.py):
  - `phase_offset_samples` already stored
  - sampled but **not stored**: `intervals` (random/missed), `missed_indices` (missed_beat mode)
  - event `times` and `amplitude` already live in `EventBatch` → **no duplication**
- `TimeJitterNode`:
  - sampled but **not stored**: jitter noise `[B, E]` (not in outputs)

### Views
- `MissingnessView` (aiono/views/missingness.py):
  - sampled but **not stored**: `keep_mask`, `apply_gap`, `gap_start`, `hold_mask` (full-size / large)
  - requirement: **do not store** masks; provide a deterministic one-call regen method from meta
- `NoiseView`:
  - sampled noise `[B,C,L]` → **do not store** (explicit requirement)
- `BaselineWanderView`, `ECGLeadsView`, `UnitsPercentOfCapacityView`:
  - already storing sampled params (no action)

## Implementation steps (single phase)
1) **Introduce process sampled namespace**
   - Add `meta["samples"]` dict (if absent) and populate per-node/process entries.
   - Use keys like `"EventTrainNode:<out_key>"` and `"TimeJitterNode:<out_key>"` for uniqueness.

2) **TrendSeasonAnomalyProcess sampled params**
   - Store the per-sample tensors in `meta["samples"]["TrendSeasonAnomalyProcess"]`:
     - `trend_slope` `[B]` (post-mask slope)
     - `trend_offset` `[B]`
     - `season_freq` `[B]`
     - `season_phase` `[B]`
     - `season_amp` `[B]` (post spiky boost)
     - `anomaly_amp` `[B]`
     - `anomaly_center` `[B]`
     - `anomaly_sigma` `[B]` or scalar (currently constant `0.03`)
   - **Do not store** `extra_noise` tensor; optionally store `extra_noise_std` and rely on seed for regen.

3) **EventTrainNode sampled params**
   - Store in `meta["samples"]["EventTrainNode:<out_key>"]`:
     - `intervals` `[B, N+1]` (normalized, before cumsum)
     - `missed_indices` `[B]` only for `mode="missed_beat"`
   - Keep `phase_offset_samples` as is (already in meta).
   - Do **not** duplicate event `times` or `amplitude` (they’re in `EventBatch`).

4) **TimeJitterNode sampled params**
   - Store `time_jitter` `[B, E]` in `meta["samples"]["TimeJitterNode:<out_key>"]`
   - Rationale: sampled parameter is not otherwise accessible, and size is bounded by events (not full `[B,C,L]`).

5) **MissingnessView deterministic mask regeneration**
   - Add `MissingnessView.sample_masks(meta, shape, device)` (static/class method).
   - Inputs: `meta` from the view (seed + params), and `shape` `[B, C, L]`.
   - Outputs: dict of masks (`dropout_mask`, `gap_mask`, `hold_mask`, plus `gap_start`/`apply_gap` if needed).
   - Add `mask_version` to meta for forward-compatibility.
   - Keep existing runtime behavior; do **not** store masks in meta.

6) **Docs & examples**
   - Update `DOCUMENTATION.md` to describe `meta["samples"]` for process-level sampled params.
   - Add a small snippet showing how to fetch sampled params and how to regenerate missingness masks.

7) **Tests**
   - `tests/test_trend_season_meta_samples.py`: assert shapes/keys for new process meta.
   - `tests/test_event_train_meta_samples.py`: assert `intervals` and `missed_indices` shapes.
   - `tests/test_time_jitter_meta_samples.py`: verify stored jitter equals (jittered - original) for a controlled input.
   - `tests/test_missingness_mask_regen.py`: generate observation, regen masks via helper, and confirm mask equivalence.

## Open questions for review
1) Are you OK with the `meta["samples"]` namespace and keying by `"<NodeClass>:<out_key>"`?
2) For `TimeJitterNode`, do you prefer storing `time_jitter` `[B,E]` directly (as above), or storing `jitter_seed` + params and regenerating via a helper?
