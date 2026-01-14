# Plan: Per-sample `num_enabled` in `EnableComponentsNode`

## Goal
Allow batches where **different samples have different numbers of enabled components**, by making `EnableComponentsNode.num_enabled` sampled **per sample** (not a single fixed `k` for the whole batch).

This enables experiments like: “within the same batch, some samples have 1 component, others have 2–3, others have all components”.

## Non-goals
- No changes to the additive component Views (`enabled_key=...`) or event gating Views.
- No new dataset/example yet (core support first).

## Proposed API change (minimal + backwards compatible)
Update `toyts/processes/nodes.py:EnableComponentsNode`:

- Change constructor from:
  - `EnableComponentsNode(component_keys: list[str], num_enabled: int)`
  to:
  - `EnableComponentsNode(component_keys: list[str], num_enabled: SamplerLike[int])`

Semantics:
- `num_enabled` is sampled as `k: int64[B]` each `forward()` call.
- `state.y["component_count"]` becomes **per-sample**: `k` (int64 `[B]`).
- `state.y["component_id"]` is emitted **only when all samples have `k == 1`** (the “single-component classification” case). Otherwise it is omitted because it is not well-defined for multi-component samples.

## Implementation details (KISS)
File: `toyts/processes/nodes.py`

### 1) Init-time normalization
- Keep existing validation for `component_keys` (non-empty, unique, non-empty strings).
- Normalize `num_enabled` via `sampler_from_value(num_enabled, name="num_enabled")`.
  - Store as `self.num_enabled_sampler`.

### 2) Forward-time sampling + validation
- Sample `k` per sample:
  - `k = sampler_sample(..., shape=(B,), dtype=torch.int64, name="num_enabled")`  # `[B]`
- Fail fast if any `k < 1` or `k > N` where `N = len(component_keys)`:
  - Raise `ValueError` with: `min(k)`, `max(k)`, and `N`.

### 3) Vectorized k-hot selection for variable k
Goal: build `enabled_matrix: bool[B, N]` where each row has exactly `k[b]` `True` values, without Python loops over `B`.

Single approach for all k (simple + deterministic):
- `scores = torch.rand((B, N), generator=rng, device=state.device, dtype=torch.float32)`  # `[B, N]`
- `order = scores.argsort(dim=1, descending=True)`  # permutation indices `[B, N]`
- `rank = torch.arange(N, device=state.device)[None, :]`  # `[1, N]`
- `keep = rank < k[:, None]`  # `[B, N]` bool: first `k[b]` ranks are kept
- `enabled_matrix = torch.zeros((B, N), device=state.device, dtype=torch.bool)`  # `[B, N]`
- `enabled_matrix.scatter_(1, order, keep)`  # `[B, N]`

### 4) Write outputs
- `state.y["component_count"] = k`  # int64 `[B]`
- If `torch.all(k == 1)`:
  - `component_id = order[:, 0].to(torch.int64)`  # `[B]`
  - `state.y["component_id"] = component_id`
  - Update `state.meta["label_names"]["component_id"] = component_keys` with the existing mismatch check.
- Write per-component masks:
  - `state.meta["enabled"][key] = enabled_matrix[:, idx]`  # bool `[B]`
  - Keep the existing “do not overwrite enabled[...]” fail-fast behavior.

## Other components to review (expected: no changes)
- `toyts/views/_enabled.py:views_resolve_enabled_mask` should already accept `bool[B]` masks; variable `k` only changes how many masks are `True`.
- Event gating (`GateEventsByEnabledNode`) already gates per-sample via `bool[B]`; variable `k` should work unchanged.

## Tests (pytest)
Add `tests/test_enable_components_node_variable_num_enabled.py` covering:
- **Shapes/dtypes**:
  - each `meta["enabled"][key]` is `torch.bool` with shape `[B]`
  - `y["component_count"]` is `torch.int64` with shape `[B]`
- **Row-sum correctness**:
  - `sum_k = stack(enabled_masks, dim=1).sum(dim=1)` equals `component_count` for all samples.
- **Determinism** (fixed generator seed):
  - same seed → same `component_count` and enabled masks.
- **`component_id` emission rule**:
  - if sampler always yields `1` (e.g. `num_enabled=1` or `ConstantSampler(1)`): `component_id` exists.
  - if sampler can yield `>1` (e.g. `RandIntSampler(1, N+1)`): `component_id` does not exist.

## Docs + examples (after core + tests)
- Update `DOCUMENTATION.md` and `README.md` to mention:
  - `EnableComponentsNode(..., num_enabled=SamplerLike[int])` enables *heterogeneous* component counts in a single batch.
  - why it exists: lets you benchmark training under variable mixture complexity without branching pipelines.
- Update `examples/06_basic_components.*` to include one extra snippet showing variable-k usage, e.g.:
  - `num_enabled = RandIntSampler(1, len(component_keys) + 1)`

## Acceptance criteria
- `uv run pytest` passes.
- Existing code using `num_enabled=int` behaves the same (including `component_id` when all `k == 1`).
- Variable-k sampler produces mixed component counts inside one batch and Views gate correctly via `enabled_key`.

## Open question (for review)
Should `component_id` be emitted for **only the samples where `k == 1`** inside a mixed batch (with a sentinel value for others), or keep KISS and emit it only when **all** samples are single-component?
