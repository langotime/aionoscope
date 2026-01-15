# Plan: Imbalanced component sampling (rare components) based on example 06

## Goal
Extend the “basic components” dataset pattern from `examples/06_basic_components.py` to simulate **class / component imbalance**, where some components appear **very rarely** across samples (long-tail / rare-event regimes), while keeping:
- Process → View separation intact (labels and selection happen in the Process).
- Determinism via explicit `torch.Generator` / seed (no global seeding).
- GPU-friendly vectorized sampling (no Python loops over `B`).

## Current state (what exists today)
- `EnableComponentsNode(component_keys, num_enabled)` samples a **uniform** k-hot mask per sample:
  - `num_enabled` is already a `SamplerLike[int]`, so `k` can vary per sample.
  - Component selection is uniform via per-sample random scores + argsort (a random permutation) and “take first k”.
  - If **all samples have `k == 1`**, it emits a categorical label `state.y["component_id"]` (uniformly distributed across components).
- Example 06 uses `EnableComponentsNode` to gate a fixed `ViewChain` (noise/trend/periodic/event render) via `enabled_key=...`.
- Docs already point to example 06 for enabled-mask sampling (`README.md`, `DOCUMENTATION.md`).

This makes it impossible to express “component X is 0.1% of the dataset” without rewriting the process.

## Proposed change (one best solution, minimal API surface)
### 1) Add a reusable **class sampler** to `EnableComponentsNode`
Instead of adding a node-specific `component_weights=...` parameter, extend `toyts/processes/nodes.py:EnableComponentsNode` with a single optional argument that is a **Sampler** over class indices:

```python
EnableComponentsNode(
    component_keys=[...],
    num_enabled=1,
    component_id=CategoricalSampler(probs=[...]),
)
```

Behavior:
- `component_id is None` → **exact existing behavior** (backwards compatible).
- `component_id is not None` → use the sampler to sample `component_id: int64[B]` with an arbitrary (possibly highly imbalanced) distribution.

Why this is better:
- The sampling policy lives in a **Sampler** (`CategoricalSampler`, `ChoiceSampler`, custom Sampler), so the same sampler can be reused anywhere else in ToyTS that needs non-uniform categorical sampling.
- `EnableComponentsNode` stays focused on one job: “turn a sampled component id / k-hot selection into `enabled` masks and labels”.

Fail-fast validation (no silent defaults):
- `component_id` is only supported when **all samples have `k == 1`**:
  - If `num_enabled` is a constant not equal to 1 → raise in `__init__`.
  - If `num_enabled` is sampled and `k` is not all-ones at runtime → raise in `forward` with a clear message.
- Sampled `component_id` must be in `[0, len(component_keys))` for all samples; otherwise raise `ValueError` with min/max values.

Sampling algorithm (vectorized, deterministic with `rng`):
- If `component_id is not None`:
  - Sample `component_id: int64[B]` via `sampler_sample(..., dtype=torch.int64)` (sampler can be `CategoricalSampler`, etc.).
  - Construct `enabled_matrix: bool[B, N]` with a single scatter per sample.
  - Set `state.y["component_count"] = ones[B]` and `state.y["component_id"] = component_id`.
- Else:
  - Keep current uniform selection logic unchanged (scores → argsort → take first k).

Meta / reproducibility:
- Store the sampler spec in process meta so the imbalance is visible from the batch:
  - e.g. `state.meta.setdefault("enabled_spec", {})["component_id"] = component_id_sampler.spec()`
  - (small + human-readable, no large tensors)

### 2) New example derived from example 06 (keep 06 as the balanced baseline)
Add:
- `examples/07_imbalanced_components.py`
- `examples/07_imbalanced_components.ipynb` (kept in sync)

Example behavior:
- Reuse `build_basic_components_process(...)` structure from example 06, but pass `component_id=...` (a sampler) into `EnableComponentsNode`.
- Use `num_enabled = 1` by default to create an **imbalanced classification dataset** over `component_id`.
- Print an empirical histogram (counts + percentages) for `component_id` over a reasonably large batch (e.g. `batch_size=4096` on CPU; smaller if CUDA).
- Save a figure grid similar to example 06 plus a simple bar plot of class frequencies:
  - `examples/figures/07_imbalanced_components_grid.png`
  - `examples/figures/07_imbalanced_components_hist.png`

Example configuration (explicit, no hidden defaults):
- Define a **class probability vector** (or weights that are normalized) and build a sampler:
  - `component_weights_by_key = {...}` (readable) → convert to `probs: list[float]` aligned with `component_keys`
  - `component_id_sampler = CategoricalSampler(probs=probs)`
  - common: trends/noises
  - rare: `spike`, `level_change`, `gaussian`

### 3) Documentation updates
Keep docs coherent with the new capability:
- `README.md`: mention `EnableComponentsNode(..., component_id=CategoricalSampler(...))` for imbalanced datasets and point to `examples/07_imbalanced_components.py`.
- `DOCUMENTATION.md`: extend “Runtime Component Gating (Enabled Masks)” with a short note + snippet showing sampler-based imbalanced sampling.

## Extension: imbalanced mixtures when `num_enabled > 1` (k-hot selection)

### Problem
For `num_enabled > 1`, we want to sample a **subset** of components per sample (k-hot mask) such that:
- some components are **rare** marginally (long-tail),
- selection is **without replacement** within a sample (no duplicate components),
- ideally supports **per-sample k** (`SamplerLike[int]`),
- stays **GPU-friendly** and deterministic under a `torch.Generator`,
- the imbalance policy lives in a **reusable Sampler** (so we can reuse it beyond this node).

### Options

#### Option A (recommended): weighted ordering sampler + `component_order=...`
Add a sampler that produces a weighted random **ordering** of components per sample, then `EnableComponentsNode` enables the first `k`.

- **New sampler (core)**: `WeightedPermutationSampler(probs=[...])` (name TBD).
  - `sample(shape=(B, N), dtype=int64)` returns `order: int64[B, N]` (a per-row permutation of `[0..N-1]`).
  - Implementation idea: Gumbel-top-k / Plackett–Luce ranking:
    - `scores = log(probs) + gumbel_noise`, then `order = argsort(scores, desc=True)`.
    - Supports zero probs (they sink to the end); fail-fast if requested `k` exceeds count of positive probs.
- **EnableComponentsNode**: add `component_order: SamplerLike[int] | None = None`.
  - If provided: use `order` instead of the current uniform `scores.argsort` path, then reuse the existing `rank < k` logic.
  - Works naturally with **variable k** (already supported by `num_enabled`).
  - Store `component_order.spec()` in `meta["enabled_spec"]["component_order"]`.
- **Pros**: most general (covers `k=1..N`), supports variable k, sampler is reusable, no Python loops.
- **Cons**: O(B·N·logN) due to sorting (typically fine for component counts like example 06; can be optimized later with `topk`).

#### Option B: weighted “top-k indices” sampler + `component_ids=...`
Add a sampler that returns only the first `Kmax` selected indices per sample (without replacement).

- **New sampler (core)**: `WithoutReplacementSampler(probs=[...], num_samples=Kmax)` (name TBD).
  - `sample(shape=(B, Kmax), dtype=int64)` returns `ids: int64[B, Kmax]` with no duplicates per row.
  - Implementation idea: `torch.multinomial(probs_row, num_samples=Kmax, replacement=False)`.
- **EnableComponentsNode**: add `component_ids: SamplerLike[int] | None = None`.
  - If provided: scatter from `ids` into `[B, N]` with a `rank < k` keep mask (for per-sample k).
  - Requires choosing `Kmax >= max(k)` and validating `Kmax <= num_positive_probs`.
- **Pros**: O(B·Kmax) sampling, avoids sorting when N is large and K is small.
- **Cons**: you must decide/validate `Kmax`; less ergonomic when k varies widely.

#### Option C: recipe-level imbalance over explicit subsets (no generic weighted-subset sampler)
Model imbalance over **combinations** rather than per-component weights.

- Use `ChoiceSampler(choices=[subset0, subset1, ...], probs=[...])`, where each subset is a tuple/list of component keys or ids.
- Add a small node `EnableComponentsFromRecipeNode` to convert a sampled subset into `enabled` masks (and optionally a `recipe_id` label in `y`).
- **Pros**: exact control over which combinations are rare/common; easy to encode domain constraints (e.g., “never mix two periodic components”).
- **Cons**: doesn’t scale when N is large; less reusable if you just want long-tail marginals over components.

#### Option D (not recommended): independent Bernoulli per component + rejection to reach k
Sample each component with `BernoulliSampler(p_i)` then resample/reject until exactly k components are enabled.
- **Reject**: introduces Python control flow / rejection loops; can be slow and awkward on GPU; determinism is brittle.

### Example impact (when implementing)
- Extend `examples/07_imbalanced_components.py` (or add `examples/08_imbalanced_mixtures.py`) to show `num_enabled > 1`:
  - histogram of **marginal enable rates** per component (since `component_id` is no longer a single label for k>1),
  - optional demonstration of variable k (e.g. `RandIntSampler(1, 4)`).

### Tests to add (when implementing)
- Determinism with fixed generator for the new sampler + node path.
- Correctness:
  - `enabled_matrix.sum(dim=1) == component_count` for all samples.
  - no duplicates within selected indices per sample.
  - zero-prob components are never selected when `k <= num_positive_probs`.
- Validation:
  - sampler output shape/dtype checks,
  - out-of-range indices error,
  - if `k > num_positive_probs` → fail fast with a clear message.

## Tests (pytest)
Add focused tests (either a new file like `tests/test_enable_components_node_component_id_sampler.py` or extend the existing `tests/test_enable_components_node_variable_num_enabled.py`):

1) **Determinism**
- Two runs with the same `torch.Generator(...).manual_seed(s)` produce identical `enabled` masks and `component_id`.

2) **Hard-zero probabilities work as expected**
- `CategoricalSampler(probs=[0.0, 1.0, 0.0])`, `num_enabled=1` → always selects class 1.

3) **Validation of compatibility with `k`**
- If `component_id` is provided and `k` is not all-ones (constant or sampled) → raises `ValueError` explaining the constraint.

4) **Validation**
- Out-of-range sampled indices (e.g. constant sampler returning `-1` or `N`) → raises `ValueError` with min/max.
- `CategoricalSampler` input probs should be finite + non-negative + sum > 0 (if current `CategoricalSampler` lacks finiteness checks, add them as part of this change).

## Validation commands (when implementing)
- `uv run pytest`
- `uv run python examples/07_imbalanced_components.py` (confirm figures saved under `examples/figures/`)

## Open questions (for your review)
1) For `num_enabled > 1`, which option do you want?
   - A) weighted ordering sampler + `component_order=...` (recommended)
   - B) weighted top-k sampler + `component_ids=...`
   - C) recipe-level `ChoiceSampler` over explicit subsets
2) Do you need biased selection together with **per-sample k** (`num_enabled` as a sampler), or only fixed `k`?
3) For k>1, should we add a multi-label target in `y` (e.g. `y["components_multi_hot"]: bool[B, N]`) so users don’t have to treat `meta["enabled"]` as a label?
