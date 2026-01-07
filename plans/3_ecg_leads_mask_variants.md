# Plan: ECGLeadsView A0 variants (selected)

## Goal
Enable three modes for the mixing mask (A0) via a single API:
1) single mask `[C, K]`, 2) batch of masks `[B, C, K]`, 3) on-the-fly random mask generation.

## Selected API (single class, flexible input)
- Keep `ECGLeadsView` but let `A0` accept either a tensor or a generator function.
- `A0` accepted forms:
  - `torch.Tensor` of shape `[C, K]` or `[B, C, K]`
  - `Callable[[int, torch.Generator, torch.device], torch.Tensor]` returning `[B, C, K]`
- Resolution order per forward: if `A0` is callable, call it to get `[B, C, K]`; else if `A0` is `[B, C, K]`, validate `B`; else use `[C, K]` and expand.

## Implementation steps (once variant is chosen)
1) Update `ECGLeadsView` (or new view classes) to validate shapes and enforce KISS behavior.
2) Ensure RNG handling uses `rng_make_generator` and passes generator into any sampler.
3) Preserve meta fields (`A0`, `A`, `seed`, etc.); for batched/sampled, store the per-batch `A` used.
4) Add/adjust tests in `tests/`:
   - static `[C, K]` path
   - batched `[B, C, K]` path
   - sampler path with deterministic generator
5) Update docs/examples to show the new usage.

## Decision notes
- Single-class API keeps surface area minimal while covering all requested modes.
