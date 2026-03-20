# Engineering Standards

These standards preserve the detailed implementation guidance that should remain stable even though [AGENTS.md](../../AGENTS.md) stays short.

- [docs/design-docs/index.md](index.md): return to the design-doc index.

## Work Planning Defaults

- By default, treat requests as single-phase feature work.
- If the user explicitly asks for an agile rollout, split work into phases where each phase delivers a working system that is immediately usable for research.
- Do not guess external APIs. Validate them with Context7 first; if that is insufficient, use Perplexity.

## Code Guidelines

- Use Python for development.
- Keep code clean with a solid separation between actual actions and presentation.
- Keep code idiomatic Python.
- Follow PEP 585 and PEP 604.
- Keep code DRY.
- Keep code minimal.
- Follow KISS:
  - implement one best solution instead of multiple options;
  - use a test or benchmark to choose that best solution when needed;
  - if the best solution requires a missing package, fail immediately rather than adding fallback behavior.
- Fail fast with human-readable errors that explain what happened and how to fix it.
- No silent defaults.
- No defensive programming.
- Use meaningful names and avoid one-letter identifiers.
- Add tensor-dimension comments to PyTorch tensors.

## Python Tooling

- Use `uv` for Python package management.
- If a required package is missing, fail immediately instead of protecting against it.
- Prefer `uv run python -m ...` for Python entrypoints to avoid import-path hacks.
- Do not use `sys.path.insert`.
- Use `pytest` for tests.

## Code Maintenance Principles

- Keep only one version of the code.
- When something can be done multiple ways, choose the best one by checking, testing, benchmarking, or asking the user.
- When reimplementing existing code, do not keep the old implementation as fallback.
- Prove compatibility or improvement with tests.
- Put tests under `tests/`.
- Clean up temporary files.

## Reproducibility

- Always accept a `torch.Generator` or seed and use it for sampling.
- Do not use global `torch.manual_seed` inside modules.
- Write seed and minimal parameters into metadata so a specific batch can be reproduced.
- Do not duplicate sampled tensors in metadata if they already appear in `LatentState` or `Observation`.
- Do not store full-size sampled noise tensors or masks in metadata; store seeds and minimal parameters to regenerate them.

## Separation Of Responsibilities

- Process is responsible for ground truth and labels.
- View is responsible for presentation and measurement distortions.
- Labels must not depend on view parameters unless the task explicitly tests robustness to that dependency.

## Anti-Shortcut / Anti-Cheating

- Labels must be functions of latent process state only, not view parameters or observation artifacts.
- View parameters such as noise, sampling rate, clipping, missingness, padding, or channel order must be sampled independently of labels by default.
- Keep tensor shapes and preprocessing identical across labels; avoid variable-length or padding cues, masks, NaN counts, and similar structural leaks.
- Avoid making labels trivially recoverable from global statistics such as mean, variance, energy, extrema, or spike counts unless that is explicitly the point of the task.
- Treat `meta` as reproducibility/debugging support, not model input.
- Do not store labels or near-label proxies in `meta` unless strictly needed for debugging and clearly labeled as such.
- When adding a new dataset or task, include a shortcut check and a label-shuffle control.

## Performance

- Avoid Python loops over `B` and `N`; prefer broadcasting, `einsum`, or `conv1d`.
- Cache `t_grid` as a buffer in modules where appropriate.
- Keep computations in `float32` unless there is a clear reason to use something else such as `bfloat16`.
- Treat noise and quantization carefully.
- Avoid huge intermediate tensors with shape `[B, N, L]` when `N` and `L` are large.
- For pulse trains, prefer impulse plus `conv1d` when possible.

## Testability

- Test shapes and dtype.
- Test determinism with a fixed `torch.Generator`.
- Test post-view ranges after clipping and quantization.
- Test for absence of shortcuts when the task claims shortcut resistance.

## Documentation And Examples

- Document each process and view: what it models, which parameters it uses, and which invariances it expects.
- Keep minimal runnable scripts in `examples/`.
- Examples should visualize results with plots or figures when appropriate.
- Matching notebooks should remain interactive and should not wrap the whole flow in `main()`.
- After changing code, review whether [ARCHITECTURE.md](../../ARCHITECTURE.md), [README.md](../../README.md), and [DOCUMENTATION.md](../../DOCUMENTATION.md) must change.
- Treat those three docs as having distinct roles:
  - [ARCHITECTURE.md](../../ARCHITECTURE.md): stable design choices, goals, invariants, core data model, metadata contracts, execution model, and extension rules.
  - [README.md](../../README.md): short human-facing onboarding, install/run basics, public capabilities, and the first examples.
  - [DOCUMENTATION.md](../../DOCUMENTATION.md): detailed usage, metadata layout, advanced composition patterns, example guide, and operational details.
- If a code change does not require updating one of those files, say so explicitly in the plan or review instead of silently skipping it.
