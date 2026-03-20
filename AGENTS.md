# Coding Guide

This file is the short entry point for agents working in this repository. Treat it as a map, not the full manual.

## Start Here

- Read [docs/index.md](docs/index.md) for the repository knowledge map.
- Use [docs/planning.md](docs/planning.md) for planning workflow. Plans live as GitHub issues created and updated with `gh`; do not create checked-in plan Markdown in `plans/`.
- Ignore [README.md](README.md) when gathering agent context; it is human-facing onboarding text.
- Use [ARCHITECTURE.md](ARCHITECTURE.md) for stable design contracts and [DOCUMENTATION.md](DOCUMENTATION.md) for operational details.

## Non-Negotiable Workflow

- Assume requests are single-phase feature work unless the user explicitly asks for an agile phased rollout.
- Do not guess external APIs. Validate them with Context7 first; if that is insufficient, use Perplexity.
- Use Python for development.
- Use `uv` for environment and package management.
- Run Python entrypoints with `uv run python -m ...` when possible.
- Use `pytest` for tests and run tests after changes before reporting completion.
- Do not keep fallback implementations. Replace old code and prove behavior with tests.
- Keep `.py` examples and matching `.ipynb` notebooks in sync.

## Engineering Rules

- Keep code minimal, idiomatic, and DRY.
- Fail fast with human-readable errors. No silent defaults. No defensive programming.
- Prefer one best implementation path.
- Use meaningful names, not one-letter identifiers.
- Add tensor-dimension comments to PyTorch tensors.
- Preserve Process -> View separation.
- Preserve reproducibility through explicit `torch.Generator` or seed handling.
- Keep labels functions of latent process state, not of view artifacts.
- Prefer vectorized tensor code over Python loops across batch or time.

See [docs/design-docs/core-beliefs.md](docs/design-docs/core-beliefs.md) for the durable beliefs, [docs/design-docs/engineering-standards.md](docs/design-docs/engineering-standards.md) for the detailed implementation standards preserved from the original guide, and [docs/design-docs/repository-operating-model.md](docs/design-docs/repository-operating-model.md) for the repository workflow model.

## Required Review Before Finishing

- Review whether [ARCHITECTURE.md](ARCHITECTURE.md), [README.md](README.md), and [DOCUMENTATION.md](DOCUMENTATION.md) must change.
- Run the standard validation path with `uv run python -m scripts.harness_validate --skip-profile` when the full harness is appropriate.
