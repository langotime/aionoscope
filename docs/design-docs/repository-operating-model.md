# Repository Operating Model

This repository is organized for agent legibility as well as human readability.

## Knowledge Layout

- [AGENTS.md](../../AGENTS.md) is the short table of contents for agents.
- [docs/index.md](../index.md) is the repository map and the starting point for deeper docs.
- [README.md](../../README.md), [ARCHITECTURE.md](../../ARCHITECTURE.md), and [DOCUMENTATION.md](../../DOCUMENTATION.md) remain the canonical top-level entry points for users and contributors.

## Planning

- Plans live as GitHub issues created and updated with `gh`; see [docs/planning.md](../planning.md).
- Checked-in execution-plan Markdown under `plans/` is not part of the active workflow.
- Related commits should use the non-closing footer format documented in [docs/planning.md](../planning.md).

## Validation

- The standard validation command is `uv run python -m scripts.harness_validate --skip-profile`.
- Repository checks are implemented in `scripts/repo_checks.py`.
- Representative smoke runs live in `scripts/smoke_examples.py`.
- Compile and profiling probes live in `scripts/compile_check.py` and `scripts/profile_generation.py`.

## Artifacts

- Stable generated documentation belongs under [docs/generated/index.md](../generated/index.md).
- Transient validation artifacts belong under `.artifacts/validation/` and should not be committed.

## Research Inputs

- External paper inputs and converted notes live under [papers/README.md](../../papers/README.md).
- Research inputs inform design work but are not part of the public package API.
