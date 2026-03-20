# Repository Knowledge Map

This `docs/` tree is the repository-local system of record for agent-operable knowledge. Start here, then read only the relevant deeper documents.

## Core Entry Points

- [docs/planning.md](planning.md): plan workflow and commit-reference convention.
- [docs/design-docs/index.md](design-docs/index.md): durable repository and library beliefs.
- [docs/design-docs/engineering-standards.md](design-docs/engineering-standards.md): detailed implementation standards for agents and maintainers.
- [docs/design-docs/planning-rules.md](design-docs/planning-rules.md): plan content requirements that complement the `gh` workflow.
- [docs/benchmark-specs/index.md](benchmark-specs/index.md): versioned benchmark semantics and benchmark-specific conventions.
- [docs/processes/index.md](processes/index.md): public process and process-node map.
- [docs/views/index.md](views/index.md): public view and observation-model map.
- [docs/references/index.md](references/index.md): short local references for agent workflows.
- [docs/generated/index.md](generated/index.md): stable generated documentation artifacts.
- [docs/research/index.md](research/index.md): research material and the role of `papers/`.
- [docs/quality-score.md](quality-score.md): current repository quality scorecard.
- [docs/tech-debt.md](tech-debt.md): tracked structural debt.

## Top-Level Docs

- [README.md](../README.md): human-facing onboarding. Agents should update it when onboarding changes, but should not use it as a primary context source.
- [ARCHITECTURE.md](../ARCHITECTURE.md): stable design contracts and durable repository model.
- [DOCUMENTATION.md](../DOCUMENTATION.md): operational usage details and example guidance.

## Validation Workflow

Use one standard validation command when you need the full repo loop:

```bash
uv run python -m scripts.harness_validate --skip-profile
```

This command runs repository checks, the test suite, representative smoke runs, and compile checks. Transient validation artifacts go to `.artifacts/validation/`; stable generated docs belong under [docs/generated/index.md](generated/index.md).
