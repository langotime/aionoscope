# Planning Rules

These rules preserve the content requirements for planning while adapting the workflow to issue-based plans documented in [docs/planning.md](../planning.md).

- [docs/design-docs/index.md](index.md): return to the design-doc index.

## Planning Medium

- Plans live as GitHub issues created and updated with `gh`.
- Do not create checked-in execution-plan Markdown under `plans/`.

## What Every Meaningful Plan Must Include

- An explicit documentation review step for [ARCHITECTURE.md](../../ARCHITECTURE.md), [README.md](../../README.md), and [DOCUMENTATION.md](../../DOCUMENTATION.md).
- A `Documentation impact` section with three explicit entries:
  - `ARCHITECTURE.md`: what stable design decisions will change, or why no change is needed.
  - `README.md`: what onboarding/public-facing text will change, or why no change is needed.
  - `DOCUMENTATION.md`: what usage/operational details will change, or why no change is needed.

## What Must Happen Before Submitting A Plan For Review

- Study the existing codebase so the resulting API and internals stay coherent, lean, clean, and logical as a whole.
- Check whether the intended code already exists in larger components and propose refactors that improve composability instead of duplicating behavior.
- Describe how the change affects library documentation and what updates are required to keep the library easy to use.
- Make updating or explicitly rejecting updates to [ARCHITECTURE.md](../../ARCHITECTURE.md), [README.md](../../README.md), and [DOCUMENTATION.md](../../DOCUMENTATION.md) a mandatory part of the plan, not an optional follow-up.

## Commit And Issue Discipline

- Each meaningful workstream gets one plan issue.
- Use meaningful commits where each commit represents one coherent slice of work.
- Reference the active plan issue in related commit messages with a non-closing footer such as `Part of #123`.
- Split unrelated work into separate commits so each commit maps cleanly to one plan issue.
