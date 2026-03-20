# Quality Score

- [docs/index.md](index.md): return to the docs map.

This scorecard is intentionally simple. It tracks areas that affect agent and human legibility.

| Area | Grade | Notes | Next action |
| --- | --- | --- | --- |
| Planning workflow | B | GitHub-issue planning exists, but older checked-in-plan guidance needed cleanup. | Keep `AGENTS.md` and `docs/planning.md` aligned. |
| Docs topology | B | `docs/` now has a map and indexes, but coverage is still concise. | Expand only where repeated work shows a gap. |
| Validation harness | B | Standard command path exists with machine-readable outputs. | Add more domain-specific probes when recurring failures justify them. |
| Repo checks | B | Docs links, example pairs, and planning consistency are enforced. | Add new checks only for durable invariants. |
| Example surface | B | Paired `.py` / `.ipynb` coverage is enforced structurally. | Improve content-level sync only if drift becomes a recurring issue. |
| Research inputs | C | `papers/` is documented, but still lightly integrated. | Promote only durable findings into stable docs. |
