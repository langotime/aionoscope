# Tech Debt

- [docs/index.md](index.md): return to the docs map.

This file tracks current structural debt that affects repository legibility or reliability.

## Active Debt

- Example `.py` / `.ipynb` sync is enforced structurally, not semantically.
- Validation artifacts are machine-readable, but compile/profile outputs are still text logs rather than richer structured summaries.
- `papers/` is documented but still separated from any curated research summaries under `docs/`.
- The validation harness currently runs one broad pytest command; if runtime grows, the suite may need more intentional partitioning.
