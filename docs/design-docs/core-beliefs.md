# Core Beliefs

These are the durable beliefs that shape both the library API and the repository workflow.

## Library Beliefs

- Process owns latent truth, labels, and process-side metadata.
- View owns presentation, rendering, sensor effects, units, sampling, clipping, missingness, and noise.
- Labels must remain functions of latent process state unless a task explicitly documents robustness testing against label-dependent observation changes.
- Reproducibility is mandatory. Sampling must accept an explicit `torch.Generator` or seed, and metadata should store only the minimal values needed to regenerate behavior.
- Synthetic tasks must resist shortcuts. Avoid label leakage through shape, padding, masks, view parameters, global statistics, or metadata proxies.
- GPU-friendly vectorized tensor code is the default. Avoid Python loops across batch or time dimensions.

## Engineering Beliefs

- Keep one implementation path. Prefer replacement plus tests over fallback code.
- KISS wins. Choose one best solution, not multiple options.
- Fail fast with human-readable errors. No silent defaults. No defensive programming.
- Use idiomatic Python, `uv`, and `pytest`.
- Examples are part of the public teaching surface. Keep `.py` and `.ipynb` pairs aligned.

## Documentation Beliefs

- Top-level docs keep distinct roles: onboarding in [README.md](../../README.md), stable contracts in [ARCHITECTURE.md](../../ARCHITECTURE.md), and operational detail in [DOCUMENTATION.md](../../DOCUMENTATION.md).
- `AGENTS.md` should stay short and point to deeper repository-local knowledge.
- `docs/` should hold the durable repo map, not ephemeral task chatter.
