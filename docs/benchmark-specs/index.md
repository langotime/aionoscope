# Benchmark Specs

- [docs/index.md](../index.md): return to the docs map.

This section tracks benchmark-specific semantics that downstream consumers should treat as stable contracts.

## Current Benchmark Contracts

- `aiono_basic_components/v1`: the public contract for the balanced basic-components benchmark family. The implementation source of truth lives in [aiono/benchmarks/aiono_basic_components.py](../../aiono/benchmarks/aiono_basic_components.py).
- PTB-XL label conventions: the public target remains `y["scp"]`, with rhythm, diagnostic, and form slices exposed through label groups as described in [ARCHITECTURE.md](../../ARCHITECTURE.md) and [DOCUMENTATION.md](../../DOCUMENTATION.md).

## When To Add Material Here

- Add a new benchmark note here when multiple repos or workflows need the same semantic contract.
- Keep benchmark-specific meaning here; keep reusable process/view mechanics in the main architecture docs.
