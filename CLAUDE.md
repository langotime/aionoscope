# Coding guide

This file provides guidance to AI agents when working with code in this repository.

## Repository Overview

This repo contains source code of the ToyTS library - PyTorch-native online synthetic time series dataset generator.

## Work planning

- By default assume that all requests are "feature requests" and should be implemented in a single phase.
- If I explicietly request to be agile, follow the agile approach: Split the work into phases where each phase delivers a working system that can be immediately used for research, with each subsequent phase building upon the previous one.
- NO API GUESSING. Use Context7 MCP to validate API of external libraries. If Context7 doesn't help - use Perplexity MCP to validate.

## Code Guidelines
- Use Python for all development
- Keep the code clean with a solid separation of actual actions and presentation
- Make sure the code is idiomatic Python
  - Follow PEP 585 and PEP 604
- Make sure the code is DRY
- Keep the code minimal
- KISS
  - To follow the KISS principle, always implement one best solution instead of implementing a multitude of options - build a unittest or a benchmark to choose which one is the best and use it. And if this option requires installing some library which is not installed, just generate a fatal error or an exception.
  - To follow the KISS principle, always implement the solution with the minimal amount of code.
- Fail fast and always return or raise a human-readable error with enough context to understand what exactly has happened and how to fix it.
- NO silent defaults.
- NO defensive programming!
- Use human-readable meaningful variable and function names. Avoid one-letter naming.
- Add comments with tensor dimensions to all PyTorch tensors.

## Project rules

## Python Development Tooling
- Use 'uv' for Python package management.
- Never protect from missing packages - it should fail right away. If the package is missing, you either forgot to install it using 'uv add' or you're not running using .venv. Use 'uv run' to run scripts.
- Use 'pytest' for writing unit tests in Python.

## Code Maintenance Principles
- Always have only one version of the code. If something can be done in two-three ways, choose the best one (google, test or ask me) and implement. 
- When reimplementing existing code, don't keep it "for fallback" - instead, use unittests to make sure the new code is performing the same way or better
- ALWAYS run unittests after changing the code and before you report that the job is done.
- Put unittests into the tests/ subdirectory.
- Use prefixing for tool naming. I.e. a tool to get an item from a knowledge base should be kb_get_item(), not get_item_from_kb().

## Development Best Practices
- ALWAYS clean up temprorary files.
- Use 'uv run python -m' to run python code to avoid module import errors. YOU MUST NOT use 'sys.path.insert'

## Engineering Best Practices

### Reproducibility
- Always accept a torch.Generator (or seed) and use it for sampling.
- Do not use global torch.manual_seed inside modules.
- Write seed/parameters into meta so a specific batch can be reproduced.

### Separation of responsibilities
- Process is responsible for ground truth and labels.
- View is responsible for presentation and measurement distortions.
- Labels must not depend on view parameters unless explicitly testing robustness.

### Anti-shortcut / anti-cheating
- Labels must be a function of the latent process state only (and its own RNG/params), never of view parameters or observation artifacts.
- View parameters (noise level, sampling rate, quantization, clipping, missingness, padding, channel order, etc.) must be sampled independently of labels by default; label-conditioned augmentation is allowed only when explicitly testing robustness and must be documented.
- Keep tensor shapes and preprocessing identical across labels; avoid variable-length/padding patterns, masks, NaN counts, or other structural cues that correlate with `y`.
- Avoid making `y` trivially recoverable from global statistics (mean/DC offset, variance, energy, max/min, number of spikes, etc.) unless the task is explicitly about those cues.
- `meta` is for reproducibility/debugging only and must not be used as model input; do not store labels or near-label proxies in `meta` unless strictly needed for debugging (and clearly namespace/label them as debug-only).
- When adding a new dataset/task, include a "shortcut check": a cheap baseline on simple per-sample stats should not reach suspiciously high accuracy; also run a label-shuffle control (accuracy ~ chance).

### Performance
- Avoid Python loops over B and N. Prefer broadcasting, einsum, or conv1d.
- Cache t_grid as a buffer in modules.
- Keep computations in float32 (or bfloat16 when needed), but treat noise/quantization carefully.
- Avoid huge intermediate tensors [B,N,L] when N and L are large.
- For pulse train, use impulse + conv1d (stage 3/optional) when possible.

### Testability
- Test shapes and dtype.
- Test determinism with a fixed torch.Generator.
- Test ranges after clipping and quantization.
- Test for absence of shortcuts (baseline features should not give overly high accuracy).

### Documentation and examples
- Document each process and view: what it models, which parameters, expected invariances.
- Keep minimal runnable scripts in examples/.

## Planning
- Write plans to files in Markdown.
- Put plans into the plans/ subridectory with a unique descriptive name and .md file type. Start each file name with "N_" prefix where N is a unique increasing counter.
