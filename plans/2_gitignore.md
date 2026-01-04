# Add `.gitignore`

## Goal

Add a basic `.gitignore` to keep local virtualenvs, caches, and build artifacts out of the ToyTS git history.

## Scope

- Add `toyts/.gitignore` with ignores for `.venv`, `uv` cache, `pytest` cache, `__pycache__`, build outputs, and coverage artifacts.

## Verification

- Run `uv run pytest` from the `toyts/` repo root.
