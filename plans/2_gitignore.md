# Add `.gitignore`

## Goal

Add a basic `.gitignore` to keep local virtualenvs, caches, and build artifacts out of the Aionoscope git history.

## Scope

- Add `.gitignore` with ignores for `.venv`, `uv` cache, `pytest` cache, `__pycache__`, build outputs, and coverage artifacts.

## Verification

- Run `uv run pytest` from the repository root.
