from __future__ import annotations


def labels_make_lookup(names: list[str]) -> dict[str, int]:
    if not names:
        raise ValueError("names must be non-empty.")
    if len(set(names)) != len(names):
        raise ValueError("names must be unique.")
    return {name: idx for idx, name in enumerate(names)}
