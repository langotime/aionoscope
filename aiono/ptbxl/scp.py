from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SCPStatement:
    code: str
    description: str
    diagnostic: bool
    form: bool
    rhythm: bool
    diagnostic_class: str | None
    diagnostic_subclass: str | None


def _parse_bool_field(value: str | None, *, field: str, code: str) -> bool:
    if value is None or value == "":
        return False
    try:
        return float(value) != 0.0
    except ValueError as exc:
        raise ValueError(
            f"Invalid PTB-XL scp_statements.csv value for field '{field}' at code '{code}': {value!r}."
        ) from exc


@functools.lru_cache(maxsize=1)
def load_scp_statements() -> dict[str, SCPStatement]:
    """Load PTB-XL SCP statements from packaged CSV (in CSV order)."""
    csv_path = Path(__file__).resolve().parents[1] / "assets" / "ptbxl" / "scp_statements.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Missing packaged PTB-XL taxonomy file. "
            f"Expected at {csv_path}."
        )

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("PTB-XL scp_statements.csv has no header.")
        index_key = reader.fieldnames[0]
        if index_key != "":
            raise ValueError(
                "PTB-XL scp_statements.csv must have a leading empty column name "
                f"(got {index_key!r}); expected the PTB-XL format."
            )

        statements: dict[str, SCPStatement] = {}
        for row in reader:
            code = (row.get(index_key) or "").strip()
            if not code:
                raise ValueError("PTB-XL scp_statements.csv contains an empty SCP code.")
            if code in statements:
                raise ValueError(f"Duplicate SCP code in PTB-XL scp_statements.csv: '{code}'.")

            description = (row.get("description") or "").strip()
            diagnostic = _parse_bool_field(row.get("diagnostic"), field="diagnostic", code=code)
            form = _parse_bool_field(row.get("form"), field="form", code=code)
            rhythm = _parse_bool_field(row.get("rhythm"), field="rhythm", code=code)
            diagnostic_class = (row.get("diagnostic_class") or "").strip() or None
            diagnostic_subclass = (row.get("diagnostic_subclass") or "").strip() or None

            statements[code] = SCPStatement(
                code=code,
                description=description,
                diagnostic=diagnostic,
                form=form,
                rhythm=rhythm,
                diagnostic_class=diagnostic_class,
                diagnostic_subclass=diagnostic_subclass,
            )

    if not statements:
        raise ValueError("PTB-XL scp_statements.csv contains no rows.")
    return statements


def ptbxl_all_codes() -> list[str]:
    """Return all 71 SCP codes in CSV order."""
    return list(load_scp_statements().keys())


def ptbxl_group_indices(*, scp_codes: list[str]) -> dict[str, list[int]]:
    """Return index lists for rhythm/diagnostic/form within scp_codes."""
    if not scp_codes:
        raise ValueError("scp_codes must be non-empty.")
    if len(set(scp_codes)) != len(scp_codes):
        raise ValueError("scp_codes must be unique.")

    all_codes = ptbxl_all_codes()
    if scp_codes != all_codes:
        raise ValueError(
            "scp_codes must match PTB-XL CSV ordering. "
            "Use ptbxl_all_codes() to obtain the canonical list."
        )

    code_to_index = {code: idx for idx, code in enumerate(scp_codes)}
    groups = {
        "rhythm": [],
        "diagnostic": [],
        "form": [],
    }
    for code, stmt in load_scp_statements().items():
        if stmt.rhythm:
            groups["rhythm"].append(code_to_index[code])
        if stmt.diagnostic:
            groups["diagnostic"].append(code_to_index[code])
        if stmt.form:
            groups["form"].append(code_to_index[code])
    return groups


def ptbxl_codes_by_group(
    group: Literal["diagnostic", "form", "rhythm"],
) -> list[str]:
    """Return SCP codes in the requested PTB-XL group (in CSV order)."""
    statements = load_scp_statements()
    if group == "diagnostic":
        return [code for code, stmt in statements.items() if stmt.diagnostic]
    if group == "form":
        return [code for code, stmt in statements.items() if stmt.form]
    if group == "rhythm":
        return [code for code, stmt in statements.items() if stmt.rhythm]
    raise ValueError(f"Unknown group '{group}'. Expected diagnostic/form/rhythm.")


def ptbxl_rhythm_codes() -> list[str]:
    """Return the 12 PTB-XL rhythm SCP codes (in CSV order)."""
    return ptbxl_codes_by_group("rhythm")
