from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)")
SKIPPED_EXAMPLE_NAMES = {"__init__"}
REQUIRED_DOCS = (
    "docs/index.md",
    "docs/design-docs/index.md",
    "docs/design-docs/core-beliefs.md",
    "docs/design-docs/engineering-standards.md",
    "docs/design-docs/planning-rules.md",
    "docs/design-docs/repository-operating-model.md",
    "docs/benchmark-specs/index.md",
    "docs/processes/index.md",
    "docs/views/index.md",
    "docs/references/index.md",
    "docs/generated/index.md",
    "docs/research/index.md",
    "docs/quality-score.md",
    "docs/tech-debt.md",
    "docs/planning.md",
    "papers/README.md",
)
REQUIRED_LINKS = {
    "AGENTS.md": (
        "docs/index.md",
        "docs/planning.md",
        "ARCHITECTURE.md",
        "README.md",
        "DOCUMENTATION.md",
    ),
    "README.md": ("docs/index.md", "ARCHITECTURE.md", "DOCUMENTATION.md"),
    "ARCHITECTURE.md": ("docs/index.md", "README.md", "DOCUMENTATION.md"),
    "DOCUMENTATION.md": ("docs/index.md", "README.md", "ARCHITECTURE.md"),
    "docs/index.md": (
        "docs/planning.md",
        "docs/design-docs/index.md",
        "docs/design-docs/engineering-standards.md",
        "docs/design-docs/planning-rules.md",
        "docs/benchmark-specs/index.md",
        "docs/processes/index.md",
        "docs/views/index.md",
        "docs/references/index.md",
        "docs/generated/index.md",
        "docs/research/index.md",
        "docs/quality-score.md",
        "docs/tech-debt.md",
    ),
    "docs/design-docs/index.md": (
        "docs/index.md",
        "docs/design-docs/core-beliefs.md",
        "docs/design-docs/engineering-standards.md",
        "docs/design-docs/planning-rules.md",
        "docs/design-docs/repository-operating-model.md",
    ),
    "docs/design-docs/engineering-standards.md": ("docs/design-docs/index.md",),
    "docs/design-docs/planning-rules.md": (
        "docs/design-docs/index.md",
        "docs/planning.md",
    ),
    "docs/benchmark-specs/index.md": ("docs/index.md",),
    "docs/processes/index.md": ("docs/index.md",),
    "docs/views/index.md": ("docs/index.md",),
    "docs/references/index.md": ("docs/index.md",),
    "docs/generated/index.md": ("docs/index.md",),
    "docs/research/index.md": ("docs/index.md", "papers/README.md"),
    "docs/quality-score.md": ("docs/index.md",),
    "docs/tech-debt.md": ("docs/index.md",),
    "papers/README.md": ("docs/research/index.md",),
}


@dataclass(frozen=True)
class CheckFinding:
    check: str
    message: str


@dataclass(frozen=True)
class CheckReport:
    ok: bool
    findings: list[CheckFinding]


def _repo_path(repo_root: Path, relative_path: str) -> Path:
    return (repo_root / relative_path).resolve()


def _normalize_local_link(source_path: Path, link_target: str, repo_root: Path) -> str | None:
    if not link_target:
        return None
    if "://" in link_target or link_target.startswith(("mailto:", "#")):
        return None
    resolved = (source_path.parent / link_target).resolve()
    return resolved.relative_to(repo_root.resolve()).as_posix()


def extract_local_links(repo_root: Path, relative_path: str) -> set[str]:
    source_path = _repo_path(repo_root, relative_path)
    text = source_path.read_text(encoding="utf-8")
    links: set[str] = set()
    for raw_target in MARKDOWN_LINK_RE.findall(text):
        normalized = _normalize_local_link(source_path, raw_target, repo_root)
        if normalized is not None:
            links.add(normalized)
    return links


def check_required_docs(repo_root: Path) -> list[CheckFinding]:
    findings: list[CheckFinding] = []
    for relative_path in REQUIRED_DOCS:
        if not _repo_path(repo_root, relative_path).exists():
            findings.append(
                CheckFinding(
                    check="required_docs",
                    message=f"Missing required documentation path: {relative_path}.",
                )
            )
    return findings


def check_required_links(repo_root: Path) -> list[CheckFinding]:
    findings: list[CheckFinding] = []
    for relative_path, required_targets in REQUIRED_LINKS.items():
        source_path = _repo_path(repo_root, relative_path)
        if not source_path.exists():
            findings.append(
                CheckFinding(
                    check="required_links",
                    message=f"Cannot validate links because {relative_path} is missing.",
                )
            )
            continue

        local_links = extract_local_links(repo_root, relative_path)
        for required_target in required_targets:
            if required_target not in local_links:
                findings.append(
                    CheckFinding(
                        check="required_links",
                        message=(
                            f"{relative_path} must link to {required_target} "
                            "to preserve docs navigation."
                        ),
                    )
                )

        for local_target in local_links:
            if not _repo_path(repo_root, local_target).exists():
                findings.append(
                    CheckFinding(
                        check="required_links",
                        message=f"{relative_path} links to missing path: {local_target}.",
                    )
                )
    return findings


def check_example_pairs(repo_root: Path) -> list[CheckFinding]:
    findings: list[CheckFinding] = []
    examples_dir = repo_root / "examples"
    script_stems = {
        path.stem
        for path in examples_dir.glob("*.py")
        if path.stem not in SKIPPED_EXAMPLE_NAMES
    }
    notebook_stems = {
        path.stem
        for path in examples_dir.glob("*.ipynb")
        if path.stem not in SKIPPED_EXAMPLE_NAMES
    }

    missing_notebooks = sorted(script_stems - notebook_stems)
    missing_scripts = sorted(notebook_stems - script_stems)
    for stem in missing_notebooks:
        findings.append(
            CheckFinding(
                check="example_pairs",
                message=f"examples/{stem}.py is missing a matching examples/{stem}.ipynb notebook.",
            )
        )
    for stem in missing_scripts:
        findings.append(
            CheckFinding(
                check="example_pairs",
                message=f"examples/{stem}.ipynb is missing a matching examples/{stem}.py script.",
            )
        )
    return findings


def check_planning_consistency(repo_root: Path) -> list[CheckFinding]:
    findings: list[CheckFinding] = []
    agents_text = _repo_path(repo_root, "AGENTS.md").read_text(encoding="utf-8")
    planning_text = _repo_path(repo_root, "docs/planning.md").read_text(encoding="utf-8")

    if "Put plans into the plans/" in agents_text:
        findings.append(
            CheckFinding(
                check="planning_consistency",
                message="AGENTS.md still instructs checked-in plan files under plans/.",
            )
        )
    if "Execution plans live as GitHub issues" not in planning_text:
        findings.append(
            CheckFinding(
                check="planning_consistency",
                message="docs/planning.md must state that execution plans live as GitHub issues.",
            )
        )
    if "`gh`" not in planning_text:
        findings.append(
            CheckFinding(
                check="planning_consistency",
                message="docs/planning.md must document GitHub CLI gh as the planning tool.",
            )
        )
    return findings


def run_repo_checks(repo_root: Path) -> CheckReport:
    findings = [
        *check_required_docs(repo_root),
        *check_required_links(repo_root),
        *check_example_pairs(repo_root),
        *check_planning_consistency(repo_root),
    ]
    return CheckReport(ok=not findings, findings=findings)


def write_report(report: CheckReport, output_path: Path) -> None:
    payload = {
        "ok": report.ok,
        "findings": [asdict(finding) for finding in report.findings],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository structure and docs checks.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to validate.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_repo_checks(args.repo_root.resolve())

    if args.json_output is not None:
        write_report(report, args.json_output)

    if report.ok:
        print("Repository checks passed.")
        return

    print("Repository checks failed:")
    for finding in report.findings:
        print(f"- [{finding.check}] {finding.message}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
