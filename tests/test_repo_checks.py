from __future__ import annotations

from pathlib import Path

from scripts.repo_checks import (
    check_example_pairs,
    check_planning_consistency,
    check_required_docs,
    check_required_links,
    extract_local_links,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_extract_local_links_resolves_relative_targets(tmp_path: Path) -> None:
    repo_root = tmp_path
    _write(repo_root / "docs/index.md", "[beliefs](design-docs/core-beliefs.md)\n")
    _write(repo_root / "docs/design-docs/core-beliefs.md", "# beliefs\n")

    links = extract_local_links(repo_root, "docs/index.md")

    assert links == {"docs/design-docs/core-beliefs.md"}


def test_required_docs_report_missing_paths(tmp_path: Path) -> None:
    findings = check_required_docs(tmp_path)

    assert findings
    assert findings[0].check == "required_docs"


def test_required_links_report_missing_cross_links(tmp_path: Path) -> None:
    _write(tmp_path / "AGENTS.md", "# agents\n")
    _write(tmp_path / "README.md", "[Documentation](DOCUMENTATION.md)\n")
    _write(tmp_path / "ARCHITECTURE.md", "[README](README.md)\n[Documentation](DOCUMENTATION.md)\n")
    _write(tmp_path / "DOCUMENTATION.md", "[README](README.md)\n[Architecture](ARCHITECTURE.md)\n")
    _write(tmp_path / "docs/planning.md", "Execution plans live as GitHub issues via `gh`.\n")
    _write(tmp_path / "docs/index.md", "# docs\n")
    _write(tmp_path / "docs/design-docs/index.md", "# design\n")
    _write(tmp_path / "docs/design-docs/core-beliefs.md", "# beliefs\n")
    _write(tmp_path / "docs/design-docs/engineering-standards.md", "# engineering\n")
    _write(tmp_path / "docs/design-docs/planning-rules.md", "# planning rules\n")
    _write(tmp_path / "docs/design-docs/repository-operating-model.md", "# ops\n")
    _write(tmp_path / "docs/benchmark-specs/index.md", "# benchmark\n")
    _write(tmp_path / "docs/processes/index.md", "# processes\n")
    _write(tmp_path / "docs/views/index.md", "# views\n")
    _write(tmp_path / "docs/references/index.md", "# refs\n")
    _write(tmp_path / "docs/generated/index.md", "# generated\n")
    _write(tmp_path / "docs/research/index.md", "# research\n")
    _write(tmp_path / "docs/quality-score.md", "# quality\n")
    _write(tmp_path / "docs/tech-debt.md", "# debt\n")
    _write(tmp_path / "papers/README.md", "# papers\n")

    findings = check_required_links(tmp_path)

    assert findings
    assert any("AGENTS.md must link to docs/index.md" in finding.message for finding in findings)


def test_example_pairs_require_matching_notebooks(tmp_path: Path) -> None:
    _write(tmp_path / "examples/demo.py", "print('demo')\n")
    _write(tmp_path / "examples/paired.py", "print('paired')\n")
    _write(tmp_path / "examples/paired.ipynb", "{}\n")
    _write(tmp_path / "examples/__init__.py", "")

    findings = check_example_pairs(tmp_path)

    assert len(findings) == 1
    assert "examples/demo.py is missing a matching examples/demo.ipynb notebook." == findings[0].message


def test_planning_consistency_accepts_issue_based_workflow(tmp_path: Path) -> None:
    _write(tmp_path / "AGENTS.md", "Use [planning](docs/planning.md).\n")
    _write(
        tmp_path / "docs/planning.md",
        "# Planning\nExecution plans live as GitHub issues.\nUse `gh` from the repository root.\n",
    )

    findings = check_planning_consistency(tmp_path)

    assert findings == []
