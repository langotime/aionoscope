from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.repo_checks import run_repo_checks, write_report


@dataclass(frozen=True)
class StepResult:
    name: str
    command: list[str]
    returncode: int
    duration_sec: float
    timed_out: bool
    stdout_path: str
    stderr_path: str


def _run_command(
    *,
    name: str,
    command: list[str],
    repo_root: Path,
    output_dir: Path,
    timeout_sec: float,
) -> StepResult:
    stdout_path = output_dir / f"{name}.stdout.txt"
    stderr_path = output_dir / f"{name}.stderr.txt"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
        stdout_text = completed.stdout
        stderr_text = completed.stderr
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        stderr_suffix = exc.stderr or ""
        stderr_text = (
            f"{stderr_suffix}\nTimed out after {timeout_sec:.1f} seconds while running: "
            f"{' '.join(command)}\n"
        )
        returncode = 124
        timed_out = True

    duration_sec = time.perf_counter() - start
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    return StepResult(
        name=name,
        command=command,
        returncode=returncode,
        duration_sec=duration_sec,
        timed_out=timed_out,
        stdout_path=stdout_path.relative_to(repo_root).as_posix(),
        stderr_path=stderr_path.relative_to(repo_root).as_posix(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard repository validation harness.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to validate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".artifacts/validation/latest"),
        help="Directory for machine-readable reports and command logs.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip the lightweight profiler step.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_report = run_repo_checks(repo_root)
    write_report(repo_report, output_dir / "repo_checks.json")

    steps: list[StepResult] = []
    command_specs = [
        ("pytest", [sys.executable, "-m", "pytest", "-q"], 180.0),
        (
            "smoke_examples",
            [
                sys.executable,
                "-m",
                "scripts.smoke_examples",
                "--json-output",
                str(output_dir / "smoke_examples.json"),
            ],
            60.0,
        ),
        (
            "compile_pulse",
            [
                sys.executable,
                "-m",
                "scripts.compile_check",
                "--process",
                "pulse",
                "--backend",
                "eager",
                "--device",
                "cpu",
                "--batch-size",
                "1",
                "--seq-len",
                "128",
                "--frequency-hz",
                "5.0",
                "--sample-rate-hz",
                "250.0",
                "--seed",
                "7",
            ],
            60.0,
        ),
        (
            "compile_trend",
            [
                sys.executable,
                "-m",
                "scripts.compile_check",
                "--process",
                "trend",
                "--backend",
                "eager",
                "--device",
                "cpu",
                "--batch-size",
                "1",
                "--seq-len",
                "100",
                "--components",
                "3",
                "--seed",
                "11",
            ],
            60.0,
        ),
    ]

    if not args.skip_profile:
        command_specs.append(
            (
                "profile_trend",
                [
                    sys.executable,
                    "-m",
                    "scripts.profile_generation",
                    "--process",
                    "trend",
                    "--device",
                    "cpu",
                    "--batch-size",
                    "1",
                    "--seq-len",
                    "100",
                    "--components",
                    "3",
                    "--steps",
                    "1",
                    "--warmup",
                    "0",
                ],
                60.0,
            )
        )

    for name, command, timeout_sec in command_specs:
        step_result = _run_command(
            name=name,
            command=command,
            repo_root=repo_root,
            output_dir=output_dir,
            timeout_sec=timeout_sec,
        )
        steps.append(step_result)

    summary = {
        "repo_checks_ok": repo_report.ok,
        "steps": [asdict(step) for step in steps],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    failed_steps = [step for step in steps if step.returncode != 0]
    if repo_report.ok and not failed_steps:
        print(f"Validation harness passed. Artifacts: {output_dir.relative_to(repo_root)}")
        return

    print(f"Validation harness failed. Artifacts: {output_dir.relative_to(repo_root)}")
    if not repo_report.ok:
        print("- Repository structure checks failed.")
    for step in failed_steps:
        timeout_suffix = " (timed out)" if step.timed_out else ""
        print(f"- Step {step.name} failed with return code {step.returncode}{timeout_suffix}.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
