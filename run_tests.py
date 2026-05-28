

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_DIR = PROJECT_ROOT / "tests"
LINE = "-" * 70
DOUBLE = "=" * 70

# pytest exit codes (https://docs.pytest.org/en/stable/reference/exit-codes.html)
EXIT_OK = 0
EXIT_NO_TESTS = 5  # not a real failure — file exists but has no collected tests


@dataclass
class TestResult:
    name: str
    exit_code: int
    duration: float

    @property
    def status(self) -> str:
        if self.exit_code == EXIT_OK:
            return "[PASS]"
        if self.exit_code == EXIT_NO_TESTS:
            return "[SKIP]"
        return "[FAIL]"


def discover_test_files() -> List[Path]:
    """Find all test_*.py files in tests/ (skips __init__.py and conftest.py)."""
    return sorted(TESTS_DIR.glob("test_*.py"))


def run_one_file(path: Path, verbose: bool, with_cov: bool) -> TestResult:
    """Run a single test file in a fresh pytest subprocess."""
    args = [sys.executable, "-m", "pytest", str(path)]
    args.append("-v" if verbose else "-q")
    args.append("--tb=short")
    if with_cov:
        # --cov-append accumulates across suites into one .coverage file
        args.extend(["--cov=src", "--cov-append", "--cov-report="])

    print(f"\n{LINE}")
    print(f"  RUNNING: {path.name}")
    print(LINE)

    start = time.time()
    result = subprocess.run(args, cwd=PROJECT_ROOT)
    duration = time.time() - start

    return TestResult(name=path.name, exit_code=result.returncode, duration=duration)


def print_summary(results: List[TestResult]) -> int:
    """
    Print final summary table.

    Returns 0 if no real failures (skips are OK), 1 otherwise.
    """
    print("\n" + DOUBLE)
    print("  TEST SUITE SUMMARY")
    print(DOUBLE)
    print(f"  {'FILE':<45} {'STATUS':<10} {'TIME':>10}")
    print(f"  {'-' * 45} {'-' * 10} {'-' * 10}")

    for r in results:
        print(f"  {r.name:<45} {r.status:<10} {r.duration:>8.2f}s")

    print(f"  {'-' * 45} {'-' * 10} {'-' * 10}")

    passed = sum(1 for r in results if r.exit_code == EXIT_OK)
    skipped = sum(1 for r in results if r.exit_code == EXIT_NO_TESTS)
    failed = sum(1 for r in results if r.exit_code not in (EXIT_OK, EXIT_NO_TESTS))
    total = len(results)
    total_time = sum(r.duration for r in results)

    if failed == 0 and skipped == 0:
        print(f"  ALL {total} SUITES PASSED in {total_time:.2f}s")
    elif failed == 0:
        print(f"  {passed}/{total} SUITES PASSED, {skipped} SKIPPED "
              f"in {total_time:.2f}s")
    else:
        print(f"  {failed}/{total} SUITES FAILED  "
              f"({passed} passed, {skipped} skipped) in {total_time:.2f}s")
        print(f"\n  Failed suites:")
        for r in results:
            if r.exit_code not in (EXIT_OK, EXIT_NO_TESTS):
                print(f"    - {r.name}")

    print(DOUBLE)
    # Skipped suites do NOT cause overall failure
    return 0 if failed == 0 else 1


def show_coverage_report() -> None:
    """Render the accumulated coverage report after all suites complete."""
    print("\n" + DOUBLE)
    print("  COVERAGE REPORT")
    print(DOUBLE)
    subprocess.run(
        [sys.executable, "-m", "coverage", "report", "-m"],
        cwd=PROJECT_ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="IntelliDesk test orchestrator")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose pytest output for each suite")
    parser.add_argument("--only",
                        help="Run only the named test file (e.g. test_generate_data.py)")
    parser.add_argument("--cov", action="store_true",
                        help="Collect coverage across all suites")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop after the first failing suite")
    args = parser.parse_args()

    if not TESTS_DIR.exists():
        print(f"[ERROR] tests/ directory not found at {TESTS_DIR}")
        return 1

    if args.only:
        target = TESTS_DIR / args.only
        if not target.exists():
            print(f"[ERROR] Test file not found: {target}")
            return 1
        files = [target]
    else:
        files = discover_test_files()

    if not files:
        print("[WARN] No test files found in tests/")
        return 1

    # Wipe stale coverage data for a clean slate
    if args.cov:
        cov_file = PROJECT_ROOT / ".coverage"
        if cov_file.exists():
            cov_file.unlink()

    print(f"Discovered {len(files)} test file(s) in {TESTS_DIR.name}/")
    for f in files:
        print(f"  - {f.name}")

    results: List[TestResult] = []
    for path in files:
        result = run_one_file(path, verbose=args.verbose, with_cov=args.cov)
        results.append(result)
        if args.fail_fast and result.status == "[FAIL]":
            print(f"\n[--fail-fast] Stopping after {path.name} failed.")
            break

    exit_code = print_summary(results)

    if args.cov:
        show_coverage_report()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())