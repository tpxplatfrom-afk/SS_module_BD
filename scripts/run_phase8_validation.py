"""
SS Tutor BD — Phase 8 Master Validation Runner
Executes regression tests, Phase 8 curriculum tests, coverage audits, dataset quality audits,
13D model evaluations, and release security checks in one command.
"""
import sys
import subprocess
import json
import time
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_step(label: str, cmd: list[str]) -> dict:
    print(f"\n{'='*65}")
    print(f"  RUNNING: {label}")
    print(f"{'='*65}")
    t0 = time.time()
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT)
    )
    elapsed = round(time.time() - t0, 2)
    passed = (res.returncode == 0)
    output = (res.stdout + res.stderr).strip()
    if output:
        for line in output.split("\n")[-25:]:
            print("  " + line)
    return {"label": label, "passed": passed, "elapsed_s": elapsed, "returncode": res.returncode}


def main():
    start_time = time.time()
    print("\n" + "="*70)
    print("  SS TUTOR BD — PHASE 8 MASTER VALIDATION RUNNER")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    print("="*70)

    py = sys.executable
    steps = []

    # 1. Phases 1-4 Complete Regression Suite (17 tests)
    steps.append(run_step("Phases 1-4 Complete Regression Suite (17 tests)", [py, "tests/run_all_tests.py"]))

    # 2. Phase 8 Curriculum & Module Test Suite (6 tests)
    steps.append(run_step("Phase 8 Curriculum & Module Suite (6 tests)", [py, "tests/test_phase8_curriculum.py"]))

    # 3. Curriculum Coverage Audit
    steps.append(run_step("Curriculum Coverage Engine Audit", [py, "core/curriculum/coverage_engine.py"]))

    # 4. Dataset Quality Audit
    steps.append(run_step("Training Dataset Quality Audit", [py, "core/curriculum/dataset_auditor.py"]))

    # 5. 13-Dimension Model Evaluation Suite
    steps.append(run_step("13-Dimension Curriculum Model Evaluation", [py, "benchmarks/phase8/curriculum_eval_suite.py"]))

    # 6. Release Security Audit
    steps.append(run_step("Release Artifact & Security Audit", [py, "scripts/audit_release.py"]))

    total = len(steps)
    passed = sum(1 for s in steps if s["passed"])
    failed = total - passed
    total_time = round(time.time() - start_time, 2)

    print("\n" + "="*70)
    print("  PHASE 8 MASTER VALIDATION SUMMARY")
    print("="*70)
    for s in steps:
        icon = "PASS" if s["passed"] else "FAIL"
        print(f"  [{icon}] {s['label']} ({s['elapsed_s']}s)")

    print(f"\n  TOTAL RESULTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print(f"  Total Execution Time: {total_time}s")

    verdict = "PHASE 8: CORE MODEL FOUNDATION & CURRICULUM ARCHITECTURE VALIDATED" if failed == 0 else f"PHASE 8: {failed} STEP(S) FAILED"
    print(f"\n  FINAL VERDICT: {verdict}")
    print("="*70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
