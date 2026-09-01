"""
SS Tutor BD — Phase 7 Master Validation & Production Release Certification Runner
Executes complete end-to-end certification across all unit tests, golden tests, security audits,
and physical 2GB Android hardware benchmarks (itel A662L).
"""
import sys
import os
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
    print("  SS TUTOR BD — PHASE 7 MASTER RELEASE CERTIFICATION RUNNER")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    print("="*70)

    py = sys.executable
    steps = []

    # 1. Full Regression Test Suite (17 tests)
    steps.append(run_step("Phases 1-4 Complete Regression Suite (17 tests)", [py, "tests/run_all_tests.py"]))

    # 2. Android Native Engine & Golden Tests (4 tests)
    steps.append(run_step("Android Native Engine & Golden Tests (4 tests)", [py, "tests/android/test_android_engine.py"]))

    # 3. Release Artifact & Security Audit
    steps.append(run_step("Release Artifact & Secret Security Audit", [py, "scripts/audit_release.py"]))

    # 4. Phase 7 Full Evaluation Suite on itel A662L Device
    steps.append(run_step("Physical 2GB Device Full Evaluation Suite (itel A662L)", [py, "scripts/phase7_evaluation_suite.py"]))

    # 5. Automatic Failure Detector
    steps.append(run_step("Automatic Failure Detector & Invariant Validation", [py, "scripts/phase7_failure_detector.py"]))

    total = len(steps)
    passed = sum(1 for s in steps if s["passed"])
    failed = total - passed
    total_time = round(time.time() - start_time, 2)

    print("\n" + "="*70)
    print("  PHASE 7 MASTER VALIDATION SUMMARY")
    print("="*70)
    for s in steps:
        icon = "PASS" if s["passed"] else "FAIL"
        print(f"  [{icon}] {s['label']} ({s['elapsed_s']}s)")

    print(f"\n  TOTAL RESULTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print(f"  Total Execution Time: {total_time}s")

    verdict = "PHASE 7: PRODUCTION CERTIFIED (Physical 2GB itel A662L Hardware Verified)" if failed == 0 else f"PHASE 7: {failed} STEP(S) FAILED"
    print(f"\n  FINAL VERDICT: {verdict}")
    print("="*70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
