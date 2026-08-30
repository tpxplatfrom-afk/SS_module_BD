"""
SS Tutor BD — Phase 6 Master Validation & Release Certification Runner
Orchestrates end-to-end verification across unit tests, golden tests, release audits,
and physical 2GB Android hardware benchmarks (itel A662L via ADB).
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
    print("  SS TUTOR BD — PHASE 6 MASTER VALIDATION & RELEASE CERTIFICATION")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    print("="*70)

    py = sys.executable
    steps = []

    # 1. Phases 1-4 Regression Suite
    steps.append(run_step("Phases 1-4 Regression Suite (17 tests)", [py, "tests/run_all_tests.py"]))

    # 2. Android Native Engine & Golden Tests
    steps.append(run_step("Android Native Engine & Golden Tests (4 tests)", [py, "tests/android/test_android_engine.py"]))

    # 3. Release Artifact & Secret Security Audit
    steps.append(run_step("Release Artifact & Security Audit", [py, "scripts/audit_release.py"]))

    # 4. Physical Android Real-Device Benchmark Suite (itel A662L via ADB)
    steps.append(run_step("Real 2GB Android Device Benchmark Suite (itel A662L)", [py, "benchmarks/android/real_device/real_device_benchmark.py"]))

    total = len(steps)
    passed = sum(1 for s in steps if s["passed"])
    failed = total - passed
    total_time = round(time.time() - start_time, 2)

    print("\n" + "="*70)
    print("  PHASE 6 VALIDATION & CERTIFICATION SUMMARY")
    print("="*70)
    for s in steps:
        icon = "PASS" if s["passed"] else "FAIL"
        print(f"  [{icon}] {s['label']} ({s['elapsed_s']}s)")

    print(f"\n  TOTAL RESULTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print(f"  Total Execution Time: {total_time}s")

    verdict = "PHASE 6: PRODUCTION CERTIFIED (Physical 2GB Device itel A662L Verified)" if failed == 0 else f"PHASE 6: {failed} STEP(S) FAILED"
    print(f"\n  FINAL VERDICT: {verdict}")
    print("="*70 + "\n")

    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_steps": total,
        "passed": passed,
        "failed": failed,
        "total_time_s": total_time,
        "steps": steps,
        "final_verdict": verdict
    }
    out = PROJECT_ROOT / "results" / "phase6" / "master_validation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
