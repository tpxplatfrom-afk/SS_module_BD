"""
SS Tutor BD — Phase 5 Master Validation Runner
Runs all non-Android validation automatically. Combines:
  - Python core regression suite (Phases 1-4)
  - Android engine unit & golden tests (Phase 5)
  - Android memory benchmark (emulated)
  - Android 100-question quality benchmark (emulated)
  - Release artifact audit

Usage:
    python scripts/run_phase5_validation.py
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
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT)
    )
    elapsed = round(time.time() - t0, 2)
    passed = result.returncode == 0
    output = (result.stdout + result.stderr).strip()
    if output:
        for line in output.split("\n")[-20:]:  # Show last 20 lines
            print("  " + line)
    return {"label": label, "passed": passed, "elapsed_s": elapsed, "returncode": result.returncode}


def run_inline(label: str, fn) -> dict:
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        elapsed = round(time.time() - t0, 2)
        return {"label": label, "passed": True, "elapsed_s": elapsed}
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"  ERROR: {e}")
        return {"label": label, "passed": False, "elapsed_s": elapsed, "error": str(e)}


def main():
    start = time.time()
    print("\n" + "="*70)
    print("  SS TUTOR BD — PHASE 5 MASTER VALIDATION RUNNER")
    print(f"  {datetime.datetime.now().isoformat()}")
    print("="*70)

    py = sys.executable
    steps = []

    # 1. Core Regression Suite (Phases 1-4)
    steps.append(run_step("Phase 1-4 Regression Suite (17 tests)", [py, "tests/run_all_tests.py"]))

    # 2. Android Engine Unit & Golden Tests
    steps.append(run_step("Android Engine Unit & Golden Tests (4 tests)", [py, "tests/android/test_android_engine.py"]))

    # 3. Android Memory Benchmark (emulated)
    steps.append(run_step("Android Memory Benchmark (Emulated)", [py, "benchmarks/android/android_memory_benchmark.py"]))

    # 4. Android Quality Benchmark (emulated)
    steps.append(run_step("Android Quality Benchmark 100Q (Emulated)", [py, "benchmarks/android/run_android_quality_benchmark.py"]))

    # 5. Release Audit
    steps.append(run_step("Release Artifact Audit", [py, "scripts/audit_release.py"]))

    # Summary
    total = len(steps)
    passed = sum(1 for s in steps if s["passed"])
    failed = total - passed
    total_time = round(time.time() - start, 2)

    print("\n" + "="*70)
    print("  PHASE 5 VALIDATION SUMMARY")
    print("="*70)
    for s in steps:
        icon = "PASS" if s["passed"] else "FAIL"
        print(f"  [{icon}] {s['label']} ({s['elapsed_s']}s)")

    print(f"\n  RESULT: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print(f"  Total Time: {total_time}s")

    verdict = "PHASE 5: ALL AUTOMATED VALIDATIONS PASS" if failed == 0 else f"PHASE 5: {failed} VALIDATION(S) FAILED"
    print(f"\n  {verdict}")
    print("\n  NOTE: Android PSS memory gates (A/B/C) require real device measurement.")
    print("  Run: adb shell dumpsys meminfo bd.sstutor.app")
    print("="*70 + "\n")

    # Save full results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_steps": total,
        "passed": passed,
        "failed": failed,
        "total_time_s": total_time,
        "steps": steps,
        "verdict": verdict
    }
    out = PROJECT_ROOT / "results" / "phase5" / "validation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Full results saved to: {out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
