"""
SS Tutor BD — Phase 8.3 Master Validation Runner
Executes comprehensive end-to-end certification across all phases and Phase 8.3 capacity suites.
"""
import sys
import os
import subprocess
import json
import time
import datetime
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = PROJECT_ROOT / "results" / "phase8.3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA256 = "bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb"


def run_step(label: str, cmd: list) -> dict:
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
        for line in output.split("\n")[-15:]:
            print("  " + line)
    return {"label": label, "passed": passed, "elapsed_s": elapsed, "returncode": res.returncode}


def main():
    start_time = time.time()
    print("\n" + "="*70)
    print("  SS TUTOR BD — PHASE 8.3 MASTER VALIDATION RUNNER")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    print("="*70)

    py = sys.executable
    steps = []

    # 1. Full Phase 1-4 Regression Suite
    steps.append(run_step("Phases 1-4 Complete Regression Suite (17 tests)", [py, "tests/run_all_tests.py"]))

    # 2. Phase 8 Curriculum Suite
    steps.append(run_step("Phase 8 Curriculum & Module Suite (6 tests)", [py, "tests/test_phase8_curriculum.py"]))

    # 3. Phase 8.2 Core Master Suite
    steps.append(run_step("Phase 8.2 Core Model Master Suite (12 tests)", [py, "tests/test_phase8_2_core_master.py"]))

    # 4. Phase 8.3 Core Capacity Suite
    steps.append(run_step("Phase 8.3 Core Model Capacity Suite (12 tests)", [py, "tests/test_phase8_3_core_capacity.py"]))

    # 5. Core Master SHA-256 Immutability Check
    master_sf = PROJECT_ROOT / "models" / "core" / "ss_bangladesh" / "model" / "model.safetensors"
    if master_sf.exists():
        actual_hash = hashlib.sha256(master_sf.read_bytes()).hexdigest()
        is_immutable = (actual_hash == EXPECTED_SHA256)
        print(f"\n{'='*65}")
        print("  RUNNING: Post-Benchmark SHA-256 Immutability Anchor Check")
        print(f"{'='*65}")
        print(f"  Expected Hash: {EXPECTED_SHA256}")
        print(f"  Actual Hash:   {actual_hash}")
        print(f"  Status:        {'VERIFIED_MATCH (Zero Drift)' if is_immutable else 'FAILED_DRIFT_DETECTED'}")
        steps.append({"label": "Core Master SHA-256 Immutability Check", "passed": is_immutable, "elapsed_s": 0.01, "returncode": 0 if is_immutable else 1})
    else:
        steps.append({"label": "Core Master SHA-256 Immutability Check", "passed": False, "elapsed_s": 0.0, "returncode": 1})

    # 6. Android Real-Device Verification Step
    steps.append(run_step("Android Real-Device Verification (itel A662L)", [py, "scripts/benchmark_android_core.py"]))

    # 7. Release Artifact & Security Audit
    steps.append(run_step("Release Artifact & Security Audit", [py, "scripts/audit_release.py"]))

    total = len(steps)
    passed_count = sum(1 for s in steps if s["passed"])
    failed_count = total - passed_count
    total_time = round(time.time() - start_time, 2)

    print("\n" + "="*70)
    print("  PHASE 8.3 MASTER VALIDATION SUMMARY")
    print("="*70)
    for s in steps:
        icon = "PASS" if s["passed"] else "FAIL"
        print(f"  [{icon}] {s['label']} ({s['elapsed_s']}s)")

    print(f"\n  TOTAL RESULTS: {passed_count} PASSED / {failed_count} FAILED / {total} TOTAL")
    print(f"  Total Execution Time: {total_time}s")

    verdict = "PHASE 8.3: FULLY CHARACTERIZED" if failed_count == 0 else f"PHASE 8.3: FAILED ({failed_count} steps failed)"
    print(f"\n  FINAL VERDICT: {verdict}")
    print("="*70 + "\n")

    summary_data = {
        "phase": "8.3",
        "title": "Core Model Master Capability Characterization & Real-Device Offline Capacity Study",
        "timestamp": datetime.datetime.now().isoformat(),
        "final_verdict": verdict,
        "total_steps": total,
        "passed_steps": passed_count,
        "failed_steps": failed_count,
        "total_execution_time_s": total_time,
        "steps": steps
    }
    with open(RESULTS_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
