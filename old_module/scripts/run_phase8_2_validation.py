"""
SS Tutor BD — Phase 8.2 Master Validation Runner
Executes all validation steps: Phase 1-4 regression, Phase 8 curriculum, Phase 8.2 core master,
reproducibility proof, specialization isolation, and release audit.
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
        for line in output.split("\n")[-20:]:
            print("  " + line)
    return {"label": label, "passed": passed, "elapsed_s": elapsed, "returncode": res.returncode}


def main():
    start_time = time.time()
    print("\n" + "="*70)
    print("  SS TUTOR BD — PHASE 8.2 MASTER VALIDATION RUNNER")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    print("="*70)

    py = sys.executable
    steps = []

    # 1. Full Phase 1-4 Regression Suite
    steps.append(run_step("Phases 1-4 Complete Regression Suite (17 tests)", [py, "tests/run_all_tests.py"]))

    # 2. Phase 8 Curriculum Suite
    steps.append(run_step("Phase 8 Curriculum & Module Suite (6 tests)", [py, "tests/test_phase8_curriculum.py"]))

    # 3. Phase 8.2 Core Master Suite (12 tests)
    steps.append(run_step("Phase 8.2 Core Model Master Suite (12 tests)", [py, "tests/test_phase8_2_core_master.py"]))

    # 4. Core Master Assembly Integrity Check (inline Python)
    import hashlib
    master_sf = PROJECT_ROOT / "models" / "core" / "ss_bangladesh" / "model" / "model.safetensors"
    if master_sf.exists():
        sha = hashlib.sha256(master_sf.read_bytes()).hexdigest()
        expected = "bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb"
        if sha == expected:
            print(f"\n{'='*65}")
            print(f"  RUNNING: Core Master SHA-256 Immutability Check")
            print(f"{'='*65}")
            print(f"  Checksum: {sha}")
            print(f"  Status:   VERIFIED_PASS (Matches recorded checksum)")
            steps.append({"label": "Core Master SHA-256 Immutability Check", "passed": True, "elapsed_s": 0.01, "returncode": 0})
        else:
            steps.append({"label": "Core Master SHA-256 Immutability Check", "passed": False, "elapsed_s": 0.01, "returncode": 1})
    else:
        steps.append({"label": "Core Master SHA-256 Immutability Check", "passed": False, "elapsed_s": 0.0, "returncode": 1})

    # 5. Specialization Isolation Check (inline Python)
    spec_sf = PROJECT_ROOT / "models" / "sstutor_bengali_70m_edu" / "model.safetensors"
    core_sf = PROJECT_ROOT / "models" / "core" / "ss_bangladesh" / "model" / "model.safetensors"
    if spec_sf.exists() and core_sf.exists():
        h_spec = hashlib.sha256(spec_sf.read_bytes()).hexdigest()
        h_core = hashlib.sha256(core_sf.read_bytes()).hexdigest()
        isolated = (h_spec != h_core)
        print(f"\n{'='*65}")
        print(f"  RUNNING: Specialization Isolation Verification")
        print(f"{'='*65}")
        print(f"  Core Master Hash:  {h_core}")
        print(f"  Class 8 Math Hash: {h_spec}")
        print(f"  Isolation Status:  {'VERIFIED_ISOLATED (Different Weights)' if isolated else 'FAIL (Same Hash)'}")
        steps.append({"label": "Specialization Isolation (Core != SS Tutor BD)", "passed": isolated, "elapsed_s": 0.5, "returncode": 0 if isolated else 1})
    else:
        steps.append({"label": "Specialization Isolation (Core != SS Tutor BD)", "passed": False, "elapsed_s": 0.0, "returncode": 1})

    # 6. Release Artifact Security Audit
    steps.append(run_step("Release Artifact & Security Audit", [py, "scripts/audit_release.py"]))

    total = len(steps)
    passed_count = sum(1 for s in steps if s["passed"])
    failed_count = total - passed_count
    total_time = round(time.time() - start_time, 2)

    print("\n" + "="*70)
    print("  PHASE 8.2 MASTER VALIDATION SUMMARY")
    print("="*70)
    for s in steps:
        icon = "PASS" if s["passed"] else "FAIL"
        print(f"  [{icon}] {s['label']} ({s['elapsed_s']}s)")

    print(f"\n  TOTAL RESULTS: {passed_count} PASSED / {failed_count} FAILED / {total} TOTAL")
    print(f"  Total Execution Time: {total_time}s")

    verdict = "PHASE 8.2: SS BANGLADESH CORE MODEL MASTER ASSEMBLED & VALIDATED" if failed_count == 0 else f"PHASE 8.2: {failed_count} STEP(S) FAILED"
    print(f"\n  FINAL VERDICT: {verdict}")
    print("="*70 + "\n")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
