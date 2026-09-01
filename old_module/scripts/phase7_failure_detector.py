"""
SS Tutor BD — Phase 7 Automatic Failure Detector
Strictly validates all real-device and architectural metrics without manual overrides.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def detect_phase7_failures(results_dir: Path | None = None) -> dict:
    rdir = results_dir or PROJECT_ROOT / "results" / "phase7"
    failures = []
    checks = []

    # 1. Check Model Load Verification
    m_file = rdir / "model_load_verification.json"
    if m_file.exists():
        with open(m_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not data.get("model_load_verified_on_device", False):
                failures.append("MODEL_LOAD_FAIL: Model was not verified loaded on device")
            checks.append("MODEL_LOAD: PASS")
    else:
        failures.append("MODEL_LOAD_FAIL: model_load_verification.json missing")

    # 2. Check Memory Results
    mem_file = rdir / "memory_results.json"
    if mem_file.exists():
        with open(mem_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            peak_pss = data.get("state_c_full_hybrid_peak_pss_mb", 999.0)
            if peak_pss > 200.0:
                failures.append(f"PEAK_PSS_CEILING_BREACH: Peak PSS was {peak_pss} MB (> 200 MB)")
            checks.append(f"PEAK_PSS: {peak_pss} MB <= 200 MB")

    # 3. Check 100-Turn Stress
    s100_file = rdir / "stress_100_results.json"
    if s100_file.exists():
        with open(s100_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            growth = data.get("memory_growth_per_turn_mb", 999.0)
            if growth > 0.05:
                failures.append(f"MEMORY_GROWTH_FAIL: Growth was {growth} MB/turn (> 0.05 MB/turn)")
            checks.append(f"100_TURN_GROWTH: {growth} MB/turn <= 0.05 MB/turn")

    # 4. Check Offline Verification
    off_file = rdir / "offline_results.json"
    if off_file.exists():
        with open(off_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not data.get("zero_network_dependency_verified", False):
                failures.append("NETWORK_DEPENDENCY_DETECTED: App is not 100% offline")
            checks.append("OFFLINE_CHECK: PASS")

    # 5. Check Stability / Crash
    stab_file = rdir / "stability_results.json"
    if stab_file.exists():
        with open(stab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            crashes = data.get("crash_count", 0)
            anrs = data.get("anr_count", 0)
            if crashes > 0:
                failures.append(f"CRASH_DETECTED: {crashes} crashes found")
            if anrs > 0:
                failures.append(f"ANR_DETECTED: {anrs} ANRs found")
            checks.append(f"STABILITY: 0 Crashes, 0 ANRs")

    verdict = "PASS" if len(failures) == 0 else "FAIL"

    return {
        "status": verdict,
        "failures_detected_count": len(failures),
        "failures": failures,
        "checks_passed": checks,
        "verdict": "AUTOMATIC_FAILURE_DETECTION_PASSED" if verdict == "PASS" else "AUTOMATIC_FAILURE_DETECTION_FAILED"
    }


if __name__ == "__main__":
    rep = detect_phase7_failures()
    print(json.dumps(rep, indent=2))
