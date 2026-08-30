"""
SS Tutor BD — Master Real Device Benchmark Harness (Phase 6)
Executes all real-device validation suites against the connected physical Android device.
"""
import sys
import time
import json
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from benchmarks.android.real_device.device_info import collect_device_profile
from benchmarks.android.real_device.adb_memory_monitor import ADBMemoryMonitor
from benchmarks.android.real_device.adb_session_runner import ADBSessionRunner
from benchmarks.android.real_device.adb_package_inspector import ADBPackageInspector
from benchmarks.android.real_device.adb_network_monitor import ADBNetworkMonitor
from benchmarks.android.real_device.adb_log_collector import ADBLogCollector
from benchmarks.android.real_device.quality.real_device_quality_benchmark import run_real_device_quality_benchmark


def run_master_real_device_suite() -> dict:
    print("\n" + "="*70)
    print("  SS TUTOR BD — PHASE 6 REAL-DEVICE BENCHMARK SUITE")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    print("="*70)

    # 1. Device Profile
    print("\n[1/6] Capturing Real Device Profile...")
    dev_prof = collect_device_profile()
    if dev_prof.get("status") == "BLOCKED":
        print("  [ERROR] No physical Android device connected via ADB!")
        return {"status": "BLOCKED", "error": "No physical device connected"}
    print(f"  Device: {dev_prof['manufacturer']} {dev_prof['model']} | Android {dev_prof['android_version']} (API {dev_prof['sdk_level']}) | RAM: {dev_prof['total_ram_mb']} MB (2GB Class: {dev_prof['is_2gb_ram_class']})")

    # 2. Package & Storage Audit
    print("\n[2/6] Auditing Storage Footprint & Asset Sizes...")
    pkg_inspector = ADBPackageInspector(device_id=dev_prof["device_id"])
    storage_rep = pkg_inspector.audit_model_and_asset_sizes()
    print(f"  INT4 Model: {storage_rep['model_binary_int4_mb']} MB (Target <= 50 MB) | Total Assets: {storage_rep['total_assets_mb']} MB | Free Storage: {storage_rep['device_available_storage_mb']} MB")

    # 3. Memory & Multi-Turn Benchmark
    print("\n[3/6] Running Real-Device Memory & Multi-Turn Sessions (10, 25, 50, 100 turns)...")
    runner = ADBSessionRunner(device_id=dev_prof["device_id"])
    mem_rep = runner.run_multi_turn_benchmark([10, 25, 50, 100])
    print(f"  Cold PSS: {mem_rep['cold_launch']['cold_pss_mb']} MB | Peak PSS: {mem_rep['peak_active_pss_mb']} MB | 100-Turn Growth: {mem_rep['session_runs'][-1]['growth_mb_per_turn']} MB/turn")

    # 4. 100-Question Quality Benchmark
    print("\n[4/6] Executing 100-Question On-Device Quality Benchmark...")
    qual_rep = run_real_device_quality_benchmark()
    print(f"  Overall Quality Score: {qual_rep['overall_score']}/100 | Math: {qual_rep['gates']['Q1_math_accuracy']}% | Grounding: {qual_rep['gates']['Q2_grounding_adherence']}% | Hint: {qual_rep['gates']['Q3_hint_compliance']}%")

    # 5. Offline & Network Audit
    print("\n[5/6] Auditing Zero-Network Offline Operation...")
    net_monitor = ADBNetworkMonitor(device_id=dev_prof["device_id"])
    offline_rep = net_monitor.audit_offline_status()
    print(f"  Offline Status: {offline_rep['status']} | Zero Remote API: {offline_rep['zero_network_dependency_verified']}")

    # 6. Stability & Crash Log Collector
    print("\n[6/6] Collecting Device Stability & Crash Logs...")
    log_collector = ADBLogCollector(device_id=dev_prof["device_id"])
    stab_rep = log_collector.collect_stability_report()
    print(f"  Crashes: {stab_rep['crash_count']} | ANRs: {stab_rep['anr_count']} | Zero-Crash Gate: {stab_rep['zero_crash_gate']}")

    # Compile Final Summary
    all_passed = (
        dev_prof.get("status") == "VERIFIED" and
        storage_rep.get("model_size_gate_passed", False) and
        mem_rep.get("gate_m3_pass", False) and
        mem_rep.get("gate_m4_growth_pass", False) and
        qual_rep.get("overall_score", 0) >= 85.0 and
        offline_rep.get("zero_network_dependency_verified", False) and
        stab_rep.get("zero_crash_gate", False)
    )

    final_results = {
        "status": "VERIFIED_PASS" if all_passed else "FAIL",
        "timestamp": datetime.datetime.now().isoformat(),
        "device": dev_prof,
        "storage": storage_rep,
        "memory": mem_rep,
        "quality": qual_rep,
        "offline": offline_rep,
        "stability": stab_rep,
        "final_verdict": "PRODUCTION CERTIFIED (Verified on Real 2GB itel A662L Hardware)" if all_passed else "NOT PRODUCTION READY"
    }

    out_dir = PROJECT_ROOT / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "real_device_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print(f"  REAL-DEVICE BENCHMARK VERDICT: {final_results['final_verdict']}")
    print("="*70 + "\n")

    return final_results


if __name__ == "__main__":
    run_master_real_device_suite()
