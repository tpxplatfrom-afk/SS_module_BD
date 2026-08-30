"""SS Tutor BD — Unit Tests: Device Profiler (Phase 3C)"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.runtime.device_profile import DeviceProfiler


def test_hardware_info_returns_dict():
    info = DeviceProfiler.get_hardware_info()
    assert "total_ram_mb" in info
    assert info["total_ram_mb"] > 0
    assert "cpu_cores_logical" in info
    print(f"test_hardware_info_returns_dict: PASSED (RAM={info['total_ram_mb']} MB)")

def test_ultra_low_tier_at_500mb():
    tier = DeviceProfiler.determine_device_tier(override_ram_mb=500.0)
    assert tier == DeviceProfiler.TIER_ULTRA_LOW
    print("test_ultra_low_tier_at_500mb: PASSED")

def test_low_tier_at_1500mb():
    tier = DeviceProfiler.determine_device_tier(override_ram_mb=1500.0)
    assert tier == DeviceProfiler.TIER_LOW
    print("test_low_tier_at_1500mb: PASSED")

def test_standard_tier_at_4000mb():
    tier = DeviceProfiler.determine_device_tier(override_ram_mb=4000.0)
    assert tier == DeviceProfiler.TIER_STANDARD
    print("test_standard_tier_at_4000mb: PASSED")

def test_ultra_low_policy_disables_neural():
    policy = DeviceProfiler.get_adaptive_policy(DeviceProfiler.TIER_ULTRA_LOW)
    assert policy["neural_generation_enabled"] == False
    assert policy["max_context_tokens"] <= 256
    print("test_ultra_low_policy_disables_neural: PASSED")

def test_low_tier_policy_enables_neural():
    policy = DeviceProfiler.get_adaptive_policy(DeviceProfiler.TIER_LOW)
    assert policy["neural_generation_enabled"] == True
    assert policy["max_context_tokens"] <= 384
    print("test_low_tier_policy_enables_neural: PASSED")

def test_standard_policy_largest_context():
    policy = DeviceProfiler.get_adaptive_policy(DeviceProfiler.TIER_STANDARD)
    assert policy["max_context_tokens"] >= 384
    print("test_standard_policy_largest_context: PASSED")

def run_all():
    print("\n--- Device Profiler Tests ---")
    test_hardware_info_returns_dict()
    test_ultra_low_tier_at_500mb()
    test_low_tier_at_1500mb()
    test_standard_tier_at_4000mb()
    test_ultra_low_policy_disables_neural()
    test_low_tier_policy_enables_neural()
    test_standard_policy_largest_context()
    print("--- All Device Profiler Tests PASSED (7 / 7) ---\n")

if __name__ == "__main__":
    run_all()
