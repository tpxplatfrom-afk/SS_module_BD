"""SS Tutor BD — Unit Tests: Memory Budget Manager (Phase 3C)"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.runtime.memory_budget import MemoryBudgetManager


def test_preferred_range_passes():
    v = MemoryBudgetManager.evaluate_rss(150.0)
    assert v["status"] == "PREFERRED"
    assert v["passed_production_ceiling"] == True
    print("test_preferred_range_passes: PASSED")

def test_acceptable_range_passes():
    v = MemoryBudgetManager.evaluate_rss(195.0)
    assert v["passed_production_ceiling"] == True
    assert "ACCEPTABLE" in v["status"]
    print("test_acceptable_range_passes: PASSED")

def test_warning_tier_fails():
    v = MemoryBudgetManager.evaluate_rss(210.0)
    assert v["passed_production_ceiling"] == False
    assert "WARNING" in v["status"]
    print("test_warning_tier_fails: PASSED")

def test_disqualified_fails():
    v = MemoryBudgetManager.evaluate_rss(260.0)
    assert v["passed_production_ceiling"] == False
    assert "DISQUALIFIED" in v["status"] or "FAIL" in v["status"]
    print("test_disqualified_fails: PASSED")

def test_budget_breakdown_sums_correctly():
    bd = MemoryBudgetManager.get_budget_breakdown()
    total = bd["total_budget_sum"]
    manual = (bd["model_weights"] + bd["runtime_engine"] + bd["kv_cache"] +
              bd["rag_storage"] + bd["compressor_buffers"] + bd["math_engine"] +
              bd["sanitizer_validator"] + bd["app_overhead"] + bd["safety_margin"])
    assert abs(total - manual) < 0.1, f"Budget mismatch: {total} != {manual}"
    print(f"test_budget_breakdown_sums_correctly: PASSED (total={total} MB)")

def test_get_current_rss_mb_is_positive():
    rss = MemoryBudgetManager.get_current_rss_mb()
    assert rss > 0, f"RSS should be > 0 MB, got {rss}"
    print(f"test_get_current_rss_mb_is_positive: PASSED ({rss} MB)")

def run_all():
    print("\n--- Memory Budget Tests ---")
    test_preferred_range_passes()
    test_acceptable_range_passes()
    test_warning_tier_fails()
    test_disqualified_fails()
    test_budget_breakdown_sums_correctly()
    test_get_current_rss_mb_is_positive()
    print("--- All Memory Budget Tests PASSED (6 / 6) ---\n")

if __name__ == "__main__":
    run_all()
