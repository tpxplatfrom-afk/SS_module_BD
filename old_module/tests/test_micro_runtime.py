"""SS Tutor BD — Unit Tests: Micro Runtime Adapter (Phase 3C)"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.runtime.micro_runtime import (
    MicroRuntimeFactory, DeterministicFallbackRuntime, MicroGenResult
)


def test_factory_creates_deterministic():
    rt = MicroRuntimeFactory.create("deterministic", "TEST")
    assert isinstance(rt, DeterministicFallbackRuntime)
    print("test_factory_creates_deterministic: PASSED")

def test_factory_unknown_raises():
    try:
        MicroRuntimeFactory.create("unknown_backend_xyz", "TEST")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("test_factory_unknown_raises: PASSED")

def test_deterministic_runtime_loads():
    rt = DeterministicFallbackRuntime()
    rt.load("")  # No model path needed
    print("test_deterministic_runtime_loads: PASSED")

def test_deterministic_runtime_generates():
    rt = DeterministicFallbackRuntime()
    rt.load("")
    result = rt.generate("sys", "user question")
    assert isinstance(result, MicroGenResult)
    assert len(result.text) > 0
    assert result.backend == "deterministic_template"
    print("test_deterministic_runtime_generates: PASSED")

def test_deterministic_runtime_uses_result_tag():
    rt = DeterministicFallbackRuntime()
    rt.load("")
    result = rt.generate("sys", "[R] মুনাফা = ১৫০০ টাকা")
    assert "১৫০০" in result.text
    print("test_deterministic_runtime_uses_result_tag: PASSED")

def test_deterministic_runtime_unloads():
    rt = DeterministicFallbackRuntime()
    rt.load("")
    rt.unload()
    # Should not raise
    print("test_deterministic_runtime_unloads: PASSED")

def test_deterministic_memory_usage_positive():
    rt = DeterministicFallbackRuntime()
    mem = rt.memory_usage_mb()
    assert mem > 0
    print(f"test_deterministic_memory_usage_positive: PASSED ({mem} MB)")

def test_micro_gen_result_dataclass():
    r = MicroGenResult(text="test", tokens_per_sec=10.0, backend="test")
    assert r.text == "test"
    assert r.tokens_per_sec == 10.0
    print("test_micro_gen_result_dataclass: PASSED")


def run_all():
    print("\n--- Micro Runtime Adapter Tests ---")
    test_factory_creates_deterministic()
    test_factory_unknown_raises()
    test_deterministic_runtime_loads()
    test_deterministic_runtime_generates()
    test_deterministic_runtime_uses_result_tag()
    test_deterministic_runtime_unloads()
    test_deterministic_memory_usage_positive()
    test_micro_gen_result_dataclass()
    print("--- All Micro Runtime Tests PASSED (8 / 8) ---\n")


if __name__ == "__main__":
    run_all()
