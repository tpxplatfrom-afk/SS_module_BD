"""
SS Tutor BD - Full Regression Test Runner (Phase 1 to Phase 4)
"""

import sys
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tests = [
    # Phase 3A regressions
    ("Math Engine",              "tests/test_math_engine.py"),
    ("Math Validator",           "tests/test_math_validator.py"),
    ("Compact Prompts",          "tests/test_compact_prompts.py"),
    ("Hybrid Tutor",             "tests/test_hybrid_tutor.py"),
    ("Sanitizer",                "tests/test_sanitizer.py"),
    ("RAG Pipeline",             "tests/test_rag.py"),
    # Phase 3C regressions
    ("Memory Budget",            "tests/test_memory_budget.py"),
    ("Device Profile",           "tests/test_device_profile.py"),
    ("Context Compressor",       "tests/test_context_compressor.py"),
    ("Micro Prompt Protocol",    "tests/test_micro_protocol.py"),
    ("Session Memory",           "tests/test_session_memory.py"),
    ("Hint Leak Detector",       "tests/test_hint_leak.py"),
    ("Repetition Detector",      "tests/test_repetition_detector.py"),
    ("Micro Runtime Adapter",    "tests/test_micro_runtime.py"),
    # Phase 4 new test suites
    ("Phase 4 Dedicated Tokenizer", "tests/test_tokenizer_phase4.py"),
    ("Phase 4 Validation Layer",    "tests/test_validation_layer.py"),
    ("Phase 4 Context Budget",      "tests/test_context_budget.py"),
]

print("\n" + "=" * 65)
print("  SS TUTOR BD — COMPLETE REGRESSION SUITE (Phases 1-4)")
print("=" * 65)

passed = 0
failed = 0
results = []

for label, script in tests:
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode == 0:
        passed += 1
        status = "PASSED"
        print(f"  [PASSED] {label}")
    else:
        failed += 1
        status = "FAILED"
        print(f"  [FAILED] {label}")
        print(f"    STDOUT: {result.stdout[-300:].strip()}")
        print(f"    STDERR: {result.stderr[-300:].strip()}")
    results.append({"test": label, "status": status})

print("-" * 65)
print(f"  RESULT:  {passed} PASSED  /  {failed} FAILED  /  {len(tests)} TOTAL")
print("=" * 65 + "\n")

sys.exit(0 if failed == 0 else 1)
