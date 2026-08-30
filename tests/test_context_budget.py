"""SS Tutor BD — Unit Tests: Phase 4 Context Budget Manager"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.runtime.context_budget import ContextBudgetManager


def test_truncation_bounds_word_count():
    long_text = "পাঠ্যপুস্তকের তথ্য " * 100
    truncated = ContextBudgetManager.truncate_text_to_token_budget(long_text, max_tokens=60)
    assert len(truncated.split()) <= 45, f"Too long: {len(truncated.split())} words"
    print("test_truncation_bounds_word_count: PASSED")


def test_empty_string_truncation():
    res = ContextBudgetManager.truncate_text_to_token_budget("", max_tokens=50)
    assert res == ""
    print("test_empty_string_truncation: PASSED")


def test_enforce_prompt_budget_structure():
    res = ContextBudgetManager.enforce_prompt_budget(
        task_code="SLV",
        question="৩/৪ + ৫/৬ এর যোগফল কত?",
        textbook_fact="ভগ্নাংশ যোগের নিয়ম: হর সমান করো।",
        computed_result="যোগফল = ১৯/১২"
    )
    assert res["task_code"] == "SLV"
    assert "৩/৪" in res["question"]
    assert res["max_context_tokens"] == 256
    print("test_enforce_prompt_budget_structure: PASSED")


def run_all():
    print("\n--- Phase 4 Context Budget Manager Tests ---")
    test_truncation_bounds_word_count()
    test_empty_string_truncation()
    test_enforce_prompt_budget_structure()
    print("--- All Context Budget Tests PASSED (3 / 3) ---\n")


if __name__ == "__main__":
    run_all()
