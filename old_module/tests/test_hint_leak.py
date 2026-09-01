"""SS Tutor BD — Unit Tests: Hint Answer-Leak Detector (Phase 3C)"""
import sys
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def contains_exact_numeric_answer(response: str, exact_answer: str) -> bool:
    """
    Detects whether the LLM's hint response leaks the exact final numeric answer.
    Checks for the exact digit sequence (Bengali or ASCII) appearing prominently.
    """
    bn_digits = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
                 "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}
    def normalize(s: str) -> str:
        for b, a in bn_digits.items():
            s = s.replace(b, a)
        return s

    norm_answer = normalize(exact_answer.strip())
    norm_response = normalize(response)
    # Check for exact number match surrounded by word boundaries
    pattern = rf"(?<!\d){re.escape(norm_answer)}(?!\d)"
    return bool(re.search(pattern, norm_response))


def test_hint_no_leak_passes():
    hint = "প্রথমে ল.সা.গু বের করো, তারপর হর সমান করো।"
    exact = "১৯"
    assert not contains_exact_numeric_answer(hint, exact)
    print("test_hint_no_leak_passes: PASSED")

def test_hint_with_leak_detected():
    hint = "উত্তর হলো ১৯/১২।"
    exact = "১৯"
    assert contains_exact_numeric_answer(hint, exact)
    print("test_hint_with_leak_detected: PASSED")

def test_hint_partial_number_not_flagged():
    hint = "১৯৮ টাকার মধ্যে কিছু অংশ সুদ।"
    exact = "১৯"
    # 198 contains 19 but as part of a larger number — should NOT be flagged
    assert not contains_exact_numeric_answer(hint, exact)
    print("test_hint_partial_number_not_flagged: PASSED")

def test_exact_answer_5050_detected():
    hint = "সব সংখ্যা যোগ করলে ৫০৫০ পাওয়া যায়।"
    exact = "৫০৫০"
    assert contains_exact_numeric_answer(hint, exact)
    print("test_exact_answer_5050_detected: PASSED")

def test_indirect_hint_not_flagged():
    hint = "গাউসের পদ্ধতিতে প্রথম ও শেষ পদ যোগ করে দেখো।"
    exact = "৫০৫০"
    assert not contains_exact_numeric_answer(hint, exact)
    print("test_indirect_hint_not_flagged: PASSED")

def test_ascii_answer_detected():
    hint = "The answer is 1500 taka."
    exact = "1500"
    assert contains_exact_numeric_answer(hint, exact)
    print("test_ascii_answer_detected: PASSED")

def run_all():
    print("\n--- Hint Leak Detection Tests ---")
    test_hint_no_leak_passes()
    test_hint_with_leak_detected()
    test_hint_partial_number_not_flagged()
    test_exact_answer_5050_detected()
    test_indirect_hint_not_flagged()
    test_ascii_answer_detected()
    print("--- All Hint Leak Tests PASSED (6 / 6) ---\n")

if __name__ == "__main__":
    run_all()
