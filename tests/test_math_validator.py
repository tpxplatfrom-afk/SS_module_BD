"""
SS Tutor BD - Unit Tests: Math Validator (Phase 3B)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.math.validator import MathValidator
from core.math.fraction import FractionHelper


def test_validator_non_math_pass():
    res = MathValidator.validate_and_correct(
        "পিথাগোরাস কী?",
        "পিথাগোরাস হল একজন গ্রিক গণিতবিদ।",
        deterministic_result=None
    )
    assert res["is_math_task"] == False
    assert res["is_valid"] == True
    assert res["corrected"] == False
    print("test_validator_non_math_pass: PASSED")

def test_validator_good_llm_output_passes():
    det_res = FractionHelper.add((3, 4), (5, 6))
    det_res["intent"] = "fraction_addition"
    det_res["final_answer"] = det_res["final_answer_bengali"]
    det_res["steps"] = det_res["steps_bengali"]
    res = MathValidator.validate_and_correct(
        "৩/৪ + ৫/৬",
        "সমাধান:\nধাপ ১: ল.সা.গু = ১২\nধাপ ২: সমান হর করে যোগ করলে ১৯/১২ = ১ সমস্ত ৭/১২।",
        deterministic_result=det_res
    )
    assert res["is_math_task"] == True
    assert res["corrected"] == False
    print("test_validator_good_llm_output_passes: PASSED")

def test_validator_corrupt_output_corrected():
    det_res = FractionHelper.add((3, 4), (5, 6))
    det_res["intent"] = "fraction_addition"
    det_res["final_answer"] = det_res["final_answer_bengali"]
    det_res["steps"] = det_res["steps_bengali"]
    res = MathValidator.validate_and_correct(
        "৩/৪ + ৫/৬",
        "কোনো কোনো কোনো",
        deterministic_result=det_res
    )
    assert res["is_math_task"] == True
    assert res["corrected"] == True
    assert len(res["verified_text"]) > 30
    print("test_validator_corrupt_output_corrected: PASSED")

def test_validator_short_output_corrected():
    det_res = {"intent": "simple_interest", "final_answer": "১৫০০ টাকা", "steps": ["I = 5000 x 0.1 x 3 = 1500"]}
    res = MathValidator.validate_and_correct(
        "সরল মুনাফা কত?",
        "মুনাফা",
        deterministic_result=det_res
    )
    assert res["corrected"] == True
    print("test_validator_short_output_corrected: PASSED")


def run_all_validator_tests():
    print("\n--- Running Math Validator Unit Tests ---")
    test_validator_non_math_pass()
    test_validator_good_llm_output_passes()
    test_validator_corrupt_output_corrected()
    test_validator_short_output_corrected()
    print("--- All Math Validator Tests PASSED (4 / 4) ---\n")


if __name__ == "__main__":
    run_all_validator_tests()
