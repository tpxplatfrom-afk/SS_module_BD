"""SS Tutor BD — Unit Tests: Phase 4 Multi-Guard Validation Layer"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.validation.grounding_validator import GroundingValidator
from core.validation.math_answer_validator import MathAnswerValidator
from core.validation.hint_validator import HintValidator
from core.validation.language_validator import LanguageValidator
from core.validation.format_validator import FormatValidator


def test_grounding_supported_passes():
    res = GroundingValidator.validate_grounding("সরল মুনাফার সূত্র হলো I = Prn।", "সরল মুনাফা I = Prn", is_unsupported_query=False)
    assert res["is_valid"] == True
    print("test_grounding_supported_passes: PASSED")


def test_grounding_unsupported_refusal_passes():
    res = GroundingValidator.validate_grounding("প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না।", "বৃত্তের ক্ষেত্রফল πr²", is_unsupported_query=True)
    assert res["is_valid"] == True
    print("test_grounding_unsupported_refusal_passes: PASSED")


def test_grounding_unsupported_hallucination_fails():
    res = GroundingValidator.validate_grounding("তিনি ১৯১৩ সালে নোবেল পুরস্কার পান।", "বৃত্তের ক্ষেত্রফল πr²", is_unsupported_query=True)
    assert res["is_valid"] == False
    print("test_grounding_unsupported_hallucination_fails: PASSED")


def test_math_validator_consistent_passes():
    res = MathAnswerValidator.validate_and_correct("যোগফল হলো ১৯/১২।", "১৯/১২")
    assert res["is_valid"] == True
    print("test_math_validator_consistent_passes: PASSED")


def test_math_validator_conflict_corrected():
    res = MathAnswerValidator.validate_and_correct("ভুল উত্তর।", "১৯/১২", steps=["ধাপ ১: লসাগু ১২"])
    assert res["was_corrected"] == True
    assert "১৯/১২" in res["final_text"]
    print("test_math_validator_conflict_corrected: PASSED")


def test_hint_validator_no_leak_passes():
    res = HintValidator.validate_hint_compliance("প্রথমে হরদ্বয়ের লসাগু বের করো।", "১৯/১২")
    assert res["is_valid"] == True
    print("test_hint_validator_no_leak_passes: PASSED")


def test_hint_validator_leak_sanitized():
    res = HintValidator.validate_hint_compliance("উত্তর হলো ১৯/১২।", "১৯/১২")
    assert res["leaked"] == True
    assert "১৯/১২" not in res["final_text"]
    print("test_hint_validator_leak_sanitized: PASSED")


def test_language_validator_clean_bengali_passes():
    res = LanguageValidator.validate_language("সরল মুনাফার ক্ষেত্রে মুনাফা = আসল × হার × সময়।")
    assert res["is_valid"] == True
    print("test_language_validator_clean_bengali_passes: PASSED")


def test_language_validator_repetition_fails():
    res = LanguageValidator.validate_language("যোগফল কত যোগফল কত যোগফল কত যোগফল কত যোগফল কত")
    assert res["has_repetition_loop"] == True
    print("test_language_validator_repetition_fails: PASSED")


def test_format_validator_strips_tags():
    res = FormatValidator.clean_output_format("[T] EXP\n[F] সূত্র\n<|im_start|>সঠিক উত্তর।<|im_end|>")
    assert "[T]" not in res["cleaned_text"]
    assert "<|im_start|>" not in res["cleaned_text"]
    assert "সঠিক উত্তর।" in res["cleaned_text"]
    print("test_format_validator_strips_tags: PASSED")


def run_all():
    print("\n--- Phase 4 Multi-Guard Validation Layer Tests ---")
    test_grounding_supported_passes()
    test_grounding_unsupported_refusal_passes()
    test_grounding_unsupported_hallucination_fails()
    test_math_validator_consistent_passes()
    test_math_validator_conflict_corrected()
    test_hint_validator_no_leak_passes()
    test_hint_validator_leak_sanitized()
    test_language_validator_clean_bengali_passes()
    test_language_validator_repetition_fails()
    test_format_validator_strips_tags()
    print("--- All Validation Layer Tests PASSED (10 / 10) ---\n")


if __name__ == "__main__":
    run_all()
