"""
Unit Tests for SS Tutor BD Output Sanitizer
Verifies that control tokens, prompt echoes, and repetition loops are cleaned
without corrupting authentic Bengali unicode characters and mathematical formulas.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sanitization.cleaner import (
    strip_control_tokens,
    strip_prompt_echo,
    truncate_repetition_loops,
    sanitize_tutor_output
)


def test_control_token_stripping():
    sample = "ভগ্নাংশের যোগ </tool_call> করতে হবে <|im_start|>assistant এবং সঠিক উত্তর হলো ৫।"
    cleaned = strip_control_tokens(sample)
    assert "</tool_call>" not in cleaned
    assert "<|im_start|>" not in cleaned
    assert "ভগ্নাংশের যোগ" in cleaned
    assert "সঠিক উত্তর হলো ৫।" in cleaned
    print("test_control_token_stripping: PASSED")


def test_prompt_echo_stripping():
    prompt = "সালোকসংশ্লেষণ কাকে বলে?"
    model_out = "প্রশ্ন: সালোকসংশ্লেষণ কাকে বলে?\nসালোকসংশ্লেষণ হলো উদ্ভিদের খাদ্য তৈরির জৈব রাসায়নিক প্রক্রিয়া।"
    cleaned = strip_prompt_echo(model_out, prompt)
    assert not cleaned.startswith("প্রশ্ন:")
    assert "সালোকসংশ্লেষণ হলো উদ্ভিদের খাদ্য তৈরির" in cleaned
    print("test_prompt_echo_stripping: PASSED")


def test_repetition_loop_truncation():
    looping_text = (
        "ধাপ ১: প্রথমে ৩ এবং ৪ যোগ করো।\n"
        "প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য"
    )
    cleaned, had_loop = truncate_repetition_loops(looping_text)
    assert had_loop is True
    assert "ধাপ ১: প্রথমে ৩ এবং ৪ যোগ করো।" in cleaned
    assert cleaned.count("প্রতিটি ধাপ বাংলা/csv দেখার জন্য") <= 2
    print("test_repetition_loop_truncation: PASSED")


def test_bengali_character_preservation():
    complex_bn = "বিজ্ঞান, সূক্ষ্ম, বণ্টন, উচ্ছ্বাস, কৃষ্ণগহ্বর, ড়, ঢ়, য়, ং, ঃ, ঁ, (a+b)^2 = a^2+2ab+b^2।"
    res = sanitize_tutor_output(complex_bn)
    assert res["cleaned_text"] == complex_bn
    assert "কৃষ্ণগহ্বর" in res["cleaned_text"]
    assert "সূক্ষ্ম" in res["cleaned_text"]
    print("test_bengali_character_preservation: PASSED")


def run_all_sanitizer_tests():
    print("\n--- Running Sanitizer Unit Tests ---")
    test_control_token_stripping()
    test_prompt_echo_stripping()
    test_repetition_loop_truncation()
    test_bengali_character_preservation()
    print("--- All Sanitizer Tests PASSED ---\n")


if __name__ == "__main__":
    run_all_sanitizer_tests()
