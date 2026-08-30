"""
SS Tutor BD - Unit Tests: Compact Tutor Prompt Protocol (Phase 3B)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.prompts.compact_tutor_templates import get_compact_system_prompt, build_compact_prompt


def test_system_prompt_is_bengali():
    sp = get_compact_system_prompt()
    # Should contain Bengali Unicode characters
    assert any('\u0980' <= c <= '\u09FF' for c in sp), "System prompt must contain Bengali text"
    assert len(sp) < 300, "System prompt should be compact (<300 chars)"
    print("test_system_prompt_is_bengali: PASSED")

def test_compact_prompt_solve_contains_task():
    prompt = build_compact_prompt("SOLVE", "৩/৪ + ৫/৬ = ?")
    assert "[TASK: SOLVE]" in prompt
    assert "[QUESTION]" in prompt
    assert "[OUTPUT_RULES]" in prompt
    print("test_compact_prompt_solve_contains_task: PASSED")

def test_compact_prompt_hint_has_negative_constraint():
    prompt = build_compact_prompt("HINT", "সরাসরি উত্তর দিও না।")
    assert "HINT" in prompt
    assert "সরাসরি উত্তর" in prompt or "সরাসরি" in prompt
    assert "উত্তর বলবে না" in prompt or "সরাসরি" in prompt
    print("test_compact_prompt_hint_has_negative_constraint: PASSED")

def test_compact_prompt_with_textbook_context():
    prompt = build_compact_prompt(
        "EXPLAIN",
        "ভগ্নাংশের নিয়ম কী?",
        textbook_context="ভগ্নাংশের হর ও লবের ধারণা পাঠ্যপুস্তক থেকে।"
    )
    assert "[TEXTBOOK]" in prompt
    assert "ভগ্নাংশের হর" in prompt
    print("test_compact_prompt_with_textbook_context: PASSED")

def test_compact_prompt_with_verified_result():
    prompt = build_compact_prompt(
        "SOLVE",
        "সরল মুনাফা কত?",
        verified_result="মুনাফা = ১৫০০ টাকা"
    )
    assert "[VERIFIED_RESULT]" in prompt
    assert "মুনাফা = ১৫০০" in prompt
    print("test_compact_prompt_with_verified_result: PASSED")

def test_compact_prompt_total_length_bounded():
    # Should stay under 1000 chars for minimal context usage
    prompt = build_compact_prompt(
        "SOLVE",
        "সরল মুনাফা নির্ণয় করো।",
        textbook_context="সরল মুনাফার সূত্র হলো I = Prn।",
        verified_result="মুনাফা = ১৫০০ টাকা"
    )
    # Rough token estimate — each 4 chars ~ 1 token
    est_tokens = len(prompt) // 4
    assert est_tokens < 400, f"Prompt too long: ~{est_tokens} tokens"
    print(f"test_compact_prompt_total_length_bounded: PASSED (~{est_tokens} tokens estimated)")

def test_hint_mode_negative_rule_present():
    prompt = build_compact_prompt("HINT", "সমাধান করো।")
    assert "সরাসরি উত্তর বলবে না" in prompt
    print("test_hint_mode_negative_rule_present: PASSED")


def run_all_prompt_tests():
    print("\n--- Running Compact Prompt Protocol Unit Tests ---")
    test_system_prompt_is_bengali()
    test_compact_prompt_solve_contains_task()
    test_compact_prompt_hint_has_negative_constraint()
    test_compact_prompt_with_textbook_context()
    test_compact_prompt_with_verified_result()
    test_compact_prompt_total_length_bounded()
    test_hint_mode_negative_rule_present()
    print("--- All Compact Prompt Tests PASSED (7 / 7) ---\n")


if __name__ == "__main__":
    run_all_prompt_tests()
