"""SS Tutor BD — Unit Tests: Micro Prompt Protocol (Phase 3C)"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.prompts.micro_protocol import (
    build_micro_prompt, get_micro_system_prompt, estimate_prompt_tokens,
    TAG_TASK, TAG_FACT, TAG_RESULT, TAG_GOAL, TAG_HINT, TAG_CONSTRAINT
)


def test_system_prompt_is_short_bengali():
    sp = get_micro_system_prompt()
    assert any('\u0980' <= c <= '\u09FF' for c in sp), "Must contain Bengali"
    assert len(sp.split()) < 25, "System prompt must be compact"
    print("test_system_prompt_is_short_bengali: PASSED")

def test_solve_contains_task_tag():
    p = build_micro_prompt("SOLVE", "৩/৪ + ৫/৬ = ?")
    assert TAG_TASK in p
    assert "SLV" in p
    print("test_solve_contains_task_tag: PASSED")

def test_hint_contains_hint_tag():
    p = build_micro_prompt("HINT", "সমীকরণ সমাধান করো")
    assert TAG_HINT in p
    assert "উত্তর সরাসরি বলা নিষেধ" in p
    print("test_hint_contains_hint_tag: PASSED")

def test_fact_included_when_provided():
    p = build_micro_prompt("EXPLAIN", "মুনাফা কী?", textbook_fact="সরল মুনাফার সূত্র I = Prn।")
    assert TAG_FACT in p
    assert "I = Prn" in p
    print("test_fact_included_when_provided: PASSED")

def test_result_included_when_provided():
    p = build_micro_prompt("SOLVE", "সরল মুনাফা", computed_result="মুনাফা = ১৫০০ টাকা")
    assert TAG_RESULT in p
    assert "১৫০০" in p
    print("test_result_included_when_provided: PASSED")

def test_constraint_tag_present():
    p = build_micro_prompt("EXPLAIN", "বীজগণিত কী?")
    assert TAG_CONSTRAINT in p
    print("test_constraint_tag_present: PASSED")

def test_overhead_below_100_tokens():
    system = get_micro_system_prompt()
    prompt = build_micro_prompt("SOLVE", "x = ?")
    overhead = estimate_prompt_tokens(system + prompt)
    assert overhead < 100, f"Overhead too large: {overhead} tokens"
    print(f"test_overhead_below_100_tokens: PASSED (~{overhead} tokens)")

def test_no_tag_fact_when_no_fact():
    p = build_micro_prompt("EXPLAIN", "মৌলিক সংখ্যা কী?")
    assert TAG_FACT not in p
    print("test_no_tag_fact_when_no_fact: PASSED")

def run_all():
    print("\n--- Micro Prompt Protocol Tests ---")
    test_system_prompt_is_short_bengali()
    test_solve_contains_task_tag()
    test_hint_contains_hint_tag()
    test_fact_included_when_provided()
    test_result_included_when_provided()
    test_constraint_tag_present()
    test_overhead_below_100_tokens()
    test_no_tag_fact_when_no_fact()
    print("--- All Micro Protocol Tests PASSED (8 / 8) ---\n")

if __name__ == "__main__":
    run_all()
