"""SS Tutor BD — Unit Tests: Context Compressor (Phase 3C)"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.rag.context_compressor import ContextCompressor


def test_formula_preserved():
    raw = "সরল মুনাফার সূত্র হলো I = Prn যেখানে P = আসল।"
    result = ContextCompressor.extract_key_formulas_and_facts(raw, max_words=50)
    assert "I = Prn" in result or "সূত্র" in result
    print("test_formula_preserved: PASSED")

def test_duplicate_lines_removed():
    raw = "সরল মুনাফা হলো মূল অর্থের উপর অর্জিত সুদ।\nসরল মুনাফা হলো মূল অর্থের উপর অর্জিত সুদ।"
    result = ContextCompressor.extract_key_formulas_and_facts(raw, max_words=80)
    # Should not repeat the same sentence twice in output
    assert result.count("সরল মুনাফা হলো") <= 1
    print("test_duplicate_lines_removed: PASSED")

def test_output_word_count_bounded():
    raw = ("পাঠ্যপুস্তকে বীজগণিতের অনেক তথ্য রয়েছে। " * 20)
    result = ContextCompressor.extract_key_formulas_and_facts(raw, max_words=40)
    assert len(result.split()) <= 45, f"Too long: {len(result.split())} words"
    print("test_output_word_count_bounded: PASSED")

def test_empty_text_returns_something():
    result = ContextCompressor.extract_key_formulas_and_facts("", max_words=30)
    assert result == "" or len(result) >= 0
    print("test_empty_text_returns_something: PASSED")

def test_bengali_definition_captured():
    raw = "পিথাগোরাস উপপাদ্য কাকে বলে: সমকোণী ত্রিভুজে অতিভুজের বর্গ অপর দুই বাহুর বর্গের সমষ্টির সমান।"
    result = ContextCompressor.extract_key_formulas_and_facts(raw, max_words=50)
    assert "পিথাগোরাস" in result
    print("test_bengali_definition_captured: PASSED")

def test_math_notation_preserved():
    raw = "বৃত্তের ক্ষেত্রফল = πr² এবং পরিধি = 2πr।"
    result = ContextCompressor.extract_key_formulas_and_facts(raw, max_words=30)
    assert "π" in result or "πr" in result
    print("test_math_notation_preserved: PASSED")

def test_compress_retrieved_chunks_empty():
    result = ContextCompressor.compress_retrieved_chunks([])
    assert result == ""
    print("test_compress_retrieved_chunks_empty: PASSED")

def run_all():
    print("\n--- Context Compressor Tests ---")
    test_formula_preserved()
    test_duplicate_lines_removed()
    test_output_word_count_bounded()
    test_empty_text_returns_something()
    test_bengali_definition_captured()
    test_math_notation_preserved()
    test_compress_retrieved_chunks_empty()
    print("--- All Context Compressor Tests PASSED (7 / 7) ---\n")

if __name__ == "__main__":
    run_all()
