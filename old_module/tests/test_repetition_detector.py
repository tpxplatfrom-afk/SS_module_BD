"""SS Tutor BD — Unit Tests: Repetition Detector (Phase 3C)"""
import sys
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def detect_bengali_repetition(text: str, min_phrase_len: int = 4, max_repeats: int = 2) -> bool:
    """Detects repetitive loop patterns in Bengali or mixed-language output."""
    if not text or len(text) < 20:
        return False

    # 1. Word n-gram repetition check
    words = text.split()
    for n in [3, 4, 5]:
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        seen = {}
        for ng in ngrams:
            seen[ng] = seen.get(ng, 0) + 1
            if seen[ng] > max_repeats:
                return True

    # 2. Repeated sentence / clause check
    sentences = re.split(r"[।!?\n]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_phrase_len]
    seen_sent = {}
    for s in sentences:
        seen_sent[s] = seen_sent.get(s, 0) + 1
        if seen_sent[s] > max_repeats:
            return True

    return False


def test_clean_text_not_flagged():
    text = "সরল মুনাফার সূত্র হলো I = Prn। এখানে P হলো আসল টাকা এবং r হলো সুদের হার।"
    assert not detect_bengali_repetition(text)
    print("test_clean_text_not_flagged: PASSED")

def test_repetitive_phrase_detected():
    text = "বাংলা বুঝিয়ে বলো বাংলা বুঝিয়ে বলো বাংলা বুঝিয়ে বলো বাংলা বুঝিয়ে বলো"
    assert detect_bengali_repetition(text)
    print("test_repetitive_phrase_detected: PASSED")

def test_repeated_sentence_detected():
    text = "সরল মুনাফা হলো মূল অর্থের উপর সুদ।\nসরল মুনাফা হলো মূল অর্থের উপর সুদ।\nসরল মুনাফা হলো মূল অর্থের উপর সুদ।"
    assert detect_bengali_repetition(text, max_repeats=1)
    print("test_repeated_sentence_detected: PASSED")

def test_normal_tutoring_response_passes():
    text = "প্রথমে ল.সা.গু নির্ণয় করো। তারপর ভগ্নাংশের হর সমান করো। এরপর লব যোগ করো। এভাবে সঠিক উত্তর পাওয়া যাবে।"
    assert not detect_bengali_repetition(text)
    print("test_normal_tutoring_response_passes: PASSED")

def test_english_repetition_detected():
    text = "the answer is the answer is the answer is the answer"
    assert detect_bengali_repetition(text)
    print("test_english_repetition_detected: PASSED")

def test_short_text_not_flagged():
    text = "হ্যাঁ"
    assert not detect_bengali_repetition(text)
    print("test_short_text_not_flagged: PASSED")

def run_all():
    print("\n--- Repetition Detector Tests ---")
    test_clean_text_not_flagged()
    test_repetitive_phrase_detected()
    test_repeated_sentence_detected()
    test_normal_tutoring_response_passes()
    test_english_repetition_detected()
    test_short_text_not_flagged()
    print("--- All Repetition Detector Tests PASSED (6 / 6) ---\n")

if __name__ == "__main__":
    run_all()
