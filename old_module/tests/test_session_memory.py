"""SS Tutor BD — Unit Tests: Session Manager (Phase 3C)"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.runtime.session_manager import SessionManager, SessionState


def test_session_creates_fresh():
    mgr = SessionManager()
    s = mgr.get_or_create("test-001")
    assert s.session_id == "test-001"
    assert s.turn_count == 0
    print("test_session_creates_fresh: PASSED")

def test_session_reuses_same_object():
    mgr = SessionManager()
    s1 = mgr.get_or_create("test-001")
    s2 = mgr.get_or_create("test-001")
    assert s1 is s2
    print("test_session_reuses_same_object: PASSED")

def test_turn_count_increments():
    mgr = SessionManager()
    s = mgr.get_or_create("test-002")
    for i in range(5):
        s.update("প্রশ্ন " + str(i), "EXPLAIN")
    assert s.turn_count == 5
    print("test_turn_count_increments: PASSED")

def test_last_question_bounded():
    mgr = SessionManager()
    s = mgr.get_or_create("test-003")
    long_q = "ক" * 500
    s.update(long_q, "EXPLAIN")
    assert s.last_question is not None
    assert len(s.last_question) <= 120
    print("test_last_question_bounded: PASSED")

def test_context_prefix_short():
    mgr = SessionManager()
    s = mgr.get_or_create("test-004")
    s.update("ভগ্নাংশ কী?", "EXPLAIN", concept="ভগ্নাংশ")
    prefix = s.get_context_prefix()
    assert len(prefix) < 60, f"Prefix too long: {prefix}"
    print("test_context_prefix_short: PASSED")

def test_reset_clears_state():
    mgr = SessionManager()
    s = mgr.get_or_create("test-005")
    s.update("প্রশ্ন", "HINT")
    mgr.clear("test-005")
    s2 = mgr.get_or_create("test-005")
    assert s2.turn_count == 0
    assert s2.last_question is None
    print("test_reset_clears_state: PASSED")

def test_no_raw_message_accumulation():
    """Verify session doesn't store a growing list of raw messages."""
    mgr = SessionManager()
    s = mgr.get_or_create("test-006")
    for i in range(100):
        s.update(f"প্রশ্ন {i}", "EXPLAIN")
    assert not hasattr(s, "message_history") or s.message_history is None
    assert s.turn_count == 100
    print("test_no_raw_message_accumulation: PASSED (100 turns, no history list)")

def run_all():
    print("\n--- Session Memory Tests ---")
    test_session_creates_fresh()
    test_session_reuses_same_object()
    test_turn_count_increments()
    test_last_question_bounded()
    test_context_prefix_short()
    test_reset_clears_state()
    test_no_raw_message_accumulation()
    print("--- All Session Memory Tests PASSED (7 / 7) ---\n")

if __name__ == "__main__":
    run_all()
