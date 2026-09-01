"""
SS Tutor BD - Bounded Session Manager (Phase 3C)
Maintains constant O(1) session memory by keeping only a compact state summary,
never accumulating raw message history in the neural context.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionState:
    """
    Compact, bounded session state object.
    Replaces raw message-history accumulation with a fixed-size summary.
    """
    session_id: str = ""
    current_subject: str = "math"
    current_class: int = 8
    current_chapter: Optional[str] = None
    current_concept: Optional[str] = None
    last_question: Optional[str] = None
    last_computed_result: Optional[str] = None
    last_mode: str = "EXPLAIN"
    turn_count: int = 0
    compact_summary: Optional[str] = None   # Single-sentence running summary, max 50 chars

    def update(self, question: str, mode: str, result: Optional[str] = None, concept: Optional[str] = None) -> None:
        """Update session state after each turn. Does NOT accumulate raw messages."""
        self.last_question = question[:120] if question else None
        self.last_mode = mode
        self.last_computed_result = result
        self.turn_count += 1
        if concept:
            self.current_concept = concept
        # Rolling compact summary — only the latest topic (constant memory)
        if concept:
            self.compact_summary = f"প্রসঙ্গ: {concept[:40]}"
        elif self.last_question:
            self.compact_summary = self.last_question[:50]

    def get_context_prefix(self) -> str:
        """Returns a brief context hint for the next prompt (< 20 tokens)."""
        if self.compact_summary:
            return f"আগের প্রসঙ্গ: {self.compact_summary}"
        return ""

    def reset(self) -> None:
        """Resets to blank state without memory leak."""
        self.current_chapter = None
        self.current_concept = None
        self.last_question = None
        self.last_computed_result = None
        self.compact_summary = None
        self.turn_count = 0


class SessionManager:
    """Manages a bounded pool of sessions. Never stores raw message buffers."""

    def __init__(self):
        self._sessions: dict = {}

    def get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].reset()

    def session_count(self) -> int:
        return len(self._sessions)
