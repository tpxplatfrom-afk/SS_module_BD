"""
SS Tutor BD - Compact Tutor Prompt Protocol
Defines ultra-low-token structured prompt templates with minimal token overhead for low-RAM micro-models.
Supports EXPLAIN, SOLVE, HINT, VERIFY, and CORRECT modes.
"""

from typing import Optional, Dict, Any


def get_compact_system_prompt() -> str:
    """Ultra-compact system prompt minimizing KV cache footprint."""
    return "তুমি বাংলাদেশ NCTB ৮ম শ্রেণির গণিত শিক্ষক। পাঠ্যবইয়ের তথ্য ও গণনার ভিত্তিতে বাংলায় সহজ ও সঠিক উত্তর দাও।"


def build_compact_prompt(
    task_mode: str,
    user_query: str,
    textbook_context: Optional[str] = None,
    verified_result: Optional[str] = None,
    student_level: str = "Class 8"
) -> str:
    """
    Constructs a structured prompt using concise delimiters.
    task_mode: 'EXPLAIN', 'SOLVE', 'HINT', 'VERIFY', 'CORRECT'
    """
    parts = [f"[TASK: {task_mode.upper()}]"]

    if textbook_context and textbook_context.strip():
        # Clean and limit textbook context length
        parts.append(f"[TEXTBOOK]\n{textbook_context.strip()}")

    if verified_result and verified_result.strip():
        parts.append(f"[VERIFIED_RESULT]\n{verified_result.strip()}")

    parts.append(f"[QUESTION]\n{user_query.strip()}")

    # Specialized compact output rule per mode
    if task_mode.upper() == "HINT":
        parts.append("[OUTPUT_RULES]\nসরাসরি উত্তর বলবে না। শুধু প্রথম ধাপের সূত্র বা নিয়মটি ইঙ্গিত দাও।")
    elif task_mode.upper() == "SOLVE":
        parts.append("[OUTPUT_RULES]\nধাপ ১, ধাপ ২ আকারে স্পষ্ট বাংলায় সমাধান দেখাও।")
    elif task_mode.upper() == "EXPLAIN":
        parts.append("[OUTPUT_RULES]\nসহজ ভাষায় মূল ধারণাটি বুঝিয়ে বলো।")
    elif task_mode.upper() == "CORRECT":
        parts.append("[OUTPUT_RULES]\nভুলটি সংশোধন করে সঠিক নিয়ম বুঝিয়ে দাও।")
    else:
        parts.append("[OUTPUT_RULES]\nসংক্ষেপে স্পষ্ট বাংলায় উত্তর দাও।")

    return "\n\n".join(parts)
