"""
SS Tutor BD - Micro-Prompt Protocol (Phase 3C)
Ultra-compact structured prompt builder using single-char delimiter tags.
Target overhead: < 70 tokens per prompt invocation.
"""

from typing import Optional

# Compact delimiter tags (single character, no English bloat)
TAG_TASK = "[T]"
TAG_FACT = "[F]"
TAG_RESULT = "[R]"
TAG_GOAL = "[G]"
TAG_HINT = "[H]"
TAG_CONSTRAINT = "[C]"

# Task mode codes
MODE_EXPLAIN = "EXP"
MODE_SOLVE = "SLV"
MODE_HINT = "HNT"
MODE_CHECK = "CHK"
MODE_SUMMARIZE = "SUM"
MODE_PRACTICE = "PRC"
MODE_CORRECT = "COR"


def get_micro_system_prompt() -> str:
    """Minimal one-line system prompt in Bengali."""
    return "তুমি একজন বাংলাদেশ NCTB গণিত শিক্ষক। শুধু বাংলায় সংক্ষিপ্ত উত্তর দাও।"


def build_micro_prompt(
    mode: str,
    question: str,
    textbook_fact: Optional[str] = None,
    computed_result: Optional[str] = None,
    hint_only: bool = False,
    extra_constraint: Optional[str] = None
) -> str:
    """
    Builds an ultra-compact structured prompt using single-character tags.
    Targets < 70 tokens overhead (excluding question and facts).

    Tags:
      [T] = Task (mode code + class level)
      [F] = Textbook fact (brief)
      [R] = Pre-computed deterministic result
      [G] = Generation goal
      [H] = Hint policy (only for HINT mode)
      [C] = Constraint
    """
    parts = []

    # Task line
    mode_code = {
        "EXPLAIN": MODE_EXPLAIN,
        "SOLVE": MODE_SOLVE,
        "HINT": MODE_HINT,
        "CHECK": MODE_CHECK,
        "SUMMARIZE": MODE_SUMMARIZE,
        "PRACTICE": MODE_PRACTICE,
        "CORRECT": MODE_CORRECT,
    }.get(mode.upper(), MODE_EXPLAIN)

    parts.append(f"{TAG_TASK} {mode_code} cl=8 bn=1")

    # Textbook fact (compressed, no duplication)
    if textbook_fact and textbook_fact.strip():
        trunc_fact = " ".join(textbook_fact.split()[:35])
        parts.append(f"{TAG_FACT} {trunc_fact}")

    # Pre-computed deterministic result (crucial for math modes)
    if computed_result and computed_result.strip():
        parts.append(f"{TAG_RESULT} {computed_result}")

    # Generation goal
    if mode.upper() == "HINT":
        parts.append(f"{TAG_GOAL} ইঙ্গিত দাও, উত্তর বলো না।")
        parts.append(f"{TAG_HINT} উত্তর সরাসরি বলা নিষেধ।")
    elif mode.upper() == "SOLVE":
        parts.append(f"{TAG_GOAL} ধাপ বাংলায় ব্যাখ্যা করো।")
    elif mode.upper() == "EXPLAIN":
        parts.append(f"{TAG_GOAL} সহজ বাংলায় বোঝাও।")
    elif mode.upper() == "CHECK":
        parts.append(f"{TAG_GOAL} ছাত্রের উত্তর যাচাই করো।")
    elif mode.upper() == "CORRECT":
        parts.append(f"{TAG_GOAL} ভুল ব্যাখ্যা করো, সঠিক দেখাও।")
    elif mode.upper() == "PRACTICE":
        parts.append(f"{TAG_GOAL} একটি অনুশীলন প্রশ্ন তৈরি করো।")
    else:
        parts.append(f"{TAG_GOAL} সংক্ষিপ্ত বাংলায় উত্তর দাও।")

    # Constraints
    constraint_text = "বাংলায়। সংক্ষিপ্ত। পাঠ্যপুস্তকের বাইরে তথ্য দিও না।"
    if extra_constraint:
        constraint_text += f" {extra_constraint}"
    parts.append(f"{TAG_CONSTRAINT} {constraint_text}")

    # Question
    parts.append(f"প্রশ্ন: {question}")

    return "\n".join(parts)


def estimate_prompt_tokens(prompt: str) -> int:
    """Rough token estimate (4 chars/token for mixed Bengali-English text)."""
    return max(1, len(prompt) // 4)
