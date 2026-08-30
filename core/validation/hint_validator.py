"""
SS Tutor BD - Socratic Hint Validator (Phase 4)
Ensures that hint responses never reveal the direct final answer.
"""

import re
from typing import Dict, Any


class HintValidator:
    BENGALI_DIGITS = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
                      "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}

    @classmethod
    def to_ascii_digits(cls, text: str) -> str:
        res = text
        for b, a in cls.BENGALI_DIGITS.items():
            res = res.replace(b, a)
        return res

    @classmethod
    def validate_hint_compliance(cls, hint_text: str, exact_numeric_answer: str) -> Dict[str, Any]:
        """
        Checks if the exact numeric answer appears in the hint response.
        If leaked, rejects and replaces with a pure pedagogical hint.
        """
        if not exact_numeric_answer:
            return {"is_valid": True, "leaked": False, "final_text": hint_text}

        norm_ans = cls.to_ascii_digits(str(exact_numeric_answer).strip())
        norm_hint = cls.to_ascii_digits(hint_text)

        # Look for the exact number with word boundaries
        pattern = rf"(?<!\d){re.escape(norm_ans)}(?!\d)"
        leaked = bool(re.search(pattern, norm_hint))

        if leaked:
            sanitized_hint = "ইঙ্গিত: সমস্যাটির মূল সূত্র ও প্রয়োজনীয় ধাপগুলো চিন্তা করো। সরাসরি উত্তর খোঁজার আগে প্রতিটি পদ চিহ্নিত করার চেষ্টা করো।"
            return {
                "is_valid": False,
                "leaked": True,
                "final_text": sanitized_hint,
                "reason": "ANSWER_LEAKAGE_DETECTED"
            }

        return {
            "is_valid": True,
            "leaked": False,
            "final_text": hint_text,
            "reason": "SOCRATIC_HINT_SECURE"
        }
