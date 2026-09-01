"""
SS Tutor BD - Language & Repetition Validator (Phase 4)
Validates Bengali linguistic fluency and eliminates degenerative repetition loops.
"""

import re
from typing import Dict, Any


class LanguageValidator:
    @staticmethod
    def contains_bengali(text: str) -> bool:
        return any('\u0980' <= c <= '\u09FF' for c in text)

    @staticmethod
    def detect_repetition_loops(text: str, max_repeats: int = 2) -> bool:
        if not text or len(text) < 20:
            return False
        words = text.split()
        for n in [3, 4, 5]:
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            seen = {}
            for ng in ngrams:
                seen[ng] = seen.get(ng, 0) + 1
                if seen[ng] > max_repeats:
                    return True
        return False

    @classmethod
    def validate_language(cls, text: str) -> Dict[str, Any]:
        has_bn = cls.contains_bengali(text)
        has_loop = cls.detect_repetition_loops(text)
        is_valid = has_bn and not has_loop

        return {
            "is_valid": is_valid,
            "contains_bengali": has_bn,
            "has_repetition_loop": has_loop,
            "verdict": "VALID_BENGALI" if is_valid else ("REPETITION_LOOP" if has_loop else "NO_BENGALI_DETECTED")
        }
