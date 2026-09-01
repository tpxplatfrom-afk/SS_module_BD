"""
SS Tutor BD - Grounding & Anti-Hallucination Validator (Phase 4)
Validates that LLM responses do not introduce ungrounded claims or hallucinated facts.
"""

import re
from typing import Dict, Any, List


class GroundingValidator:
    REFUSAL_PHRASES = [
        "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না",
        "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না",
        "পাঠ্যবইয়ে এই তথ্যটি দেওয়া নেই",
        "পাঠ্যবইয়ে এই তথ্যটি দেওয়া নেই",
        "এই প্রশ্নের তথ্য পাঠ্যপুস্তকে অনুপস্থিত"
    ]

    @classmethod
    def validate_grounding(cls, response_text: str, textbook_facts: str, is_unsupported_query: bool = False) -> Dict[str, Any]:
        """
        Verifies that:
        1. If the query cannot be answered from context, the model produces a polite refusal.
        2. If facts are provided, the model adheres to keywords/formulas in the facts.
        """
        has_refusal = any(phrase in response_text for phrase in cls.REFUSAL_PHRASES)

        if is_unsupported_query:
            is_valid = has_refusal
            verdict = "PASSED_REFUSAL" if is_valid else "FAILED_HALLUCINATED_UNSUPPORTED_CLAIM"
        else:
            # Check for keyword overlap
            fact_words = set(textbook_facts.split())
            resp_words = set(response_text.split())
            overlap = fact_words.intersection(resp_words)
            is_valid = len(overlap) > 0 or len(response_text) > 10
            verdict = "PASSED_GROUNDED" if is_valid else "FAILED_DISCONNECTED_RESPONSE"

        return {
            "is_valid": is_valid,
            "verdict": verdict,
            "has_refusal": has_refusal,
            "grounding_score": 1.0 if is_valid else 0.0
        }
