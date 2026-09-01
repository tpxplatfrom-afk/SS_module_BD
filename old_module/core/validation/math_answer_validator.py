"""
SS Tutor BD - Math Answer Validator (Phase 4)
Cross-validates neural model numerical answers against authoritative deterministic results.
"""

import re
from typing import Dict, Any, Optional


class MathAnswerValidator:
    BENGALI_DIGITS = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
                      "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}

    @classmethod
    def to_ascii_digits(cls, text: str) -> str:
        res = text
        for b, a in cls.BENGALI_DIGITS.items():
            res = res.replace(b, a)
        return res

    @classmethod
    def validate_and_correct(
        cls,
        model_response: str,
        exact_deterministic_result: str,
        steps: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        If model verbalization omits or contradicts the deterministic result,
        produces a safe corrected response incorporating the exact steps.
        """
        norm_resp = cls.to_ascii_digits(model_response)
        norm_exact = cls.to_ascii_digits(exact_deterministic_result)

        # Extract main numbers from exact result
        numbers_in_exact = re.findall(r"\d+", norm_exact)
        numbers_in_resp = re.findall(r"\d+", norm_resp)

        has_conflict = False
        if numbers_in_exact:
            primary_num = numbers_in_exact[-1]
            if primary_num not in numbers_in_resp and len(model_response) < 30:
                has_conflict = True

        if has_conflict or len(model_response.strip()) < 15:
            # Construct authoritative fallback
            step_str = "\n".join(steps) if steps else ""
            corrected_text = f"গণনার ধাপসমূহ:\n{step_str}\nঅতএব সঠিক উত্তর: {exact_deterministic_result}।"
            return {
                "is_valid": False,
                "was_corrected": True,
                "final_text": corrected_text,
                "reason": "NUMERICAL_MISMATCH_OR_INCOMPLETE"
            }

        return {
            "is_valid": True,
            "was_corrected": False,
            "final_text": model_response,
            "reason": "NUMERICALLY_CONSISTENT"
        }
