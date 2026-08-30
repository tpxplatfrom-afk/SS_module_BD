"""
SS Tutor BD - Bengali Mathematical Intent & Expression Parser
Detects whether a query contains deterministic math problems (fractions, arithmetic, interest, pythagoras)
and extracts clean structured parameters.
"""

import re
from typing import Dict, Any, Optional, Tuple


def convert_bengali_to_english_digits(text: str) -> str:
    bengali_digits = "০১২৩৪৫৬৭৮৯"
    res = ""
    for ch in text:
        if ch in bengali_digits:
            res += str(bengali_digits.index(char := ch))
        else:
            res += ch
    return res


class ExpressionParser:
    @staticmethod
    def detect_math_intent(query: str) -> Dict[str, Any]:
        """Classifies mathematical intent and extracts parameters."""
        q_eng = convert_bengali_to_english_digits(query)

        # 1. Fraction Addition / Operation
        # Matches e.g. "3/4 + 5/6" or "৩/৪ এবং ৫/৬ এর যোগফল"
        frac_matches = re.findall(r"(\d+)\s*/\s*(\d+)", q_eng)
        if len(frac_matches) >= 2 and any(w in query for w in ["যোগ", "+", "সমষ্টি", "যোগফল"]):
            return {
                "intent": "fraction_addition",
                "fraction1": (int(frac_matches[0][0]), int(frac_matches[0][1])),
                "fraction2": (int(frac_matches[1][0]), int(frac_matches[1][1]))
            }

        # 2. Simple Interest: principal, rate, time
        # Look for numbers with %, টাকা, বছর
        if any(w in query for w in ["মুনাফা", "সুদ", "মুনাফা-আসল"]) and not any(w in query for w in ["চক্রবৃদ্ধি", "compound"]):
            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", q_eng)]
            if len(nums) >= 3:
                # Typically [rate, principal, time] or [principal, rate, time]
                # Find the one that's <= 30 as rate, the large one as principal, the small one as years
                sorted_nums = sorted(nums)
                time_y = sorted_nums[0]
                rate = sorted_nums[1] if len(sorted_nums) >= 2 and sorted_nums[1] <= 35 else sorted_nums[0]
                principal = sorted_nums[-1]
                return {
                    "intent": "simple_interest",
                    "principal": principal,
                    "rate_pct": rate,
                    "time_years": time_y
                }

        # 3. Compound Interest
        if any(w in query for w in ["চক্রবৃদ্ধি"]):
            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", q_eng)]
            if len(nums) >= 3:
                sorted_nums = sorted(nums)
                time_y = sorted_nums[0]
                rate = sorted_nums[1]
                principal = sorted_nums[-1]
                return {
                    "intent": "compound_interest",
                    "principal": principal,
                    "rate_pct": rate,
                    "time_years": time_y
                }

        # 4. Arithmetic Series Sum (1 + 2 + ... + 100 or 1 থেকে 50 পর্যন্ত)
        if any(w in query for w in ["ধারা", "ক্রমিক", "স্বাভাবিক সংখ্যার যোগফল", "পর্যন্ত"]) and any(w in query for w in ["যোগফল", "সমষ্টি"]):
            nums = [int(n) for n in re.findall(r"\d+", q_eng)]
            if len(nums) >= 2:
                return {
                    "intent": "series_sum",
                    "first_term": min(nums),
                    "last_term": max(nums)
                }

        # 5. Pythagoras (সমকোণী ত্রিভুজ)
        if any(w in query for w in ["সমকোণী", "অতিভুজ", "পিথাগোরাস"]):
            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", q_eng)]
            if len(nums) >= 2:
                if "অতিভুজ" in query and any(w in query for w in ["ভূমি", "লম্ব"]):
                    # If hypotenuse and leg are given, find other leg
                    hyp = max(nums)
                    leg = min(nums)
                    return {"intent": "pythagoras_leg", "hypotenuse": hyp, "leg": leg}
                else:
                    # Legs given, find hypotenuse
                    return {"intent": "pythagoras_hypotenuse", "leg1": nums[0], "leg2": nums[1]}

        # 6. Circle metrics (বৃত্তের ব্যাসার্ধ/ব্যাস)
        if any(w in query for w in ["বৃত্ত", "ব্যাসার্ধ", "পরিধি", "বৃত্তের ক্ষেত্রফল"]):
            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", q_eng)]
            if len(nums) >= 1:
                r = nums[0]
                if "ব্যাস" in query and "ব্যাসার্ধ" not in query:
                    r = r / 2.0
                return {"intent": "circle_metrics", "radius": r}

        # 7. Quadratic Factorization (x^2 + 5x + 6)
        quad_match = re.search(r"x\^2\s*([\+\-]\s*\d+)?x\s*([\+\-]\s*\d+)", q_eng.replace(" ", ""))
        if quad_match:
            b_str = quad_match.group(1) or "+1"
            c_str = quad_match.group(2)
            b_val = int(b_str.replace("+", "")) if b_str else 1
            c_val = int(c_str.replace("+", ""))
            return {"intent": "factorization", "b": b_val, "c": c_val}

        return {"intent": "general_or_concept"}
