"""
SS Tutor BD - Deterministic Fraction Engine
Handles exact fraction arithmetic, algebraic fraction operations, reduction,
and generates step-by-step educational explanations in natural Bengali.
"""

import math
from typing import Tuple, Dict, Any, List


class FractionHelper:
    @staticmethod
    def parse_fraction(text: str) -> Tuple[int, int]:
        """Parses string like '3/4' or '৩/৪' into (numerator, denominator)."""
        # Convert Bengali digits to English
        bengali_digits = "০১২৩৪৫৬৭৮৯"
        converted = ""
        for char in text:
            if char in bengali_digits:
                converted += str(bengali_digits.index(char))
            else:
                converted += char
        
        parts = converted.strip().split("/")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        elif len(parts) == 1 and parts[0].isdigit():
            return int(parts[0]), 1
        raise ValueError(f"Invalid fraction format: {text}")

    @staticmethod
    def to_bengali_number(num: int) -> str:
        """Converts English integer to Bengali digits."""
        bengali_digits = "০১২৩৪৫৬৭৮৯"
        s = str(num)
        res = ""
        for ch in s:
            if ch.isdigit():
                res += bengali_digits[int(ch)]
            else:
                res += ch
        return res

    @staticmethod
    def reduce(num: int, den: int) -> Tuple[int, int]:
        """Reduces fraction to simplest form."""
        if den == 0:
            raise ZeroDivisionError("হর কখনো শূন্য হতে পারে না।")
        g = math.gcd(num, den)
        return num // g, den // g

    @staticmethod
    def to_mixed_fraction(num: int, den: int) -> Dict[str, Any]:
        """Converts improper fraction to mixed fraction (মিশ্র ভগ্নাংশ)."""
        num, den = FractionHelper.reduce(num, den)
        if den == 1:
            return {"type": "integer", "value": num, "bengali": FractionHelper.to_bengali_number(num)}
        if abs(num) < den:
            return {
                "type": "proper",
                "num": num,
                "den": den,
                "bengali": f"{FractionHelper.to_bengali_number(num)}/{FractionHelper.to_bengali_number(den)}"
            }
        
        whole = num // den
        rem = num % den
        return {
            "type": "mixed",
            "whole": whole,
            "num": rem,
            "den": den,
            "bengali": f"{FractionHelper.to_bengali_number(whole)} সমস্ত {FractionHelper.to_bengali_number(rem)}/{FractionHelper.to_bengali_number(den)}"
        }

    @staticmethod
    def add(f1: Tuple[int, int], f2: Tuple[int, int]) -> Dict[str, Any]:
        """Adds two fractions with complete NCTB Bengali step-by-step derivation."""
        n1, d1 = f1
        n2, d2 = f2
        
        # LCM of denominators
        lcm_den = (d1 * d2) // math.gcd(d1, d2)
        m1 = lcm_den // d1
        m2 = lcm_den // d2
        
        step_n1 = n1 * m1
        step_n2 = n2 * m2
        total_num = step_n1 + step_n2
        
        red_num, red_den = FractionHelper.reduce(total_num, lcm_den)
        mixed = FractionHelper.to_mixed_fraction(red_num, red_den)
        
        steps = [
            f"ধাপ ১: হরদ্বয় {FractionHelper.to_bengali_number(d1)} এবং {FractionHelper.to_bengali_number(d2)} এর ল.সা.গু = {FractionHelper.to_bengali_number(lcm_den)}।",
            f"ধাপ ২: সাধারণ হরবিশিষ্টকরণ করে পাই: {FractionHelper.to_bengali_number(step_n1)}/{FractionHelper.to_bengali_number(lcm_den)} + {FractionHelper.to_bengali_number(step_n2)}/{FractionHelper.to_bengali_number(lcm_den)}।",
            f"ধাপ ৩: লবদ্বয় যোগ করে পাই: ({FractionHelper.to_bengali_number(step_n1)} + {FractionHelper.to_bengali_number(step_n2)})/{FractionHelper.to_bengali_number(lcm_den)} = {FractionHelper.to_bengali_number(total_num)}/{FractionHelper.to_bengali_number(lcm_den)}।"
        ]
        if red_den != lcm_den:
            steps.append(f"ধাপ ৪: লঘিষ্ঠকরণ করে পাই: {FractionHelper.to_bengali_number(red_num)}/{FractionHelper.to_bengali_number(red_den)}।")
        if mixed["type"] == "mixed":
            steps.append(f"ধাপ ৫: মিশ্র ভগ্নাংশে রূপান্তর: {mixed['bengali']}।")

        final_ans = mixed['bengali'] if mixed["type"] == "mixed" else f"{FractionHelper.to_bengali_number(red_num)}/{FractionHelper.to_bengali_number(red_den)}"

        return {
            "operation": "addition",
            "result_improper": (red_num, red_den),
            "result_mixed": mixed,
            "final_answer_bengali": final_ans,
            "steps_bengali": steps
        }
