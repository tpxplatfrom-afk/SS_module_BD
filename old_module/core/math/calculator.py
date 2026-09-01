"""
SS Tutor BD - Deterministic Arithmetic & Financial Math Engine
Calculates exact arithmetic, sequences, profit/loss, simple & compound interest,
and geometry formulas with stepwise Bengali explanations.
"""

import math
from typing import Dict, Any, List, Optional
from core.math.fraction import FractionHelper


class MathCalculator:
    @staticmethod
    def to_bengali(val: Any) -> str:
        if isinstance(val, (int, float)):
            if isinstance(val, float) and val.is_integer():
                val = int(val)
        return FractionHelper.to_bengali_number(val)

    @staticmethod
    def simple_interest(principal: float, rate_pct: float, time_years: float) -> Dict[str, Any]:
        """Calculates Simple Interest I = Prn and Amount A = P + I."""
        r = rate_pct / 100.0
        interest = principal * r * time_years
        amount = principal + interest
        
        steps = [
            f"দেওয়া আছে: আসল P = {MathCalculator.to_bengali(principal)} টাকা, মুনাফার হার r = {MathCalculator.to_bengali(rate_pct)}% = {rate_pct}/১০০, সময় n = {MathCalculator.to_bengali(time_years)} বছর।",
            f"আমরা জানি, সরল মুনাফা I = P × r × n",
            f"I = {MathCalculator.to_bengali(principal)} × ({rate_pct}/১০০) × {MathCalculator.to_bengali(time_years)} = {MathCalculator.to_bengali(interest)} টাকা।",
            f"মুনাফা-আসল A = P + I = {MathCalculator.to_bengali(principal)} + {MathCalculator.to_bengali(interest)} = {MathCalculator.to_bengali(amount)} টাকা।"
        ]
        return {
            "principal": principal,
            "rate_pct": rate_pct,
            "time_years": time_years,
            "interest": interest,
            "total_amount": amount,
            "steps": steps,
            "final_answer": f"মুনাফা {MathCalculator.to_bengali(interest)} টাকা এবং মুনাফা-আসল {MathCalculator.to_bengali(amount)} টাকা"
        }

    @staticmethod
    def compound_interest(principal: float, rate_pct: float, time_years: float) -> Dict[str, Any]:
        """Calculates Compound Amount C = P(1 + r)^n and Compound Interest C - P."""
        r = rate_pct / 100.0
        compound_amount = principal * ((1.0 + r) ** time_years)
        compound_interest = compound_amount - principal
        
        steps = [
            f"দেওয়া আছে: মূলধন P = {MathCalculator.to_bengali(principal)} টাকা, হার r = {MathCalculator.to_bengali(rate_pct)}% = {r}, সময় n = {MathCalculator.to_bengali(time_years)} বছর।",
            f"আমরা জানি, চক্রবৃদ্ধি মূলধন C = P(১ + r)^n",
            f"C = {MathCalculator.to_bengali(principal)} × (১ + {r})^{MathCalculator.to_bengali(time_years)} = {MathCalculator.to_bengali(round(compound_amount, 2))} টাকা।",
            f"চক্রবৃদ্ধি মুনাফা = C - P = {MathCalculator.to_bengali(round(compound_amount, 2))} - {MathCalculator.to_bengali(principal)} = {MathCalculator.to_bengali(round(compound_interest, 2))} টাকা।"
        ]
        return {
            "principal": principal,
            "rate_pct": rate_pct,
            "time_years": time_years,
            "compound_amount": round(compound_amount, 2),
            "compound_interest": round(compound_interest, 2),
            "steps": steps,
            "final_answer": f"চক্রবৃদ্ধি মূলধন {MathCalculator.to_bengali(round(compound_amount, 2))} টাকা ও মুনাফা {MathCalculator.to_bengali(round(compound_interest, 2))} টাকা"
        }

    @staticmethod
    def series_sum(first_term: int, last_term: int, n_terms: Optional[int] = None) -> Dict[str, Any]:
        """Calculates arithmetic series sum S_n = ((1st + last) * n) / 2."""
        if n_terms is None:
            n_terms = (last_term - first_term + 1)
        total_sum = ((first_term + last_term) * n_terms) // 2
        steps = [
            f"১ম পদ = {MathCalculator.to_bengali(first_term)}, শেষ পদ = {MathCalculator.to_bengali(last_term)}, পদসংখ্যা = {MathCalculator.to_bengali(n_terms)}।",
            f"সমষ্টি সূত্র: S = [({MathCalculator.to_bengali(first_term)} + {MathCalculator.to_bengali(last_term)}) × {MathCalculator.to_bengali(n_terms)}] / ২",
            f"= ({MathCalculator.to_bengali(first_term + last_term)} × {MathCalculator.to_bengali(n_terms)}) / ২ = {MathCalculator.to_bengali(total_sum)}।"
        ]
        return {
            "first_term": first_term,
            "last_term": last_term,
            "n_terms": n_terms,
            "sum": total_sum,
            "steps": steps,
            "final_answer": f"{MathCalculator.to_bengali(total_sum)}"
        }

    @staticmethod
    def pythagoras(a: Optional[float] = None, b: Optional[float] = None, c: Optional[float] = None) -> Dict[str, Any]:
        """Calculates hypotenuse (c) or leg (a or b) using c^2 = a^2 + b^2."""
        if c is None and a is not None and b is not None:
            c = math.sqrt(a**2 + b**2)
            steps = [
                f"দেওয়া আছে: লম্ব a = {MathCalculator.to_bengali(a)}, ভূমি b = {MathCalculator.to_bengali(b)}।",
                f"পিথাগোরাসের উপপাদ্য অনুসারে: অতিভুজ c^2 = a^2 + b^2 = {MathCalculator.to_bengali(a)}^2 + {MathCalculator.to_bengali(b)}^2",
                f"c^2 = {MathCalculator.to_bengali(a**2)} + {MathCalculator.to_bengali(b**2)} = {MathCalculator.to_bengali(a**2 + b**2)}",
                f"c = √{MathCalculator.to_bengali(a**2 + b**2)} = {MathCalculator.to_bengali(c)} একক।"
            ]
            return {"hypotenuse": c, "steps": steps, "final_answer": f"{MathCalculator.to_bengali(c)} একক"}
        elif c is not None and a is not None and b is None:
            b = math.sqrt(c**2 - a**2)
            steps = [
                f"দেওয়া আছে: অতিভুজ c = {MathCalculator.to_bengali(c)}, অপর বাহু a = {MathCalculator.to_bengali(a)}।",
                f"পিথাগোরাসের সূত্র মতে: b^2 = c^2 - a^2 = {MathCalculator.to_bengali(c)}^2 - {MathCalculator.to_bengali(a)}^2",
                f"b^2 = {MathCalculator.to_bengali(c**2)} - {MathCalculator.to_bengali(a**2)} = {MathCalculator.to_bengali(c**2 - a**2)}",
                f"b = √{MathCalculator.to_bengali(c**2 - a**2)} = {MathCalculator.to_bengali(b)} একক।"
            ]
            return {"leg": b, "steps": steps, "final_answer": f"{MathCalculator.to_bengali(b)} একক"}
        raise ValueError("Invalid Pythagoras inputs.")

    @staticmethod
    def circle_metrics(radius: float, pi_fraction: bool = True) -> Dict[str, Any]:
        """Calculates Circle Circumference 2*pi*r and Area pi*r^2."""
        if pi_fraction:
            circumference = 2.0 * (22.0 / 7.0) * radius
            area = (22.0 / 7.0) * (radius ** 2)
            pi_str = "২২/৭"
        else:
            circumference = 2.0 * 3.1416 * radius
            area = 3.1416 * (radius ** 2)
            pi_str = "৩.১৪১৬"

        steps = [
            f"দেওয়া আছে: বৃত্তের ব্যাসার্ধ r = {MathCalculator.to_bengali(radius)} সেমি, π ≈ {pi_str}।",
            f"পরিধি সূত্র: C = 2 × π × r = 2 × {pi_str} × {MathCalculator.to_bengali(radius)} = {MathCalculator.to_bengali(round(circumference, 2))} সেমি।",
            f"ক্ষেত্রফল সূত্র: A = π × r^2 = {pi_str} × ({MathCalculator.to_bengali(radius)})^2 = {MathCalculator.to_bengali(round(area, 2))} বর্গসেমি।"
        ]
        return {
            "radius": radius,
            "circumference": round(circumference, 2),
            "area": round(area, 2),
            "steps": steps,
            "final_answer": f"পরিধি {MathCalculator.to_bengali(round(circumference, 2))} সেমি এবং ক্ষেত্রফল {MathCalculator.to_bengali(round(area, 2))} বর্গসেমি"
        }
