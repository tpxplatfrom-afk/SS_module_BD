"""
SS Tutor BD - Deterministic Equation Solver
Solves linear simultaneous systems (প্রতিস্থাপন ও অপনয়ন) and quadratic factorizations with step-by-step Bengali output.
"""

import math
from typing import Dict, Any, Tuple, Optional
from core.math.fraction import FractionHelper


class EquationSolver:
    @staticmethod
    def solve_linear_2x2(
        a1: float, b1: float, c1: float,
        a2: float, b2: float, c2: float,
        method: str = "elimination"
    ) -> Dict[str, Any]:
        """
        Solves 2x2 system:
        a1*x + b1*y = c1
        a2*x + b2*y = c2
        """
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            raise ValueError("সমীকরণ জোটটির কোনো একক সমাধান নেই (Determinant is 0)।")

        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det

        x_int = int(x) if x.is_integer() else round(x, 2)
        y_int = int(y) if y.is_integer() else round(y, 2)

        steps = [
            f"প্রদত্ত সমীকরণদ্বয়:\n(১) {a1}x + {b1}y = {c1}\n(২) {a2}x + {b2}y = {c2}",
            f"অপনয়ন/প্রতিস্থাপন পদ্ধতিতে সমাধান করে পাই:",
            f"x এর মান = {FractionHelper.to_bengali_number(x_int)}",
            f"y এর মান = {FractionHelper.to_bengali_number(y_int)}"
        ]

        return {
            "x": x_int,
            "y": y_int,
            "bengali_solution": f"(x, y) = ({FractionHelper.to_bengali_number(x_int)}, {FractionHelper.to_bengali_number(y_int)})",
            "steps": steps
        }

    @staticmethod
    def factorize_quadratic(b: int, c: int) -> Dict[str, Any]:
        """
        Factorizes x^2 + bx + c using middle-term break.
        Finds p, q such that p*q = c and p+q = b -> (x + p)(x + q).
        """
        found = False
        p_res, q_res = 0, 0
        limit = int(math.isqrt(abs(c))) + abs(b) + 10
        
        for p in range(-limit, limit + 1):
            if p == 0:
                continue
            if c % p == 0:
                q = c // p
                if p + q == b:
                    found = True
                    p_res, q_res = p, q
                    break

        if not found:
            return {
                "factorizable": False,
                "message": "রাশিটিকে সাধারণ পূর্ণসংখ্যায় মধ্যপদ বিভাজন করা যায় না।"
            }

        p_str = f"+ {FractionHelper.to_bengali_number(p_res)}" if p_res >= 0 else f"- {FractionHelper.to_bengali_number(abs(p_res))}"
        q_str = f"+ {FractionHelper.to_bengali_number(q_res)}" if q_res >= 0 else f"- {FractionHelper.to_bengali_number(abs(q_res))}"

        steps = [
            f"রাশি: x^2 + ({FractionHelper.to_bengali_number(b)})x + ({FractionHelper.to_bengali_number(c)})",
            f"এখানে ধ্রুবক {FractionHelper.to_bengali_number(c)} = {FractionHelper.to_bengali_number(p_res)} × {FractionHelper.to_bengali_number(q_res)} এবং যোগফল {FractionHelper.to_bengali_number(p_res)} + {FractionHelper.to_bengali_number(q_res)} = {FractionHelper.to_bengali_number(b)}।",
            f"মধ্যপদ বিভাজন করে পাই: x^2 + {p_res}x + {q_res}x + {c}",
            f"= x(x {p_str}) {q_str}(x {p_str})",
            f"= (x {p_str})(x {q_str})।"
        ]

        return {
            "factorizable": True,
            "factors": (p_res, q_res),
            "bengali_expression": f"(x {p_str})(x {q_str})",
            "steps": steps
        }
