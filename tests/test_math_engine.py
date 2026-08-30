"""
SS Tutor BD - Unit Tests: Deterministic Math Engine (Phase 3B)
"""

import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.math.fraction import FractionHelper
from core.math.calculator import MathCalculator
from core.math.equation_solver import EquationSolver
from core.math.expression_parser import ExpressionParser


def test_fraction_add_basic():
    result = FractionHelper.add((3, 4), (5, 6))
    assert result["result_improper"] == (19, 12), f"Expected (19,12), got {result['result_improper']}"
    assert result["result_mixed"]["type"] == "mixed", "Should be a mixed fraction"
    assert result["result_mixed"]["whole"] == 1
    assert result["result_mixed"]["num"] == 7
    assert result["result_mixed"]["den"] == 12
    print("test_fraction_add_basic: PASSED")

def test_fraction_add_reduces():
    result = FractionHelper.add((1, 2), (1, 2))
    assert result["result_improper"] == (1, 1), f"Expected (1,1), got {result['result_improper']}"
    print("test_fraction_add_reduces: PASSED")

def test_fraction_add_steps_bengali():
    result = FractionHelper.add((2, 3), (4, 5))
    steps = result["steps_bengali"]
    assert len(steps) >= 3
    assert "ল.সা.গু" in steps[0]
    print("test_fraction_add_steps_bengali: PASSED")

def test_simple_interest():
    res = MathCalculator.simple_interest(5000, 10, 3)
    assert res["interest"] == 1500.0, f"Expected 1500, got {res['interest']}"
    assert res["total_amount"] == 6500.0, f"Expected 6500, got {res['total_amount']}"
    print("test_simple_interest: PASSED")

def test_compound_interest():
    res = MathCalculator.compound_interest(8000, 10, 2)
    assert abs(res["compound_amount"] - 9680.0) < 0.1, f"Expected 9680, got {res['compound_amount']}"
    print("test_compound_interest: PASSED")

def test_compound_interest_5000():
    res = MathCalculator.compound_interest(5000, 10, 3)
    # C = 5000 * 1.1^3 = 5000 * 1.331 = 6655
    assert abs(res["compound_amount"] - 6655.0) < 0.5, f"Expected ~6655, got {res['compound_amount']}"
    print("test_compound_interest_5000: PASSED")

def test_series_sum_1_to_100():
    res = MathCalculator.series_sum(1, 100)
    assert res["sum"] == 5050, f"Expected 5050, got {res['sum']}"
    print("test_series_sum_1_to_100: PASSED")

def test_series_sum_1_to_50():
    res = MathCalculator.series_sum(1, 50)
    assert res["sum"] == 1275, f"Expected 1275, got {res['sum']}"
    print("test_series_sum_1_to_50: PASSED")

def test_pythagoras_hypotenuse():
    res = MathCalculator.pythagoras(a=6.0, b=8.0)
    assert abs(res["hypotenuse"] - 10.0) < 0.01, f"Expected 10, got {res['hypotenuse']}"
    print("test_pythagoras_hypotenuse: PASSED")

def test_pythagoras_leg():
    res = MathCalculator.pythagoras(c=13.0, a=5.0)
    assert abs(res["leg"] - 12.0) < 0.01, f"Expected 12, got {res['leg']}"
    print("test_pythagoras_leg: PASSED")

def test_pythagoras_17_8_15():
    res = MathCalculator.pythagoras(c=17.0, a=8.0)
    assert abs(res["leg"] - 15.0) < 0.01, f"Expected 15, got {res['leg']}"
    print("test_pythagoras_17_8_15: PASSED")

def test_circle_radius_7():
    res = MathCalculator.circle_metrics(7.0)
    assert abs(res["circumference"] - 44.0) < 0.2, f"Expected 44, got {res['circumference']}"
    assert abs(res["area"] - 154.0) < 0.2, f"Expected 154, got {res['area']}"
    print("test_circle_radius_7: PASSED")

def test_factorize_quadratic_basic():
    res = EquationSolver.factorize_quadratic(7, 12)
    assert res["factorizable"] == True
    assert set(res["factors"]) == {3, 4}
    print("test_factorize_quadratic_basic: PASSED")

def test_factorize_quadratic_5_6():
    res = EquationSolver.factorize_quadratic(5, 6)
    assert res["factorizable"] == True
    assert set(res["factors"]) == {2, 3}
    print("test_factorize_quadratic_5_6: PASSED")

def test_expression_parser_fraction():
    intent = ExpressionParser.detect_math_intent("৩/৪ + ৫/৬ এর যোগফল নির্ণয় করো")
    assert intent["intent"] == "fraction_addition", f"Got {intent['intent']}"
    print("test_expression_parser_fraction: PASSED")

def test_expression_parser_simple_interest():
    intent = ExpressionParser.detect_math_intent("৫০০০ টাকায় ১০% হারে ৩ বছরের সরল মুনাফা কত?")
    assert intent["intent"] == "simple_interest", f"Got {intent['intent']}"
    print("test_expression_parser_simple_interest: PASSED")

def test_expression_parser_series():
    intent = ExpressionParser.detect_math_intent("১ থেকে ১০০ পর্যন্ত ক্রমিক সংখ্যার যোগফল কত?")
    assert intent["intent"] == "series_sum", f"Got {intent['intent']}"
    print("test_expression_parser_series: PASSED")

def test_expression_parser_general():
    intent = ExpressionParser.detect_math_intent("পিথাগোরাস উপপাদ্য কী?")
    assert intent["intent"] == "general_or_concept", f"Got {intent['intent']}"
    print("test_expression_parser_general: PASSED")


def run_all_math_tests():
    print("\n--- Running SS Tutor BD Math Engine Unit Tests ---")
    test_fraction_add_basic()
    test_fraction_add_reduces()
    test_fraction_add_steps_bengali()
    test_simple_interest()
    test_compound_interest()
    test_compound_interest_5000()
    test_series_sum_1_to_100()
    test_series_sum_1_to_50()
    test_pythagoras_hypotenuse()
    test_pythagoras_leg()
    test_pythagoras_17_8_15()
    test_circle_radius_7()
    test_factorize_quadratic_basic()
    test_factorize_quadratic_5_6()
    test_expression_parser_fraction()
    test_expression_parser_simple_interest()
    test_expression_parser_series()
    test_expression_parser_general()
    print(f"--- All Math Engine Tests PASSED ({18} / 18) ---\n")


if __name__ == "__main__":
    run_all_math_tests()
