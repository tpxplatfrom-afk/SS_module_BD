"""
THSA-2B Socratic Math & UI Screen-Safe Rendering Benchmark
============================================================
Tests the pedagogical response pipeline for:
  - Class 9 Math Chapter 3, Exercise 3.1, Question 2(a)
  - Calculation-first, explanation-second sequence
  - Markdown/LaTeX structure integrity (preventing screen layout breaks in Android)
  - Socratic empathy and tolerance alignment
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add engine directory to path
MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.engine.socratic_math_engine import SocraticMathEngine

def run_test():
    print("=" * 80)
    print("THSA-2.41B: SOCRATIC MATH & SCREEN-SAFE RENDERING BENCHMARK")
    print("=" * 80)
    
    engine = SocraticMathEngine()
    test_query = "Can you explain the number 'a' in Chapter 3, Exercise 3.1 of 2 of Class 9 Maths book?"
    print(f"\n[Input Query]: {test_query}\n")
    
    result = engine.solve_and_explain(test_query)
    
    # 1. Verification of Calculation-First Structure
    print("[Test 1] Structure Check: Calculation First -> Explanation Second")
    assert "১. গাণিতিক সমাধান" in result["formatted_markdown"], "Failed: Missing Calculation Header"
    assert "২. সহজ ভাষায় বিস্তারিত ব্যাখ্যা" in result["formatted_markdown"], "Failed: Missing Explanation Header"
    assert "৩. সক্রেটিক সহনশীল ইঙ্গিত" in result["formatted_markdown"], "Failed: Missing Socratic Hint"
    
    calc_idx = result["formatted_markdown"].find("১. গাণিতিক সমাধান")
    exp_idx = result["formatted_markdown"].find("২. সহজ ভাষায় বিস্তারিত ব্যাখ্যা")
    assert calc_idx < exp_idx, "Failed: Explanation appeared before calculation!"
    print("  --> PASS: Sequence strictly adheres to Calculation First -> Explanation Second.")

    # 2. Mathematical Exactness Check
    print("\n[Test 2] Mathematical Exactness Check (NCTB Ex 3.1 Q 2a)")
    assert "(7q - p)^2" in result["calculation_block"] or "(7q - p)²" in result["calculation_block"], "Failed: Final simplification $(7q - p)^2$ not found!"
    assert "a^2 - 2ab + b^2" in result["calculation_block"], "Failed: Core formula not identified!"
    print("  --> PASS: Exact algebraic solution verified: (7q - p)^2.")

    # 3. Android Screen Safety & Markdown Delimiter Integrity Check
    print("\n[Test 3] Android UI Screen Safety & Markdown Encapsulation Check")
    # Check that raw unstructured equation text doesn't overflow without markdown tags
    assert result["is_screen_safe"] is True, "Failed: Screen safety flag not set"
    assert "```math" in result["formatted_markdown"], "Failed: Math not wrapped in codeblock/markdown delimiter"
    print("  --> PASS: Payload fully encapsulated in Markdown (.md format) preventing Android screen text breakage.")

    # 4. Socratic Empathy & Tolerance Tone Check
    print("\n[Test 4] Empathy, Tolerance & Socratic Scaffolding Tone Check")
    assert "💡" in result["socratic_hint"], "Failed: Socratic hint icon missing"
    assert "চেষ্টা" in result["socratic_hint"] or "বুঝতে" in result["socratic_hint"], "Failed: Encouraging tone words missing"
    print("  --> PASS: Socratic tone provides encouraging scaffolding for students.")

    # Print Live Formatted Output for Inspection
    print("\n" + "=" * 80)
    print("LIVE MODULE GENERATED RESPONSE (Markdown Payload):")
    print("=" * 80)
    print(result["formatted_markdown"])
    print("=" * 80)
    print("\n[ALL 4 PEDAGOGICAL & UI TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    run_test()
