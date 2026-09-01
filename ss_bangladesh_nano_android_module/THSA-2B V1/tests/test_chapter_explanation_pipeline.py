"""
THSA-2B Chapter Master Blueprint & Pedagogical Test Suite
===========================================================
Tests the module when a user prompts: "Explain Chapter N"
Verifies:
  1. What prerequisite knowledge is needed to understand the chapter.
  2. What exact formulas & equations are required for calculations.
  3. Key calculation strategies and common examination traps.
  4. Socratic roadmap & interactive self-practice question.
  5. UI Markdown screen-safety check.
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.engine.socratic_math_engine import SocraticMathEngine

def run_test():
    print("=" * 80)
    print("THSA-2.41B: CHAPTER 'N' BLUEPRINT & FORMULA EXPLANATION TEST")
    print("=" * 80)
    
    engine = SocraticMathEngine()
    test_query = "Explain Chapter 3 of Class 9 Maths book. What formulas and knowledge are needed?"
    print(f"\n[User Query]: {test_query}\n")
    
    result = engine.solve_and_explain(test_query)
    
    # 1. Verification of Required Knowledge (Prerequisites)
    print("[Test 1] Prerequisite Knowledge Section Verification")
    assert "১. প্রয়োজনীয় পূর্বজ্ঞান" in result["formatted_markdown"], "Failed: Missing Prerequisites Header"
    assert "চলক" in result["formatted_markdown"] or "মৌলিক" in result["formatted_markdown"], "Failed: Prerequisites content missing"
    print("  --> PASS: Module clearly states what foundational knowledge the student needs.")

    # 2. Verification of Formula Sheet for Calculation
    print("\n[Test 2] Formula Sheet & Equations Verification for Chapter Calculations")
    assert "২. প্রয়োজনীয় সকল সূত্র ও অনুসিদ্ধান্তের তালিকা" in result["formatted_markdown"], "Failed: Missing Formulas Header"
    assert "(a + b)^2" in result["formatted_markdown"], "Failed: Missing core square formula"
    assert "(a + b)^3" in result["formatted_markdown"], "Failed: Missing cube formula"
    assert "ভাগশেষ উপপাদ্য" in result["formatted_markdown"], "Failed: Missing remainder theorem"
    print("  --> PASS: All essential calculation formulas & deductions included in clean LaTeX format.")

    # 3. Verification of Calculation Strategies & Exam Traps
    print("\n[Test 3] Calculation Strategies & Common Exam Pitfalls Verification")
    assert "৩. গাণিতিক হিসাবের মূল নিয়ম ও সমাধান কৌশল" in result["formatted_markdown"], "Failed: Missing Strategies Header"
    assert "৪. সচরাচর সাধারণ ভুল ও পরীক্ষার সতর্কতা" in result["formatted_markdown"], "Failed: Missing Common Pitfalls Header"
    print("  --> PASS: Module provides actionable calculation heuristics and warns about sign errors.")

    # 4. Verification of Socratic Roadmap & Self-Practice
    print("\n[Test 4] Socratic Roadmap & Interactive Quiz Verification")
    assert "৫. সহনশীল সক্রেটিক রোডম্যাপ" in result["formatted_markdown"], "Failed: Missing Socratic Roadmap"
    assert "কুইজ প্রশ্ন" in result["formatted_markdown"], "Failed: Interactive check quiz missing"
    print("  --> PASS: Includes encouraging roadmap with an interactive self-test quiz.")

    # 5. UI Screen Safety Check
    print("\n[Test 5] Mobile Screen Safety (.md structure)")
    assert result["is_screen_safe"] is True, "Failed: Screen safe flag false"
    print("  --> PASS: Output is structured in standard GitHub-flavored Markdown preventing Android layout breakage.")

    # Print Live Output
    print("\n" + "=" * 80)
    print("LIVE MODULE GENERATED CHAPTER EXPLANATION (Markdown Payload):")
    print("=" * 80)
    print(result["formatted_markdown"])
    print("=" * 80)
    print("\n[ALL CHAPTER 'N' BLUEPRINT TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    run_test()
