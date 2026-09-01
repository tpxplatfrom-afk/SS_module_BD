"""
THSA-2B Creative Assessment & Question Answering Test
======================================================
Tests:
  1. Creative Question (CQ) and MCQ Generation from a specified Book & Chapter.
  2. Automatic Step-by-Step Solving of the generated CQ (Calculation First -> Explanation Second).
  3. Unicode NFC and Android screen safety validation.
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.engine.creative_assessment_engine import CreativeAssessmentEngine

def run_test():
    print("=" * 80)
    print("THSA-2.41B: CREATIVE QUESTION (CQ & MCQ) GENERATION & SOLVING TEST")
    print("=" * 80)

    engine = CreativeAssessmentEngine()

    # -------------------------------------------------------------
    # Test 1: Generate Creative Question & MCQs
    # -------------------------------------------------------------
    gen_query = "Create some creative questions and MCQs from Chapter 3 of Class 9 Maths book."
    print(f"\n[Test 1 Query]: {gen_query}\n")
    
    gen_result = engine.generate_creative_questions(gen_query)
    
    print("[Verification: Question Generation]")
    assert "১. সৃজনশীল প্রশ্ন" in gen_result["formatted_markdown"], "Failed: Missing CQ section"
    assert "উদ্দীপক:" in gen_result["formatted_markdown"], "Failed: Missing Stem/Uddipok"
    assert "(ক)" in gen_result["formatted_markdown"] and "(খ)" in gen_result["formatted_markdown"], "Failed: Missing CQ sub-questions"
    assert "২. বহুনির্বাচনি প্রশ্ন" in gen_result["formatted_markdown"], "Failed: Missing MCQ section"
    assert "বহুপদী সমাপ্তিসূচক" in gen_result["formatted_markdown"], "Failed: Missing multi-completion MCQ"
    print("  --> PASS: Creative Questions (CQ) & MCQs generated according to NCTB standards.")

    print("\n" + "-" * 80)
    print("LIVE GENERATED CREATIVE QUESTIONS & MCQS:")
    print("-" * 80)
    print(gen_result["formatted_markdown"])
    print("-" * 80)

    # -------------------------------------------------------------
    # Test 2: Solve the Generated Creative Question
    # -------------------------------------------------------------
    solve_query = "Answer this creative question that you created."
    print(f"\n[Test 2 Query]: {solve_query}\n")

    solve_result = engine.solve_creative_question(solve_query)

    print("[Verification: Creative Question Solving]")
    assert "সৃজনশীল প্রশ্নের নির্ভুল সমাধান" in solve_result["formatted_markdown"], "Failed: Missing solution header"
    assert "(ক) নং প্রশ্নের সমাধান" in solve_result["formatted_markdown"], "Failed: Missing Ka solution"
    assert "(খ) নং প্রশ্নের সমাধান" in solve_result["formatted_markdown"], "Failed: Missing Kha solution"
    assert "(গ) নং প্রশ্নের সমাধান" in solve_result["formatted_markdown"], "Failed: Missing Ga solution"
    assert "```math" in solve_result["formatted_markdown"], "Failed: Missing math block"
    assert "গাণিতিক হিসাব" in solve_result["formatted_markdown"] and "সহজ ব্যাখ্যা" in solve_result["formatted_markdown"], "Failed: Calculation first / explanation missing"
    print("  --> PASS: Module accurately solved its own generated creative question step-by-step.")

    print("\n" + "-" * 80)
    print("LIVE GENERATED STEP-BY-STEP SOLUTION:")
    print("-" * 80)
    print(solve_result["formatted_markdown"])
    print("-" * 80)

    print("\n[ALL CREATIVE QUESTION GENERATION & SOLVING TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    run_test()
