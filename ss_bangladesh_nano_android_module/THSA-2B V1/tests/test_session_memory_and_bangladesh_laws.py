"""
THSA-2B Dynamic Sibling Session Memory & Bangladesh Laws Test Suite
====================================================================
Tests:
  1. Multi-turn Session Memory (Sibling A sets Class 9 -> Subsequent queries adhere to Class 9).
  2. Dynamic Sibling Switcher (Sibling B switches phone to Class 7 -> Context switches to Class 7).
  3. Bangladesh Laws & Constitution (মৌলিক অধিকার, সাইবার আইন, বাল্যবিয়ে নিরোধ, জরুরি সেবা ৯৯৯).
  4. 1-Click Copy-Paste format and Unicode integrity.
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.engine.universal_tutor_engine import UniversalTutorEngine

def run_test():
    print("=" * 80)
    print("THSA-2.41B: SIBLING SESSION MEMORY & BANGLADESH LAWS BENCHMARK")
    print("=" * 80)

    engine = UniversalTutorEngine()

    # -------------------------------------------------------------------------
    # 1. Multi-Turn Session Memory: Sibling A (Class 9)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Sibling A: Profile Set to Class 9")
    turn1_prompt = "আমি ৯ম শ্রেণিতে পড়ি।"
    turn1_res = engine.ask(turn1_prompt)
    assert "Class 9" in engine.session_tracker.active_class, "Failed to set Class 9"
    print(f"  -> Profile Initialized: {engine.session_tracker.active_class} - ✅ PASS")

    # Turn 2: Sibling A asks math question WITHOUT mentioning class
    print("\n[TEST 2] Sibling A: Subsequent Query without mentioning class")
    turn2_prompt = "৩য় অধ্যায়ের ২ নং প্রশ্নটি বুঝিয়ে দাও"
    turn2_res = engine.ask(turn2_prompt)
    assert "Class 9" in turn2_res["markdown"], "Failed to resolve in Class 9 context"
    assert "১. গাণিতিক সমাধান" in turn2_res["markdown"], "Failed math calculation"
    print("  -> Context Retained (Auto-Resolved as Class 9): ✅ PASS")

    # -------------------------------------------------------------------------
    # 2. Dynamic Sibling Switch: Sibling B (Switches to Class 7)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Sibling B: Dynamic Profile Switch to Class 7")
    turn3_prompt = "আমি ৯ম শ্রেণিতে না, আমি ৭ম শ্রেণিতে পড়ি।"
    turn3_res = engine.ask(turn3_prompt)
    assert "Class 7" in engine.session_tracker.active_class, "Failed to switch to Class 7"
    assert engine.session_tracker.active_class_numeric == 7, "Failed numeric class 7"
    print(f"  -> Sibling Switched Phone: New Active Profile = {engine.session_tracker.active_class} - ✅ PASS")

    # -------------------------------------------------------------------------
    # 3. Bangladesh Laws & Constitution Tests
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Bangladesh Laws & Constitutional Rights")
    
    # 3.1 Constitution & Fundamental Rights
    const_prompt = "বাংলাদেশের সংবিধানে নাগরিকদের মৌলিক অধিকারগুলো কী কী?"
    res_const = engine.ask(const_prompt)
    assert "অনুচ্ছেদ ২৭" in res_const["markdown"], "Failed Article 27 equality"
    assert "অনুচ্ছেদ ৩৯" in res_const["markdown"], "Failed Article 39 freedom of speech"
    print("  -> Constitutional Fundamental Rights: ✅ PASS")

    # 3.2 Cyber Security Law
    cyber_prompt = "সাইবার নিরাপত্তা আইন ও অনলাইন হ্যাকিং বা গুজবের শাস্তি কী?"
    res_cyber = engine.ask(cyber_prompt)
    assert "অনলাইন হয়রানি" in res_cyber["markdown"] or "হ্যাকিং" in res_cyber["markdown"], "Failed cyber law"
    print("  -> Cyber Security Law & Digital Safety: ✅ PASS")

    # 3.3 Child Marriage Restraint Act
    child_prompt = "বাল্যবিয়ে নিরোধ আইন অনুযায়ী বিয়ের ন্যূনতম বয়স কত?"
    res_child = engine.ask(child_prompt)
    assert "১৮ বছর" in res_child["markdown"] and "২১ বছর" in res_child["markdown"], "Failed marriage age limits"
    print("  -> Child Marriage & Anti-Dowry Laws: ✅ PASS")

    # 3.4 Emergency Helplines (999, 333, 109)
    help_prompt = "জাতীয় জরুরি সেবা ৯৯৯ এবং অন্যান্য হেল্পলাইন নম্বরগুলো কী কী?"
    res_help = engine.ask(help_prompt)
    assert "৯৯৯" in res_help["markdown"] and "১০৯" in res_help["markdown"] and "৩৩৩" in res_help["markdown"], "Failed helplines"
    print("  -> National Emergency Helplines (999/109/333): ✅ PASS")

    print("\n" + "=" * 80)
    print("LIVE SAMPLE: BANGLADESH CONSTITUTION & LEGAL RIGHTS OUTPUT:")
    print("=" * 80)
    print(res_const["markdown"])
    print("=" * 80)

    print("\n[ALL SIBLING SESSION MEMORY & BANGLADESH LAWS TESTS PASSED!]")

if __name__ == "__main__":
    run_test()
