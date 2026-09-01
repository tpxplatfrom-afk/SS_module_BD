"""
THSA-2B Multi-Disciplinary Quantitative Math & Scientific Nomenclature Test Suite
==================================================================================
Tests:
  1. Physics Quantitative Math (Work, Power, Energy & Mechanical Conservation)
  2. Chemistry Quantitative Math (Molarity Calculation & Acid-Base Titration)
  3. Biology Scientific Names & Linnaean Taxonomy Lookup (ইলিশ, রুই মাছ, কাঁঠাল, শাপলা, মানুষ, ম্যালেরিয়া)
  4. Calculation-First & Screen-Safe Markdown Format Verification
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.engine.creative_assessment_engine import CreativeAssessmentEngine
from src.engine.scientific_nomenclature_engine import ScientificNomenclatureEngine

def run_test():
    print("=" * 80)
    print("THSA-2.41B: MULTI-DISCIPLINARY MATH & SCIENTIFIC NOMENCLATURE BENCHMARK")
    print("=" * 80)

    cq_engine = CreativeAssessmentEngine()
    tax_engine = ScientificNomenclatureEngine()

    # -------------------------------------------------------------------------
    # 1. Physics Board Assessment Test (Work, Power, Energy)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Physics Board Assessment & Quantitative Math Test")
    phy_query = "Create a board standard CQ from Class 9 Physics Chapter 4 on Work, Power & Energy."
    phy_gen = cq_engine.generate_creative_questions(phy_query)
    print(f"  -> Generated: {phy_gen['subject']} | {phy_gen['chapter']}")
    assert "৫০ কেজি" in phy_gen["formatted_markdown"], "Failed: Stem missing"
    assert "শক্তির মাত্রা" in phy_gen["formatted_markdown"], "Failed: Question missing"
    print("  -> CQ Generation: PASS")

    phy_solve = cq_engine.solve_creative_question(phy_query)
    assert "19600" in phy_solve["formatted_markdown"], "Failed: Kinetic energy 19600 J calculation missing"
    assert "ML^2T^{-2}" in phy_solve["formatted_markdown"], "Failed: Dimension formula missing"
    assert "সংরক্ষণশীলতা" in phy_solve["formatted_markdown"], "Failed: Conservation law analysis missing"
    print("  -> Quantitative Physics Math (19600 J & Conservation): PASS")

    # -------------------------------------------------------------------------
    # 2. Chemistry Quantitative Math & Titration Test
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Chemistry Stoichiometry & Titration Quantitative Math Test")
    chem_query = "Create a board standard CQ from HSC Chemistry 2nd Paper Quantitative Chemistry on Molarity and Titration."
    chem_gen = cq_engine.generate_creative_questions(chem_query)
    print(f"  -> Generated: {chem_gen['subject']} | {chem_gen['chapter']}")
    assert "মোলারিটি" in chem_gen["formatted_markdown"], "Failed: Question missing"
    assert "২৫০ mL" in chem_gen["formatted_markdown"], "Failed: Volume missing"
    print("  -> CQ Generation: PASS")

    chem_solve = cq_engine.solve_creative_question(chem_query)
    assert "0.4" in chem_solve["formatted_markdown"], "Failed: Molarity 0.4 M calculation missing"
    assert "100" in chem_solve["formatted_markdown"], "Failed: Neutralization 100 mL HCl calculation missing"
    print("  -> Quantitative Chemistry Math (0.4 M Molarity & 100 mL Titration): PASS")

    # -------------------------------------------------------------------------
    # 3. Biology Scientific Names & Linnaean Taxonomy Lookup Test
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Scientific Nomenclature (দ্বিপদ নামকরণ) & Taxonomy Test")
    
    species_tests = [
        ("ইলিশ মাছের বৈজ্ঞানিক নাম কী এবং এর বৈশিষ্ট্য কী?", "Tenualosa ilisha", "Chordata"),
        ("রুই মাছের বৈজ্ঞানিক নাম ও শ্রেণিবিন্যাস বলো", "Labeo rohita", "Actinopterygii"),
        ("কাঁঠালের বৈজ্ঞানিক নাম কী?", "Artocarpus heterophyllus", "Moraceae"),
        ("সাদা শাপলার বৈজ্ঞানিক নাম কী?", "Nymphaea nouchali", "Plantae"),
        ("মানুষের বৈজ্ঞানিক নাম কী?", "Homo sapiens", "Mammalia"),
        ("ম্যালেরিয়া পরজীবীর বৈজ্ঞানিক নাম কী?", "Plasmodium vivax", "Apicomplexa")
    ]

    for q, exp_sci, exp_rank in species_tests:
        res = tax_engine.lookup_species(q)
        assert exp_sci.lower() in res["scientific_name"].lower(), f"Failed: {exp_sci} not in {res['scientific_name']}"
        assert exp_rank.lower() in res["formatted_markdown"].lower(), f"Failed: {exp_rank} not in taxonomic profile"
        print(f"  -> Lookup '{res['bangla_name']}' -> *{res['scientific_name']}* ({exp_rank}): ✅ PASS")

    # Print Live Scientific Name Profile
    print("\n" + "=" * 80)
    print("SAMPLE LIVE GENERATED TAXONOMIC PROFILE (ইলিশ মাছ):")
    print("=" * 80)
    sample_profile = tax_engine.lookup_species("ইলিশ মাছের বৈজ্ঞানিক নাম ও শ্রেণিবিন্যাস")
    print(sample_profile["formatted_markdown"])
    print("=" * 80)

    print("\n[ALL MULTI-DISCIPLINARY MATH & SCIENTIFIC NOMENCLATURE TESTS PASSED!]")

if __name__ == "__main__":
    run_test()
