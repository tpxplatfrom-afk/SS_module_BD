"""
SS Tutor BD - 100-Point Scoring Engine & Sequential Gate Validator
Evaluates benchmark outputs against deterministic and heuristic educational rules.
"""

from typing import Dict, Any, List, Optional
import json
import re


def detect_repetition_loop(text: str) -> bool:
    """Detects repetitive degeneration loops in model output."""
    if not text or len(text) < 30:
        return False
    # Check for repeated substring sequences (e.g. phrases repeating 3+ times)
    if re.search(r"(.{6,})\1{2,}", text):
        return True
    # Check if a single line or sentence repeats multiple times
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 3 and len(set(lines)) <= len(lines) // 2:
        return True
    return False


def score_bengali_quality(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scores Category A (Bengali Language & Dialogue): Max 20 Points."""
    bn_tests = [t for t in test_results if t["id"].startswith("BN-")]
    if not bn_tests:
        return {"score": 0.0, "max_score": 20.0, "percentage": 0.0, "details": "No BN tests found"}

    passed_count = 0
    total_count = len(bn_tests)

    for test in bn_tests:
        output = test.get("output", "")
        has_bengali = bool(re.search(r"[\u0980-\u09FF]", output))
        is_looping = detect_repetition_loop(output)
        
        keywords = test.get("expected_keywords", [])
        kw_matches = sum(1 for kw in keywords if kw in output) if keywords else 1
        kw_ratio = kw_matches / len(keywords) if keywords else 1.0

        neg_pass = True
        if "no_english_sentences" in test.get("negative_constraints", []):
            eng_words = len(re.findall(r"[a-zA-Z]{3,}", output))
            if eng_words > 3:
                neg_pass = False

        if has_bengali and (kw_ratio >= 0.5) and neg_pass and not is_looping and len(output) > 10:
            test["passed"] = True
            test["item_score"] = 1.0
            passed_count += 1
        else:
            test["passed"] = False
            test["item_score"] = 0.0
            reasons = []
            if is_looping:
                reasons.append("Repetitive degeneration loop")
            if not has_bengali:
                reasons.append("Missing Bengali unicode")
            if kw_ratio < 0.5:
                reasons.append("Missing required pedagogical keywords")
            if not neg_pass:
                reasons.append("Failed negative constraint")
            test["failure_reason"] = ", ".join(reasons) if reasons else "Sub-optimal quality"

    ratio = passed_count / total_count
    score = round(ratio * 20.0, 2)
    return {
        "score": score,
        "max_score": 20.0,
        "percentage": round(ratio * 100, 1),
        "passed_items": passed_count,
        "total_items": total_count
    }


def score_educational_reasoning(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scores Category B & C (Mathematics & Science Reasoning): Max 25 Points."""
    math_sci_tests = [t for t in test_results if t["id"].startswith("MATH-") or t["id"].startswith("SCI-")]
    if not math_sci_tests:
        return {"score": 0.0, "max_score": 25.0, "percentage": 0.0, "details": "No MATH/SCI tests"}

    passed_count = 0
    total_count = len(math_sci_tests)

    for test in math_sci_tests:
        output = test.get("output", "")
        exp_ans = test.get("expected_answer")
        exp_concepts = test.get("expected_concepts", [])
        is_looping = detect_repetition_loop(output)

        is_valid = False
        if not is_looping:
            if exp_ans:
                ans_clean = str(exp_ans).replace(" ", "")
                out_clean = output.replace(" ", "")
                if ans_clean in out_clean or any(k in output for k in str(exp_ans).split()):
                    is_valid = True
            elif exp_concepts:
                matches = sum(1 for c in exp_concepts if any(part in output for part in c.split()))
                if matches >= 1:
                    is_valid = True
            else:
                is_valid = len(output) > 20

        if is_valid:
            test["passed"] = True
            test["item_score"] = 1.0
            passed_count += 1
        else:
            test["passed"] = False
            test["item_score"] = 0.0
            test["failure_reason"] = "Repetitive degeneration loop" if is_looping else "Incorrect reasoning or missing answer"

    ratio = passed_count / total_count
    score = round(ratio * 25.0, 2)
    return {
        "score": score,
        "max_score": 25.0,
        "percentage": round(ratio * 100, 1),
        "passed_items": passed_count,
        "total_items": total_count
    }


def score_mobile_efficiency(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Scores Mobile Resource Efficiency: Max 20 Points."""
    peak_rss = metrics.get("peak_rss_mb", 999.0)
    tok_per_sec = metrics.get("tokens_per_second", 0.0)

    ram_pts = 0.0
    if peak_rss <= 500.0:
        ram_pts = 10.0
    elif peak_rss <= 650.0:
        ram_pts = 8.0
    elif peak_rss <= 750.0:
        ram_pts = 5.0
    else:
        ram_pts = 0.0

    speed_pts = 0.0
    if tok_per_sec >= 12.0:
        speed_pts = 10.0
    elif tok_per_sec >= 8.0:
        speed_pts = 8.0
    elif tok_per_sec >= 4.0:
        speed_pts = 6.0
    elif tok_per_sec >= 2.0:
        speed_pts = 3.0
    else:
        speed_pts = 1.0

    score = round(ram_pts + speed_pts, 2)
    return {
        "score": score,
        "max_score": 20.0,
        "percentage": round((score / 20.0) * 100, 1),
        "ram_component_pts": ram_pts,
        "speed_component_pts": speed_pts,
        "peak_rss_mb": peak_rss,
        "tokens_per_sec": tok_per_sec
    }


def score_instruction_following(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scores Category D (Pedagogical & Instruction Following): Max 15 Points."""
    ped_tests = [t for t in test_results if t["id"].startswith("PED-")]
    if not ped_tests:
        return {"score": 0.0, "max_score": 15.0, "percentage": 0.0}

    passed_count = 0
    total_count = len(ped_tests)

    for test in ped_tests:
        output = test.get("output", "")
        neg_constraints = test.get("negative_constraints", [])
        is_looping = detect_repetition_loop(output)
        
        passed_neg = True
        if "do_not_reveal_final_answer_x_equals_2_or_3" in neg_constraints:
            if "x = 2" in output or "x = 3" in output or "২ এবং ৩" in output:
                passed_neg = False
        if "do_not_answer_with_simple_yes_no" in neg_constraints:
            if output.strip() in ["হ্যাঁ", "না", "Yes", "No"]:
                passed_neg = False

        if passed_neg and not is_looping and len(output) > 15:
            test["passed"] = True
            test["item_score"] = 1.0
            passed_count += 1
        else:
            test["passed"] = False
            test["item_score"] = 0.0
            test["failure_reason"] = "Repetitive degeneration loop" if is_looping else "Violated negative pedagogical constraint"

    ratio = passed_count / total_count
    score = round(ratio * 15.0, 2)
    return {
        "score": score,
        "max_score": 15.0,
        "percentage": round(ratio * 100, 1),
        "passed_items": passed_count,
        "total_items": total_count
    }


def score_grounding_and_hallucination(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scores Category E (Knowledge Grounding & Anti-Hallucination): Max 10 Points."""
    ground_tests = [t for t in test_results if t["id"].startswith("GROUND-")]
    if not ground_tests:
        return {"score": 0.0, "max_score": 10.0, "percentage": 0.0}

    passed_count = 0
    total_count = len(ground_tests)

    for test in ground_tests:
        output = test.get("output", "")
        is_looping = detect_repetition_loop(output)
        passed = False
        
        if not is_looping:
            if test["id"] == "GROUND-002":
                if "নেই" in output or "নাই" in output or "১৭টি" in output or "17" in output:
                    passed = True
            elif test["id"] == "GROUND-003":
                if "নেই" in output or "অপরিচিত" in output or "পাওয়া যায়নি" in output or "কোনো" in output:
                    passed = True
            elif test["id"] == "GROUND-004":
                if "অসম্পূর্ণ" in output or "প্রস্থ" in output or "মান প্রয়োজন" in output or "দেওয়া নেই" in output:
                    passed = True
            elif test["id"] == "GROUND-005":
                if "3.14" in output or "২২/৭" in output or "সঠিক নয়" in output:
                    passed = True
            else:
                if len(output) > 15:
                    passed = True

        if passed:
            test["passed"] = True
            test["item_score"] = 1.0
            passed_count += 1
        else:
            test["passed"] = False
            test["item_score"] = 0.0
            test["failure_reason"] = "Repetitive loop" if is_looping else "Hallucination or missed grounding context"

    ratio = passed_count / total_count
    score = round(ratio * 10.0, 2)
    return {
        "score": score,
        "max_score": 10.0,
        "percentage": round(ratio * 100, 1),
        "passed_items": passed_count,
        "total_items": total_count
    }


def score_license_freedom(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Scores License & Redistribution Freedom: Max 10 Points."""
    status = candidate.get("license_status", "")
    lic_type = candidate.get("license", "")

    if status == "LICENSE_PASSED" and ("Apache" in lic_type or "MIT" in lic_type):
        score = 10.0
    elif status == "LICENSE_PASSED":
        score = 8.0
    elif status == "REQUIRES_VERIFICATION":
        score = 4.0
    else:
        score = 0.0

    return {
        "score": score,
        "max_score": 10.0,
        "license": lic_type,
        "status": status
    }


def calculate_composite_scorecard(
    candidate: Dict[str, Any],
    test_results: List[Dict[str, Any]],
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes the full 100-point scorecard and validates sequential gates."""
    bn_score = score_bengali_quality(test_results)
    reason_score = score_educational_reasoning(test_results)
    mobile_score = score_mobile_efficiency(metrics)
    instruct_score = score_instruction_following(test_results)
    ground_score = score_grounding_and_hallucination(test_results)
    lic_score = score_license_freedom(candidate)

    total_score = round(
        bn_score["score"] +
        reason_score["score"] +
        mobile_score["score"] +
        instruct_score["score"] +
        ground_score["score"] +
        lic_score["score"],
        2
    )

    # Load configured gate thresholds
    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.json"
    conf_gates = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                conf_gates = json.load(f).get("sequential_gates", {})
        except Exception:
            pass

    th_gate1 = conf_gates.get("gate_1_license", "LICENSE_PASSED")
    th_gate2 = conf_gates.get("gate_2_bengali_min_score", 12.0)
    th_gate3 = conf_gates.get("gate_3_reasoning_min_score", 15.0)
    th_gate4 = conf_gates.get("gate_4_max_peak_rss_mb", 750.0)
    th_gate5 = conf_gates.get("gate_5_min_tokens_per_sec", 4.0)
    th_gate6 = conf_gates.get("gate_6_min_total_score", 70.0)

    gate_1 = (candidate.get("license_status") == th_gate1)
    gate_2 = (bn_score["score"] >= th_gate2)
    gate_3 = (reason_score["score"] >= th_gate3)
    gate_4 = (metrics.get("peak_rss_mb", 999.0) <= th_gate4)
    gate_5 = (metrics.get("tokens_per_second", 0.0) >= th_gate5)
    gate_6 = (total_score >= th_gate6)

    all_gates_passed = (gate_1 and gate_2 and gate_3 and gate_4 and gate_5 and gate_6)

    return {
        "total_score": total_score,
        "max_total": 100.0,
        "passing_threshold": th_gate6,
        "passed": total_score >= th_gate6,
        "all_gates_passed": all_gates_passed,
        "breakdown": {
            "bengali_linguistic_quality": bn_score,
            "educational_reasoning": reason_score,
            "mobile_resource_efficiency": mobile_score,
            "instruction_constraint_following": instruct_score,
            "knowledge_grounding_anti_hallucination": ground_score,
            "license_redistribution_freedom": lic_score
        },
        "gates": {
            "gate_1_license": {"status": "PASS" if gate_1 else "FAIL", "required": th_gate1},
            "gate_2_bengali": {"status": "PASS" if gate_2 else "FAIL", "score": bn_score["score"], "threshold": th_gate2},
            "gate_3_reasoning": {"status": "PASS" if gate_3 else "FAIL", "score": reason_score["score"], "threshold": th_gate3},
            "gate_4_memory": {"status": "PASS" if gate_4 else "FAIL", "peak_rss_mb": metrics.get("peak_rss_mb"), "threshold_max": th_gate4},
            "gate_5_speed": {"status": "PASS" if gate_5 else "FAIL", "tok_per_sec": metrics.get("tokens_per_second"), "threshold_min": th_gate5},
            "gate_6_total_score": {"status": "PASS" if gate_6 else "FAIL", "total_score": total_score, "threshold_min": th_gate6}
        }
    }
