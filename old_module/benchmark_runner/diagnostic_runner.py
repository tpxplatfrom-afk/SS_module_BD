"""
SS Tutor BD - Phase 2 Diagnostic Benchmark Runner
Executes targeted tests for Bengali orthography, repetition resistance, negative constraints,
Socratic recovery, and textbook grounding.
"""

import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIAG_PATH = PROJECT_ROOT / "benchmarks" / "phase2_diagnostics" / "diagnostic_prompts.json"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
RAW_DIR = PROJECT_ROOT / "results" / "raw"

from runtimes.base import ModelRuntime
from runtimes.mock_runtime import MockRuntime
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from models.manager import get_active_model, get_candidate


def load_diagnostic_items() -> List[Dict[str, Any]]:
    with open(DIAG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for cat in data.get("categories", []):
        cat_code = cat.get("category_code")
        cat_name = cat.get("category_name")
        for item in cat.get("items", []):
            item["category_code"] = cat_code
            item["category_name"] = cat_name
            items.append(item)
    return items


def evaluate_diagnostic_item(item: Dict[str, Any], output: str) -> Dict[str, Any]:
    cat_code = item["category_code"]
    has_bengali = bool(re.search(r"[\u0980-\u09FF]", output))
    
    # 1. Control token / Artifact check
    has_tool_leak = bool(re.search(r"</?tool_call>|<\|im_start\|>|<\|im_end\|>|\[a-zA-Z]{5,}", output))
    if "</tool_call>" in output:
        has_tool_leak = True

    # 2. Repetition loop check
    has_repetition = bool(re.search(r"(.{6,})\1{2,}", output))
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    if len(lines) >= 3 and len(set(lines)) <= len(lines) // 2:
        has_repetition = True

    # 3. Expected keywords
    kw_list = item.get("expected_keywords", [])
    kw_hits = sum(1 for kw in kw_list if kw in output) if kw_list else 0
    kw_ratio = (kw_hits / len(kw_list)) if kw_list else 1.0

    # 4. Expected answer
    exp_ans = item.get("expected_answer")
    ans_hit = (str(exp_ans) in output) if exp_ans else True

    # 5. Negative constraints
    neg_pass = True
    neg_reasons = []
    for neg in item.get("negative_constraints", []):
        if neg == "do_not_reveal_x_equals_7" or neg == "do_not_say_7":
            if "x = 7" in output or "x=7" in output or "৭" in output or " 7 " in output:
                neg_pass = False
                neg_reasons.append("Revealed prohibited answer 7")
        elif neg == "must_have_exactly_3_numbered_steps":
            step_hits = len(re.findall(r"ধাপ\s*[১১২৩123]", output))
            if step_hits < 3:
                neg_pass = False
                neg_reasons.append("Missing 3 numbered steps")
        elif neg == "no_english_words":
            eng_matches = re.findall(r"[a-zA-Z]{3,}", output)
            if len(eng_matches) > 1:
                neg_pass = False
                neg_reasons.append(f"Found English words: {eng_matches[:3]}")

    # Calculate pass status
    passed = False
    reasons = []
    if not has_bengali:
        reasons.append("No Bengali text")
    if has_tool_leak:
        reasons.append("Leaked control tokens or excessive English")
    if has_repetition:
        reasons.append("Repetitive degeneration loop")
    if not neg_pass:
        reasons.extend(neg_reasons)
    if kw_list and kw_ratio < 0.5:
        reasons.append(f"Keyword match ratio too low ({kw_hits}/{len(kw_list)})")
    if exp_ans and not ans_hit:
        reasons.append(f"Missing expected answer '{exp_ans}'")

    if not reasons and len(output) > 10:
        passed = True

    return {
        "item_id": item["id"],
        "category_code": cat_code,
        "title": item.get("title", ""),
        "passed": passed,
        "failure_reasons": reasons,
        "has_tool_leak": has_tool_leak,
        "has_repetition": has_repetition,
        "kw_ratio": round(kw_ratio, 2),
        "output": output
    }


def run_diagnostic_suite(candidate_id: str, runtime_type: str = "llama_cpp") -> Dict[str, Any]:
    candidate = get_candidate(candidate_id)
    if not candidate:
        raise ValueError(f"Candidate {candidate_id} not found.")

    if runtime_type == "mock":
        runtime = MockRuntime(model_id=candidate["id"])
        runtime.load("mock")
    else:
        active = get_active_model()
        if not active or not Path(active["file_path"]).exists():
            raise FileNotFoundError(f"Active model binary not found for {candidate_id}")
        runtime = LlamaCppRuntime(
            model_id=candidate["id"],
            quantization=candidate.get("quantization", "Q4_K_M"),
            threads=2,
            tokenizer_repo=candidate.get("tokenizer_repo_id")
        )
        runtime.load(active["file_path"])

    items = load_diagnostic_items()
    print(f"\n[Diagnostic Runner] Running Phase 2 Diagnostics for {candidate['id']} ({len(items)} items)...")

    results = []
    system_prompt = (
        "You are SS Tutor BD, an expert AI tutor for Bangladesh High School (Class 6-10). "
        "Follow all instructions carefully. Answer in clear natural Bengali."
    )

    for idx, item in enumerate(items, 1):
        prompt = item["prompt"]
        if item.get("context"):
            prompt = f"প্রদত্ত পাঠ্যপুস্তক তথ্য:\n{item['context']}\n\nপ্রশ্ন:\n{prompt}"

        gen = runtime.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=256,
            temperature=0.0
        )
        eval_res = evaluate_diagnostic_item(item, gen.text)
        eval_res["latency_s"] = gen.generation_time_s
        eval_res["tokens_per_sec"] = gen.tokens_per_sec
        results.append(eval_res)
        print(f"  [{idx}/{len(items)}] {item['id']} ({item['title']}): {'PASS' if eval_res['passed'] else 'FAIL'}")

    runtime.unload()

    total_passed = sum(1 for r in results if r["passed"])
    pass_pct = round((total_passed / len(results)) * 100, 1)

    # Category summaries
    cat_summary = {}
    for r in results:
        code = r["category_code"]
        if code not in cat_summary:
            cat_summary[code] = {"total": 0, "passed": 0}
        cat_summary[code]["total"] += 1
        if r["passed"]:
            cat_summary[code]["passed"] += 1

    summary_payload = {
        "candidate_id": candidate["id"],
        "total_items": len(results),
        "total_passed": total_passed,
        "pass_percentage": pass_pct,
        "category_summary": cat_summary,
        "results": results
    }

    # Save raw and report
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{candidate['id']}_diagnostics.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    report_path = REPORTS_DIR / f"{candidate['id']}_diagnostics_report.md"
    _generate_diagnostic_report(report_path, candidate, summary_payload)

    print(f"\n[Diagnostics Complete] Pass Rate: {total_passed}/{len(results)} ({pass_pct}%)")
    print(f"  Raw:    {raw_path}")
    print(f"  Report: {report_path}\n")

    return summary_payload


def _generate_diagnostic_report(report_path: Path, candidate: Dict[str, Any], summary: Dict[str, Any]):
    md = f"""# SS Tutor BD — Phase 2 Diagnostic Report: {candidate['name']}

**Candidate ID:** `{candidate['id']}`  
**Model Name:** `{candidate['name']}`  
**Parameters:** {candidate['parameters_billion']}B  
**Diagnostic Items:** {summary['total_items']}  
**Passed Items:** {summary['total_passed']} / {summary['total_items']} ({summary['pass_percentage']}%)  

---

## 1. Category Diagnostic Breakdown

| Category | Category Description | Passed / Total | Pass Rate | Status |
| :--- | :--- | :--- | :--- | :--- |
"""
    for code, data in summary["category_summary"].items():
        rate = round((data["passed"] / data["total"]) * 100, 1)
        status = "PASS" if rate >= 70.0 else "FAIL"
        md += f"| `{code}` | {code} | {data['passed']} / {data['total']} | {rate}% | {status} |\n"

    md += """
---

## 2. Detailed Diagnostic Outputs & Observations

"""
    for r in summary["results"]:
        status_badge = "✅ PASS" if r["passed"] else "❌ FAIL"
        md += f"### {r['item_id']}: {r['title']} — {status_badge}\n\n"
        if r["failure_reasons"]:
            md += f"> **Failure Diagnostic:** {', '.join(r['failure_reasons'])}\n\n"
        md += f"**Model Output:**\n```\n{r['output']}\n```\n\n---\n\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "CAND-02"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "llama_cpp"
    run_diagnostic_suite(cid, rtype)
