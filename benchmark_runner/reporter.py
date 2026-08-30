"""
SS Tutor BD - Benchmark Result & Report Generator
Generates canonical JSON artifacts, human-readable Markdown reports, and structured failure logs.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FAILURES_DIR = PROJECT_ROOT / "results" / "failures"


def ensure_directories():
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)


def save_raw_result(candidate_id: str, quantization: str, result_data: Dict[str, Any]) -> Path:
    ensure_directories()
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    filename = f"{candidate_id}_{quantization}_{timestamp_str}.json"
    target_path = RAW_RESULTS_DIR / filename
    
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
        
    return target_path


def extract_and_save_failures(candidate_id: str, test_results: List[Dict[str, Any]]) -> Path:
    ensure_directories()
    failures = []
    for test in test_results:
        if not test.get("passed", True) or test.get("failure_reason"):
            failures.append({
                "test_id": test["id"],
                "category": test.get("category", "UNKNOWN"),
                "prompt": test.get("prompt"),
                "output": test.get("output"),
                "reason": test.get("failure_reason", "Response did not satisfy expected pedagogical/arithmetic criteria"),
                "severity": test.get("severity", "MEDIUM")
            })

    target_path = FAILURES_DIR / f"{candidate_id}_failures.json"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
        
    return target_path


def generate_markdown_report(
    candidate: Dict[str, Any],
    scorecard: Dict[str, Any],
    metrics: Dict[str, Any],
    test_results: List[Dict[str, Any]]
) -> Path:
    ensure_directories()
    cand_id = candidate["id"]
    cand_name = candidate["name"].replace("/", "_")
    quant = candidate.get("quantization", "Q4_K_M")
    report_filename = f"{cand_id}_{cand_name}_{quant}.md"
    report_path = REPORTS_DIR / report_filename

    total_score = scorecard["total_score"]
    all_gates = "PASSED ALL GATES" if scorecard["all_gates_passed"] else "DID NOT PASS ALL GATES"
    
    md_content = f"""# SS Tutor BD — Benchmark Report: {candidate['name']}

**Candidate ID:** `{cand_id}`  
**Model Name:** `{candidate['name']}`  
**Publisher:** {candidate.get('publisher', 'Unknown')}  
**Parameter Count:** {candidate.get('parameters_billion')}B  
**Quantization Tested:** `{quant}`  
**Evaluation Date:** {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Overall Benchmark Status:** **{all_gates} (Total Score: {total_score} / 100)**

---

## 1. Executive Summary & Recommendation

| Assessment Dimension | Result | Target Benchmark Status |
| :--- | :--- | :--- |
| **Composite Score** | **{total_score} / 100** | {'PASS (>= 70)' if total_score >= 70 else 'FAIL (< 70)'} |
| **License Compliance** | `{candidate.get('license_status')}` | {'PASS' if candidate.get('license_status') == 'LICENSE_PASSED' else 'REQUIRES REVIEW'} |
| **Peak Memory (RSS)** | **{metrics.get('peak_rss_mb', 'N/A')} MB** | {'PASS (<= 750 MB)' if metrics.get('peak_rss_mb', 999) <= 750 else 'FAIL (> 750 MB)'} |
| **Generation Speed** | **{metrics.get('tokens_per_second', 'N/A')} tok/s** | {'PASS (>= 4.0 tok/s)' if metrics.get('tokens_per_second', 0) >= 4.0 else 'FAIL (< 4.0)'} |
| **Bengali Linguistic Score** | **{scorecard['breakdown']['bengali_linguistic_quality']['score']} / 20** | {'PASS' if scorecard['breakdown']['bengali_linguistic_quality']['score'] >= 10 else 'NEEDS IMPROVEMENT'} |
| **Educational Reasoning Score** | **{scorecard['breakdown']['educational_reasoning']['score']} / 25** | {'PASS' if scorecard['breakdown']['educational_reasoning']['score'] >= 12 else 'NEEDS IMPROVEMENT'} |

---

## 2. 100-Point Scorecard Breakdown

```
========================================================================================
CATEGORY                                SCORE / MAX        PERCENTAGE    STATUS
========================================================================================
1. Bengali Linguistic Quality           {scorecard['breakdown']['bengali_linguistic_quality']['score']:<6} / 20.0        {scorecard['breakdown']['bengali_linguistic_quality']['percentage']:<6}%     {'PASS' if scorecard['breakdown']['bengali_linguistic_quality']['score'] >= 10 else 'FAIL'}
2. Educational Reasoning (Math/Sci)     {scorecard['breakdown']['educational_reasoning']['score']:<6} / 25.0        {scorecard['breakdown']['educational_reasoning']['percentage']:<6}%     {'PASS' if scorecard['breakdown']['educational_reasoning']['score'] >= 12 else 'FAIL'}
3. Mobile Resource Efficiency           {scorecard['breakdown']['mobile_resource_efficiency']['score']:<6} / 20.0        {scorecard['breakdown']['mobile_resource_efficiency']['percentage']:<6}%     {'PASS' if scorecard['breakdown']['mobile_resource_efficiency']['score'] >= 10 else 'FAIL'}
4. Instruction & Socratic Scaffolding   {scorecard['breakdown']['instruction_constraint_following']['score']:<6} / 15.0        {scorecard['breakdown']['instruction_constraint_following']['percentage']:<6}%     {'PASS' if scorecard['breakdown']['instruction_constraint_following']['score'] >= 8 else 'FAIL'}
5. Knowledge Grounding & Anti-Halluc.   {scorecard['breakdown']['knowledge_grounding_anti_hallucination']['score']:<6} / 10.0        {scorecard['breakdown']['knowledge_grounding_anti_hallucination']['percentage']:<6}%     {'PASS' if scorecard['breakdown']['knowledge_grounding_anti_hallucination']['score'] >= 5 else 'FAIL'}
6. License & Redistribution Freedom     {scorecard['breakdown']['license_redistribution_freedom']['score']:<6} / 10.0        100.0%     PASS
----------------------------------------------------------------------------------------
TOTAL COMPOSITE SCORE                   {total_score:<6} / 100.0       {round((total_score/100)*100, 1)}%      {'PASSED' if total_score >= 70 else 'FAILED'}
========================================================================================
```

---

## 3. Sequential Gate Validation

* **Gate 1 (License Gate):** `{scorecard['gates']['gate_1_license']['status']}` (Declared: `{candidate.get('license')}`)
* **Gate 2 (Bengali Linguistic Gate):** `{scorecard['gates']['gate_2_bengali']['status']}` (Score: {scorecard['gates']['gate_2_bengali']['score']} / 20, Min: 10.0)
* **Gate 3 (Educational Reasoning Gate):** `{scorecard['gates']['gate_3_reasoning']['status']}` (Score: {scorecard['gates']['gate_3_reasoning']['score']} / 25, Min: 12.0)
* **Gate 4 (Mobile Memory Gate):** `{scorecard['gates']['gate_4_memory']['status']}` (Peak RSS: {scorecard['gates']['gate_4_memory']['peak_rss_mb']} MB, Cap: 750 MB)
* **Gate 5 (Mobile Speed Gate):** `{scorecard['gates']['gate_5_speed']['status']}` (Throughput: {scorecard['gates']['gate_5_speed']['tok_per_sec']} tok/s, Min: 4.0 tok/s)
* **Gate 6 (Composite Threshold Gate):** `{scorecard['gates']['gate_6_total_score']['status']}` (Total: {total_score} / 100, Min: 70.0)

---

## 4. Resource & Latency Measurements

* **Model File Size:** `{metrics.get('model_file_size_mb', 'N/A')} MB`
* **Model Load Time:** `{metrics.get('load_time_ms', 'N/A')} ms`
* **Time-to-First-Token (TTFT):** `{metrics.get('ttft_ms', 'N/A')} ms`
* **Average Generation Throughput:** `{metrics.get('tokens_per_second', 'N/A')} tokens/second`
* **Peak Resident Set Size (RSS):** `{metrics.get('peak_rss_mb', 'N/A')} MB`
* **Host Platform:** Windows 10 Pro (x64) — Intel i5-6500 @ 3.20GHz

---

## 5. Sample Evaluation Outputs

### Sample A: Bengali Explanation (BN-002)
> *Prompt:* `ভাইয়া, আমার সূচকের অংকগুলো বুঝতে খুব সমস্যা হচ্ছে। সহজ ভাষায় সূচক (Exponent) কী একটু বুঝিয়ে বলবেন?`  
> *Model Output:*  
> {next((t.get('output') for t in test_results if t['id'] == 'BN-002'), 'N/A')}

### Sample B: Mathematics Step-by-Step (MATH-001)
> *Prompt:* `3/4 + 5/6 এর যোগফল নির্ণয় করো এবং অপ্রকৃত ভগ্নাংশ থেকে মিশ্র ভগ্নাংশে রূপান্তর করো।`  
> *Model Output:*  
> {next((t.get('output') for t in test_results if t['id'] == 'MATH-001'), 'N/A')}

### Sample C: Negative Constraint / Socratic Hint (PED-001)
> *Prompt:* `প্রশ্ন: x^2 - 5x + 6 = 0 এর সমাধান কী? আমাকে কিন্তু সরাসরি উত্তর বলবে না, শুধু ইঙ্গিত দাও।`  
> *Model Output:*  
> {next((t.get('output') for t in test_results if t['id'] == 'PED-001'), 'N/A')}

---

## 6. Failure Analysis Summary

Total Failed / Sub-optimal Test Cases Recorded: **{sum(1 for t in test_results if t.get('score', 1.0) < 0.5)}**  
Detailed failure dump saved to: `results/failures/{cand_id}_failures.json`

---

## 7. Next Architectural Recommendation

Based on this evaluation:
* If the candidate passed all gates, it qualifies for **Phase 2 (Class 8 Math Retrieval & Prototype)**.
* If memory or speed gates failed, evaluate **more aggressive quantization (Q3_K_M / IQ2_M)** or step down to an alternative candidate.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report_path
