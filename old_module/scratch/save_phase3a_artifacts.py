"""
SS Tutor BD - Extract Failures and Performance Summary for Phase 3A
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3a"
TUTOR_RESULTS_FILE = RESULTS_DIR / "tutor_results_CAND-01.json"
RETRIEVAL_RESULTS_FILE = RESULTS_DIR / "retrieval_results.json"
FAILURES_FILE = RESULTS_DIR / "failures.json"
PERF_FILE = RESULTS_DIR / "performance_results.json"

with open(TUTOR_RESULTS_FILE, "r", encoding="utf-8") as f:
    tutor_data = json.load(f)

with open(RETRIEVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
    ret_data = json.load(f)

failures = []
for r in tutor_data.get("results", []):
    if r["total_score"] < 8 or r["rubric_breakdown"]["math_correctness"] < 2 or r["rubric_breakdown"]["bengali_quality"] < 2:
        failures.append({
            "question_id": r["question_id"],
            "mode": r["mode"],
            "topic": r["topic"],
            "total_score": r["total_score"],
            "rubric_breakdown": r["rubric_breakdown"],
            "had_repetition": r["had_repetition"],
            "had_corruption": r["had_corruption"],
            "response_text": r["response_text"]
        })

with open(FAILURES_FILE, "w", encoding="utf-8") as f:
    json.dump({"total_suboptimal_cases": len(failures), "failures": failures}, f, indent=2, ensure_ascii=False)

perf_summary = {
    "candidate_id": "CAND-01",
    "model_name": "Qwen2.5-0.5B-Instruct",
    "quantization": "Q4_K_M",
    "model_file_size_mb": 468.64,
    "peak_rss_mb": tutor_data["performance"]["peak_rss_mb"],
    "target_rss_mb": 750.0,
    "rss_passed": tutor_data["performance"]["peak_rss_mb"] <= 750.0,
    "retrieval": {
        "recall_at_1_pct": ret_data["metrics"]["recall_at_1_pct"],
        "recall_at_3_pct": ret_data["metrics"]["recall_at_3_pct"],
        "recall_at_5_pct": ret_data["metrics"]["recall_at_5_pct"],
        "avg_retrieval_latency_ms": ret_data["metrics"]["avg_latency_ms"]
    },
    "inference": {
        "tokens_per_second": tutor_data["performance"]["tokens_per_second"],
        "total_tokens": tutor_data["performance"]["total_tokens_generated"],
        "total_inference_time_s": tutor_data["performance"]["total_inference_time_s"],
        "avg_time_per_question_s": round(tutor_data["performance"]["total_inference_time_s"] / tutor_data["total_questions"], 2)
    },
    "scorecard": {
        "grounded_tutor_score_pct": tutor_data["score_percentage"],
        "math_correctness_pct": tutor_data["rubric_breakdown"]["math_correctness"]["percentage"],
        "bengali_quality_pct": tutor_data["rubric_breakdown"]["bengali_quality"]["percentage"],
        "grounding_adherence_pct": tutor_data["rubric_breakdown"]["grounding"]["percentage"],
        "pedagogy_pct": tutor_data["rubric_breakdown"]["pedagogy"]["percentage"],
        "hint_compliance_rate_pct": tutor_data["hint_compliance_rate_pct"]
    }
}

with open(PERF_FILE, "w", encoding="utf-8") as f:
    json.dump(perf_summary, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(failures)} sub-optimal cases to {FAILURES_FILE}")
print(f"Saved performance summary to {PERF_FILE}")
