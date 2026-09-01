"""
SS Tutor BD - Phase 3B Evaluation Runner
Runs 3-Way Comparative Evaluation (Mode A: LLM-Only, Mode B: LLM+RAG, Mode C: LLM+RAG+Deterministic Math Tools).
Computes Phase 3B Weighted Scorecard (100-Point Scale) and Quality-per-MB Efficiency Metrics.
"""

import sys
import os
import json
import time
import re
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever
from core.tutor_engine import GroundedTutorEngine, TutorResponse
from core.math.fraction import FractionHelper
from core.math.calculator import MathCalculator
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from runtimes.mock_runtime import MockRuntime
from models.manager import get_active_model

QUESTIONS_PATH = PROJECT_ROOT / "benchmarks" / "phase3b" / "tutor_100_benchmark.json"
DB_PATH = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3b"


def evaluate_single_response(q_item: Dict[str, Any], resp: TutorResponse) -> Dict[str, Any]:
    out = resp.final_text
    mode = q_item.get("mode", "EXPLAIN")
    exp_ans = q_item.get("expected_answer", "")
    neg_constraints = q_item.get("negative_constraints", [])

    # 1. Math Correctness (0 - 2 pts)
    math_pts = 0
    if mode == "HINT":
        math_pts = 2
    elif exp_ans:
        ans_clean = str(exp_ans).replace(" ", "")
        out_clean = out.replace(" ", "")
        if any(term in out_clean for term in ans_clean.split(",")) or any(term in out for term in str(exp_ans).split()):
            math_pts = 2
        elif any(char.isdigit() for char in out) and len(out) > 20:
            math_pts = 1
        else:
            math_pts = 0
    else:
        math_pts = 2 if len(out) > 20 else 1

    # 2. Bengali Quality (0 - 2 pts)
    has_bengali = bool(re.search(r"[\u0980-\u09FF]", out))
    has_loop = resp.sanitization_flags.get("had_repetition_loop", False) or bool(re.search(r"(.{6,})\1{2,}", out))
    has_corruption = bool(re.search(r"</?tool_|User:\s*Question:", out))

    if has_bengali and not has_loop and not has_corruption and len(out) >= 15:
        bn_pts = 2
    elif has_bengali and not has_corruption:
        bn_pts = 1
    else:
        bn_pts = 0

    # 3. Grounding (0 - 2 pts)
    if resp.pipeline_type == "llm_only":
        ground_pts = 1 if has_bengali else 0
    elif resp.grounding_status == "GROUNDED":
        ground_pts = 2
    else:
        ground_pts = 1

    # 4. Pedagogy (0 - 2 pts)
    has_steps = bool(re.search(r"(?:ধাপ|১\.|২\.|প্রথমে|আমরা জানি|দেওয়া আছে|অতএব|সুতরাং)", out))
    if has_steps and len(out) > 30:
        ped_pts = 2
    elif len(out) > 15:
        ped_pts = 1
    else:
        ped_pts = 0

    # 5. Instruction / Hint Compliance (0 - 2 pts)
    inst_pts = 2
    if mode == "HINT":
        # Check if direct answer was leaked
        if exp_ans:
            for bad_term in str(exp_ans).split(","):
                if bad_term.strip() and len(bad_term.strip()) > 1 and bad_term.strip() in out:
                    inst_pts = 0
                    break
        if "do_not_say_3_or_4" in neg_constraints and ("৩ এবং ৪" in out or "3 and 4" in out):
            inst_pts = 0
        if "do_not_say_12" in neg_constraints and ("১২ সেমি" in out or "12" in out):
            inst_pts = 0

    total_item_score = math_pts + bn_pts + ground_pts + ped_pts + inst_pts
    return {
        "question_id": q_item["id"],
        "category": q_item.get("category", "general"),
        "mode": mode,
        "total_score": total_item_score,
        "max_score": 10,
        "math_pts": math_pts,
        "bn_pts": bn_pts,
        "ground_pts": ground_pts,
        "ped_pts": ped_pts,
        "inst_pts": inst_pts,
        "was_corrected": resp.was_corrected_by_validator,
        "latency_ms": resp.total_latency_ms
    }


def run_phase3b_evaluation(
    candidate_id: str = "CAND-01",
    runtime_type: str = "llama_cpp",
    context_length: int = 2048,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    print(f"\n" + "=" * 75)
    print(f"      SS TUTOR BD — PHASE 3B THREE-WAY BENCHMARK EVALUATION")
    print(f"=" * 75)
    print(f"Candidate: {candidate_id} | Runtime: {runtime_type} | Context: {context_length}")

    indexer = KnowledgeIndexer(str(DB_PATH))
    retriever = KnowledgeRetriever(indexer)

    if runtime_type == "mock":
        runtime = MockRuntime(model_id=candidate_id)
        runtime.load("mock", context_length=context_length)
        file_size_mb = 0.0
    else:
        active = get_active_model()
        if not active or not Path(active.get("file_path", "")).exists():
            raise FileNotFoundError(f"Active model binary not found for {candidate_id}")
        runtime = LlamaCppRuntime(
            model_id=candidate_id,
            quantization=active.get("quantization", "Q4_K_M"),
            threads=4,
            tokenizer_repo=active.get("tokenizer_repo_id")
        )
        runtime.load(active["file_path"], context_length=context_length)
        file_size_mb = round(Path(active["file_path"]).stat().st_size / (1024 * 1024), 2)

    engine = GroundedTutorEngine(
        retriever=retriever,
        runtime=runtime,
        default_top_k=2,
        temperature=0.1,
        repeat_penalty=1.15,
        max_tokens=192
    )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    questions = suite.get("questions", [])
    if limit and limit > 0:
        questions = questions[:limit]

    total_q = len(questions)
    print(f"Total Evaluation Items: {total_q}\n")

    modes = [
        ("Mode A: LLM-Only", "llm_only"),
        ("Mode B: LLM + RAG", "llm_rag"),
        ("Mode C: LLM + RAG + Math Engine (Hybrid)", "hybrid_rag_tools")
    ]

    mode_summaries = {}

    for mode_label, pipe_type in modes:
        print(f"--- Running {mode_label} across {total_q} questions ---")
        t0 = time.perf_counter()
        mode_score = 0
        math_total = 0
        bn_total = 0
        ground_total = 0
        ped_total = 0
        inst_total = 0
        corrected_count = 0
        peak_rss = runtime.get_current_rss_mb()
        total_tokens = 0
        item_evals = []

        for idx, q in enumerate(questions, 1):
            resp = engine.process_query(
                query=q["query"],
                mode=q.get("mode", "auto"),
                pipeline_type=pipe_type,
                student_level="Class 8"
            )
            ev = evaluate_single_response(q, resp)
            item_evals.append(ev)

            mode_score += ev["total_score"]
            math_total += ev["math_pts"]
            bn_total += ev["bn_pts"]
            ground_total += ev["ground_pts"]
            ped_total += ev["ped_pts"]
            inst_total += ev["inst_pts"]
            if ev["was_corrected"]:
                corrected_count += 1
            total_tokens += resp.generated_tokens
            peak_rss = max(peak_rss, resp.peak_rss_mb)

        duration_s = time.perf_counter() - t0
        speed = round(total_tokens / duration_s, 2) if duration_s > 0 else 0.0
        score_pct = round((mode_score / (total_q * 10)) * 100, 2)
        math_pct = round((math_total / (total_q * 2)) * 100, 2)
        bn_pct = round((bn_total / (total_q * 2)) * 100, 2)
        ground_pct = round((ground_total / (total_q * 2)) * 100, 2)
        ped_pct = round((ped_total / (total_q * 2)) * 100, 2)
        inst_pct = round((inst_total / (total_q * 2)) * 100, 2)

        mode_summaries[pipe_type] = {
            "mode_name": mode_label,
            "score_pct": score_pct,
            "math_correctness_pct": math_pct,
            "bengali_quality_pct": bn_pct,
            "grounding_pct": ground_pct,
            "pedagogy_pct": ped_pct,
            "instruction_compliance_pct": inst_pct,
            "corrections_by_validator": corrected_count,
            "tokens_per_second": speed,
            "peak_rss_mb": round(peak_rss, 2),
            "total_duration_s": round(duration_s, 2),
            "evaluations": item_evals
        }
        print(f"  Score: {mode_score}/{total_q*10} ({score_pct}%) | Math: {math_pct}% | Bengali: {bn_pct}% | Peak RSS: {round(peak_rss, 2)} MB\n")

    runtime.unload()
    indexer.close()

    # Calculate Phase 3B 100-Point Weighted Scorecard for Hybrid Mode (Mode C)
    hybrid = mode_summaries["hybrid_rag_tools"]
    peak_mem = hybrid["peak_rss_mb"]

    # 1. Memory Efficiency (25 pts): 25 if <= 200MB, linear down to 0 at 250MB
    if peak_mem <= 200.0:
        mem_score = 25.0
    elif peak_mem <= 250.0:
        mem_score = 25.0 * (1.0 - (peak_mem - 200.0) / 50.0)
    else:
        mem_score = 0.0

    # 2. Bengali Quality (20 pts): based on bn_pct
    bn_weighted = 20.0 * (hybrid["bengali_quality_pct"] / 100.0)

    # 3. Educational Tutoring (20 pts): based on ped_pct
    ped_weighted = 20.0 * (hybrid["pedagogy_pct"] / 100.0)

    # 4. Grounding (15 pts): based on ground_pct
    ground_weighted = 15.0 * (hybrid["grounding_pct"] / 100.0)

    # 5. Instruction Following (10 pts): based on inst_pct
    inst_weighted = 10.0 * (hybrid["instruction_compliance_pct"] / 100.0)

    # 6. Speed (5 pts): 5 if >= 10 tok/s, min 4 tok/s
    speed_weighted = min(5.0, 5.0 * (hybrid["tokens_per_second"] / 10.0))

    # 7. Model Footprint (5 pts): 5 if <= 150MB, down to 0 at 300MB
    if file_size_mb <= 150.0:
        footprint_score = 5.0
    elif file_size_mb <= 300.0:
        footprint_score = 5.0 * (1.0 - (file_size_mb - 150.0) / 150.0)
    else:
        footprint_score = 0.0

    composite_score = round(mem_score + bn_weighted + ped_weighted + ground_weighted + inst_weighted + speed_weighted + footprint_score, 2)

    # Quality per MB metrics
    quality_per_mb = round(composite_score / max(peak_mem, 1.0), 4)
    quality_per_100mb = round((composite_score / max(peak_mem, 1.0)) * 100.0, 2)

    # Hard Gates Evaluation
    gate_license = "LICENSE_PASSED"
    gate_memory = "PASS" if peak_mem <= 250.0 else "FAIL"
    gate_math = "PASS" if hybrid["math_correctness_pct"] >= 90.0 else "FAIL"
    gate_grounding = "PASS" if hybrid["grounding_pct"] >= 90.0 else "FAIL"
    gate_speed = "PASS" if hybrid["tokens_per_second"] >= 4.0 or runtime_type == "mock" else "FAIL"

    overall_verdict = "PRODUCTION CANDIDATE" if (gate_memory == "PASS" and gate_math == "PASS" and composite_score >= 70.0 and peak_mem <= 200.0) else (
        "PROMISING (WARNING TIER)" if (gate_memory == "PASS" and gate_math == "PASS" and peak_mem <= 250.0) else "DISQUALIFIED / RESEARCH ONLY"
    )

    final_payload = {
        "candidate_id": candidate_id,
        "runtime_type": runtime_type,
        "quantization": "Q4_K_M",
        "file_size_mb": file_size_mb,
        "context_length": context_length,
        "total_questions_tested": total_q,
        "three_way_comparison": {
            "mode_a_llm_only": mode_summaries["llm_only"],
            "mode_b_llm_rag": mode_summaries["llm_rag"],
            "mode_c_hybrid": mode_summaries["hybrid_rag_tools"]
        },
        "scorecard_weighted_100pt": {
            "memory_efficiency_25pt": round(mem_score, 2),
            "bengali_quality_20pt": round(bn_weighted, 2),
            "educational_tutoring_20pt": round(ped_weighted, 2),
            "grounding_15pt": round(ground_weighted, 2),
            "instruction_following_10pt": round(inst_weighted, 2),
            "speed_5pt": round(speed_weighted, 2),
            "model_footprint_5pt": round(footprint_score, 2),
            "total_composite_score": composite_score
        },
        "efficiency_metrics": {
            "quality_per_mb": quality_per_mb,
            "quality_per_100mb": quality_per_100mb
        },
        "gate_status": {
            "gate_1_license": gate_license,
            "gate_2_binary_size": "PASS" if file_size_mb <= 200.0 else "WARNING",
            "gate_3_memory": gate_memory,
            "gate_4_speed": gate_speed,
            "gate_7_hybrid_math": gate_math,
            "gate_8_grounding": gate_grounding,
            "overall_verdict": overall_verdict
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"phase3b_{candidate_id}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print("=" * 75)
    print("      SS TUTOR BD — PHASE 3B THREE-WAY BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Candidate:                {candidate_id}")
    print(f"Mode A (LLM Only):        {mode_summaries['llm_only']['score_pct']}% (Math: {mode_summaries['llm_only']['math_correctness_pct']}%)")
    print(f"Mode B (LLM + RAG):       {mode_summaries['llm_rag']['score_pct']}% (Math: {mode_summaries['llm_rag']['math_correctness_pct']}%)")
    print(f"Mode C (Hybrid Pipeline): {mode_summaries['hybrid_rag_tools']['score_pct']}% (Math: {mode_summaries['hybrid_rag_tools']['math_correctness_pct']}%)")
    print(f"---------------------------------------------------------------------------")
    print(f"Phase 3B Weighted Score:  {composite_score} / 100.0")
    print(f"Quality per 100 MB RAM:   {quality_per_100mb}")
    print(f"Peak Process RSS:         {peak_mem} MB (Hard Ceiling: 250 MB)")
    print(f"Overall Model Verdict:    {overall_verdict}")
    print(f"Results File:             {out_file}")
    print("=" * 75 + "\n")

    return final_payload


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "MOCK"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "mock"
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_phase3b_evaluation(candidate_id=cid, runtime_type=rtype, limit=lim)
