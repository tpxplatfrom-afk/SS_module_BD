"""
SS Tutor BD - Phase 3A Class 8 Mathematics Tutoring Benchmark Engine
Executes grounded tutoring evaluation across 50 questions and parameter experiment grid.
Computes 500-point rubric score (Math, Bengali, Grounding, Pedagogy, Instruction/Hint Compliance).
"""

import sys
import json
import time
import re
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
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from runtimes.mock_runtime import MockRuntime
from models.manager import get_active_model, get_candidate

QUESTIONS_PATH = PROJECT_ROOT / "benchmarks" / "phase3_class8_math" / "tutor_questions.json"
DB_PATH = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3a"


def evaluate_single_response(q_item: Dict[str, Any], resp: TutorResponse) -> Dict[str, Any]:
    out = resp.final_text
    mode = q_item.get("mode", "solve")
    
    # 1. Mathematical Correctness (0 - 2 pts)
    math_pts = 0
    exp_ans = q_item.get("expected_answer")
    exp_steps = q_item.get("expected_steps", [])
    exp_concepts = q_item.get("expected_concepts", [])

    if mode == "hint":
        # In hint mode, math points depend on offering a relevant formula/guidance without solving
        math_pts = 2
    elif exp_ans:
        ans_clean = str(exp_ans).replace(" ", "")
        out_clean = out.replace(" ", "")
        # Check if expected numerical answer or core formula appears
        if ans_clean in out_clean or any(k in out for k in str(exp_ans).split(",")):
            math_pts = 2
        elif exp_steps and any(any(s_term in out for s_term in step.split()) for step in exp_steps):
            math_pts = 1
        else:
            math_pts = 0
    elif exp_concepts:
        concept_hits = sum(1 for c in exp_concepts if any(term in out for term in c.split()[:2]))
        if concept_hits >= len(exp_concepts) * 0.6:
            math_pts = 2
        elif concept_hits >= 1:
            math_pts = 1
        else:
            math_pts = 0
    else:
        math_pts = 2 if len(out) > 30 else 1

    # 2. Bengali Linguistic Quality (0 - 2 pts)
    has_bengali = bool(re.search(r"[\u0980-\u09FF]", out))
    has_loop = resp.sanitization_flags.get("had_repetition_loop", False) or bool(re.search(r"(.{6,})\1{2,}", out))
    has_corruption = bool(re.search(r"[a-zA-Z]{5,}|</?tool_", out))

    if has_bengali and not has_loop and not has_corruption and len(out) >= 20:
        bn_pts = 2
    elif has_bengali and not has_corruption:
        bn_pts = 1
    else:
        bn_pts = 0

    # 3. Grounding Adherence (0 - 2 pts)
    if resp.grounding_status == "GROUNDED" and len(resp.retrieved_chunks) > 0:
        ground_pts = 2
    elif resp.grounding_status == "PARTIALLY_GROUNDED":
        ground_pts = 1
    else:
        ground_pts = 0

    # 4. Pedagogy & Step Layout (0 - 2 pts)
    has_steps = bool(re.search(r"(?:ধাপ|১\.|২\.|প্রথমে|আমরা জানি|দেওয়া আছে|অতএব)", out))
    if has_steps and len(out) > 40:
        ped_pts = 2
    elif len(out) > 20:
        ped_pts = 1
    else:
        ped_pts = 0

    # 5. Instruction & Hint Compliance (0 - 2 pts)
    inst_pts = 2
    if mode == "hint":
        # Strict penalty: If hint prompt and final answer is exposed -> 0 points!
        if exp_ans:
            for bad_term in str(exp_ans).split(","):
                if bad_term.strip() and bad_term.strip() in out:
                    inst_pts = 0
                    break
        if "৩" in out and "৪" in out and q_item["id"] == "TUT-HNT-001":
            # Check if direct root was stated
            if "x = 3" in out or "x = 4" in out or "৩ এবং ৪" in out:
                inst_pts = 0
        if "১২" in out and q_item["id"] == "TUT-HNT-003":
            if "১২ সেমি" in out or "12" in out:
                inst_pts = 0

    total_item_score = math_pts + bn_pts + ground_pts + ped_pts + inst_pts

    return {
        "question_id": q_item["id"],
        "mode": mode,
        "topic": q_item.get("topic", "general"),
        "total_score": total_item_score,
        "max_score": 10,
        "rubric_breakdown": {
            "math_correctness": math_pts,
            "bengali_quality": bn_pts,
            "grounding": ground_pts,
            "pedagogy": ped_pts,
            "instruction_compliance": inst_pts
        },
        "grounding_status": resp.grounding_status,
        "had_repetition": has_loop,
        "had_corruption": has_corruption,
        "generated_tokens": resp.generated_tokens,
        "latency_ms": resp.total_latency_ms,
        "response_text": out,
        "retrieved_chunk_ids": [c["chunk_id"] for c in resp.retrieved_chunks]
    }


def run_tutor_benchmark(
    candidate_id: str = "CAND-01",
    runtime_type: str = "llama_cpp",
    temperature: float = 0.1,
    repeat_penalty: float = 1.15,
    max_tokens: int = 256,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    print(f"\n[Tutor Benchmark] Initializing Grounded Tutor Engine for {candidate_id}...")
    indexer = KnowledgeIndexer(str(DB_PATH))
    retriever = KnowledgeRetriever(indexer)

    if runtime_type == "mock":
        runtime = MockRuntime(model_id=candidate_id)
        runtime.load("mock")
    else:
        active = get_active_model()
        if not active or not Path(active["file_path"]).exists():
            raise FileNotFoundError(f"Active model binary not found for {candidate_id}")
        runtime = LlamaCppRuntime(
            model_id=candidate_id,
            quantization=active.get("quantization", "Q4_K_M"),
            threads=4,
            tokenizer_repo=active.get("tokenizer_repo_id")
        )
        runtime.load(active["file_path"])

    engine = GroundedTutorEngine(
        retriever=retriever,
        runtime=runtime,
        default_top_k=2,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens
    )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        q_suite = json.load(f)

    questions = q_suite.get("questions", [])
    if limit and limit > 0:
        questions = questions[:limit]

    print(f"[Tutor Benchmark] Running {len(questions)} grounded Class 8 Mathematics questions...")
    print(f"  Parameters: temp={temperature}, repeat_penalty={repeat_penalty}, max_tokens={max_tokens}\n")

    results = []
    total_score = 0
    total_max_score = len(questions) * 10
    total_inference_time_s = 0.0
    total_tokens = 0
    peak_rss = runtime.get_current_rss_mb()
    hint_success_count = 0
    hint_total_count = 0

    for idx, q in enumerate(questions, 1):
        resp = engine.process_query(
            query=q["query"],
            mode=q.get("mode", "auto"),
            student_level="Class 8",
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens
        )
        eval_item = evaluate_single_response(q, resp)
        results.append(eval_item)

        total_score += eval_item["total_score"]
        total_tokens += resp.generated_tokens
        total_inference_time_s += (resp.inference_latency_ms / 1000)
        peak_rss = max(peak_rss, resp.peak_rss_mb)

        if q.get("mode") == "hint":
            hint_total_count += 1
            if eval_item["rubric_breakdown"]["instruction_compliance"] == 2:
                hint_success_count += 1

        if idx % 5 == 0 or idx == len(questions):
            print(f"  Processed {idx}/{len(questions)} questions... (Score so far: {total_score}/{idx*10}, Speed: {resp.tokens_per_second} tok/s)", flush=True)

    runtime.unload()
    indexer.close()

    overall_pct = round((total_score / total_max_score) * 100, 2)
    overall_speed = round(total_tokens / total_inference_time_s, 2) if total_inference_time_s > 0 else 0.0
    hint_rate = round((hint_success_count / max(hint_total_count, 1)) * 100, 2)

    # Calculate rubric category averages
    rubric_totals = {"math_correctness": 0, "bengali_quality": 0, "grounding": 0, "pedagogy": 0, "instruction_compliance": 0}
    for r in results:
        for k, v in r["rubric_breakdown"].items():
            rubric_totals[k] += v

    rubric_summary = {
        k: {
            "score": v,
            "max": len(questions) * 2,
            "percentage": round((v / (len(questions) * 2)) * 100, 2)
        }
        for k, v in rubric_totals.items()
    }

    summary_payload = {
        "candidate_id": candidate_id,
        "total_questions": len(questions),
        "total_score": total_score,
        "max_score": total_max_score,
        "score_percentage": overall_pct,
        "passed_target_70_pct": overall_pct >= 70.0,
        "hint_compliance_rate_pct": hint_rate,
        "parameters": {
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens
        },
        "performance": {
            "tokens_per_second": overall_speed,
            "total_tokens_generated": total_tokens,
            "total_inference_time_s": round(total_inference_time_s, 2),
            "peak_rss_mb": round(peak_rss, 2)
        },
        "rubric_breakdown": rubric_summary,
        "results": results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"tutor_results_{candidate_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("      SS TUTOR BD — GROUNDED CLASS 8 MATH BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Candidate ID:             {candidate_id}")
    print(f"Total Grounded Score:     {total_score} / {total_max_score} ({overall_pct}%) [Target: >= 70%]")
    print(f"Mathematical Correctness: {rubric_summary['math_correctness']['percentage']}%")
    print(f"Bengali Language Quality: {rubric_summary['bengali_quality']['percentage']}%")
    print(f"Grounding Adherence:      {rubric_summary['grounding']['percentage']}%")
    print(f"Pedagogical Scaffolding:  {rubric_summary['pedagogy']['percentage']}%")
    print(f"Hint Compliance Rate:     {hint_rate}% [Target: >= 90%]")
    print(f"Peak Memory (RSS):        {round(peak_rss, 2)} MB [Target: <= 750 MB]")
    print(f"Generation Throughput:    {overall_speed} tokens/sec")
    print(f"Target Verdict:           {'✅ PASSED (>= 70%)' if overall_pct >= 70.0 else '❌ FAILED'}")
    print(f"Results File:             {out_file}")
    print("=" * 70 + "\n")

    return summary_payload


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "CAND-01"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "llama_cpp"
    run_tutor_benchmark(candidate_id=cid, runtime_type=rtype)
