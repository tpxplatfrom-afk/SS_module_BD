"""
SS Tutor BD - Generation Parameter Experimentation Grid
Evaluates small controlled configurations (temperature, repeat_penalty)
over a representative 10-question NCTB subset to measure quality vs latency tradeoffs.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever
from core.tutor_engine import GroundedTutorEngine
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from runtimes.mock_runtime import MockRuntime
from models.manager import get_active_model

QUESTIONS_PATH = PROJECT_ROOT / "benchmarks" / "phase3_class8_math" / "tutor_questions.json"
DB_PATH = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3a"

CONFIGS = [
    {"name": "Config A (Baseline)", "temp": 0.0, "repeat_penalty": 1.0},
    {"name": "Config B (Low Temp)", "temp": 0.1, "repeat_penalty": 1.0},
    {"name": "Config C (Repeat Penalty)", "temp": 0.0, "repeat_penalty": 1.15},
    {"name": "Config D (Optimal RAG + Scaffold)", "temp": 0.1, "repeat_penalty": 1.15}
]


def run_parameter_experiments(candidate_id: str = "CAND-01", runtime_type: str = "llama_cpp") -> Dict[str, Any]:
    print(f"\n[Parameter Experiments] Starting Grid Search for {candidate_id}...")
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
            threads=2,
            tokenizer_repo=active.get("tokenizer_repo_id")
        )
        runtime.load(active["file_path"])

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        all_q = json.load(f).get("questions", [])

    # Pick 10 representative questions across modes
    subset_ids = [
        "TUT-EXP-001", "TUT-EXP-003", "TUT-EXP-009",
        "TUT-ARI-001", "TUT-ARI-003", "TUT-ARI-007",
        "TUT-ALG-002", "TUT-ALG-004",
        "TUT-GEO-001", "TUT-HNT-001"
    ]
    test_questions = [q for q in all_q if q["id"] in subset_ids]

    experiment_results = []

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        t = cfg["temp"]
        rp = cfg["repeat_penalty"]
        print(f"\n--- Testing {cfg_name}: temp={t}, repeat_penalty={rp} ---")

        engine = GroundedTutorEngine(
            retriever=retriever,
            runtime=runtime,
            default_top_k=3,
            temperature=t,
            repeat_penalty=rp,
            max_tokens=256
        )

        cfg_total_score = 0
        cfg_total_tokens = 0
        cfg_total_inf_time = 0.0
        loop_count = 0

        for q in test_questions:
            resp = engine.process_query(
                query=q["query"],
                mode=q.get("mode", "auto"),
                student_level="Class 8",
                temperature=t,
                repeat_penalty=rp
            )
            # Evaluate
            from benchmark_runner.tutor_benchmark import evaluate_single_response
            eval_res = evaluate_single_response(q, resp)
            cfg_total_score += eval_res["total_score"]
            cfg_total_tokens += resp.generated_tokens
            cfg_total_inf_time += (resp.inference_latency_ms / 1000)
            if eval_res["had_repetition"]:
                loop_count += 1

        speed = round(cfg_total_tokens / cfg_total_inf_time, 2) if cfg_total_inf_time > 0 else 0.0
        pct = round((cfg_total_score / (len(test_questions) * 10)) * 100, 2)

        cfg_summary = {
            "config_name": cfg_name,
            "temperature": t,
            "repeat_penalty": rp,
            "total_score": cfg_total_score,
            "max_score": len(test_questions) * 10,
            "score_pct": pct,
            "tokens_per_second": speed,
            "repetition_loops_detected": loop_count,
            "avg_latency_s": round(cfg_total_inf_time / len(test_questions), 2)
        }
        experiment_results.append(cfg_summary)
        print(f"  Score: {cfg_total_score}/{len(test_questions)*10} ({pct}%) | Speed: {speed} tok/s | Loops: {loop_count}")

    runtime.unload()
    indexer.close()

    out_file = RESULTS_DIR / "parameter_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"candidate_id": candidate_id, "experiments": experiment_results}, f, indent=2, ensure_ascii=False)

    print(f"\n[Parameter Experiments Complete] Saved report to {out_file}\n")
    return {"candidate_id": candidate_id, "experiments": experiment_results}


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "CAND-01"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "llama_cpp"
    run_parameter_experiments(cid, rtype)
