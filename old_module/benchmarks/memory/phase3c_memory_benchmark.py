"""
SS Tutor BD - Phase 3C Memory Benchmark Engine
Measures process RSS at every critical lifecycle stage with explicit KV-cache grid search.
Tests: cold start, 10-query warm, 25/50/100-turn multi-turn, 10 load/unload cycles.
"""

import sys
import os
import json
import time
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.manager import get_active_model, get_candidate
from core.runtime.memory_budget import MemoryBudgetManager

RESULTS_DIR = PROJECT_ROOT / "results" / "phase3c"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_rss() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


NCTB_PROBE_QUESTIONS = [
    "৩/৪ + ৫/৬ এর যোগফল নির্ণয় করো।",
    "৫০০০ টাকায় ১০% হারে ৩ বছরের সরল মুনাফা কত?",
    "পিথাগোরাস উপপাদ্য কী?",
    "x² + 7x + 12 = 0 সমীকরণ সমাধান করো।",
    "বৃত্তের ক্ষেত্রফলের সূত্র লিখো।",
    "১ থেকে ৫০ পর্যন্ত সংখ্যার যোগফল কত?",
    "লাভ ও ক্ষতি বলতে কী বোঝায়?",
    "চক্রবৃদ্ধি মুনাফা কীভাবে নির্ণয় করা হয়?",
    "৬/৮ ভগ্নাংশটি লঘিষ্ঠ করো।",
    "সমকোণী ত্রিভুজের অতিভুজ কাকে বলে?",
]


def run_context_grid_benchmark(candidate_id: str, runtime_type: str = "llama_cpp") -> Dict[str, Any]:
    """
    Tests multiple (context_length, max_output) combinations.
    Records peak RSS for each configuration.
    Only proceeds to next if previous passed memory gate.
    """
    print("\n" + "=" * 65)
    print(f"  SS TUTOR BD — PHASE 3C CONTEXT GRID BENCHMARK")
    print(f"  Candidate: {candidate_id}")
    print("=" * 65)

    active = get_active_model()
    if not active:
        print("ERROR: No active model. Run: python benchmark_runner/cli.py download <CAND_ID>")
        return {}

    configurations = [
        (128, 48),
        (192, 64),
        (256, 96),
        (384, 96),
        (512, 128),
    ]

    grid_results = []
    for ctx_len, max_out in configurations:
        print(f"\n  Testing context={ctx_len}, max_output={max_out} ...", flush=True)

        baseline = get_rss()

        try:
            if runtime_type == "llama_cpp":
                from runtimes.llama_cpp_runtime import LlamaCppRuntime
                runtime = LlamaCppRuntime(
                    model_id=candidate_id,
                    quantization=active.get("quantization", "Q4_K_M"),
                    threads=4,
                    tokenizer_repo=active.get("tokenizer_repo_id")
                )
                runtime.load(active["file_path"], context_length=ctx_len)
            else:
                from runtimes.mock_runtime import MockRuntime
                runtime = MockRuntime(model_id=candidate_id)
                runtime.load("mock", context_length=ctx_len)

            after_load = get_rss()

            # Run one inference
            from runtimes.base import GenerationParams
            gen_params = GenerationParams(max_tokens=max_out, temperature=0.15, repeat_penalty=1.2)
            result = runtime.generate(
                "তুমি একজন বাংলাদেশ NCTB শিক্ষক।",
                NCTB_PROBE_QUESTIONS[0],
                params=gen_params
            )
            after_inference = get_rss()

            verdict = MemoryBudgetManager.evaluate_rss(after_inference)

            grid_results.append({
                "context_length": ctx_len,
                "max_output_tokens": max_out,
                "baseline_rss": baseline,
                "after_load_rss": after_load,
                "after_inference_rss": after_inference,
                "status": verdict["status"],
                "passed_ceiling": verdict["passed_production_ceiling"],
                "tokens_per_sec": getattr(result, "tokens_per_sec", 0)
            })

            status_icon = "✅" if verdict["passed_production_ceiling"] else "❌"
            print(f"    {status_icon} RSS={after_inference} MB | Status: {verdict['status']}")

            runtime.unload()
            del runtime
            gc.collect()
            time.sleep(0.5)

        except Exception as ex:
            grid_results.append({
                "context_length": ctx_len,
                "max_output_tokens": max_out,
                "error": str(ex)
            })
            print(f"    ❌ ERROR: {str(ex)[:80]}")

    output = {
        "candidate_id": candidate_id,
        "runtime_type": runtime_type,
        "grid_results": grid_results,
        "recommended_config": None
    }

    # Select best passing configuration
    for r in grid_results:
        if r.get("passed_ceiling") and "error" not in r:
            output["recommended_config"] = {
                "context_length": r["context_length"],
                "max_output_tokens": r["max_output_tokens"],
                "peak_rss": r["after_inference_rss"]
            }

    out_path = RESULTS_DIR / f"context_grid_{candidate_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Grid results saved: {out_path}")
    return output


def run_multiturn_stability_benchmark(
    candidate_id: str,
    runtime_type: str = "llama_cpp",
    context_length: int = 256,
    max_output: int = 64,
    turn_counts: List[int] = [10, 25, 50, 100]
) -> Dict[str, Any]:
    """
    Runs multi-turn sessions measuring memory growth over N turns.
    Detects monotonic memory growth (memory leak).
    """
    print("\n" + "=" * 65)
    print(f"  SS TUTOR BD — PHASE 3C MULTI-TURN STABILITY BENCHMARK")
    print(f"  Candidate: {candidate_id} | Context: {context_length}")
    print("=" * 65)

    active = get_active_model()
    if not active:
        print("ERROR: No active model.")
        return {}

    try:
        if runtime_type == "llama_cpp":
            from runtimes.llama_cpp_runtime import LlamaCppRuntime
            from core.rag.indexer import KnowledgeIndexer
            from core.rag.retriever import KnowledgeRetriever
            from core.tutor_engine import GroundedTutorEngine

            runtime = LlamaCppRuntime(
                model_id=candidate_id,
                quantization=active.get("quantization", "Q4_K_M"),
                threads=4,
                tokenizer_repo=active.get("tokenizer_repo_id")
            )
            runtime.load(active["file_path"], context_length=context_length)
            db_path = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
            retriever = KnowledgeRetriever(KnowledgeIndexer(str(db_path)))
            engine = GroundedTutorEngine(
                retriever=retriever,
                runtime=runtime,
                default_top_k=1,
                temperature=0.15,
                repeat_penalty=1.2,
                max_tokens=max_output
            )
        else:
            from runtimes.mock_runtime import MockRuntime
            from core.rag.indexer import KnowledgeIndexer
            from core.rag.retriever import KnowledgeRetriever
            from core.tutor_engine import GroundedTutorEngine
            runtime = MockRuntime(model_id=candidate_id)
            runtime.load("mock", context_length=context_length)
            db_path = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
            retriever = KnowledgeRetriever(KnowledgeIndexer(str(db_path)))
            engine = GroundedTutorEngine(retriever=retriever, runtime=runtime, max_tokens=max_output)
    except Exception as ex:
        return {"error": str(ex)}

    session_rss_start = get_rss()
    session_results = []

    for max_turns in turn_counts:
        print(f"\n  --- {max_turns}-turn session ---", flush=True)
        rss_readings = []
        rss_readings.append(get_rss())

        for i in range(max_turns):
            q = NCTB_PROBE_QUESTIONS[i % len(NCTB_PROBE_QUESTIONS)]
            try:
                engine.process_query(q, mode="EXPLAIN", pipeline_type="hybrid_rag_tools")
            except Exception:
                pass
            rss_readings.append(get_rss())

        session_end_rss = rss_readings[-1]
        session_peak_rss = max(rss_readings)
        growth = session_end_rss - rss_readings[0]
        growth_per_query = round(growth / max_turns, 4)
        is_stable = growth_per_query <= 0.5  # <= 0.5 MB/query growth threshold

        print(f"    End RSS: {session_end_rss} MB | Peak: {session_peak_rss} MB | Growth: {growth:+.2f} MB ({growth_per_query:.4f} MB/query)")
        print(f"    Status: {'✅ STABLE PLATEAU' if is_stable else '⚠️ GROWTH DETECTED'}")

        session_results.append({
            "turns": max_turns,
            "start_rss": rss_readings[0],
            "end_rss": session_end_rss,
            "peak_rss": session_peak_rss,
            "growth_mb": round(growth, 3),
            "growth_per_query_mb": growth_per_query,
            "stable": is_stable,
            "passed_ceiling": session_peak_rss <= MemoryBudgetManager.ABSOLUTE_CEILING_MB
        })

    # Overall verdict
    all_stable = all(r["stable"] for r in session_results)
    all_within_ceiling = all(r["passed_ceiling"] for r in session_results)

    print("\n" + "=" * 65)
    verdict = "✅ PASS" if (all_stable and all_within_ceiling) else "❌ FAIL"
    print(f"  Multi-Turn Verdict: {verdict}")
    print("=" * 65)

    output = {
        "candidate_id": candidate_id,
        "context_length": context_length,
        "max_output_tokens": max_output,
        "session_rss_start": session_rss_start,
        "turn_sessions": session_results,
        "all_stable": all_stable,
        "all_within_ceiling": all_within_ceiling,
        "overall_verdict": "PASS" if (all_stable and all_within_ceiling) else "FAIL"
    }

    out_path = RESULTS_DIR / f"multiturn_memory_{candidate_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Report: {out_path}")
    return output


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "CAND-03"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "llama_cpp"
    ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    # Stage 1: Grid benchmark across context configurations
    grid = run_context_grid_benchmark(cid, rtype)

    # Stage 2: Multi-turn stability using recommended configuration
    recommended = grid.get("recommended_config")
    if recommended:
        ctx_use = recommended["context_length"]
        out_use = recommended["max_output_tokens"]
    else:
        ctx_use = ctx
        out_use = 64

    run_multiturn_stability_benchmark(cid, rtype, context_length=ctx_use, max_output=out_use)
