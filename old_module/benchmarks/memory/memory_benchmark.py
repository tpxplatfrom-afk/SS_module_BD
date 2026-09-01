"""
SS Tutor BD - Production Memory Benchmark Engine
Accurately measures process RSS across baseline, runtime load, tokenizer, RAG, inference, and peak allocations.
Enforces the Phase 3B Production Memory Contract (Preferred: <=200 MB, Warning: 200-250 MB, Fail: >250 MB).
"""

import sys
import os
import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from runtimes.mock_runtime import MockRuntime
from models.manager import get_active_model

DB_PATH = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


def get_process_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def evaluate_memory_verdict(peak_rss: float) -> str:
    if peak_rss <= 200.0:
        return "PREFERRED (<= 200 MB)"
    elif peak_rss <= 250.0:
        return "WARNING (200 - 250 MB)"
    else:
        return "DISQUALIFIED / FAIL (> 250 MB)"


def run_memory_benchmark(
    candidate_id: str = "CAND-01",
    runtime_type: str = "llama_cpp",
    context_length: int = 1024,
    test_prompt: str = "৩/৪ + ৫/৬ এর যোগফল নির্ণয় করো এবং ধাপগুলো বাংলায় বুঝিয়ে বলো।"
) -> Dict[str, Any]:
    print(f"\n============================================================")
    print(f"      SS TUTOR BD — PRODUCTION MEMORY BENCHMARK")
    print(f"============================================================")
    print(f"Target: Candidate {candidate_id} | Context Length: {context_length} | Runtime: {runtime_type}")

    # Stage 1: Baseline Process Memory
    rss_baseline = get_process_rss_mb()
    print(f"1. Baseline Process RSS:          {rss_baseline:.2f} MB")

    # Stage 2: Runtime Initialization
    t0 = time.perf_counter()
    if runtime_type == "mock":
        runtime = MockRuntime(model_id=candidate_id)
    else:
        active = get_active_model()
        if not active or not Path(active.get("file_path", "")).exists():
            raise FileNotFoundError(f"Active model binary not found for {candidate_id}. Download model first.")
        runtime = LlamaCppRuntime(
            model_id=candidate_id,
            quantization=active.get("quantization", "Q4_K_M"),
            threads=4,
            tokenizer_repo=active.get("tokenizer_repo_id")
        )
    rss_runtime_init = get_process_rss_mb()
    runtime_delta = round(rss_runtime_init - rss_baseline, 2)
    print(f"2. Runtime Init RSS:              {rss_runtime_init:.2f} MB (Delta: +{runtime_delta} MB)")

    # Stage 3: Model Weights Load
    if runtime_type == "mock":
        runtime.load("mock", context_length=context_length)
    else:
        runtime.load(active["file_path"], context_length=context_length)
    rss_model_loaded = get_process_rss_mb()
    model_delta = round(rss_model_loaded - rss_runtime_init, 2)
    print(f"3. Model Loaded RSS:              {rss_model_loaded:.2f} MB (Delta: +{model_delta} MB)")

    # Stage 4: RAG Engine & SQLite FTS5 Initialization
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Knowledge index not found at {DB_PATH}. Run ingestion first.")
    indexer = KnowledgeIndexer(str(DB_PATH))
    retriever = KnowledgeRetriever(indexer)
    rss_rag_init = get_process_rss_mb()
    rag_delta = round(rss_rag_init - rss_model_loaded, 2)
    print(f"4. RAG / FTS5 Initialized RSS:    {rss_rag_init:.2f} MB (Delta: +{rag_delta} MB)")

    # Stage 5: Inference Execution & Peak Memory
    peak_tracker = [rss_rag_init]

    # Warm-up query
    retriever.retrieve(test_prompt, top_k=2)
    
    t_inf_start = time.perf_counter()
    gen_res = runtime.generate(
        prompt=test_prompt,
        system_prompt="You are a helpful NCTB Class 8 Bengali Mathematics tutor.",
        max_tokens=128,
        temperature=0.1
    )
    inf_duration_s = time.perf_counter() - t_inf_start

    rss_inference = get_process_rss_mb()
    peak_tracker.append(rss_inference)
    peak_rss = max(peak_tracker)
    inference_delta = round(peak_rss - rss_rag_init, 2)
    print(f"5. Post-Inference RSS:            {rss_inference:.2f} MB (Inference Delta: +{inference_delta} MB)")
    print(f"6. Measured Peak RSS:             {peak_rss:.2f} MB")

    # Cleanup / Unload
    runtime.unload()
    indexer.close()
    rss_post_unload = get_process_rss_mb()
    print(f"7. Post-Unload RSS:               {rss_post_unload:.2f} MB")

    verdict = evaluate_memory_verdict(peak_rss)

    summary = {
        "candidate_id": candidate_id,
        "runtime_type": runtime_type,
        "context_length": context_length,
        "memory_profile": {
            "baseline_rss_mb": rss_baseline,
            "runtime_init_rss_mb": rss_runtime_init,
            "model_loaded_rss_mb": rss_model_loaded,
            "rag_init_rss_mb": rss_rag_init,
            "post_inference_rss_mb": rss_inference,
            "peak_rss_mb": peak_rss,
            "post_unload_rss_mb": rss_post_unload,
            "deltas": {
                "runtime_delta_mb": runtime_delta,
                "model_delta_mb": model_delta,
                "rag_delta_mb": rag_delta,
                "inference_delta_mb": inference_delta,
                "total_peak_over_baseline_mb": round(peak_rss - rss_baseline, 2)
            }
        },
        "performance": {
            "ttft_ms": round(gen_res.ttft_ms, 2) if hasattr(gen_res, "ttft_ms") else 0.0,
            "tokens_per_second": round(gen_res.tokens_per_sec, 2) if hasattr(gen_res, "tokens_per_sec") else 0.0,
            "generated_tokens": gen_res.generated_tokens,
            "inference_duration_s": round(inf_duration_s, 2)
        },
        "contract_assessment": {
            "preferred_target_mb": 200.0,
            "hard_ceiling_mb": 250.0,
            "status": verdict,
            "passed_production_ceiling": peak_rss <= 250.0,
            "passed_preferred_target": peak_rss <= 200.0
        }
    }

    print(f"------------------------------------------------------------")
    print(f"Production Contract Verdict:      {verdict}")
    print(f"============================================================\n")

    return summary


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "MOCK"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "mock"
    ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    res = run_memory_benchmark(candidate_id=cid, runtime_type=rtype, context_length=ctx)
    print(json.dumps(res, indent=2, ensure_ascii=False))
