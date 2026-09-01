"""
SS Tutor BD - Memory Stability & Leak Detection Engine
Runs continuous multi-turn tutoring sessions (10, 20, 50 queries) to verify memory plateau stability.
Enforces that memory does not exhibit linear unbounded growth (leakage).
"""

import sys
import os
import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever
from core.tutor_engine import GroundedTutorEngine
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from runtimes.mock_runtime import MockRuntime
from models.manager import get_active_model

DB_PATH = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
QUESTIONS_PATH = PROJECT_ROOT / "benchmarks" / "phase3b" / "tutor_100_benchmark.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3b"


def get_current_rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


def run_memory_stability_test(
    candidate_id: str = "MOCK",
    runtime_type: str = "mock",
    session_lengths: List[int] = [10, 20, 30]
) -> Dict[str, Any]:
    print(f"\n============================================================")
    print(f"      SS TUTOR BD — MEMORY STABILITY & LEAK TEST")
    print(f"============================================================")
    print(f"Candidate: {candidate_id} | Runtime: {runtime_type} | Sessions: {session_lengths}")

    indexer = KnowledgeIndexer(str(DB_PATH))
    retriever = KnowledgeRetriever(indexer)

    if runtime_type == "mock":
        runtime = MockRuntime(model_id=candidate_id)
        runtime.load("mock")
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
        runtime.load(active["file_path"], context_length=2048)

    engine = GroundedTutorEngine(
        retriever=retriever,
        runtime=runtime,
        default_top_k=2,
        temperature=0.1,
        repeat_penalty=1.15,
        max_tokens=128
    )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        all_q = json.load(f).get("questions", [])

    session_reports = []
    initial_rss = get_current_rss_mb()
    print(f"Session Initial Process RSS: {initial_rss:.2f} MB\n")

    for n_queries in session_lengths:
        print(f"--- Running {n_queries}-query continuous session ---")
        t0 = time.perf_counter()
        session_start_rss = get_current_rss_mb()
        peak_during_session = session_start_rss
        query_slice = (all_q * ((n_queries // len(all_q)) + 1))[:n_queries]

        rss_trace = []
        for i, q in enumerate(query_slice, 1):
            resp = engine.process_query(
                query=q["query"],
                mode=q.get("mode", "auto"),
                pipeline_type="hybrid_rag_tools"
            )
            cur_rss = get_current_rss_mb()
            peak_during_session = max(peak_during_session, cur_rss)
            if i % 5 == 0 or i == n_queries:
                rss_trace.append({"query_num": i, "rss_mb": cur_rss})

        session_end_rss = get_current_rss_mb()
        duration_s = round(time.perf_counter() - t0, 2)
        growth_mb = round(session_end_rss - session_start_rss, 2)
        growth_per_query = round(growth_mb / n_queries, 4)

        report = {
            "session_length_queries": n_queries,
            "duration_s": duration_s,
            "session_start_rss_mb": session_start_rss,
            "session_end_rss_mb": session_end_rss,
            "peak_rss_mb": peak_during_session,
            "total_growth_mb": growth_mb,
            "growth_per_query_mb": growth_per_query,
            "is_stable_plateau": abs(growth_per_query) < 0.25,
            "rss_trace": rss_trace
        }
        session_reports.append(report)
        print(f"  Session End RSS: {session_end_rss:.2f} MB | Peak: {peak_during_session:.2f} MB | Growth: +{growth_mb} MB ({growth_per_query} MB/query)")
        print(f"  Status: {'✅ STABLE PLATEAU' if report['is_stable_plateau'] else '⚠️ LEAK DETECTED'}\n")

    runtime.unload()
    indexer.close()

    final_payload = {
        "candidate_id": candidate_id,
        "runtime_type": runtime_type,
        "initial_rss_mb": initial_rss,
        "sessions": session_reports,
        "overall_stability": all(s["is_stable_plateau"] for s in session_reports)
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"memory_stability_{candidate_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Overall Stability Verdict: {'✅ PASSED (No Memory Leak)' if final_payload['overall_stability'] else '❌ FAILED'}")
    print(f"Report File:               {out_file}")
    print("=" * 60 + "\n")

    return final_payload


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "MOCK"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "mock"
    run_memory_stability_test(candidate_id=cid, runtime_type=rtype)
