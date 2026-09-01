"""
SS Tutor BD - Retrieval Benchmark Runner
Evaluates 60 Class 8 Mathematics retrieval test cases over SQLite FTS5 index.
Computes Recall@1, Recall@3, Recall@5, and retrieval latency metrics.
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

TESTS_PATH = PROJECT_ROOT / "benchmarks" / "phase3_class8_math" / "retrieval_tests.json"
DB_PATH = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3a"


def run_retrieval_benchmark() -> Dict[str, Any]:
    print(f"[Retrieval Benchmark] Loading database from {DB_PATH}...")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run ingestion first.")

    indexer = KnowledgeIndexer(str(DB_PATH))
    retriever = KnowledgeRetriever(indexer)

    with open(TESTS_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    tests = suite.get("tests", [])
    total_tests = len(tests)
    relevant_tests = [t for t in tests if t["expected_chapter"] != "NONE"]
    irrelevant_tests = [t for t in tests if t["expected_chapter"] == "NONE"]

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    total_latency_ms = 0.0
    detailed_results = []

    print(f"[Retrieval Benchmark] Running {total_tests} queries across categories...\n")

    for t in tests:
        q = t["query"]
        exp_ch = t["expected_chapter"]
        exp_kws = t.get("expected_keywords", [])

        t0 = time.perf_counter()
        results = retriever.retrieve(q, top_k=5)
        duration_ms = (time.perf_counter() - t0) * 1000
        total_latency_ms += duration_ms

        retrieved_chapters = [r["chunk"].chapter_id for r in results]
        retrieved_texts = [r["chunk"].content_text for r in results]

        is_hit_1 = False
        is_hit_3 = False
        is_hit_5 = False

        if exp_ch == "NONE":
            # For irrelevant query: success if 0 results returned or low relevance
            if len(results) == 0:
                is_hit_1 = is_hit_3 = is_hit_5 = True
        else:
            if len(retrieved_chapters) >= 1 and exp_ch in retrieved_chapters[:1]:
                is_hit_1 = True
            elif len(results) >= 1 and any(any(kw in t for kw in exp_kws) for t in retrieved_texts[:1]):
                is_hit_1 = True

            if len(retrieved_chapters) >= 1 and exp_ch in retrieved_chapters[:3]:
                is_hit_3 = True
            elif len(results) >= 1 and any(any(kw in t for kw in exp_kws) for t in retrieved_texts[:3]):
                is_hit_3 = True

            if len(retrieved_chapters) >= 1 and exp_ch in retrieved_chapters[:5]:
                is_hit_5 = True
            elif len(results) >= 1 and any(any(kw in t for kw in exp_kws) for t in retrieved_texts[:5]):
                is_hit_5 = True

        if is_hit_1:
            hit_at_1 += 1
        if is_hit_3:
            hit_at_3 += 1
        if is_hit_5:
            hit_at_5 += 1

        detailed_results.append({
            "test_id": t["id"],
            "type": t["type"],
            "query": q,
            "expected_chapter": exp_ch,
            "hit_at_1": is_hit_1,
            "hit_at_3": is_hit_3,
            "hit_at_5": is_hit_5,
            "retrieved_count": len(results),
            "top_chunk_id": results[0]["chunk"].chunk_id if results else None,
            "top_chapter": results[0]["chunk"].chapter_id if results else None,
            "top_score": results[0]["score"] if results else None,
            "latency_ms": round(duration_ms, 2)
        })

    recall_1 = round((hit_at_1 / total_tests) * 100, 2)
    recall_3 = round((hit_at_3 / total_tests) * 100, 2)
    recall_5 = round((hit_at_5 / total_tests) * 100, 2)
    avg_latency = round(total_latency_ms / total_tests, 2)

    # Category breakdown
    cat_breakdown = {}
    for r in detailed_results:
        ctype = r["type"]
        if ctype not in cat_breakdown:
            cat_breakdown[ctype] = {"total": 0, "hit_1": 0, "hit_3": 0, "hit_5": 0}
        cat_breakdown[ctype]["total"] += 1
        if r["hit_at_1"]:
            cat_breakdown[ctype]["hit_1"] += 1
        if r["hit_at_3"]:
            cat_breakdown[ctype]["hit_3"] += 1
        if r["hit_at_5"]:
            cat_breakdown[ctype]["hit_5"] += 1

    summary = {
        "suite_name": suite.get("suite_name"),
        "total_queries": total_tests,
        "metrics": {
            "recall_at_1_pct": recall_1,
            "recall_at_3_pct": recall_3,
            "recall_at_5_pct": recall_5,
            "avg_latency_ms": avg_latency,
            "target_recall_at_5_pct": 90.0,
            "passed_target": recall_5 >= 90.0
        },
        "category_breakdown": cat_breakdown,
        "results": detailed_results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "retrieval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("      SS TUTOR BD — CLASS 8 RETRIEVAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Test Queries:       {total_tests}")
    print(f"Recall@1:                 {recall_1}% ({hit_at_1}/{total_tests})")
    print(f"Recall@3:                 {recall_3}% ({hit_at_3}/{total_tests})")
    print(f"Recall@5:                 {recall_5}% ({hit_at_5}/{total_tests}) [Target: >= 90%]")
    print(f"Average Retrieval Speed:  {avg_latency} ms / query")
    print(f"Target Status:            {'✅ PASSED (>= 90%)' if recall_5 >= 90.0 else '❌ FAILED'}")
    print(f"Report File:              {out_file}")
    print("=" * 70 + "\n")

    indexer.close()
    return summary


if __name__ == "__main__":
    run_retrieval_benchmark()
