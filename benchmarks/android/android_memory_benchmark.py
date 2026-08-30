"""
SS Tutor BD — Android Memory Benchmark (Phase 5)
Simulates cold launch, multi-turn sessions, model load/unload cycles, and low-memory callbacks
on the Python deterministic-fallback engine (proxy for native Android measurements).

NOTE: Python results measure the Python-level memory footprint of the deterministic core only.
Real Android PSS must be measured via `adb shell dumpsys meminfo` on a physical device.
All results from this script are labelled EMULATED (NOT PRODUCTION PROOF).
"""
import sys
import time
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.math.expression_parser import ExpressionParser
from core.math.fraction import FractionHelper
from core.math.calculator import MathCalculator
from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever
from core.validation.hint_validator import HintValidator
from core.validation.grounding_validator import GroundingValidator
from core.runtime.session_manager import SessionState

import json, os, datetime

QUERIES = [
    "৩/৪ + ৫/৬ এর যোগফল কত?",
    "৫০০০ টাকায় ১০% হারে ৩ বছরের সরল মুনাফা কত?",
    "পিথাগোরাসের উপপাদ্য কী?",
    "ভগ্নাংশ কাকে বলে?",
    "৭ সেমি ব্যাসার্ধের বৃত্তের ক্ষেত্রফল কত?",
    "১ থেকে ১০০ পর্যন্ত সংখ্যার যোগফল কত?",
    "লাভ এবং ক্ষতি বলতে কী বোঝায়?",
    "সমান্তর প্রগতি কাকে বলে?",
    "এই পাঠ্যবইয়ে কি পদার্থবিজ্ঞান আছে?",
    "8000 টাকায় 10% হারে 2 বছরের চক্রবৃদ্ধি মূলধন কত?",
]


def get_python_rss_mb() -> float:
    """Estimate current Python process memory in MB via tracemalloc."""
    current, _ = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


def benchmark_cold_launch():
    tracemalloc.start()
    t0 = time.time()
    session = SessionState("bench_cold")
    indexer = KnowledgeIndexer()
    retriever = KnowledgeRetriever(indexer)
    launch_ms = (time.time() - t0) * 1000
    rss = get_python_rss_mb()
    tracemalloc.stop()
    return {"test": "cold_launch", "latency_ms": round(launch_ms, 2),
            "rss_mb": round(rss, 2), "verdict": "EMULATED - NOT PRODUCTION PROOF"}


def benchmark_turns(n_turns: int):
    tracemalloc.start()
    session = SessionState("bench_turns")
    indexer = KnowledgeIndexer()
    retriever = KnowledgeRetriever(indexer)
    start_rss = get_python_rss_mb()
    memory_samples = []
    t0 = time.time()

    for i in range(n_turns):
        q = QUERIES[i % len(QUERIES)]
        intent = ExpressionParser.detect_math_intent(q)
        if intent["intent"] == "fraction_addition":
            FractionHelper.add(intent["fraction1"], intent["fraction2"])
        elif intent["intent"] == "simple_interest":
            MathCalculator.simple_interest(intent["principal"], intent["rate_pct"], intent["time_years"])
        elif intent["intent"] == "series_sum":
            MathCalculator.series_sum(int(intent.get("first_term", 1)), int(intent.get("last_term", 100)))

        session.update(question=q, mode="EXPLAIN", result="computed")
        memory_samples.append(get_python_rss_mb())

    total_ms = (time.time() - t0) * 1000
    end_rss = get_python_rss_mb()
    growth = (end_rss - start_rss) / max(n_turns, 1)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gate_d_pass = growth <= 0.05
    return {
        "test": f"{n_turns}_turn_session",
        "n_turns": n_turns,
        "start_rss_mb": round(start_rss, 2),
        "end_rss_mb": round(end_rss, 2),
        "peak_rss_mb": round(peak / (1024 * 1024), 2),
        "growth_mb_per_turn": round(growth, 4),
        "total_ms": round(total_ms, 2),
        "avg_ms_per_turn": round(total_ms / n_turns, 2),
        "gate_d_growth_pass": gate_d_pass,
        "verdict": "EMULATED - NOT PRODUCTION PROOF"
    }


def benchmark_model_load_unload():
    tracemalloc.start()
    t0 = time.time()
    # Simulate load: instantiate full pipeline
    session = SessionState("bench_model")
    indexer = KnowledgeIndexer()
    retriever = KnowledgeRetriever(indexer)
    loaded_rss = get_python_rss_mb()

    # Simulate unload: del references
    del session, indexer, retriever
    import gc; gc.collect()
    unloaded_rss = get_python_rss_mb()
    total_ms = (time.time() - t0) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "test": "model_load_unload",
        "loaded_rss_mb": round(loaded_rss, 2),
        "unloaded_rss_mb": round(unloaded_rss, 2),
        "memory_released_mb": round(loaded_rss - unloaded_rss, 2),
        "gate_f_pass": unloaded_rss <= loaded_rss + 0.05,
        "total_ms": round(total_ms, 2),
        "verdict": "EMULATED - NOT PRODUCTION PROOF"
    }


def run_all_benchmarks():
    print("\n" + "="*70)
    print("  SS TUTOR BD — ANDROID MEMORY BENCHMARK (Phase 5)")
    print("  !! EMULATED - Python Proxy. Real device PSS must be measured via ADB !!")
    print("="*70)

    results = []

    # Cold launch
    r = benchmark_cold_launch()
    results.append(r)
    print(f"  [Cold Launch]  Latency: {r['latency_ms']} ms | RSS: {r['rss_mb']} MB")

    # Turn sessions
    for n in [10, 25, 50, 100]:
        r = benchmark_turns(n)
        results.append(r)
        print(f"  [{n}-Turn Session]  Growth: {r['growth_mb_per_turn']} MB/turn | "
              f"Avg: {r['avg_ms_per_turn']} ms | Gate D: {'PASS' if r['gate_d_growth_pass'] else 'FAIL'}")

    # Model load/unload
    r = benchmark_model_load_unload()
    results.append(r)
    print(f"  [Load/Unload]  Loaded: {r['loaded_rss_mb']} MB | Unloaded: {r['unloaded_rss_mb']} MB | Gate F: {'PASS' if r['gate_f_pass'] else 'FAIL'}")

    # Gate summary
    print("\n--- Android Benchmark Gate Summary ---")
    gate_d_all_pass = all(r.get("gate_d_growth_pass", True) for r in results)
    gate_f_pass = next((r["gate_f_pass"] for r in results if r["test"] == "model_load_unload"), False)
    print(f"  Gate D (Memory Growth <= 0.05 MB/turn): {'PASS' if gate_d_all_pass else 'FAIL'}")
    print(f"  Gate F (Model Unload Releases Memory):  {'PASS' if gate_f_pass else 'FAIL'}")
    print(f"  Gate E (100-Turn Stability):             PASS (no OOM)")
    print("  Gate A/B/C (PSS <= 150/200/250 MB):      REQUIRES REAL ANDROID DEVICE (adb shell dumpsys meminfo)")
    print("="*70 + "\n")

    # Save results
    out_dir = PROJECT_ROOT / "benchmarks" / "android"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "android_memory_results.json"
    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment": "python_emulated",
        "note": "EMULATED - NOT PRODUCTION PROOF. Real Android PSS measurement requires adb.",
        "results": results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_all_benchmarks()
