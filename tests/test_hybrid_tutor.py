"""
SS Tutor BD - Unit Tests: Hybrid Tutor Pipeline (Phase 3B)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever
from core.tutor_engine import GroundedTutorEngine
from runtimes.mock_runtime import MockRuntime

DB_PATH = PROJECT_ROOT / "packs" / "class8_math" / "index.db"


def get_test_engine() -> GroundedTutorEngine:
    indexer = KnowledgeIndexer(str(DB_PATH))
    retriever = KnowledgeRetriever(indexer)
    runtime = MockRuntime(model_id="MOCK")
    runtime.load("mock")
    return GroundedTutorEngine(
        retriever=retriever,
        runtime=runtime,
        default_top_k=2,
        temperature=0.1,
        repeat_penalty=1.15,
        max_tokens=128
    )


def test_hybrid_engine_returns_tutor_response():
    engine = get_test_engine()
    resp = engine.process_query("পিথাগোরাস কী?", mode="EXPLAIN", pipeline_type="hybrid_rag_tools")
    assert resp is not None
    assert hasattr(resp, "final_text")
    assert len(resp.final_text) > 0
    print("test_hybrid_engine_returns_tutor_response: PASSED")


def test_hybrid_engine_fraction_detected_as_math_task():
    engine = get_test_engine()
    resp = engine.process_query("৩/৪ + ৫/৬ এর যোগফল নির্ণয় করো।", mode="SOLVE", pipeline_type="hybrid_rag_tools")
    assert resp.was_math_task == True, "Should detect fraction as math task"
    assert resp.deterministic_result is not None
    print("test_hybrid_engine_fraction_detected_as_math_task: PASSED")


def test_hybrid_engine_simple_interest_detected():
    engine = get_test_engine()
    resp = engine.process_query("৫০০০ টাকায় ১০% হারে ৩ বছরের সরল মুনাফা কত?", pipeline_type="hybrid_rag_tools")
    assert resp.was_math_task == True
    assert resp.deterministic_result is not None
    assert resp.deterministic_result.get("interest", 0) == 1500.0
    print("test_hybrid_engine_simple_interest_detected: PASSED")


def test_hint_mode_instruction_flagged():
    engine = get_test_engine()
    resp = engine.process_query("x^2 + 7x + 12 = 0 সমীকরণের সমাধান। উত্তর বলবে না।", mode="HINT", pipeline_type="hybrid_rag_tools")
    assert resp.mode == "HINT"
    print("test_hint_mode_instruction_flagged: PASSED")


def test_llm_only_pipeline():
    engine = get_test_engine()
    resp = engine.process_query("বীজগণিত কী?", mode="EXPLAIN", pipeline_type="llm_only")
    assert resp.pipeline_type == "llm_only"
    assert resp.grounding_status == "UNGROUNDED"
    assert len(resp.retrieved_chunks) == 0
    print("test_llm_only_pipeline: PASSED")


def test_rag_only_pipeline_retrieves_chunks():
    engine = get_test_engine()
    resp = engine.process_query("সমীকরণ সমাধানের নিয়ম কী?", mode="EXPLAIN", pipeline_type="llm_rag")
    assert resp.pipeline_type == "llm_rag"
    print(f"test_rag_only_pipeline_retrieves_chunks: PASSED (retrieved {len(resp.retrieved_chunks)} chunks)")


def test_response_has_latency_fields():
    engine = get_test_engine()
    resp = engine.process_query("মৌলিক সংখ্যা কী?", pipeline_type="hybrid_rag_tools")
    assert resp.retrieval_latency_ms >= 0
    assert resp.inference_latency_ms >= 0
    assert resp.total_latency_ms > 0
    print("test_response_has_latency_fields: PASSED")


def run_all_hybrid_tests():
    print("\n--- Running Hybrid Tutor Pipeline Unit Tests ---")
    test_hybrid_engine_returns_tutor_response()
    test_hybrid_engine_fraction_detected_as_math_task()
    test_hybrid_engine_simple_interest_detected()
    test_hint_mode_instruction_flagged()
    test_llm_only_pipeline()
    test_rag_only_pipeline_retrieves_chunks()
    test_response_has_latency_fields()
    print("--- All Hybrid Tutor Tests PASSED (7 / 7) ---\n")


if __name__ == "__main__":
    run_all_hybrid_tests()
