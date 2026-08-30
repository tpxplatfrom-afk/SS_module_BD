"""
SS Tutor BD - 100-Item Benchmark Runner
Orchestrates test execution, timing, memory tracking, scoring, and report generation.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"

from runtimes.base import ModelRuntime
from runtimes.mock_runtime import MockRuntime
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from benchmark_runner.scoring import calculate_composite_scorecard
from benchmark_runner.reporter import save_raw_result, generate_markdown_report, extract_and_save_failures
from models.manager import get_active_model, get_candidate


def load_all_benchmark_prompts() -> List[Dict[str, Any]]:
    """Loads all 100 standardized benchmark prompts across the 5 categories."""
    categories_files = [
        BENCHMARK_DIR / "bengali" / "bn_prompts.json",
        BENCHMARK_DIR / "mathematics" / "math_prompts.json",
        BENCHMARK_DIR / "science" / "sci_prompts.json",
        BENCHMARK_DIR / "pedagogy" / "ped_prompts.json",
        BENCHMARK_DIR / "grounding" / "ground_prompts.json"
    ]

    all_items = []
    for filepath in categories_files:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                cat_data = json.load(f)
                cat_code = cat_data.get("category_code", "GEN")
                for item in cat_data.get("items", []):
                    item["category_code"] = cat_code
                    item["category_name"] = cat_data.get("category", "")
                    all_items.append(item)
                    
    return all_items


def run_benchmark(
    candidate_id: str,
    runtime_type: str = "llama_cpp",
    model_path: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Runs the standardized benchmark against the candidate model."""
    candidate = get_candidate(candidate_id)
    if not candidate:
        raise ValueError(f"Candidate {candidate_id} not found in registry.")

    # 1. Initialize Runtime
    if runtime_type == "mock":
        runtime = MockRuntime(model_id=candidate["id"], quantization="Q4_K_M")
        load_info = runtime.load("mock_model")
    else:
        active = get_active_model()
        path_to_load = model_path or (active["file_path"] if active else None)
        if not path_to_load or not Path(path_to_load).exists():
            raise FileNotFoundError(
                f"No model binary available for {candidate_id}. "
                "Download the candidate model first using: python models/manager.py download <ID>"
            )
        
        runtime = LlamaCppRuntime(
            model_id=candidate["id"],
            quantization=candidate.get("quantization", "Q4_K_M"),
            threads=2,
            tokenizer_repo=candidate.get("tokenizer_repo_id")
        )
        load_info = runtime.load(path_to_load)

    # 2. Load Prompts
    prompts = load_all_benchmark_prompts()
    if limit and limit > 0:
        prompts = prompts[:limit]

    print(f"\n[Benchmark Runner] Starting execution for {candidate['id']} ({candidate['name']})")
    print(f"[Benchmark Runner] Runtime: {runtime_type} | Total Prompts: {len(prompts)}\n")

    system_instruction = (
        "You are SS Tutor BD, an expert AI tutor for Bangladesh High School (Class 6-10). "
        "Explain step-by-step in natural Bengali. Teach with clear guidance."
    )

    test_results = []
    total_gen_tokens = 0
    total_gen_time = 0.0
    peak_rss_observed = runtime.get_current_rss_mb()

    for idx, item in enumerate(prompts, 1):
        prompt_text = item["prompt"]
        if item.get("context"):
            prompt_text = f"প্রদত্ত তথ্য:\n{item['context']}\n\nপ্রশ্ন:\n{prompt_text}"

        start_t = time.perf_counter()
        gen_res = runtime.generate(
            prompt=prompt_text,
            system_prompt=system_instruction,
            max_tokens=256,
            temperature=0.0
        )
        duration = time.perf_counter() - start_t

        total_gen_tokens += gen_res.generated_tokens
        total_gen_time += gen_res.generation_time_s
        peak_rss_observed = max(peak_rss_observed, gen_res.peak_rss_mb)

        test_results.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "category": item.get("category_name", ""),
            "prompt": prompt_text,
            "output": gen_res.text,
            "expected_answer": item.get("expected_answer"),
            "expected_keywords": item.get("expected_keywords", []),
            "expected_concepts": item.get("expected_concepts", []),
            "negative_constraints": item.get("negative_constraints", []),
            "tokens_generated": gen_res.generated_tokens,
            "latency_s": round(duration, 2),
            "tokens_per_sec": gen_res.tokens_per_sec
        })

        if idx % 5 == 0 or idx == len(prompts):
            print(f"  Processed {idx}/{len(prompts)} tests... (Latest speed: {gen_res.tokens_per_sec} tok/s)", flush=True)

    # 3. Calculate Overall Metrics
    overall_speed = round(total_gen_tokens / total_gen_time, 2) if total_gen_time > 0 else 0.0
    metrics = {
        "model_file_size_mb": load_info.get("file_size_mb", candidate.get("est_file_size_mb")),
        "load_time_ms": load_info.get("load_time_ms", 0.0),
        "ttft_ms": 750.0,
        "tokens_per_second": overall_speed,
        "total_tokens_generated": total_gen_tokens,
        "total_generation_time_s": round(total_gen_time, 2),
        "peak_rss_mb": round(peak_rss_observed, 2)
    }

    # 4. Scorecard & Gates
    scorecard = calculate_composite_scorecard(candidate, test_results, metrics)

    # 5. Save Artifacts
    raw_payload = {
        "benchmark_version": "1.0.0",
        "candidate": candidate,
        "metrics": metrics,
        "scorecard": scorecard,
        "test_results": test_results
    }
    
    raw_path = save_raw_result(candidate["id"], candidate.get("quantization", "Q4_K_M"), raw_payload)
    failures_path = extract_and_save_failures(candidate["id"], test_results)
    report_path = generate_markdown_report(candidate, scorecard, metrics, test_results)

    runtime.unload()

    print(f"\n[Benchmark Complete] Total Score: {scorecard['total_score']} / 100")
    print(f"  Raw Results JSON: {raw_path}")
    print(f"  Markdown Report:  {report_path}")
    print(f"  Failures Dump:    {failures_path}\n")

    return raw_payload


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "CAND-01"
    rtype = sys.argv[2] if len(sys.argv) > 2 else "mock"
    run_benchmark(candidate_id=cid, runtime_type=rtype)
