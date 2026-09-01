"""
SS Tutor BD - Automated Benchmark Execution Script
Runs the complete Phase 1 benchmark pipeline for a specified candidate model.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.manager import get_candidate, download_candidate, purge_active_model
from benchmarks.tokenizer.benchmark import evaluate_tokenizer, print_comparison_table
from benchmark_runner.runner import run_benchmark
from scripts.check_disk import check_disk_health


def execute_pipeline(candidate_id: str, use_mock: bool = False):
    print("\n" + "#" * 80)
    print(f"  SS TUTOR BD — PIPELINE EXECUTION: {candidate_id}")
    print("#" * 80 + "\n")

    # Step 1: Storage Health
    print("[1/5] Checking Storage Health...")
    check_disk_health()

    # Step 2: License Gate
    print(f"\n[2/5] Checking Gate 1 (License) for {candidate_id}...")
    cand = get_candidate(candidate_id)
    if not cand:
        print(f"Error: Candidate {candidate_id} not in registry.")
        return
    
    if cand.get("license_status") != "LICENSE_PASSED":
        print(f"ABORTED: Candidate {candidate_id} has license status '{cand.get('license_status')}'.")
        print("Must pass primary-source license audit before downloading.")
        return
    print(f"  Passed Gate 1: License is {cand['license']} ({cand['license_status']})")

    # Step 3: Tokenizer Evaluation
    print(f"\n[3/5] Running Bengali Tokenizer Efficiency Benchmark...")
    tok_repo = cand.get("tokenizer_repo_id", cand["name"])
    try:
        tok_res = evaluate_tokenizer(tok_repo)
        print_comparison_table([tok_res])
    except Exception as e:
        print(f"Tokenizer benchmark warning: {e}")

    # Step 4: Model Download / Preparation
    if not use_mock:
        print(f"\n[4/5] Downloading Model Weights for {candidate_id}...")
        try:
            dl_res = download_candidate(candidate_id)
            print(f"  Downloaded: {dl_res['file_size_mb']} MB to {dl_res['file_path']}")
        except Exception as e:
            print(f"Download failed: {e}")
            return
    else:
        print(f"\n[4/5] Running in MOCK Mode (Skipping GGUF weight download)...")

    # Step 5: Full 100-Item Benchmark & Scoring
    print(f"\n[5/5] Executing 100-Item Benchmark Suite...")
    rtype = "mock" if use_mock else "llama_cpp"
    try:
        run_benchmark(candidate_id=candidate_id, runtime_type=rtype)
    except Exception as e:
        print(f"Benchmark execution failed: {e}")


if __name__ == "__main__":
    target_cand = sys.argv[1] if len(sys.argv) > 1 else "CAND-01"
    is_mock = "--mock" in sys.argv or (len(sys.argv) > 2 and sys.argv[2] == "mock")
    execute_pipeline(target_cand, use_mock=is_mock)
