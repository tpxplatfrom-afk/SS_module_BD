"""
SS Tutor BD - Unified Benchmarking CLI
Entry point for managing models, running tokenizer evaluations, executing 100-item benchmarks, and viewing reports.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.manager import (
    list_candidates,
    get_candidate,
    get_active_model,
    download_candidate,
    purge_active_model,
    get_disk_free_mb
)
from benchmarks.tokenizer.benchmark import evaluate_tokenizer, print_comparison_table
from benchmark_runner.runner import run_benchmark


def print_banner():
    print("\n" + "=" * 80)
    print("      SS TUTOR BD — MODEL BENCHMARKING & RESEARCH HARNESS (PHASE 1)")
    print("=" * 80)


def print_help():
    print_banner()
    print("""
Usage: python benchmark_runner/cli.py <command> [arguments]

Commands:
  list                     List all registered candidate models and license gate statuses
  status                   Show currently active model and host disk space
  verify <CAND_ID>         Audit license gate compliance for a candidate
  download <CAND_ID>       Download a single candidate model (enforces Gate 1 and disk limits)
  tokenizer [CAND_IDs...]  Run the Bengali tokenizer efficiency benchmark on specified models
  benchmark <CAND_ID>      Execute the full 100-item offline benchmark on active/candidate model
  benchmark-mock <CAND_ID> Execute dry-run 100-item benchmark (fast validation)
  purge                    Safely delete active model weights and free disk space
  help                     Show this help message
""")


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "list":
        print_banner()
        candidates = list_candidates()
        print(f"{'ID':<10} {'Name':<28} {'Params':<8} {'License':<18} {'Gate 1 (License)':<20} {'Est MB'}")
        print("-" * 92)
        for c in candidates:
            print(f"{c['id']:<10} {c['name']:<28} {c['parameters_billion']:<8} {c['license']:<18} {c['license_status']:<20} {c['est_file_size_mb']}")
        print(f"\nAvailable Disk Space (Drive C:): {round(get_disk_free_mb(PROJECT_ROOT), 2)} MB\n")

    elif cmd == "status":
        print_banner()
        active = get_active_model()
        if active:
            print(f"Active Model:     {active['id']} ({active['name']})")
            print(f"Quantization:     {active.get('quantization', 'Q4_K_M')}")
            print(f"File Size:        {active['file_size_mb']} MB")
            print(f"Local Path:       {active['file_path']}")
        else:
            print("No model currently active. (Directory models/active/ is clean)")
        print(f"Available Disk:   {round(get_disk_free_mb(PROJECT_ROOT), 2)} MB\n")

    elif cmd == "verify" and len(sys.argv) > 2:
        cid = sys.argv[2].upper()
        cand = get_candidate(cid)
        if not cand:
            print(f"Error: Candidate '{cid}' not found in registry.")
            sys.exit(1)
        print(f"Candidate:        {cand['id']} ({cand['name']})")
        print(f"Declared License: {cand['license']}")
        print(f"Gate 1 Status:    {cand['license_status']}")
        if cand['license_status'] == "LICENSE_PASSED":
            print("Verdict:          ELIGIBLE FOR BENCHMARKING (Passed Gate 1)")
        else:
            print("Verdict:          BLOCKED (Requires Primary-Source Legal Verification)")

    elif cmd == "download" and len(sys.argv) > 2:
        cid = sys.argv[2].upper()
        try:
            res = download_candidate(cid)
            print(f"\nSuccessfully downloaded {res['candidate_id']}.")
            print(f"File Size: {res['file_size_mb']} MB | Saved to: {res['file_path']}")
            print(f"Remaining Disk Space: {res['remaining_disk_mb']} MB\n")
        except Exception as e:
            print(f"\nDownload Failed: {str(e)}\n")
            sys.exit(1)

    elif cmd == "tokenizer":
        cands = sys.argv[2:] if len(sys.argv) > 2 else [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ]
        results = []
        for c in cands:
            cand_meta = get_candidate(c)
            tok_repo = cand_meta.get("tokenizer_repo_id", c) if cand_meta else c
            print(f"Evaluating Tokenizer: {tok_repo}...")
            try:
                results.append(evaluate_tokenizer(tok_repo))
            except Exception as e:
                print(f"Failed {tok_repo}: {e}")
        print_comparison_table(results)

    elif cmd == "benchmark" and len(sys.argv) > 2:
        cid = sys.argv[2].upper()
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
        try:
            run_benchmark(candidate_id=cid, runtime_type="llama_cpp", limit=limit)
        except Exception as e:
            print(f"\nBenchmark Failed: {str(e)}\n")
            sys.exit(1)

    elif cmd == "benchmark-mock":
        cid = sys.argv[2].upper() if len(sys.argv) > 2 else "CAND-01"
        run_benchmark(candidate_id=cid, runtime_type="mock")

    elif cmd == "purge":
        res = purge_active_model()
        print(f"\nPurge Complete. Removed files: {res['files_removed']}")
        print(f"Freed: {res['freed_mb']} MB | Current Free Disk: {res['disk_free_mb']} MB\n")

    else:
        print_help()


if __name__ == "__main__":
    main()
