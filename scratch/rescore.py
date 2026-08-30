import sys
import json
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_runner.scoring import calculate_composite_scorecard
from benchmark_runner.reporter import save_raw_result, generate_markdown_report, extract_and_save_failures

def rescore_raw(raw_file: Path):
    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidate = data["candidate"]
    metrics = data["metrics"]
    test_results = data["test_results"]

    scorecard = calculate_composite_scorecard(candidate, test_results, metrics)
    data["scorecard"] = scorecard

    # Overwrite raw file
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    failures_path = extract_and_save_failures(candidate["id"], test_results)
    report_path = generate_markdown_report(candidate, scorecard, metrics, test_results)

    print(f"Rescored {candidate['id']} - Total Score: {scorecard['total_score']} / 100")
    print(f"Updated Report: {report_path}")
    print(f"Updated Failures: {failures_path}")

if __name__ == "__main__":
    cand = sys.argv[1] if len(sys.argv) > 1 else "CAND-02"
    raw_files = list((PROJECT_ROOT / "results" / "raw").glob(f"{cand}_*.json"))
    if raw_files:
        latest = sorted(raw_files, key=lambda p: p.stat().st_mtime)[-1]
        rescore_raw(latest)
    else:
        print(f"No raw files found for {cand}")
