"""
SS Tutor BD — Release Artifact Audit (Phase 5)
Scans the repository for unauthorized training checkpoints, optimizer states,
duplicate model files, real API keys, and other release-prohibited artifacts.
"""
import sys
import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROHIBITED_FILENAMES = [
    r"optimizer\.pt$", r"optimizer\.bin$", r"checkpoint-\d+", r"checkpoint_epoch",
    r"trainer_state\.json$", r"training_args\.bin$",
    r"qwen.*\.gguf$", r"qwen.*\.bin$", r"smollm.*\.gguf$", r"smollm.*\.bin$",
    r"tinyllama.*\.gguf$", r"tinyllama.*\.bin$",
    r"__pycache__", r"\.pyc$", r"\.ipynb$"
]

API_KEY_PATTERNS = [
    r"sk-[a-zA-Z0-9]{24,}",
    r"hf_[a-zA-Z0-9]{24,}",
    r"api_key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]",
]

LARGE_FILE_THRESHOLD_MB = 100

EXCLUDED_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "sstutor_bengali_70m_edu",  # Training weights directory (excluded from production APK)
}

APPROVED_LARGE_PATHS = [
    "models/export_int4",
    "models/sstutor_bengali_70m_edu",
]


def audit_release(report_path: Path | None = None) -> dict:
    print("\n" + "="*70)
    print("  SS TUTOR BD — RELEASE ARTIFACT AUDIT")
    print("="*70)

    issues = []
    warnings = []
    large_files = []
    scanned = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]

        for fname in files:
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(PROJECT_ROOT))
            scanned += 1

            # Check prohibited filename patterns
            for pat in PROHIBITED_FILENAMES:
                if re.search(pat, fname, re.IGNORECASE):
                    issues.append({"type": "PROHIBITED_FILE", "path": rel, "pattern": pat})

            # Check large files
            try:
                size_mb = fpath.stat().st_size / (1024 * 1024)
                if size_mb > LARGE_FILE_THRESHOLD_MB:
                    approved = any(ap in rel.replace("\\", "/") for ap in APPROVED_LARGE_PATHS)
                    entry = {"path": rel, "size_mb": round(size_mb, 2), "approved": approved}
                    large_files.append(entry)
                    if not approved:
                        warnings.append({"type": "LARGE_UNAPPROVED_FILE", **entry})
            except OSError:
                pass

            # Check text files for genuine API key leakage (skip audit script itself)
            if fname != "audit_release.py" and fname.endswith((".py", ".json", ".md", ".txt", ".yaml", ".yml", ".env")):
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    for pat in API_KEY_PATTERNS:
                        if re.search(pat, content):
                            issues.append({"type": "API_KEY_LEAK", "path": rel, "pattern": pat})
                            break
                except Exception:
                    pass

    # Summary
    print(f"\n  Files Scanned:     {scanned}")
    print(f"  Issues Found:      {len(issues)}")
    print(f"  Warnings Found:    {len(warnings)}")
    print(f"  Large Files:       {len(large_files)}")

    if issues:
        print("\n  ISSUES:")
        for issue in issues[:10]:
            print(f"    [!] {issue['type']}: {issue['path']}")
    else:
        print("\n  No prohibited artifacts or API keys detected.")

    if warnings:
        print("\n  WARNINGS (large unapproved files):")
        for w in warnings[:5]:
            print(f"    [W] {w['path']} ({w['size_mb']} MB)")

    verdict = "PASS" if len(issues) == 0 else "FAIL"
    print(f"\n  AUDIT VERDICT: {verdict}")
    print("="*70 + "\n")

    result = {
        "files_scanned": scanned,
        "issues": issues,
        "warnings": warnings,
        "large_files": large_files,
        "verdict": verdict
    }

    # Save report
    out = report_path or PROJECT_ROOT / "results" / "phase5" / "release_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Audit report saved to: {out}")
    return result


if __name__ == "__main__":
    audit_release()
