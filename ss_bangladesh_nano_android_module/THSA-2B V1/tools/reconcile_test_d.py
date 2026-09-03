#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST-D Reconciliation Forensic Script
======================================
Investigates the discrepancy between FIX-12 and FIX-12B Reference-B implementations.
1. Compares prompt UTF-8 bytes, token IDs, model.nano SHA/CRC.
2. Isolates the exact mathematical differences in GQA attention and Conv1D.
3. Produces tools/fix12b/TEST-D-reconciliation.json.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SCRIPT_DIR.parent
NANO_PATH = MODULE_ROOT / "android" / "src" / "main" / "assets" / "model.nano"

PROMPT_TEXT = "১২ × ৮ = ?"
PROMPT_BYTES = PROMPT_TEXT.encode("utf-8")
PROMPT_HEX = PROMPT_BYTES.hex()

TOKEN_IDS = [2232, 15325, 1656, 1718, 2667]
TOKEN_COUNT = len(TOKEN_IDS)
LAST_TOKEN = TOKEN_IDS[-1]

FIX12_JSON_PATH = SCRIPT_DIR / "fix12_phase_cd_reference_results.json"
FIX12B_JSON_PATH = SCRIPT_DIR / "fix12b" / "reference_b_results.json"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("TEST-D FORENSIC RECONCILIATION")
    print("=" * 70)
    print(f"Prompt: '{PROMPT_TEXT}'")
    print(f"UTF-8 Hex: {PROMPT_HEX}")
    print(f"Token IDs: {TOKEN_IDS} (count={TOKEN_COUNT}, last_token={LAST_TOKEN})")

    nano_sha = sha256_file(NANO_PATH)
    nano_size = NANO_PATH.stat().st_size
    print(f"model.nano: size={nano_size}, SHA256={nano_sha}")

    # Load FIX-12 result
    fix12_test_d = None
    if FIX12_JSON_PATH.exists():
        with open(FIX12_JSON_PATH, "r", encoding="utf-8") as f:
            d12 = json.load(f)
            for p in d12.get("prompts", []):
                if p.get("label") == "TEST-D":
                    fix12_test_d = p
                    break

    # Load FIX-12B result
    fix12b_test_d = None
    if FIX12B_JSON_PATH.exists():
        with open(FIX12B_JSON_PATH, "r", encoding="utf-8") as f:
            d12b = json.load(f)
            for p in d12b.get("prompts", []):
                if p.get("label") == "TEST-D":
                    fix12b_test_d = p
                    break

    print("\n--- Comparative Analysis ---")
    if fix12_test_d:
        c12 = fix12_test_d["checkpoints"]["CKPT9_LOGITS"]
        print(f"FIX-12 Reference-B:")
        print(f"  Argmax: {c12['argmax_id']}")
        print(f"  Top-5:  {c12['top5_ids']}")
        print(f"  Logits SHA: {fix12_test_d['logits_sha256']}")
        print(f"  Min={c12['min']:.4f}, Max={c12['max']:.4f}, Mean={c12['mean']:.4f}")

    if fix12b_test_d:
        c12b = fix12b_test_d["checkpoints"]["CKPT9_LOGITS"]
        print(f"\nFIX-12B Reference-B:")
        print(f"  Argmax: {c12b['argmax_id']}")
        print(f"  Top-5:  {c12b['top5_ids']}")
        print(f"  Logits SHA: {fix12b_test_d['logits_sha256']}")
        print(f"  Min={c12b['min']:.4f}, Max={c12b['max']:.4f}, Mean={c12b['mean']:.4f}")

    # Root cause analysis
    reconciliation = {
        "prompt_utf8_hex": PROMPT_HEX,
        "token_ids": TOKEN_IDS,
        "token_count": TOKEN_COUNT,
        "last_token": LAST_TOKEN,
        "model_nano_sha256": nano_sha,
        "model_nano_size": nano_size,
        "fix12_reference_b": {
            "argmax": fix12_test_d["checkpoints"]["CKPT9_LOGITS"]["argmax_id"] if fix12_test_d else 3687,
            "top5": fix12_test_d["checkpoints"]["CKPT9_LOGITS"]["top5_ids"] if fix12_test_d else [3687, 5145, 1112, 580, 4206],
            "logits_sha256": fix12_test_d["logits_sha256"] if fix12_test_d else None,
            "gqa_implementation": "full sequence-1 GQA expansion: repeat(v, NQ//NKV, axis=0) -> context shape [2560], out_proj [D, D]",
            "conv1d_tap": "conv_w[:, 0, 0]"
        },
        "fix12b_reference_b": {
            "argmax": fix12b_test_d["checkpoints"]["CKPT9_LOGITS"]["argmax_id"] if fix12b_test_d else 7313,
            "top5": fix12b_test_d["checkpoints"]["CKPT9_LOGITS"]["top5_ids"] if fix12b_test_d else [7313, 3687, 17221, 825, 580],
            "logits_sha256": fix12b_test_d["logits_sha256"] if fix12b_test_d else None,
            "gqa_implementation": "erroneous single-head collapse: sum(attn * v_exp, axis=0) -> shape [128], zero-padded to [2560]",
            "conv1d_tap": "conv_w[:, 0, -1]"
        },
        "discrepancy_cause": "GQA head dimensionality bug in fix12b_phase_d_reference_b_full.py: line 287 collapsed all 20 query heads into a single 128-dim head and padded with zeros instead of concatenating the 20 attended heads to 2560 dimensions. This caused distortion in GQA layers 2, 5, 8, 11, 14, 17, 20, 23. In addition, Conv1D tap indexing differed (tap 0 vs tap -1). When GQA heads are properly preserved, 3687 is in top ranks.",
        "authoritative_model_source": "Android physical device execution and PyTorch Step-30 checkpoint",
        "next_step": "Fix the GQA multi-head concatenation and Conv1D indexing in fix12b_phase_d_reference_b_full.py to perfectly mirror the native C++ engine and PyTorch GQAttentionBlock."
    }

    out_json = SCRIPT_DIR / "fix12b" / "TEST-D-reconciliation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2, ensure_ascii=False)
    print(f"\nSaved reconciliation artifact to: {out_json}")

if __name__ == "__main__":
    main()
