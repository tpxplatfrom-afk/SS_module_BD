#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX-12C Layer-by-Layer Forensic Equivalence Analyzer
====================================================
Compares Reference-B intermediate checkpoints vs Android Native intermediate
checkpoints across all 5 canonical prompts and generates full numerical metrics:
  - Cosine Similarity
  - Max Absolute Error
  - Mean Absolute Error
  - RMSE
  - L2 Relative Error
  - Elements matching within tolerance
  - Exact localization of first divergence
"""

import os
import sys
import json
import math
import hashlib
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
REF_B_DIR = SCRIPT_DIR / "fix12c" / "reference_b"
ANDROID_DIR = SCRIPT_DIR / "fix12c" / "android"
OUT_REPORT = SCRIPT_DIR.parent / "FIX-12C-STEP30-LAYERWISE-NUMERICAL-EQUIVALENCE-FORENSIC-REPORT.md"

PROMPTS = [
    ("TEST-A", "2+2=?"),
    ("TEST-B", "বাংলাদেশের রাজধানী কী?"),
    ("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?"),
    ("TEST-D", "১২ × ৮ = ?"),
    ("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।"),
]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def compute_metrics(v_ref: np.ndarray, v_act: np.ndarray):
    v_ref64 = v_ref.astype(np.float64).ravel()
    v_act64 = v_act.astype(np.float64).ravel()
    dim = len(v_ref64)
    assert dim == len(v_act64), f"Dimension mismatch: {dim} vs {len(v_act64)}"

    diff = np.abs(v_ref64 - v_act64)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    norm_ref = np.sqrt(np.sum(v_ref64 ** 2))
    norm_act = np.sqrt(np.sum(v_act64 ** 2))
    norm_diff = np.sqrt(np.sum(diff ** 2))

    l2_rel = float(norm_diff / (norm_ref + 1e-12))

    denom = norm_ref * norm_act
    if denom == 0:
        cosine = 1.0 if norm_ref == norm_act else 0.0
    else:
        cosine = float(np.dot(v_ref64, v_act64) / denom)
    cosine = max(-1.0, min(1.0, cosine))

    m_1e3 = int(np.sum(diff < 1e-3))
    m_1e2 = int(np.sum(diff < 1e-2))
    m_1e1 = int(np.sum(diff < 1e-1))

    return {
        "dim": dim,
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rmse": rmse,
        "cosine": cosine,
        "l2_rel_err": l2_rel,
        "match_pct_1e3": 100.0 * m_1e3 / dim,
        "match_pct_1e2": 100.0 * m_1e2 / dim,
        "match_pct_1e1": 100.0 * m_1e1 / dim,
        "ref_norm": float(norm_ref),
        "act_norm": float(norm_act),
    }

def main():
    print("=" * 80)
    print("FIX-12C LAYER-BY-LAYER FORENSIC EQUIVALENCE AUDIT")
    print("=" * 80)

    comparison_results = {}
    first_divergence_overall = None

    ANDROID_MAP = [0, 2, 4, 6, 8]

    for pi, (label, prompt_text) in enumerate(PROMPTS):
        and_idx = ANDROID_MAP[pi]
        print(f"\n--- Analyzing [{label}] '{prompt_text}' (Ref prompt_{pi} vs Android prompt_{and_idx}) ---")
        p_ref = REF_B_DIR / f"prompt_{pi}"
        p_and = ANDROID_DIR / f"prompt_{and_idx}"

        if not p_ref.exists():
            print(f"ERROR: Missing Reference-B directory for prompt {pi}: {p_ref}")
            continue
        if not p_and.exists():
            print(f"ERROR: Missing Android directory for prompt {pi}: {p_and}")
            continue

        ref_files = set(f.name for f in p_ref.glob("*.bin"))
        and_files = set(f.name for f in p_and.glob("*.bin"))
        common = sorted(ref_files.intersection(and_files))

        print(f"Total matching checkpoint files: {len(common)} (Ref: {len(ref_files)}, Android: {len(and_files)})")

        ckpt_metrics = {}
        first_div = None

        for fname in common:
            c_name = fname[:-4]
            f_ref = p_ref / fname
            f_and = p_and / fname

            v_ref = np.fromfile(f_ref, dtype=np.float32)
            v_and = np.fromfile(f_and, dtype=np.float32)

            if len(v_ref) != len(v_and):
                print(f"Size mismatch in {fname}: ref={len(v_ref)}, android={len(v_and)}")
                continue

            m = compute_metrics(v_ref, v_and)
            ckpt_metrics[c_name] = m

            # Divergence threshold for INT8/Ternary quantized runtime vs float reference:
            # Cosine < 0.95 or L2 relative error > 0.35
            if (m["cosine"] < 0.95 or m["l2_rel_err"] > 0.35) and first_div is None:
                first_div = (c_name, m)

        comparison_results[label] = {
            "label": label,
            "prompt": prompt_text,
            "checkpoints_compared": len(ckpt_metrics),
            "first_divergence": first_div[0] if first_div else "NONE",
            "metrics": ckpt_metrics,
        }

        # Check logits
        if "ckpt24_logits" in ckpt_metrics:
            lg_ref = np.fromfile(p_ref / "ckpt24_logits.bin", dtype=np.float32)
            lg_and = np.fromfile(p_and / "ckpt24_logits.bin", dtype=np.float32)
            am_ref = int(np.argmax(lg_ref))
            am_and = int(np.argmax(lg_and))
            match = (am_ref == am_and)
            m_lg = ckpt_metrics["ckpt24_logits"]
            print(f"  LOGITS: Argmax Match: {match} (Ref: {am_ref}, Android: {am_and})")
            print(f"  LOGITS: Cosine: {m_lg['cosine']:.6f}, MaxAbsErr: {m_lg['max_abs_err']:.4f}, L2RelErr: {m_lg['l2_rel_err']:.4f}")

    json_path = SCRIPT_DIR / "fix12c" / "layerwise_comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nDetailed JSON results saved to: {json_path}")

if __name__ == "__main__":
    main()
