#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX-12B Phase E/F/J — Full 65,536 Logits Multi-Source Comparison Forensic
==========================================================================
Compares:
  REFERENCE-A (Original Step-30 PyTorch Checkpoint from Colab)
      vs
  REFERENCE-B (Nano V2 Python Emulation from production model.nano)
      vs
  ANDROID NATIVE (On-device execution from physical phone)

Computes for each prompt (all 65,536 dimensions):
  - Max Absolute Error
  - Mean Absolute Error
  - Root Mean Square Error (RMSE)
  - Cosine Similarity
  - Top-1 Match
  - Top-5 Overlap
  - Top-10 Overlap
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import struct
import hashlib
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIX12B_DIR = SCRIPT_DIR / "fix12b"
REPORT_DIR = SCRIPT_DIR.parent

PROMPTS = [
    ("TEST-A", "2+2=?"),
    ("TEST-B", "বাংলাদেশের রাজধানী কী?"),
    ("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?"),
    ("TEST-D", "১২ × ৮ = ?"),
    ("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।"),
]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def top_k_indices(arr: np.ndarray, k: int = 10) -> list:
    idx = np.argpartition(arr, -k)[-k:]
    return [int(x) for x in idx[np.argsort(arr[idx])[::-1]]]

def compare_vectors(vec1: np.ndarray, vec2: np.ndarray, name1: str, name2: str) -> dict:
    assert vec1.shape == vec2.shape == (65536,), f"Shape mismatch: {vec1.shape} vs {vec2.shape}"
    diff = np.abs(vec1.astype(np.float64) - vec2.astype(np.float64))
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    cos = cosine_similarity(vec1, vec2)

    top10_1 = top_k_indices(vec1, 10)
    top10_2 = top_k_indices(vec2, 10)

    top1_match = bool(top10_1[0] == top10_2[0])
    top5_overlap = len(set(top10_1[:5]) & set(top10_2[:5]))
    top10_overlap = len(set(top10_1) & set(top10_2))

    return {
        "pair": f"{name1}_vs_{name2}",
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "cosine_similarity": cos,
        "top1_match": top1_match,
        "top1_1": top10_1[0],
        "top1_2": top10_2[0],
        "top5_1": top10_1[:5],
        "top5_2": top10_2[:5],
        "top5_overlap": top5_overlap,
        "top10_overlap": top10_overlap,
    }

def main():
    print("=" * 75)
    print("FIX-12B FULL 65,536 LOGITS COMPARISON FORENSIC")
    print("=" * 75)

    ref_b_files = [FIX12B_DIR / f"reference_b_logits_p{i}.bin" for i in range(5)]
    ref_a_files = [FIX12B_DIR / f"reference_a_logits_p{i}.bin" for i in range(5)]
    android_files = [FIX12B_DIR / f"android_logits_p{i}.bin" for i in range(5)]

    has_ref_b = all(p.exists() for p in ref_b_files)
    has_ref_a = all(p.exists() for p in ref_a_files)
    has_android = all(p.exists() for p in android_files)

    print(f"Reference-B (Nano Python):     {'READY (' + str(len(ref_b_files)) + ' files)' if has_ref_b else 'MISSING'}")
    print(f"Reference-A (Step-30 PyTorch): {'READY (' + str(len(ref_a_files)) + ' files)' if has_ref_a else 'NOT FOUND (Requires Colab step)'}")
    print(f"Android Native:                {'READY (' + str(len(android_files)) + ' files)' if has_android else 'PENDING DEVICE PULL'}")

    results = {}

    for i, (label, prompt) in enumerate(PROMPTS):
        print(f"\n[{label}] '{prompt}'")
        prompt_res = {"label": label, "prompt": prompt}

        v_b = np.fromfile(ref_b_files[i], dtype=np.float32) if has_ref_b else None
        v_a = np.fromfile(ref_a_files[i], dtype=np.float32) if has_ref_a else None
        v_and = np.fromfile(android_files[i], dtype=np.float32) if has_android else None

        if v_b is not None and v_a is not None:
            comp_ab = compare_vectors(v_a, v_b, "RefA", "RefB")
            prompt_res["A_vs_B"] = comp_ab
            print(f"  A <-> B: Cosine={comp_ab['cosine_similarity']:.6f} MaxAbs={comp_ab['max_abs_error']:.4f} Top1={'MATCH' if comp_ab['top1_match'] else 'DIFF'} Top5Overlap={comp_ab['top5_overlap']}/5")

        if v_b is not None and v_and is not None:
            comp_band = compare_vectors(v_b, v_and, "RefB", "Android")
            prompt_res["B_vs_Android"] = comp_band
            print(f"  B <-> Android: Cosine={comp_band['cosine_similarity']:.6f} MaxAbs={comp_band['max_abs_error']:.4f} Top1={'MATCH' if comp_band['top1_match'] else 'DIFF'} Top5Overlap={comp_band['top5_overlap']}/5")

        if v_a is not None and v_and is not None:
            comp_aand = compare_vectors(v_a, v_and, "RefA", "Android")
            prompt_res["A_vs_Android"] = comp_aand
            print(f"  A <-> Android: Cosine={comp_aand['cosine_similarity']:.6f} MaxAbs={comp_aand['max_abs_error']:.4f} Top1={'MATCH' if comp_aand['top1_match'] else 'DIFF'}")

        results[label] = prompt_res

    out_json = FIX12B_DIR / "full_logits_comparison_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "has_ref_a": has_ref_a,
            "has_ref_b": has_ref_b,
            "has_android": has_android,
            "results": results
        }, f, indent=2)
    print(f"\nComparison results written to {out_json}")

if __name__ == "__main__":
    main()
