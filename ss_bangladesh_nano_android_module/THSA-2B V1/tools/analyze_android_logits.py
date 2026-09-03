#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect and compare all 10 Android logit files with Reference-B.
"""

import os
import sys
import numpy as np
import hashlib
from pathlib import Path

FIX12B = Path(r"ss_bangladesh_nano_android_module/THSA-2B V1/tools/fix12b")
VOCAB = 65536

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def main():
    print("=" * 80)
    print("ANDROID LOGIT DUMPS INSPECTION (10 FILES)")
    print("=" * 80)

    android_logits = {}
    for i in range(10):
        fname = f"fix12_logits_p{i}.bin"
        p = FIX12B / fname
        if not p.exists():
            print(f"Missing {fname}")
            continue
        data = open(p, "rb").read()
        sz = len(data)
        if sz != VOCAB * 4:
            print(f"{fname}: UNEXPECTED SIZE {sz} bytes (expected {VOCAB*4})")
            continue
        arr = np.frombuffer(data, dtype=np.float32)
        am = int(np.argmax(arr))
        vmax = float(arr[am])
        vmin = float(np.min(arr))
        vmean = float(np.mean(arr))
        top5 = np.argsort(arr)[-5:][::-1].tolist()
        top5_vals = [round(float(arr[idx]), 4) for idx in top5]
        h = sha256(data)
        android_logits[i] = {
            "file": fname,
            "argmax": am,
            "max": vmax,
            "min": vmin,
            "mean": vmean,
            "top5": top5,
            "top5_vals": top5_vals,
            "sha256": h,
            "array": arr
        }
        print(f"p{i}: argmax={am:5d}  max={vmax:8.4f}  min={vmin:8.4f}  mean={vmean:8.4f}  top5={top5}  sha256={h[:16]}...")

    print("\n" + "=" * 80)
    print("REFERENCE-B LOGIT DUMPS INSPECTION (5 FILES)")
    print("=" * 80)

    ref_b = {}
    for i in range(5):
        fname = f"reference_b_logits_p{i}.bin"
        p = FIX12B / fname
        if not p.exists():
            print(f"Missing {fname}")
            continue
        data = open(p, "rb").read()
        arr = np.frombuffer(data, dtype=np.float32)
        am = int(np.argmax(arr))
        vmax = float(arr[am])
        vmin = float(np.min(arr))
        vmean = float(np.mean(arr))
        top5 = np.argsort(arr)[-5:][::-1].tolist()
        top5_vals = [round(float(arr[idx]), 4) for idx in top5]
        h = sha256(data)
        ref_b[i] = {
            "file": fname,
            "argmax": am,
            "max": vmax,
            "min": vmin,
            "mean": vmean,
            "top5": top5,
            "top5_vals": top5_vals,
            "sha256": h,
            "array": arr
        }
        print(f"RefB p{i}: argmax={am:5d}  max={vmax:8.4f}  min={vmin:8.4f}  mean={vmean:8.4f}  top5={top5}  sha256={h[:16]}...")

    # Now compare each Reference-B file (0..4 for TEST-A..TEST-E)
    # with candidate Android logit files to find the exact match!
    print("\n" + "=" * 80)
    print("CROSS-CORRELATION: MATCHING REFERENCE-B TO ANDROID LOGIT FILES")
    print("=" * 80)

    labels = ["TEST-A", "TEST-B", "TEST-C", "TEST-D", "TEST-E"]
    for r_idx, label in enumerate(labels):
        if r_idx not in ref_b:
            continue
        r_arr = ref_b[r_idx]["array"].astype(np.float64)
        r_norm = np.linalg.norm(r_arr)
        r_am = ref_b[r_idx]["argmax"]
        print(f"\n--- {label} (RefB p{r_idx}, target argmax={r_am}) ---")
        best_cos = -1.0
        best_a_idx = -1
        for a_idx in sorted(android_logits.keys()):
            a_arr = android_logits[a_idx]["array"].astype(np.float64)
            a_norm = np.linalg.norm(a_arr)
            cos = float(np.dot(r_arr, a_arr) / (r_norm * a_norm + 1e-12))
            diff = np.abs(r_arr - a_arr)
            mae = float(np.mean(diff))
            max_ae = float(np.max(diff))
            a_am = android_logits[a_idx]["argmax"]
            match_str = "MATCH" if a_am == r_am else "DIFF "
            print(f"  vs Android p{a_idx}: cos={cos:.6f}  mae={mae:.6f}  max_ae={max_ae:.6f}  argmax={a_am} [{match_str}]")
            if cos > best_cos:
                best_cos = cos
                best_a_idx = a_idx
        print(f"  => Best Match for {label}: Android p{best_a_idx} with cosine={best_cos:.6f}")

if __name__ == "__main__":
    main()
