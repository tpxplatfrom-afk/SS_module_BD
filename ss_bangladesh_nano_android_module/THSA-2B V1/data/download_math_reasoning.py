#!/usr/bin/env python3
"""
THSA-2B Pre-Training Dataset: Mathematical & Step-by-Step Logic
===============================================================
Dataset:    openai/gsm8k (Grade School Math 8K)
Target:     data/raw/math_reasoning/math_gsm8k.txt
Purpose:    Teaches the model step-by-step arithmetic, logic, and question-answering.
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "raw", "math_reasoning")
OUT_FILE = os.path.join(OUT_DIR, "math_gsm8k.txt")
LOG_FILE = os.path.join(OUT_DIR, "download_log.txt")
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    log("=" * 60)
    log("THSA-2B DATASET PIPELINE — Mathematical & Logical Reasoning")
    log("=" * 60)
    log("Streaming GSM8K (Grade School Math Reasoning)...")

    written = 0

    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for item in ds:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()

            out.write(f"Question: {q}\n")
            out.write(f"Reasoning & Solution: {a}\n\n")
            written += 1

            if written % 2500 == 0:
                log(f"  Math problems written: {written:,}")

    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    log("=" * 60)
    log("GSM8K MATH DATASET DOWNLOAD COMPLETE")
    log("=" * 60)
    log(f"  Output file:          {OUT_FILE}")
    log(f"  Problems written:     {written:,}")
    log(f"  Final File Size:      {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
