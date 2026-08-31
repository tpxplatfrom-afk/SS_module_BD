#!/usr/bin/env python3
"""
THSA-2B Pre-Training Dataset: Step 1 of 5
Download & Extract Bengali Wikipedia
======================================
Dataset:    wikimedia/wikipedia (20231101.bn)
Target:     data/raw/bengali_wikipedia/bengali_wiki.txt
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "raw", "bengali_wikipedia")
OUT_FILE = os.path.join(OUT_DIR, "bengali_wiki.txt")
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
    log("THSA-2B DATASET PIPELINE — STEP 1: Bengali Wikipedia")
    log("=" * 60)
    log("Streaming Bengali Wikipedia from HuggingFace (20231101.bn)...")

    MIN_CHARS = 150
    written = 0
    skipped = 0

    ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True)

    with open(OUT_FILE, "w", encoding="utf-8", buffering=1024*1024) as out:
        for article in ds:
            text = article.get("text", "").strip()
            title = article.get("title", "").strip()

            if len(text) < MIN_CHARS:
                skipped += 1
                continue

            # Format article with clean document boundaries
            out.write(f"=== {title} ===\n")
            out.write(text)
            out.write("\n\n")
            written += 1

            if written % 5000 == 0:
                size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
                log(f"  Processed articles: {written + skipped:,} | Written: {written:,} | Current Size: {size_mb:.2f} MB")

    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    log("=" * 60)
    log("STEP 1 COMPLETE: Bengali Wikipedia Downloaded & Extracted")
    log("=" * 60)
    log(f"  Output file:          {OUT_FILE}")
    log(f"  Articles written:     {written:,}")
    log(f"  Short stubs skipped:  {skipped:,}")
    log(f"  Final File Size:      {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
