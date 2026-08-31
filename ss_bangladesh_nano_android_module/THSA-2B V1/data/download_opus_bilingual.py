#!/usr/bin/env python3
"""
THSA-2B Pre-Training Dataset: Step 2 of 5
Download & Extract Bilingual Bengali-English Parallel Corpus (OPUS-100)
==============================================================================
Dataset:    Helsinki-NLP/opus-100 (bn-en)
Target:     data/raw/bilingual/bilingual_bn_en.txt
Purpose:    Enables seamless English <-> Bengali translation and code-switching.
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "raw", "bilingual")
OUT_FILE = os.path.join(OUT_DIR, "bilingual_bn_en.txt")
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
    log("THSA-2B DATASET PIPELINE — STEP 2: Bilingual (English-Bengali)")
    log("=" * 60)
    log("Streaming Helsinki-NLP/opus-100 (bn-en parallel pairs)...")

    written = 0
    skipped = 0

    ds = load_dataset("Helsinki-NLP/opus-100", "bn-en", split="train", streaming=True)

    with open(OUT_FILE, "w", encoding="utf-8", buffering=1024*1024) as out:
        for item in ds:
            trans = item.get("translation", {})
            en = trans.get("en", "").strip()
            bn = trans.get("bn", "").strip()

            # Skip empty or ultra-short noise
            if len(en) < 3 or len(bn) < 3:
                skipped += 1
                continue

            # Format parallel pairs for pre-training
            out.write(f"English: {en}\n")
            out.write(f"Bengali: {bn}\n\n")
            written += 1

            if written % 25000 == 0:
                size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
                log(f"  Parallel pairs written: {written:,} | File Size: {size_mb:.2f} MB")

    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    log("=" * 60)
    log("STEP 2 COMPLETE: Bilingual Corpus Downloaded & Formatted")
    log("=" * 60)
    log(f"  Output file:      {OUT_FILE}")
    log(f"  Pairs written:    {written:,}")
    log(f"  Noise skipped:    {skipped:,}")
    log(f"  Final File Size:  {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
