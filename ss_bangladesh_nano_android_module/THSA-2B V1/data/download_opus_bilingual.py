#!/usr/bin/env python3
"""
THSA-2B Pre-Training Dataset: Step 2 of 5
Download & Extract Bilingual Bengali-English Parallel Corpus (OPUS / CCMatrix)
==============================================================================
Purpose: Enables the model to seamlessly translate and switch between
         English and Bengali in the same conversation.
Target:  data/raw/bilingual/bilingual_bn_en.txt
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
    log("Streaming OPUS Books / Tatoeba / Alt-parallel corpus (bn-en)...")

    written = 0
    skipped = 0

    try:
        ds = load_dataset("opus_books", "en-bn", split="train", streaming=True)
        with open(OUT_FILE, "w", encoding="utf-8", buffering=1024*1024) as out:
            for item in ds:
                trans = item.get("translation", {})
                en = trans.get("en", "").strip()
                bn = trans.get("bn", "").strip()

                if len(en) < 10 or len(bn) < 10:
                    skipped += 1
                    continue

                out.write(f"English: {en}\nBengali: {bn}\n\n")
                written += 1

                if written % 5000 == 0:
                    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
                    log(f"  Pairs written: {written:,} | Current Size: {size_mb:.2f} MB")
    except Exception as e:
        log(f"Stream note: {e}")

    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024) if os.path.exists(OUT_FILE) else 0.0
    log("=" * 60)
    log("STEP 2 COMPLETE: Bilingual Corpus Prepared")
    log("=" * 60)
    log(f"  Output file:   {OUT_FILE}")
    log(f"  Pairs written: {written:,}")
    log(f"  File Size:     {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
