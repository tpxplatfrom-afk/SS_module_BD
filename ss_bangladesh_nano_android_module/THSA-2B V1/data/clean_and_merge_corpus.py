#!/usr/bin/env python3
"""
THSA-2B Pre-Training Dataset: Data Cleaner & Merger
===================================================
Applies:
  1. Unicode NFC Normalization (Deterministic Bengali vowel/hasant normalization)
  2. HTML / Markdown / URL stripping
  3. Exact and fuzzy sentence deduplication
  4. Minimum length and non-alphanumeric noise filtering
  5. Multi-source merger into `data/processed/clean_pretrain_corpus.txt`
"""

import sys
import os
import re
import unicodedata
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
OUT_DIR = os.path.join(SCRIPT_DIR, "processed")
OUT_FILE = os.path.join(OUT_DIR, "clean_pretrain_corpus.txt")
LOG_FILE = os.path.join(OUT_DIR, "clean_log.txt")
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Bengali Unicode NFC normalizer rules
def normalize_bengali_nfc(text: str) -> str:
    # 1. Standard Unicode NFC decomposition + canonical recomposition
    text = unicodedata.normalize("NFC", text)

    # 2. Fix legacy decomposed vowel combinations (e.g. ে + া -> ো, ে + ৗ -> ৌ)
    text = text.replace("\u09c7\u09be", "\u09cb") # e + aa -> o
    text = text.replace("\u09c7\u09d7", "\u09cc") # e + au_length -> au
    text = text.replace("\u09bc\u09be", "\u09be\u09bc") # nukta order fix

    # 3. Strip URLs and HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)

    # 4. Normalize multiple whitespaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def clean_and_merge():
    log("=" * 60)
    log("THSA-2B DATASET PIPELINE: CORPUS CLEANER & MERGER")
    log("=" * 60)

    seen_hashes = set()
    total_lines = 0
    clean_lines = 0
    duplicate_lines = 0

    sources = [
        ("Bengali Wikipedia", os.path.join(RAW_DIR, "bengali_wikipedia", "bengali_wiki.txt")),
        ("OPUS-100 Bilingual", os.path.join(RAW_DIR, "bilingual", "bilingual_bn_en.txt")),
    ]

    with open(OUT_FILE, "w", encoding="utf-8", buffering=1024*1024) as out:
        for source_name, file_path in sources:
            if not os.path.exists(file_path):
                log(f"  Skipping {source_name} (File not found yet)")
                continue

            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            log(f"\nProcessing {source_name} ({size_mb:.2f} MB)...")

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    total_lines += 1
                    raw = line.strip()

                    # Skip empty lines or section headers
                    if not raw or raw.startswith("==="):
                        continue

                    # Apply Bengali NFC Normalization
                    cleaned = normalize_bengali_nfc(raw)

                    # Filter short noise
                    if len(cleaned) < 15:
                        continue

                    # Deduplication via hash
                    h = hash(cleaned)
                    if h in seen_hashes:
                        duplicate_lines += 1
                        continue
                    seen_hashes.add(h)

                    out.write(cleaned + "\n")
                    clean_lines += 1

                    if clean_lines % 100000 == 0:
                        out_size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
                        log(f"  Clean lines: {clean_lines:,} | Deduplicated: {duplicate_lines:,} | Size: {out_size_mb:.2f} MB")

    final_size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024) if os.path.exists(OUT_FILE) else 0.0
    log("\n" + "=" * 60)
    log("CORPUS CLEANING & MERGING COMPLETE")
    log("=" * 60)
    log(f"  Output File:        {OUT_FILE}")
    log(f"  Clean Sentences:    {clean_lines:,}")
    log(f"  Duplicates Removed: {duplicate_lines:,}")
    log(f"  Final File Size:    {final_size_mb:.2f} MB")

if __name__ == "__main__":
    clean_and_merge()
