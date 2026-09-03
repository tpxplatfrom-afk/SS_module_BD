#!/usr/bin/env python3
"""
FIX-12 Phase B — Python Tokenizer Equivalence
==============================================
Encodes all 5 required test prompts using the authoritative
THSA SentencePiece tokenizer (thsa_tokenizer.model) and records
exact token ID sequences for comparison with Android native tokenizer.

Output: fix12_phase_b_python_tokens.json

Run from THSA-2B V1 root:
    python tools/fix12_phase_b_tokenizer.py
"""

import sys
import os
import json
import hashlib
import struct

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)  # THSA-2B V1 root
TOKENIZER_MODEL = os.path.join(MODULE_ROOT, "tokenizer", "thsa_tokenizer.model")
VOCAB_FILE      = os.path.join(MODULE_ROOT, "tokenizer", "thsa_tokenizer.vocab")
OUTPUT_JSON     = os.path.join(MODULE_ROOT, "tools", "fix12_phase_b_python_tokens.json")

# ── 5 authoritative test prompts (FIX-12 §7) ────────────────────────────────
TEST_PROMPTS = [
    ("TEST-A", "2+2=?"),
    ("TEST-B", "বাংলাদেশের রাজধানী কী?"),
    ("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?"),
    ("TEST-D", "১২ × ৮ = ?"),
    ("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।"),
]

def sha256_str(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def token_ids_sha256(ids):
    """SHA-256 of token ID sequence serialized as LE int32 array."""
    buf = struct.pack(f"<{len(ids)}i", *ids)
    return sha256_str(buf)

def load_vocab_size(vocab_file):
    """Count lines in .vocab file = vocabulary size."""
    if not os.path.isfile(vocab_file):
        return -1
    count = 0
    with open(vocab_file, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

def main():
    print("=" * 70)
    print("FIX-12 PHASE B — PYTHON TOKENIZER EQUIVALENCE")
    print("=" * 70)

    # ── Validate tokenizer exists ────────────────────────────────────────────
    print(f"\n[1] Tokenizer model : {TOKENIZER_MODEL}")
    if not os.path.isfile(TOKENIZER_MODEL):
        print(f"    ERROR: Not found: {TOKENIZER_MODEL}")
        sys.exit(1)
    tok_size = os.path.getsize(TOKENIZER_MODEL)
    tok_sha  = sha256_str(open(TOKENIZER_MODEL, "rb").read())
    print(f"    Size  : {tok_size:,} bytes")
    print(f"    SHA256: {tok_sha}")

    vocab_size = load_vocab_size(VOCAB_FILE)
    print(f"\n[2] Vocabulary file : {VOCAB_FILE}")
    print(f"    Vocab size      : {vocab_size:,} tokens")

    # ── Load SentencePiece tokenizer ─────────────────────────────────────────
    print("\n[3] Loading SentencePiece tokenizer ...")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(TOKENIZER_MODEL)
        sp_vocab_size = sp.GetPieceSize()
        print(f"    SP vocab size   : {sp_vocab_size:,}")
        if sp_vocab_size != 65536:
            print(f"    WARNING: Expected 65536, got {sp_vocab_size}")
    except Exception as e:
        print(f"    ERROR loading sentencepiece: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("TOKENIZER ENCODING — ALL 5 PROMPTS")
    print(f"{'='*70}")

    results = []
    all_pass = True

    for (label, prompt) in TEST_PROMPTS:
        print(f"\n[{label}] Prompt: {repr(prompt)}")
        try:
            # Encode using sentencepiece
            ids = sp.EncodeAsIds(prompt)
            pieces = sp.EncodeAsPieces(prompt)
            decoded = sp.Decode(ids)

            sha = token_ids_sha256(ids)

            print(f"    Token count : {len(ids)}")
            print(f"    Token IDs   : {ids}")
            print(f"    Pieces      : {pieces}")
            print(f"    Decoded     : {repr(decoded)}")
            print(f"    ID SHA256   : {sha}")

            result = {
                "label": label,
                "prompt": prompt,
                "token_count": len(ids),
                "token_ids": ids,
                "pieces": pieces,
                "decoded": decoded,
                "token_ids_sha256": sha,
                "status": "OK"
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {"label": label, "prompt": prompt, "status": f"ERROR: {e}"}
            all_pass = False

        results.append(result)

    # ── Write output JSON ─────────────────────────────────────────────────────
    output = {
        "fix_version": "FIX-12",
        "phase": "B",
        "description": "Python SentencePiece tokenizer encoding",
        "tokenizer_model_path": TOKENIZER_MODEL,
        "tokenizer_model_size": tok_size,
        "tokenizer_model_sha256": tok_sha,
        "sp_vocab_size": sp_vocab_size,
        "vocab_file_lines": vocab_size,
        "prompts": results,
        "all_prompts_encoded": all_pass,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"PHASE B OUTPUT: {OUTPUT_JSON}")
    print(f"ALL PROMPTS ENCODED: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*70}")

    # ── Machine-readable summary ──────────────────────────────────────────────
    print("\nMACHINE-READABLE:")
    print(f"FIX12_PHASE_B_TOKENIZER_MODEL_SHA={tok_sha}")
    print(f"FIX12_PHASE_B_SP_VOCAB_SIZE={sp_vocab_size}")
    print(f"FIX12_PHASE_B_ALL_PROMPTS_ENCODED={'PASS' if all_pass else 'FAIL'}")
    for r in results:
        label = r["label"]
        if r.get("status") == "OK":
            print(f"FIX12_PHASE_B_{label}_TOKEN_COUNT={r['token_count']}")
            print(f"FIX12_PHASE_B_{label}_IDS={r['token_ids']}")
            print(f"FIX12_PHASE_B_{label}_SHA={r['token_ids_sha256']}")

if __name__ == "__main__":
    main()
