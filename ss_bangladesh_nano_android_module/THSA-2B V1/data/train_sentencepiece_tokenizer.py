#!/usr/bin/env python3
"""
THSA-2B Tokenizer Training Pipeline
====================================
Trains the official SentencePiece BPE Tokenizer model (V = 65,536)
directly on the cleaned Bengali + English pre-training corpus.
Outputs:
  - data/processed/thsa_tokenizer.model (2.1 MB)
  - data/processed/thsa_tokenizer.vocab
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sentencepiece as spm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(SCRIPT_DIR, "processed", "clean_pretrain_corpus.txt")
MODEL_PREFIX = os.path.join(SCRIPT_DIR, "processed", "thsa_tokenizer")

def main():
    print("=" * 60)
    print("THSA-2B: SENTENCEPIECE BPE TOKENIZER TRAINING")
    print("=" * 60)

    if not os.path.exists(CORPUS_FILE):
        print(f"Error: Clean corpus not found at {CORPUS_FILE}")
        sys.exit(1)

    size_mb = os.path.getsize(CORPUS_FILE) / (1024 * 1024)
    print(f"Training Input Corpus: {CORPUS_FILE} ({size_mb:.2f} MB)")
    print(f"Vocabulary Target:    V = 65,536 tokens")
    print(f"Model Type:           BPE with Byte Fallback + NFKC Normalization")
    print("Training started (running on CPU threads)...\n")

    t0 = time.perf_counter()

    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=65536,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        byte_fallback=True,
        normalization_rule_name="nfkc",
        num_threads=8,
        input_sentence_size=5000000,
        shuffle_input_sentence=True
    )

    t1 = time.perf_counter()
    model_path = f"{MODEL_PREFIX}.model"
    vocab_path = f"{MODEL_PREFIX}.vocab"

    print("\n" + "=" * 60)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Trained Model:    {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
    print(f"  Vocabulary File:  {vocab_path}")
    print(f"  Training Time:    {(t1 - t0):.2f} seconds")

if __name__ == "__main__":
    main()
