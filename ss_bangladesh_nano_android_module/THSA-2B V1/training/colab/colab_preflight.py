#!/usr/bin/env python3
"""
THSA-2B V1: Google Colab Preflight Verification Script
======================================================
Validates all runtime prerequisites before initiating 2B training on GPU:
  1. CUDA GPU detection, VRAM check, CUDA version, and BF16/FP16 capability
  2. Architecture instantiation (2,050,296,320 parameters)
  3. SentencePiece tokenizer verification (65,536 vocabulary)
  4. Dataset corpus availability
  5. Required Python ecosystem imports
"""

import os
import sys
import json
import psutil
from pathlib import Path

# Setup module paths
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
MODULE_ROOT = TRAINING_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

def run_preflight():
    print("=" * 80)
    print("THSA-2B V1: GOOGLE COLAB PREFLIGHT AUDIT GATE")
    print("=" * 80)

    preflight_passed = True
    reasons = []

    # 1. Hardware & CUDA Check
    import torch
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    cuda_version = torch.version.cuda if cuda_avail else "N/A"
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_avail else False
    fp16_supported = cuda_avail
    
    print(f"GPU Name:            {gpu_name}")
    print(f"CUDA Available:      {cuda_avail}")
    print(f"CUDA Version:        {cuda_version}")
    print(f"GPU Total VRAM:      {gpu_vram_gb:.2f} GB")
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"Python Version:      {sys.version.split()[0]}")
    print(f"BF16 Supported:      {bf16_supported}")
    print(f"FP16 Supported:      {fp16_supported}")
    
    if bf16_supported:
        print("Precision Policy:    bfloat16 (Native Ampere / Ada / Hopper Tensor Core support)")
    elif fp16_supported:
        print("Precision Policy:    float16 (Native Turing T4 / Volta V100 Tensor Core support)")
    else:
        print("Precision Policy:    float32 (CPU Fallback)")

    if not cuda_avail:
        preflight_passed = False
        reasons.append("CUDA GPU is not available in current runtime.")
    elif gpu_vram_gb < 11.0:
        print(f"[WARN] GPU VRAM ({gpu_vram_gb:.2f} GB) is under 12 GB. Batch size=1 and gradient checkpointing required.")

    # 2. Package Dependency Checks
    try:
        import transformers
        import sentencepiece
        import datasets
        print(f"Transformers:        {transformers.__version__}")
        print(f"SentencePiece:       {sentencepiece.__version__}")
        print(f"Datasets:            {datasets.__version__}")
    except ImportError as e:
        preflight_passed = False
        reasons.append(f"Missing required package: {e}")

    # 3. Model Architecture Check
    config_path = TRAINING_DIR / "config" / "thsa_2b_config.json"
    if not config_path.exists():
        preflight_passed = False
        reasons.append(f"Missing configuration file: {config_path}")
        config = {}
    else:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)

    print(f"Model Architecture:  {config.get('model_id', 'UNKNOWN')}")
    print(f"Target Blocks:       {config.get('total_blocks')} ({config.get('state_blocks')} State / {config.get('gqa_blocks')} GQA)")
    print(f"Hidden Dimension:    {config.get('d_model')}")

    try:
        from models.thsa_hybrid_model import THSAHybridForCausalLM
        # Use meta device to verify parameter count without allocating host RAM
        with torch.device("meta"):
            model = THSAHybridForCausalLM(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameter Count:     {param_count:,} ({param_count/1e9:.3f}B)")
        if param_count != 2050296320:
            preflight_passed = False
            reasons.append(f"Parameter count mismatch: expected 2,050,296,320, got {param_count:,}")
    except Exception as e:
        preflight_passed = False
        reasons.append(f"Failed to instantiate model architecture: {e}")

    # 4. Tokenizer Verification
    tok_path = MODULE_ROOT / "tokenizer" / "thsa_tokenizer.vocab"
    tok_model = MODULE_ROOT / "tokenizer" / "thsa_tokenizer.model"
    if not tok_path.exists() or not tok_model.exists():
        preflight_passed = False
        reasons.append(f"Missing tokenizer files in {MODULE_ROOT / 'tokenizer'}")
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(tok_model))
        vocab_sz = sp.get_piece_size()
        print(f"Tokenizer Vocab:     {vocab_sz} tokens (Expected: 65,536)")
        if vocab_sz != 65536:
            preflight_passed = False
            reasons.append(f"Tokenizer vocabulary mismatch: expected 65536, got {vocab_sz}")

    # 5. Dataset Verification
    corpus_path = MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt"
    if not corpus_path.exists():
        print(f"[Dataset] Notice: Local corpus {corpus_path} not found. Checking curriculum packs...")
        curriculum_dir = MODULE_ROOT / "data" / "curriculum"
        if not curriculum_dir.exists():
            preflight_passed = False
            reasons.append(f"No local training dataset found at {corpus_path} or {curriculum_dir}")
        else:
            print(f"Dataset:             NCTB Curriculum Knowledge Packs in {curriculum_dir}")
    else:
        print(f"Dataset:             {corpus_path} ({os.path.getsize(corpus_path):,} bytes)")

    # 6. Teacher Model Target
    teacher_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Teacher:             {teacher_name}")

    # 7. Checkpoint Directory Check
    output_dir = TRAINING_DIR / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint Dir:      {output_dir}")

    print("=" * 80)
    if preflight_passed:
        print("COLAB_PREFLIGHT_PASS")
        print("=" * 80)
        return 0
    else:
        print("COLAB_PREFLIGHT_BLOCKED")
        print("Blockers encountered:")
        for r in reasons:
            print(f"  - {r}")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(run_preflight())
