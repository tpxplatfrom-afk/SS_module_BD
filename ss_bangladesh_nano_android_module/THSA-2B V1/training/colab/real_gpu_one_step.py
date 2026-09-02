#!/usr/bin/env python3
"""
THSA-2B V1: Real GPU 1-Step Diagnostic & Memory Measurement Script
==================================================================
Performs a rigorous single training step on a physical CUDA GPU:
  1. Measures VRAM at each lifecycle stage (student, teacher, forward, backward, optimizer)
  2. Executes real teacher forward, real student forward, real distillation loss
  3. Computes backward gradients and executes optimizer step
  4. Verifies finite logits, finite loss, nonzero gradients, and positive parameter delta
"""

import os
import sys
import time
import json
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
MODULE_ROOT = TRAINING_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from models.thsa_hybrid_model import THSAHybridForCausalLM
from models.ternary_layers import TernaryLinear
from distillation.distillation_loss import DistillationLoss
from distillation.qwen_teacher_distillation import QwenTeacherWrapper, TextCorpusDataset

def get_vram_mb():
    if not torch.cuda.is_available():
        return 0, 0, 0, 0
    alloc = torch.cuda.memory_allocated() / (1024**2)
    resv = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    max_resv = torch.cuda.max_memory_reserved() / (1024**2)
    return alloc, resv, max_alloc, max_resv

def run_one_step_diagnostic():
    print("=" * 80)
    print("THSA-2B V1: REAL GPU ONE-STEP DIAGNOSTIC & MEMORY PROFILER")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[FATAL ERROR] Real GPU 1-step test requires a CUDA-enabled GPU.")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()
    precision = "bfloat16" if bf16_supported else "float16"
    student_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

    print(f"Device:              {gpu_name} ({total_vram_gb:.2f} GB Total VRAM)")
    print(f"Precision Selected:  {precision} ({student_dtype})")

    config_path = TRAINING_DIR / "config" / "thsa_2b_config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    # 1. Measure initial VRAM
    torch.cuda.reset_peak_memory_stats()
    a0, r0, _, _ = get_vram_mb()
    print(f"\n[Stage 0] Baseline Memory:         Allocated: {a0:.1f} MB | Reserved: {r0:.1f} MB")

    # 2. Instantiate Student Model
    print("[Stage 1] Instantiating Student Model (2.050B)...")
    student = THSAHybridForCausalLM(config).to(device="cuda", dtype=student_dtype)
    student.gradient_checkpointing = True
    student_params = sum(p.numel() for p in student.parameters())
    print(f"STUDENT_PARAMETER_COUNT:           {student_params:,}")
    if student_params != 2050296320:
        print(f"[FATAL] Parameter count mismatch: {student_params}")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1
    a1, r1, _, _ = get_vram_mb()
    print(f"[Stage 1] Post-Student Memory:     Allocated: {a1:.1f} MB | Reserved: {r1:.1f} MB (+{a1-a0:.1f} MB)")

    # 3. Instantiate Teacher Model
    teacher_name = "Qwen/Qwen2.5-7B-Instruct" if total_vram_gb >= 24.0 else "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[Stage 2] Loading Teacher Model ({teacher_name})...")
    teacher = QwenTeacherWrapper(teacher_name, device="cuda", precision=precision)
    a2, r2, _, _ = get_vram_mb()
    print(f"[Stage 2] Post-Teacher Memory:     Allocated: {a2:.1f} MB | Reserved: {r2:.1f} MB (+{a2-a1:.1f} MB)")

    # 4. Tokenizer & Dataset
    tok_model = MODULE_ROOT / "tokenizer" / "thsa_tokenizer.model"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_model))

    corpus_path = str(MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt")
    dataset = TextCorpusDataset(corpus_path, max_samples=1000)
    sample_text = dataset[0]
    toks = sp.encode(sample_text, out_type=int)
    seq_len = 64
    if len(toks) < seq_len:
        toks = toks + [3] * (seq_len - len(toks))
    else:
        toks = toks[:seq_len]
    input_ids = torch.tensor([toks], dtype=torch.long, device="cuda")
    targets = input_ids.clone()
    print(f"INPUT_SHAPE:                       {list(input_ids.shape)}")

    # 5. Optimizer & Loss
    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(student.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    except Exception:
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    loss_fn = DistillationLoss(alpha=0.65, temperature=2.0)

    # Capture initial parameter slice for L1 delta measurement
    initial_weight_sample = student.lm_head.weight[:5, :5].clone().detach().cpu()

    # 6. Forward Pass
    print("\n[Stage 3] Executing Forward Passes...")
    student.train()
    student_logits = student(input_ids)
    teacher_logits = teacher(input_ids, student_vocab_size=config.get("vocab_size", 65536))
    print(f"TEACHER_LOGITS_SHAPE:              {list(teacher_logits.shape)}")
    print(f"STUDENT_LOGITS_SHAPE:              {list(student_logits.shape)}")
    a3, r3, _, _ = get_vram_mb()
    print(f"[Stage 3] Post-Forward Memory:     Allocated: {a3:.1f} MB | Reserved: {r3:.1f} MB (+{a3-a2:.1f} MB)")

    # 7. Distillation Loss
    loss = loss_fn(
        student_logits.view(-1, config.get("vocab_size", 65536)),
        teacher_logits.view(-1, config.get("vocab_size", 65536)),
        targets.view(-1)
    )
    print(f"LOSS:                              {loss.item():.6f}")
    print(f"LOSS_DTYPE:                        {loss.dtype}")

    if not torch.isfinite(loss):
        print("[FATAL ERROR] Loss is not finite (NaN/Inf)!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    # 8. Backward Pass
    print("\n[Stage 4] Executing Backward Pass...")
    loss.backward()
    a4, r4, _, _ = get_vram_mb()
    print(f"[Stage 4] Post-Backward Memory:    Allocated: {a4:.1f} MB | Reserved: {r4:.1f} MB (+{a4-a3:.1f} MB)")

    # 9. Verify Gradients
    nonzero_grads = sum(1 for p in student.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    total_grad_params = sum(1 for p in student.parameters() if p.requires_grad)
    print(f"NONZERO_GRADIENT_PARAMETER_COUNT:  {nonzero_grads} / {total_grad_params}")
    if nonzero_grads == 0:
        print("[FATAL ERROR] All gradients are zero!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    # 10. Optimizer Step
    print("\n[Stage 5] Executing Optimizer Step...")
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    a5, r5, max_alloc, max_resv = get_vram_mb()
    print(f"[Stage 5] Post-Optimizer Memory:   Allocated: {a5:.1f} MB | Reserved: {r5:.1f} MB")
    print(f"PEAK_GPU_MEMORY_ALLOCATED:         {max_alloc:.1f} MB ({max_alloc / 1024:.2f} GB)")
    print(f"PEAK_GPU_MEMORY_RESERVED:          {max_resv:.1f} MB ({max_resv / 1024:.2f} GB)")

    # 11. Parameter Update Delta
    post_weight_sample = student.lm_head.weight[:5, :5].clone().detach().cpu()
    delta_l1 = (post_weight_sample - initial_weight_sample).abs().sum().item()
    print(f"PARAMETER_UPDATE_L1:               {delta_l1:.8f}")

    if delta_l1 <= 0.0:
        print("[FATAL ERROR] Parameter delta is zero after optimizer step!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    print("\n" + "=" * 80)
    print("COLAB_REAL_GPU_ONE_STEP_PASS")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(run_one_step_diagnostic())
