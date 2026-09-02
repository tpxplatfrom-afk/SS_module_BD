#!/usr/bin/env python3
"""
THSA-2B V1: Real GPU 1-Step Diagnostic & Memory Profiler
========================================================
Performs a rigorous single training step on a physical CUDA GPU with the
authoritative production teacher (Qwen/Qwen2.5-7B-Instruct):
  1. Measures exact VRAM & CPU RAM at each lifecycle stage
  2. Inspects teacher device_map and offload topology
  3. Executes real teacher forward, real student forward, real distillation loss
  4. Computes backward gradients and dynamically inspects all 219 parameters
  5. Executes optimizer.step() and measures L1/max parameter update across all tensors
  6. Reports exact PASS / BLOCKED verdict without hardcoded claims
"""

import os
import sys
import time
import json
import psutil
import argparse
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
from distillation.distillation_loss import DistillationLoss
from distillation.qwen_teacher_distillation import QwenTeacherWrapper, TextCorpusDataset

def get_vram_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / (1024**2)
    resv = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    max_resv = torch.cuda.max_memory_reserved() / (1024**2)
    return alloc, resv, max_alloc, max_resv

def get_ram_mb():
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / (1024**2), vm.total / (1024**2)

def run_one_step_diagnostic(teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    print("=" * 80)
    print("THSA-2B V1: REAL GPU ONE-STEP DIAGNOSTIC & EXECUTION AUDIT")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[FATAL ERROR] Real GPU 1-step test requires a CUDA-enabled GPU.")
        print("REAL_GPU_EXECUTION_NOT_YET_PROVEN")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()
    precision = "bfloat16" if bf16_supported else "float16"
    student_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

    print(f"Device:                 {gpu_name}")
    print(f"GPU Total VRAM:         {total_vram_gb:.2f} GB")
    print(f"CUDA Version:           {torch.version.cuda}")
    print(f"BF16 Supported:         {bf16_supported}")
    print(f"Selected Precision:     {precision} ({student_dtype})")
    print(f"Authoritative Teacher:  {teacher_model_name}")

    config_path = TRAINING_DIR / "config" / "thsa_2b_config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    # 1. Measure Baseline Memory
    torch.cuda.reset_peak_memory_stats()
    a0, r0, _, _ = get_vram_mb()
    ram_used0, ram_tot = get_ram_mb()
    print(f"\n[Stage 0] Baseline:       VRAM Alloc: {a0:6.1f} MB | VRAM Resv: {r0:6.1f} MB | CPU RAM: {ram_used0:6.1f} / {ram_tot:6.1f} MB")

    # 2. Instantiate Student Model (2.050B)
    print("\n[Stage 1] Instantiating THSA-2B Student Model...")
    t_s0 = time.perf_counter()
    student = THSAHybridForCausalLM(config).to(device="cuda", dtype=student_dtype)
    student.gradient_checkpointing = True
    t_s_init = time.perf_counter() - t_s0

    student_params = sum(p.numel() for p in student.parameters())
    student_trainable_tensors = sum(1 for p in student.parameters() if p.requires_grad)
    print(f"STUDENT_PARAMETER_COUNT:  {student_params:,} ({student_params/1e9:.3f}B)")
    print(f"STUDENT_TRAINABLE_TENSORS:{student_trainable_tensors}")
    print(f"Student Init Time:        {t_s_init:.2f} s")

    if student_params != 2050296320:
        print(f"[FATAL] Parameter count mismatch: expected 2,050,296,320, got {student_params}")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    a1, r1, _, _ = get_vram_mb()
    ram_used1, _ = get_ram_mb()
    print(f"[Stage 1] Post-Student:   VRAM Alloc: {a1:6.1f} MB | VRAM Resv: {r1:6.1f} MB (+{a1-a0:.1f} MB) | CPU RAM: {ram_used1:6.1f} MB")

    # 3. Instantiate Authoritative Teacher Model
    print(f"\n[Stage 2] Loading Authoritative Teacher Model ({teacher_model_name})...")
    t_t0 = time.perf_counter()
    teacher = QwenTeacherWrapper(teacher_model_name, device="cuda", precision=precision)
    t_t_load = time.perf_counter() - t_t0
    print(f"Teacher Load Time:        {t_t_load:.2f} s")

    # Inspect teacher device map
    teacher_device_map = getattr(teacher.teacher_model, "hf_device_map", "cuda:0")
    print(f"TEACHER_DEVICE_MAP:       {teacher_device_map}")

    a2, r2, _, _ = get_vram_mb()
    ram_used2, _ = get_ram_mb()
    print(f"[Stage 2] Post-Teacher:   VRAM Alloc: {a2:6.1f} MB | VRAM Resv: {r2:6.1f} MB (+{a2-a1:.1f} MB) | CPU RAM: {ram_used2:6.1f} MB (+{ram_used2-ram_used1:.1f} MB)")

    # 4. Tokenizer & Dataset
    tok_model = MODULE_ROOT / "tokenizer" / "thsa_tokenizer.model"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_model))
    vocab_sz = sp.get_piece_size()
    print(f"\nTokenizer Vocabulary:     {vocab_sz} tokens")

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
    print(f"INPUT_SHAPE:              {list(input_ids.shape)}")

    # 5. Optimizer & Loss Setup
    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(student.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
        print("Optimizer Activated:      Adafactor (Memory-Factored)")
    except Exception:
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
        print("Optimizer Activated:      AdamW")

    loss_fn = DistillationLoss(alpha=0.65, temperature=2.0)

    # Checksum / snapshot all trainable parameters before forward & optimizer
    initial_weights = {name: p.clone().detach().cpu() for name, p in student.named_parameters() if p.requires_grad}

    # 6. Forward Passes
    print("\n[Stage 3] Executing Forward Passes...")
    student.train()

    t_fwd_s0 = time.perf_counter()
    student_logits = student(input_ids)
    t_student_fwd = time.perf_counter() - t_fwd_s0

    t_fwd_t0 = time.perf_counter()
    teacher_logits = teacher(input_ids, student_vocab_size=config.get("vocab_size", 65536))
    t_teacher_fwd = time.perf_counter() - t_fwd_t0

    print(f"STUDENT_LOGITS_SHAPE:     {list(student_logits.shape)} (Latency: {t_student_fwd*1000:.1f} ms)")
    print(f"TEACHER_LOGITS_SHAPE:     {list(teacher_logits.shape)} (Latency: {t_teacher_fwd*1000:.1f} ms)")

    if not torch.isfinite(student_logits).all():
        print("[FATAL ERROR] Student logits contain NaN or Inf!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    if not torch.isfinite(teacher_logits).all():
        print("[FATAL ERROR] Teacher logits contain NaN or Inf!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    a3, r3, _, _ = get_vram_mb()
    print(f"[Stage 3] Post-Forward:   VRAM Alloc: {a3:6.1f} MB | VRAM Resv: {r3:6.1f} MB (+{a3-a2:.1f} MB)")

    # 7. Distillation Loss
    loss = loss_fn(
        student_logits.view(-1, config.get("vocab_size", 65536)),
        teacher_logits.view(-1, config.get("vocab_size", 65536)),
        targets.view(-1)
    )
    print(f"LOSS:                     {loss.item():.6f}")
    print(f"LOSS_DTYPE:               {loss.dtype}")

    if not torch.isfinite(loss):
        print("[FATAL ERROR] Loss is not finite (NaN/Inf)!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    # 8. Backward Pass
    print("\n[Stage 4] Executing Backward Pass...")
    t_bwd0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t_bwd0
    print(f"Backward Latency:         {t_bwd*1000:.1f} ms")

    a4, r4, _, _ = get_vram_mb()
    print(f"[Stage 4] Post-Backward:  VRAM Alloc: {a4:6.1f} MB | VRAM Resv: {r4:6.1f} MB (+{a4-a3:.1f} MB)")

    # 9. Dynamic Gradient Inspection across all parameters
    grad_norms = []
    nonzero_grad_tensors = 0
    zero_grad_tensors = 0
    for name, p in student.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                g_norm = p.grad.float().norm().item()
                grad_norms.append(g_norm)
                if g_norm > 0.0:
                    nonzero_grad_tensors += 1
                else:
                    zero_grad_tensors += 1
            else:
                zero_grad_tensors += 1

    print(f"\n--- DYNAMIC GRADIENT AUDIT ---")
    print(f"TOTAL_TRAINABLE_TENSORS:  {student_trainable_tensors}")
    print(f"NONZERO_GRAD_TENSORS:     {nonzero_grad_tensors}")
    print(f"ZERO_GRAD_TENSORS:        {zero_grad_tensors}")
    if grad_norms:
        print(f"MAX_GRAD_NORM:            {max(grad_norms):.6f}")
        print(f"MEAN_GRAD_NORM:           {sum(grad_norms)/len(grad_norms):.6f}")

    if nonzero_grad_tensors == 0:
        print("[FATAL ERROR] All parameter gradients are zero!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    # 10. Optimizer Step
    print("\n[Stage 5] Executing Optimizer Step...")
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    a5, r5, max_alloc, max_resv = get_vram_mb()
    print(f"[Stage 5] Post-Optimizer: VRAM Alloc: {a5:6.1f} MB | VRAM Resv: {r5:6.1f} MB")
    print(f"PEAK_GPU_MEMORY_ALLOCATED:{max_alloc:6.1f} MB ({max_alloc / 1024:.2f} GB)")
    print(f"PEAK_GPU_MEMORY_RESERVED: {max_resv:6.1f} MB ({max_resv / 1024:.2f} GB)")

    # 11. Parameter Update Measurement Across All Tensors
    total_l1_delta = 0.0
    max_tensor_delta = 0.0
    changed_tensors = 0

    for name, p in student.named_parameters():
        if p.requires_grad:
            old_w = initial_weights[name]
            new_w = p.detach().cpu().float()
            diff = (new_w - old_w.float()).abs()
            l1 = diff.sum().item()
            m = diff.max().item()
            total_l1_delta += l1
            if m > max_tensor_delta:
                max_tensor_delta = m
            if l1 > 0.0:
                changed_tensors += 1

    print(f"\n--- PARAMETER UPDATE AUDIT ---")
    print(f"PARAMETER_UPDATE_L1:      {total_l1_delta:.8f}")
    print(f"PARAMETER_UPDATE_MAX:     {max_tensor_delta:.8f}")
    print(f"CHANGED_PARAMETER_TENSORS:{changed_tensors} / {student_trainable_tensors}")

    if total_l1_delta <= 0.0 or changed_tensors == 0:
        print("[FATAL ERROR] Parameter delta is zero after optimizer step!")
        print("COLAB_REAL_GPU_ONE_STEP_BLOCKED")
        return 1

    print("\n" + "=" * 80)
    print("COLAB_REAL_GPU_ONE_STEP_PASS")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THSA-2B Real GPU 1-Step Diagnostic")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Authoritative teacher model")
    args = parser.parse_args()
    sys.exit(run_one_step_diagnostic(teacher_model_name=args.teacher))
