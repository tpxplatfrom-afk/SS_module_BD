#!/usr/bin/env python3
"""
THSA-2B V1: Real GPU Backward Memory & OOM Forensic Test Script
==============================================================
Validates backward pass execution on physical CUDA GPU with the authoritative
production teacher (Qwen/Qwen2.5-7B-Instruct) and 2.05B student model:
  1. Measures VRAM & CPU RAM across all lifecycle stages
  2. Enforces teacher memory cap on T4 (<= 4.0GB GPU, remainder on CPU) to guarantee student backward headroom
  3. Executes real teacher forward under torch.no_grad()
  4. Executes real student forward with gradient checkpointing
  5. Computes real distillation loss in float32
  6. Executes loss.backward() and verifies nonzero gradients
  7. Executes optimizer.step() and measures parameter delta L1 norm
"""

import os
import sys
import time
import json
import psutil
import argparse
import torch
from pathlib import Path

# Enable expandable segments to eliminate CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def run_backward_memory_test(
    teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_teacher_gpu_gb: float = 4.0
):
    print("=" * 80)
    print("THSA-2B V1: REAL GPU BACKWARD MEMORY & OOM DIAGNOSTIC TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[FATAL ERROR] Real GPU backward test requires a CUDA-enabled GPU.")
        print("REAL_GPU_EXECUTION_NOT_YET_PROVEN")
        print("COLAB_REAL_GPU_BACKWARD_BLOCKED")
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
    print(f"Teacher GPU Cap:        {max_teacher_gpu_gb:.1f} GB")

    config_path = TRAINING_DIR / "config" / "thsa_2b_config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    # Stage 0: Baseline
    torch.cuda.reset_peak_memory_stats()
    a0, r0, _, _ = get_vram_mb()
    ram0, ram_tot = get_ram_mb()
    print(f"\n[Stage 0] Baseline:       VRAM Alloc: {a0:6.1f} MB | VRAM Resv: {r0:6.1f} MB | CPU RAM: {ram0:6.1f} / {ram_tot:6.1f} MB")

    # Stage 1: Student Instantiation (2.05B)
    print("\n[Stage 1] Instantiating THSA-2B Student Model (2,050,296,320 parameters)...")
    student = THSAHybridForCausalLM(config).to(device="cuda", dtype=student_dtype)
    student.gradient_checkpointing = True
    student_params = sum(p.numel() for p in student.parameters())
    student_trainable_tensors = sum(1 for p in student.parameters() if p.requires_grad)
    print(f"STUDENT_PARAMETER_COUNT:  {student_params:,} ({student_params/1e9:.3f}B)")
    print(f"STUDENT_TRAINABLE_TENSORS:{student_trainable_tensors}")

    if student_params != 2050296320:
        print(f"[FATAL] Parameter count mismatch: expected 2,050,296,320, got {student_params}")
        print("BACKWARD_MEMORY_REPAIR_FAILED")
        return 1

    a1, r1, _, _ = get_vram_mb()
    ram1, _ = get_ram_mb()
    print(f"[Stage 1] Post-Student:   VRAM Alloc: {a1:6.1f} MB | VRAM Resv: {r1:6.1f} MB (+{a1-a0:.1f} MB) | CPU RAM: {ram1:6.1f} MB")

    # Stage 2: Teacher Loading with explicit memory cap
    print(f"\n[Stage 2] Loading Authoritative Teacher ({teacher_model_name}) with GPU cap {max_teacher_gpu_gb:.1f} GB...")
    try:
        teacher = QwenTeacherWrapper(
            teacher_model_name,
            device="cuda",
            precision=precision,
            max_gpu_memory_gb=max_teacher_gpu_gb
        )
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load teacher model: {e}")
        print("BACKWARD_MEMORY_REPAIR_FAILED")
        return 1

    teacher_device_map = getattr(teacher.teacher_model, "hf_device_map", "cuda:0")
    print(f"TEACHER_DEVICE_MAP:       {teacher_device_map}")
    a2, r2, _, _ = get_vram_mb()
    ram2, _ = get_ram_mb()
    print(f"[Stage 2] Post-Teacher:   VRAM Alloc: {a2:6.1f} MB | VRAM Resv: {r2:6.1f} MB (+{a2-a1:.1f} MB) | CPU RAM: {ram2:6.1f} MB (+{ram2-ram1:.1f} MB)")

    # Available Headroom Check
    free_headroom_mb = (total_vram_gb * 1024) - a2
    print(f"REMAINING_VRAM_HEADROOM:  {free_headroom_mb:.1f} MB ({free_headroom_mb/1024:.2f} GB)")

    # Stage 3: Tokenizer & Dataset
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
    print(f"INPUT_SHAPE:              {list(input_ids.shape)}")

    # Optimizer & Loss
    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(student.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    except Exception:
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)

    loss_fn = DistillationLoss(alpha=0.65, temperature=2.0)

    # Snapshot initial parameters
    initial_weights = {name: p.clone().detach().cpu() for name, p in student.named_parameters() if p.requires_grad}

    # Stage 4: Teacher Forward
    print("\n[Stage 3] Executing Teacher Forward under torch.no_grad()...")
    t_tf0 = time.perf_counter()
    with torch.no_grad():
        teacher_logits = teacher(input_ids, student_vocab_size=config.get("vocab_size", 65536)).detach()
    t_tf = time.perf_counter() - t_tf0
    torch.cuda.empty_cache()
    a3, r3, _, _ = get_vram_mb()
    print(f"[Stage 3] Post-Teacher-Fwd: VRAM Alloc: {a3:6.1f} MB | VRAM Resv: {r3:6.1f} MB (Latency: {t_tf*1000:.1f} ms)")

    # Stage 5: Student Forward
    print("\n[Stage 4] Executing Student Forward (Building Autograd Graph)...")
    student.train()
    t_sf0 = time.perf_counter()
    student_logits = student(input_ids)
    t_sf = time.perf_counter() - t_sf0
    a4, r4, _, _ = get_vram_mb()
    print(f"[Stage 4] Post-Student-Fwd: VRAM Alloc: {a4:6.1f} MB | VRAM Resv: {r4:6.1f} MB (+{a4-a3:.1f} MB, Latency: {t_sf*1000:.1f} ms)")

    # Stage 6: Distillation Loss
    loss = loss_fn(
        student_logits.view(-1, config.get("vocab_size", 65536)),
        teacher_logits.view(-1, config.get("vocab_size", 65536)),
        targets.view(-1)
    )
    del teacher_logits # Immediate memory release
    print(f"LOSS:                     {loss.item():.6f}")

    # Stage 7: Backward Pass
    print("\n[Stage 5] Executing Backward Pass (Activation Checkpoint Recomputation)...")
    t_b0 = time.perf_counter()
    try:
        loss.backward()
        t_b = time.perf_counter() - t_b0
        a5, r5, max_a, max_r = get_vram_mb()
        print(f"[Stage 5] Post-Backward:  VRAM Alloc: {a5:6.1f} MB | VRAM Resv: {r5:6.1f} MB (Latency: {t_b*1000:.1f} ms)")
        print(f"PEAK_GPU_MEMORY_ALLOCATED:{max_a:6.1f} MB ({max_a/1024:.2f} GB)")
        print(f"PEAK_GPU_MEMORY_RESERVED: {max_r:6.1f} MB ({max_r/1024:.2f} GB)")
    except torch.OutOfMemoryError as oom:
        print(f"\n[BACKWARD OOM FAILURE] {oom}")
        print("FREE_T4_7B_TEACHER_BACKWARD_MEMORY_BLOCKED")
        return 1

    # Stage 8: Dynamic Gradient Inspection
    nonzero_grads = sum(1 for p in student.parameters() if p.grad is not None and p.grad.float().norm().item() > 0)
    print(f"\nNONZERO_GRADIENT_TENSORS: {nonzero_grads} / {student_trainable_tensors}")

    if nonzero_grads == 0:
        print("[FATAL ERROR] Zero gradients produced during backward pass!")
        print("BACKWARD_MEMORY_REPAIR_FAILED")
        return 1

    # Stage 9: Optimizer Step
    print("\n[Stage 6] Executing Optimizer Step...")
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Compute parameter delta L1
    total_l1_delta = 0.0
    changed_tensors = 0
    for name, p in student.named_parameters():
        if p.requires_grad:
            old_w = initial_weights[name].float()
            new_w = p.detach().cpu().float()
            diff = (new_w - old_w).abs().sum().item()
            total_l1_delta += diff
            if diff > 0.0:
                changed_tensors += 1

    print(f"PARAMETER_UPDATE_L1:      {total_l1_delta:.8f}")
    print(f"CHANGED_TENSORS:          {changed_tensors} / {student_trainable_tensors}")

    if total_l1_delta <= 0.0:
        print("[FATAL ERROR] Parameters did not update!")
        print("BACKWARD_MEMORY_REPAIR_FAILED")
        return 1

    print("\n" + "=" * 80)
    print("BACKWARD_MEMORY_REPAIR_PASS")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THSA-2B Real GPU Backward Memory Test")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Authoritative teacher model")
    parser.add_argument("--max_teacher_gpu_gb", type=float, default=4.0, help="Max GPU memory for teacher (GB)")
    args = parser.parse_args()
    sys.exit(run_backward_memory_test(
        teacher_model_name=args.teacher,
        max_teacher_gpu_gb=args.max_teacher_gpu_gb
    ))
