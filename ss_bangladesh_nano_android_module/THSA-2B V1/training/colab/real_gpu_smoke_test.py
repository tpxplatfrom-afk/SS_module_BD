#!/usr/bin/env python3
"""
THSA-2B V1: Real GPU 10-Step Training Smoke Test Script
=======================================================
Executes 10 real optimizer training steps on a physical CUDA GPU with the
authoritative production teacher (Qwen/Qwen2.5-7B-Instruct) and 2.05B student:
  1. Instantiates production THSA-2B student model (2,050,296,320 parameters)
  2. Loads authoritative teacher (Qwen/Qwen2.5-7B-Instruct) with 4.0GB GPU cap for T4 headroom
  3. Uses lightweight on-GPU sampled parameter tracking to avoid host CPU RAM exhaustion
  4. Emits STEP_<N>_OPTIMIZER_UPDATE_COMPLETE heartbeat after every step
  5. Implements explicit KeyboardInterrupt handling and forensic exit telemetry
  6. Saves checkpoint to Google Drive and verifies state restoration
  7. Prints REAL_10_STEP_TRAINING_PASS or REAL_10_STEP_TRAINING_FAIL
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
from models.ternary_layers import TernaryLinear
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

def run_real_10_step_smoke_test(
    teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_teacher_gpu_gb: float = 4.0
):
    print("=" * 80)
    print("THSA-2B V1: REAL GPU 10-STEP TRAINING SMOKE TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[FATAL ERROR] Real GPU smoke test requires a CUDA-enabled GPU.")
        print("REAL_GPU_EXECUTION_NOT_YET_PROVEN")
        print("REAL_10_STEP_TRAINING_FAIL: CUDA not available on host.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()
    precision = "bfloat16" if bf16_supported else "float16"
    student_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

    # Output directory (Google Drive preferred)
    drive_dir = Path("/content/drive/MyDrive/THSA-2B/checkpoints")
    output_dir = drive_dir if drive_dir.parent.exists() else (TRAINING_DIR / "checkpoints" / "smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:                 {gpu_name} ({total_vram_gb:.2f} GB Total VRAM)")
    print(f"CUDA Version:           {torch.version.cuda}")
    print(f"BF16 Supported:         {bf16_supported}")
    print(f"Precision:              {precision} ({student_dtype})")
    print(f"Authoritative Teacher:  {teacher_model_name}")
    print(f"Teacher GPU Cap:        {max_teacher_gpu_gb:.1f} GB")
    print(f"Output Directory:       {output_dir}")

    config_path = TRAINING_DIR / "config" / "thsa_2b_config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    # 1. Instantiate Student Model (2.050B)
    print("\n[Init] Instantiating THSA-2B Student Model (2,050,296,320 parameters)...")
    student = THSAHybridForCausalLM(config).to(device="cuda", dtype=student_dtype)
    student.gradient_checkpointing = True
    student_params = sum(p.numel() for p in student.parameters())
    student_trainable_tensors = sum(1 for p in student.parameters() if p.requires_grad)
    print(f"STUDENT_PARAMETER_COUNT:  {student_params:,} ({student_params/1e9:.3f}B)")
    print(f"STUDENT_TRAINABLE_TENSORS:{student_trainable_tensors}")

    if student_params != 2050296320:
        print(f"[FATAL] Parameter count mismatch: expected 2,050,296,320, got {student_params}")
        print("REAL_10_STEP_TRAINING_FAIL: Parameter count mismatch.")
        return 1

    # 2. Instantiate Teacher Model
    print(f"\n[Init] Loading Authoritative Teacher Model ({teacher_model_name})...")
    teacher = QwenTeacherWrapper(
        teacher_model_name,
        device="cuda",
        precision=precision,
        max_gpu_memory_gb=max_teacher_gpu_gb
    )
    teacher_device_map = getattr(teacher.teacher_model, "hf_device_map", "cuda:0")
    print(f"TEACHER_DEVICE_MAP:       {teacher_device_map}")

    # 3. Tokenizer & Dataset
    tok_model = MODULE_ROOT / "tokenizer" / "thsa_tokenizer.model"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_model))
    vocab_sz = sp.get_piece_size()
    print(f"\nTokenizer Vocabulary:     {vocab_sz} tokens")

    corpus_path = str(MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt")
    dataset = TextCorpusDataset(corpus_path, max_samples=10000)
    print(f"Dataset Loaded:           {len(dataset):,} sentences from NCTB curriculum/corpus.")

    # 4. Optimizer & Loss Setup
    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(student.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
        print("Optimizer:                Adafactor (Memory-Factored)")
    except Exception:
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
        print("Optimizer:                AdamW")

    loss_fn = DistillationLoss(alpha=0.65, temperature=2.0)

    # 5. Setup Lightweight On-GPU Parameter Sampling (Avoids 16.4 GB CPU RAM Exhaustion)
    # Sample 6 representative parameter slices across diverse architectural layers:
    tracked_samples = {
        "embed_tokens": student.embed_tokens.weight[:32, :32],
        "layer0_conv1d": student.layers[0].mixer.conv1d.weight,
        "layer0_gate_proj": student.layers[0].ffn.gate_proj.weight[:32, :32],
        "layer2_q_proj": student.layers[2].mixer.q_proj.weight[:32, :32],
        "final_norm": student.final_norm.weight[:64],
        "lm_head": student.lm_head.weight[:32, :32]
    }
    initial_sample_snapshots = {k: v.clone().detach() for k, v in tracked_samples.items()}
    prev_sample_snapshots = {k: v.clone().detach() for k, v in tracked_samples.items()}

    print(f"Sampled Tracking:         6 representative layer tensors ({sum(v.numel() for v in tracked_samples.values()):,} parameters on-GPU, zero host RAM overhead)")

    print("\n" + "=" * 80)
    print("EXECUTING 10 REAL OPTIMIZER TRAINING STEPS ON CUDA")
    print("=" * 80)

    step_records = []
    total_steps = 10
    seq_len = 64
    current_step = 0

    student.train()

    try:
        for step in range(1, total_steps + 1):
            current_step = step
            t0 = time.perf_counter()
            
            # Dynamic QAT Annealing beta
            beta = 1.0 + (99.0 * step / total_steps)
            for m in student.modules():
                if isinstance(m, TernaryLinear):
                    m.beta = beta

            # Snapshot sampled parameters before step on GPU
            for k in tracked_samples:
                prev_sample_snapshots[k].copy_(tracked_samples[k])

            # Prepare batch input
            line = dataset[step % len(dataset)]
            toks = sp.encode(line, out_type=int)
            if len(toks) < seq_len:
                toks = toks + [3] * (seq_len - len(toks))
            else:
                toks = toks[:seq_len]
            input_ids = torch.tensor([toks], dtype=torch.long, device="cuda")
            targets = input_ids.clone()

            # Step A: Teacher Forward under torch.no_grad()
            with torch.no_grad():
                teacher_logits = teacher(input_ids, student_vocab_size=vocab_sz).detach()
            torch.cuda.empty_cache()

            # Step B: Student Forward
            student_logits = student(input_ids)

            # Step C: Distillation Loss
            loss = loss_fn(
                student_logits.view(-1, vocab_sz),
                teacher_logits.view(-1, vocab_sz),
                targets.view(-1)
            )
            del teacher_logits # Immediate memory release

            loss_val = loss.item()
            if not torch.isfinite(loss):
                print(f"\n[FATAL ERROR] Loss became non-finite (NaN/Inf) at step {step}: {loss_val}")
                print(f"REAL_10_STEP_TRAINING_FAIL: Non-finite loss at step {step}.")
                return 1

            # Step D: Backward Pass
            loss.backward()

            # Step E: Dynamic Gradient Audit (on-GPU norm check)
            nonzero_grads = sum(1 for p in student.parameters() if p.grad is not None and p.grad.norm().item() > 0)
            
            if nonzero_grads == 0:
                print(f"\n[FATAL ERROR] All gradients are zero at step {step}!")
                print(f"REAL_10_STEP_TRAINING_FAIL: Zero gradients at step {step}.")
                return 1

            # Step F: Optimizer Step & Gradient Zeroing
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Step G: Emit Heartbeat immediately after optimizer.step()
            print(f"  [HEARTBEAT] STEP_{step}_OPTIMIZER_UPDATE_COMPLETE")

            t_elapsed = time.perf_counter() - t0
            alloc_mb, resv_mb, max_alloc_mb, max_resv_mb = get_vram_mb()
            ram_used_mb, ram_total_mb = get_ram_mb()

            # Step H: Measure Sampled Parameter Delta directly on GPU
            step_sampled_l1_delta = 0.0
            for k in tracked_samples:
                delta = (tracked_samples[k].float() - prev_sample_snapshots[k].float()).abs().sum().item()
                step_sampled_l1_delta += delta

            print(f"  Step {step:2d}/10 | Loss: {loss_val:.6f} | Grads: {nonzero_grads:3d}/{student_trainable_tensors} | Sampled L1 Delta: {step_sampled_l1_delta:10.4f} | VRAM: {alloc_mb:.0f}/{resv_mb:.0f} MB (Peak: {max_alloc_mb:.0f} MB) | CPU RAM: {ram_used_mb:.0f}/{ram_total_mb:.0f} MB | Latency: {t_elapsed:.2f}s")

            step_records.append({
                "step": step,
                "loss": loss_val,
                "nonzero_grads": nonzero_grads,
                "sampled_l1_delta": step_sampled_l1_delta,
                "vram_allocated_mb": alloc_mb,
                "vram_reserved_mb": resv_mb,
                "peak_vram_allocated_mb": max_alloc_mb,
                "cpu_ram_used_mb": ram_used_mb,
                "latency_sec": t_elapsed
            })

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("REAL_10_STEP_TRAINING_INTERRUPTED")
        print(f"INTERRUPTED_AT_STEP: {current_step}")
        print("INTERRUPTION_TYPE:   KeyboardInterrupt (SIGINT)")
        print("=" * 80)
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print("REAL_10_STEP_TRAINING_INTERRUPTED")
        print(f"INTERRUPTED_AT_STEP: {current_step}")
        print(f"INTERRUPTION_TYPE:   Exception ({type(e).__name__}: {e})")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return 1

    # Total 10-step cumulative sampled parameter delta
    cumulative_sampled_delta = 0.0
    for k in tracked_samples:
        diff = (tracked_samples[k].float() - initial_sample_snapshots[k].float()).abs().sum().item()
        cumulative_sampled_delta += diff

    print("\n" + "=" * 80)
    print("10-STEP CUMULATIVE NUMERICAL AUDIT")
    print("=" * 80)
    print(f"CUMULATIVE_SAMPLED_L1_DELTA:{cumulative_sampled_delta:.4f}")
    print(f"FINAL_STEP_LOSS:            {step_records[-1]['loss']:.6f}")
    print(f"PEAK_GPU_MEMORY:            {max(r['peak_vram_allocated_mb'] for r in step_records):.1f} MB")
    print(f"MEAN_STEP_LATENCY:          {sum(r['latency_sec'] for r in step_records)/len(step_records):.2f} s/step")

    if cumulative_sampled_delta <= 0.0:
        print("[FATAL ERROR] Sampled parameters did not change across 10 steps!")
        print("REAL_10_STEP_TRAINING_FAIL: Zero cumulative parameter delta.")
        return 1

    # 6. Save Smoke Checkpoint
    save_path = output_dir / "checkpoint_step_000010.pt"
    tmp_path = output_dir / "checkpoint_step_000010.pt.tmp"
    checkpoint_dict = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": 10,
        "config": config,
        "distillation_meta": {
            "teacher": teacher_model_name,
            "alpha": 0.65,
            "temperature": 2.0,
            "steps": 10,
            "cumulative_sampled_l1_delta": cumulative_sampled_delta,
            "step_records": step_records
        }
    }
    torch.save(checkpoint_dict, str(tmp_path))
    if save_path.exists():
        save_path.unlink()
    tmp_path.rename(save_path)
    print(f"\n[Checkpoint] Saved 10-step smoke checkpoint to: {save_path} ({os.path.getsize(save_path):,} bytes)")

    # 7. Checkpoint Reload Verification
    print("[Checkpoint] Reloading checkpoint into fresh model to verify state restoration...")
    fresh_student = THSAHybridForCausalLM(config).to(dtype=student_dtype)
    ckpt = torch.load(str(save_path), map_location="cpu", weights_only=False)
    fresh_student.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["global_step"] == 10, f"Expected global_step 10, got {ckpt['global_step']}"
    print("[Checkpoint] State restoration verified successfully (global_step == 10)!")

    print("\n" + "=" * 80)
    print("REAL_10_STEP_TRAINING_PASS")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THSA-2B Real GPU 10-Step Training Smoke Test")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Authoritative teacher model")
    parser.add_argument("--max_teacher_gpu_gb", type=float, default=4.0, help="Max GPU memory for teacher (GB)")
    args = parser.parse_args()
    sys.exit(run_real_10_step_smoke_test(
        teacher_model_name=args.teacher,
        max_teacher_gpu_gb=args.max_teacher_gpu_gb
    ))
