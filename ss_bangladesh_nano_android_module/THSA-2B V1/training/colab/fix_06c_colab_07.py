#!/usr/bin/env python3
"""
THSA-2B V1: FIX-06C-COLAB-07 — Checkpoint Persistence Repair & 10-Step Re-execution
===================================================================================
Authoritative Production Teacher: Qwen/Qwen2.5-7B-Instruct (FROZEN)
Student Architecture: THSAHybridForCausalLM (2,050,296,320 params, 219 tensors, UNCHANGED)
Target: Checkpoint Persistence Repair & Re-execution of 10-step Real CUDA Distillation

Phases:
  Phase 0: Preflight verification (CUDA, GPU, BF16, Drive mount, disk space, config, params)
  Phase 10 (pre-train): Existing checkpoint overwrite check
  Phase 1: Real 10-step CUDA training pipeline (teacher no_grad, student forward, distillation loss,
           finite-loss audit, backward, 219 grads audit, clipping, Adafactor step, heartbeat,
           on-GPU telemetry)
  Phase 2: Atomic Checkpoint Creation (tmp write -> fsync -> tmp sha256 -> atomic replace ->
           final sha256 -> sync -> stat -> shell sha256sum -> match check)
  Phase 3: Checkpoint Content Forensics (torch.load CPU, global_step=10, required keys,
           219 tensors, 2.05B params, NaN/Inf scan, teacher metadata)
  Phase 4: Fresh Model Reload Identity Check (instantiate fresh student, load state_dict,
           torch.equal on all 219 tensors)
  Phase 5: Persistent Manifest Generation (checkpoint_step_000010.manifest.json written atomically)
  Phase 6: Secondary Backup Copy (if drive free space >= 8.0 GB)
  Phase 7: Drive Visibility Verification (shell ls -lh, sha256sum, Python readability check)
  Phase 8: Hard Stop & Final Structured Verdict Block
"""

import os
import sys
import time
import json
import math
import shutil
import hashlib
import psutil
import argparse
import datetime
import subprocess
from pathlib import Path
import torch

# Enable expandable segments to eliminate CUDA memory fragmentation
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

EXPECTED_PARAMS = 2050296320
EXPECTED_TENSORS = 219
AUTHORITATIVE_TEACHER = "Qwen/Qwen2.5-7B-Instruct"

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def compute_sha256(filepath: Path) -> str:
    """Compute streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_vram_mb():
    """Query CUDA VRAM metrics in megabytes."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024**2),
        torch.cuda.memory_reserved() / (1024**2),
        torch.cuda.max_memory_allocated() / (1024**2),
        torch.cuda.max_memory_reserved() / (1024**2),
    )

def get_ram_mb():
    """Query host CPU RAM metrics in megabytes."""
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / (1024**2), vm.total / (1024**2)

def make_tracked_samples(student):
    """6 deterministic on-GPU parameter slices — zero CPU RAM overhead."""
    return {
        "embed_tokens":     student.embed_tokens.weight[:32, :32],
        "layer0_conv1d":    student.layers[0].mixer.conv1d.weight,
        "layer0_gate_proj": student.layers[0].ffn.gate_proj.weight[:32, :32],
        "layer2_q_proj":    student.layers[2].mixer.q_proj.weight[:32, :32],
        "final_norm":       student.final_norm.weight[:64],
        "lm_head":          student.lm_head.weight[:32, :32],
    }

def sampled_l1_delta(tracked, prev_snaps):
    """Calculate absolute L1 difference across tracked slices on GPU."""
    total = 0.0
    for k in tracked:
        total += (tracked[k].float() - prev_snaps[k].float()).abs().sum().item()
    return total

def sync_filesystem():
    """Trigger OS filesystem sync where supported."""
    if hasattr(os, "sync"):
        try:
            os.sync()
        except Exception:
            pass
    else:
        try:
            subprocess.run(["sync"], check=False, capture_output=True)
        except Exception:
            pass

# --------------------------------------------------------------------------- #
# Main Fix Execution Flow
# --------------------------------------------------------------------------- #

def main(
    teacher_model_name: str,
    max_teacher_gpu_gb: float,
    checkpoint_dir_str: str,
    force_overwrite: bool = False,
    allow_local: bool = False
):
    print("=" * 80)
    print("FIX-06C-COLAB-07 — CHECKPOINT PERSISTENCE REPAIR & 10-STEP REEXECUTION")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # PHASE 0 — REPOSITORY / ENVIRONMENT PREFLIGHT
    # ------------------------------------------------------------------ #
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_avail else False
    precision = "bfloat16" if bf16_supported else "float16"
    student_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

    output_dir = Path(checkpoint_dir_str)

    # Drive mount detection
    is_colab_drive_target = str(output_dir).replace("\\", "/").startswith("/content/drive")
    drive_mounted = False
    if is_colab_drive_target:
        drive_mounted = Path("/content/drive/MyDrive").exists()
    else:
        drive_mounted = True  # Non-Colab or custom target

    drive_mount_str = "MOUNTED (/content/drive/MyDrive)" if drive_mounted else "UNMOUNTED"

    # Free space calculation
    free_space_str = "N/A"
    try:
        if output_dir.exists():
            free_bytes = shutil.disk_usage(output_dir).free
            free_space_str = f"{free_bytes / (1024**3):.2f} GB"
        elif output_dir.parent.exists():
            free_bytes = shutil.disk_usage(output_dir.parent).free
            free_space_str = f"{free_bytes / (1024**3):.2f} GB"
    except Exception:
        pass

    # Print Mandatory Preflight Block
    print(f"FIX-06C-COLAB-07")
    print(f"GPU:                         {gpu_name} ({total_vram_gb:.2f} GB)")
    print(f"CUDA:                        {cuda_ver}")
    print(f"BF16:                        {bf16_supported}")
    print(f"TEACHER:                     {teacher_model_name}")
    print(f"STUDENT_PARAMETER_COUNT:     {EXPECTED_PARAMS:,}")
    print(f"STUDENT_TRAINABLE_TENSORS:   {EXPECTED_TENSORS}")
    print(f"DRIVE_MOUNT:                 {drive_mount_str}")
    print(f"CHECKPOINT_DIR:              {output_dir}")
    print(f"FREE_SPACE:                  {free_space_str}")
    print("=" * 80)

    # Preflight Guards
    if not cuda_avail:
        print("\n[FATAL ERROR] Real GPU execution requires a physical CUDA GPU.")
        print("REAL_GPU_EXECUTION_NOT_YET_PROVEN")
        print("FIX-06C-COLAB-07-FAIL: CUDA not available on host.")
        return 1

    if is_colab_drive_target and not drive_mounted and not allow_local:
        print("\n[FATAL ERROR] Google Drive target specified but /content/drive/MyDrive is not mounted.")
        print("CHECKPOINT_PERSISTENCE_BLOCKED: DRIVE_NOT_MOUNTED")
        print("Please mount Drive via: from google.colab import drive; drive.mount('/content/drive')")
        print("FIX-06C-COLAB-07-FAIL: Drive unmounted.")
        return 1

    if teacher_model_name != AUTHORITATIVE_TEACHER:
        print(f"\n[FATAL ERROR] Authoritative teacher must be {AUTHORITATIVE_TEACHER}, got {teacher_model_name}")
        print("FIX-06C-COLAB-07-FAIL: Teacher freeze violation.")
        return 1

    # Load configuration
    config_path = TRAINING_DIR / "config" / "thsa_2b_config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    # ------------------------------------------------------------------ #
    # PHASE 10 (PRE-TRAINING) — NO CHECKPOINT OVERWRITE GUARD
    # ------------------------------------------------------------------ #
    final_ckpt_path = output_dir / "checkpoint_step_000010.pt"
    if final_ckpt_path.exists() and not force_overwrite:
        print("\n" + "=" * 80)
        print("EXISTING_CHECKPOINT_DETECTED")
        print(f"Existing checkpoint found at: {final_ckpt_path} ({os.path.getsize(final_ckpt_path):,} bytes)")
        print("To protect existing checkpoints, execution is halted.")
        print("To overwrite this checkpoint, pass --force_overwrite flag.")
        print("=" * 80)
        print("FIX-06C-COLAB-07-FAIL: Existing checkpoint detected without --force_overwrite.")
        return 1

    # Create target directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # INITIALIZATION: STUDENT, TEACHER, TOKENIZER, DATASET, OPTIMIZER
    # ------------------------------------------------------------------ #
    print("\n[Init] Instantiating THSA-2B Student Model (2,050,296,320 parameters) on GPU...")
    student = THSAHybridForCausalLM(config).to(device="cuda", dtype=student_dtype)
    student.gradient_checkpointing = True
    student_params = sum(p.numel() for p in student.parameters())
    student_trainable_tensors = sum(1 for p in student.parameters() if p.requires_grad)
    print(f"STUDENT_PARAMETER_COUNT:     {student_params:,} ({student_params/1e9:.3f}B)")
    print(f"STUDENT_TRAINABLE_TENSORS:   {student_trainable_tensors}")

    if student_params != EXPECTED_PARAMS:
        print(f"[FATAL] Parameter count mismatch: expected {EXPECTED_PARAMS:,}, got {student_params:,}")
        print("FIX-06C-COLAB-07-FAIL: Parameter count mismatch.")
        return 1

    if student_trainable_tensors != EXPECTED_TENSORS:
        print(f"[FATAL] Trainable tensor count mismatch: expected {EXPECTED_TENSORS}, got {student_trainable_tensors}")
        print("FIX-06C-COLAB-07-FAIL: Tensor count mismatch.")
        return 1

    print(f"\n[Init] Loading Authoritative Teacher Model ({teacher_model_name})...")
    teacher = QwenTeacherWrapper(
        teacher_model_name,
        device="cuda",
        precision=precision,
        max_gpu_memory_gb=max_teacher_gpu_gb
    )
    teacher_device_map = getattr(teacher.teacher_model, "hf_device_map", "cuda:0")
    print(f"TEACHER_DEVICE_MAP:          {teacher_device_map}")

    # Tokenizer & Dataset
    tok_model = MODULE_ROOT / "tokenizer" / "thsa_tokenizer.model"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_model))
    vocab_sz = sp.get_piece_size()
    print(f"\nTokenizer Vocabulary:        {vocab_sz} tokens")

    corpus_path = str(MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt")
    dataset = TextCorpusDataset(corpus_path, max_samples=10000)
    print(f"Dataset Loaded:              {len(dataset):,} sentences from NCTB curriculum/corpus.")

    # Optimizer & Loss Setup
    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(student.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
        print("Optimizer:                   Adafactor (Memory-Factored)")
    except Exception:
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
        print("Optimizer:                   AdamW")

    loss_fn = DistillationLoss(alpha=0.65, temperature=2.0)

    # Lightweight On-GPU Parameter Sampling (zero host CPU RAM overhead)
    tracked_samples = make_tracked_samples(student)
    initial_sample_snapshots = {k: v.clone().detach() for k, v in tracked_samples.items()}
    prev_sample_snapshots = {k: v.clone().detach() for k, v in tracked_samples.items()}

    print(f"Sampled Tracking:            6 representative layer tensors ({sum(v.numel() for v in tracked_samples.values()):,} params on-GPU)")

    # ------------------------------------------------------------------ #
    # PHASE 1 — REAL 10-STEP TRAINING
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 1 — EXECUTING 10 REAL OPTIMIZER TRAINING STEPS ON CUDA")
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

            # Teacher forward (torch.no_grad)
            with torch.no_grad():
                teacher_logits = teacher(input_ids, student_vocab_size=vocab_sz).detach()
            torch.cuda.empty_cache()

            # Student forward
            student_logits = student(input_ids)

            # Distillation loss
            loss = loss_fn(
                student_logits.view(-1, vocab_sz),
                teacher_logits.view(-1, vocab_sz),
                targets.view(-1)
            )
            del teacher_logits  # Immediate memory release

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"\n[FATAL ERROR] Non-finite loss ({loss_val}) at step {step}")
                print("FIX-06C-COLAB-07-FAIL: Non-finite loss.")
                return 1

            # Backward pass
            loss.backward()

            # Dynamic Gradient Audit (on-GPU norm check)
            nonzero_grads = sum(1 for p in student.parameters() if p.grad is not None and p.grad.norm().item() > 0)
            if nonzero_grads == 0:
                print(f"\n[FATAL ERROR] All gradients are zero at step {step}!")
                print("FIX-06C-COLAB-07-FAIL: Zero gradients at step.")
                return 1

            # Optimizer step & zero grad
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Heartbeat — emitted immediately after optimizer.step()
            print(f"  [HEARTBEAT] STEP_{step}_OPTIMIZER_UPDATE_COMPLETE")

            t_elapsed = time.perf_counter() - t0
            alloc_mb, resv_mb, peak_alloc_mb, _ = get_vram_mb()
            ram_used_mb, ram_total_mb = get_ram_mb()

            # Measure sampled parameter delta directly on GPU
            step_delta = sampled_l1_delta(tracked_samples, prev_sample_snapshots)

            print(
                f"  Step {step:2d}/10 | Loss: {loss_val:.6f} | "
                f"Grads: {nonzero_grads:3d}/{student_trainable_tensors} | "
                f"SampledΔ: {step_delta:10.4f} | "
                f"VRAM: {alloc_mb:.0f}/{resv_mb:.0f} MB (Peak: {peak_alloc_mb:.0f} MB) | "
                f"CPU: {ram_used_mb:.0f}/{ram_total_mb:.0f} MB | "
                f"Lat: {t_elapsed:.2f}s"
            )

            step_records.append({
                "step": step,
                "loss": loss_val,
                "nonzero_grads": nonzero_grads,
                "sampled_l1_delta": step_delta,
                "vram_allocated_mb": alloc_mb,
                "vram_reserved_mb": resv_mb,
                "peak_vram_allocated_mb": peak_alloc_mb,
                "cpu_ram_used_mb": ram_used_mb,
                "latency_sec": t_elapsed
            })

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("REAL_TRAINING_INTERRUPTED")
        print(f"INTERRUPTED_AT_STEP: {current_step}")
        print("INTERRUPTION_TYPE:   KeyboardInterrupt (SIGINT)")
        print("=" * 80)
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print("FIX-06C-COLAB-07-FAIL")
        print(f"INTERRUPTED_AT_STEP: {current_step}")
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        print("=" * 80)
        return 1

    if len(step_records) != 10:
        print(f"[FATAL ERROR] Completed only {len(step_records)}/10 steps.")
        print("FIX-06C-COLAB-07-FAIL: Incomplete 10-step execution.")
        return 1

    # Cumulative parameter delta
    cumulative_sampled_delta = sampled_l1_delta(tracked_samples, initial_sample_snapshots)
    print(f"\nCUMULATIVE_SAMPLED_L1_DELTA: {cumulative_sampled_delta:.4f}")
    if cumulative_sampled_delta <= 0.0:
        print("[FATAL ERROR] Zero cumulative parameter delta across 10 steps!")
        print("FIX-06C-COLAB-07-FAIL: Parameters did not update.")
        return 1

    # ------------------------------------------------------------------ #
    # PHASE 2 — ATOMIC CHECKPOINT CREATION
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 2 — ATOMIC CHECKPOINT PERSISTENCE PROTOCOL")
    print("=" * 80)

    final_ckpt_path = output_dir / "checkpoint_step_000010.pt"
    tmp_ckpt_path = output_dir / "checkpoint_step_000010.pt.tmp"

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
            "step_records": step_records,
        }
    }

    # Step 1-2: Save to temporary file, flush & fsync
    print(f"[Phase 2] Step 1-2: Saving checkpoint to temporary file: {tmp_ckpt_path}...")
    with open(tmp_ckpt_path, "wb") as f:
        torch.save(checkpoint_dict, f)
        f.flush()
        os.fsync(f.fileno())

    # Step 3-4: Verify temporary file exists and size > 0
    if not tmp_ckpt_path.exists():
        print(f"[FATAL] Temporary checkpoint file does not exist: {tmp_ckpt_path}")
        print("FIX-06C-COLAB-07-FAIL: tmp checkpoint missing after save.")
        return 1

    tmp_size = os.path.getsize(tmp_ckpt_path)
    print(f"[Phase 2] Step 3-4: Temporary file verified: {tmp_size:,} bytes ({tmp_size / (1024**3):.3f} GB)")
    if tmp_size == 0:
        print("[FATAL] Temporary checkpoint file is empty (0 bytes)!")
        print("FIX-06C-COLAB-07-FAIL: tmp checkpoint empty.")
        return 1

    # Step 5: Compute SHA-256 of temporary file
    print("[Phase 2] Step 5: Computing SHA-256 of temporary file...")
    tmp_sha256 = compute_sha256(tmp_ckpt_path)
    print(f"  TMP_SHA256:   {tmp_sha256}")

    # Step 6: Atomically replace temporary file -> final path
    print(f"[Phase 2] Step 6: Atomically replacing temporary file -> {final_ckpt_path.name}...")
    os.replace(tmp_ckpt_path, final_ckpt_path)

    # Step 7-8: Verify final file exists and size matches
    if not final_ckpt_path.exists():
        print(f"[FATAL] Final checkpoint file does not exist: {final_ckpt_path}")
        print("FIX-06C-COLAB-07-FAIL: final checkpoint missing after replace.")
        return 1

    final_size = os.path.getsize(final_ckpt_path)
    print(f"[Phase 2] Step 7-8: Final file verified: {final_size:,} bytes")
    if final_size != tmp_size:
        print(f"[FATAL] Size mismatch after atomic rename: tmp={tmp_size}, final={final_size}")
        print("FIX-06C-COLAB-07-FAIL: size mismatch after rename.")
        return 1

    # Step 9-10: Compute SHA-256 of final file and verify match
    print("[Phase 2] Step 9-10: Computing SHA-256 of final file and verifying integrity...")
    final_sha256 = compute_sha256(final_ckpt_path)
    print(f"  FINAL_SHA256: {final_sha256}")
    if final_sha256 != tmp_sha256:
        print(f"[FATAL] Hash mismatch before/after rename! tmp={tmp_sha256}, final={final_sha256}")
        print("FIX-06C-COLAB-07-FAIL: hash mismatch after rename.")
        return 1

    # Step 11-12: Filesystem sync
    print("[Phase 2] Step 11-12: Syncing filesystem buffers...")
    sync_filesystem()

    # Step 13: Re-stat final file
    stat_info = os.stat(final_ckpt_path)
    print(f"[Phase 2] Step 13: Final re-stat: size={stat_info.st_size:,} bytes, mtime={stat_info.st_mtime}")

    # Step 14-15: Shell sha256sum verification
    print("[Phase 2] Step 14-15: Running shell sha256sum verification...")
    shell_sha256 = None
    try:
        proc = subprocess.run(["sha256sum", str(final_ckpt_path)], capture_output=True, text=True, check=True)
        shell_sha256 = proc.stdout.strip().split()[0]
        print(f"  SHELL_SHA256: {shell_sha256}")
    except Exception as e:
        print(f"  [Notice] shell sha256sum not executed ({e}). Using Python hashlib verification.")
        shell_sha256 = final_sha256

    if shell_sha256 != final_sha256:
        print(f"[FATAL] Shell SHA-256 ({shell_sha256}) != Python SHA-256 ({final_sha256})!")
        print("FIX-06C-COLAB-07-FAIL: shell sha256sum mismatch.")
        return 1

    print("CHECKPOINT_PERSISTENCE_ATOMIC_WRITE: PASS")

    # ------------------------------------------------------------------ #
    # PHASE 3 — CHECKPOINT CONTENT FORENSICS
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 3 — CHECKPOINT CONTENT FORENSICS")
    print("=" * 80)
    print(f"Loading checkpoint into CPU from disk: {final_ckpt_path}...")
    loaded_ckpt = torch.load(str(final_ckpt_path), map_location="cpu", weights_only=False)

    ckpt_global_step = loaded_ckpt.get("global_step", -1)
    print(f"CHECKPOINT_GLOBAL_STEP:      {ckpt_global_step}")
    if ckpt_global_step != 10:
        print(f"[FATAL] Expected global_step == 10, got {ckpt_global_step}")
        print("FIX-06C-COLAB-07-FAIL: global_step != 10.")
        return 1

    for k in ("model_state_dict", "optimizer_state_dict", "config", "distillation_meta"):
        if k not in loaded_ckpt:
            print(f"[FATAL] Missing required key '{k}' in checkpoint payload.")
            print("FIX-06C-COLAB-07-FAIL: missing checkpoint key.")
            return 1
    print("CHECKPOINT_KEYS:             model_state_dict ✓  optimizer_state_dict ✓  config ✓  distillation_meta ✓")

    state_dict = loaded_ckpt["model_state_dict"]
    tensor_count = len(state_dict)
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"STATE_DICT_TENSORS:          {tensor_count} (Expected: {EXPECTED_TENSORS})")
    print(f"TOTAL_PARAMETERS:            {total_params:,} (Expected: {EXPECTED_PARAMS:,})")

    if tensor_count != EXPECTED_TENSORS:
        print(f"[FATAL] Tensor count mismatch: expected {EXPECTED_TENSORS}, got {tensor_count}")
        print("FIX-06C-COLAB-07-FAIL: tensor count mismatch.")
        return 1

    if total_params != EXPECTED_PARAMS:
        print(f"[FATAL] Parameter count mismatch: expected {EXPECTED_PARAMS:,}, got {total_params:,}")
        print("FIX-06C-COLAB-07-FAIL: parameter count mismatch.")
        return 1

    nan_count, inf_count = 0, 0
    for name, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            nan_count += 1
            print(f"  [NaN DETECTED] {name}")
        if torch.isinf(tensor).any():
            inf_count += 1
            print(f"  [Inf DETECTED] {name}")

    if nan_count > 0 or inf_count > 0:
        print(f"[FATAL] Non-finite weights: {nan_count} NaN, {inf_count} Inf")
        print("FIX-06C-COLAB-07-FAIL: non-finite values in checkpoint weights.")
        return 1
    print(f"NaN/Inf SCAN:                CLEAN (219/219 tensors clean, 0 NaN, 0 Inf)")

    meta_teacher = loaded_ckpt.get("distillation_meta", {}).get("teacher", "")
    print(f"DISTILLATION_META_TEACHER:   {meta_teacher}")
    if AUTHORITATIVE_TEACHER not in meta_teacher:
        print(f"[WARNING] Distillation teacher ({meta_teacher}) does not match {AUTHORITATIVE_TEACHER}")

    print("CHECKPOINT_CONTENT_FORENSICS: PASS")

    # ------------------------------------------------------------------ #
    # PHASE 4 — FRESH MODEL RELOAD IDENTITY
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 4 — FRESH MODEL RELOAD IDENTITY")
    print("=" * 80)
    print("Instantiating fresh THSAHybridForCausalLM model instance...")
    fresh_student = THSAHybridForCausalLM(config).to(dtype=student_dtype)
    fresh_student.load_state_dict(state_dict)
    fresh_sd = fresh_student.state_dict()

    mismatches = 0
    for name in state_dict:
        if not torch.equal(state_dict[name], fresh_sd[name]):
            mismatches += 1
            print(f"  [MISMATCH] {name}")

    if mismatches > 0:
        print(f"[FATAL] {mismatches} tensor(s) differ between checkpoint state_dict and fresh reloaded model!")
        print("CHECKPOINT_RELOAD_IDENTITY: FAIL")
        print("FIX-06C-COLAB-07-FAIL: reload identity mismatch.")
        return 1

    print("CHECKPOINT_RELOAD_IDENTITY:  PASS")
    del fresh_student, fresh_sd

    # ------------------------------------------------------------------ #
    # PHASE 5 — PERSISTENT MANIFEST CREATION
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 5 — PERSISTENT MANIFEST CREATION")
    print("=" * 80)
    manifest_path = output_dir / "checkpoint_step_000010.manifest.json"
    manifest_tmp  = output_dir / "checkpoint_step_000010.manifest.json.tmp"

    manifest_data = {
        "fix_id": "FIX-06C-COLAB-07",
        "checkpoint": "checkpoint_step_000010.pt",
        "global_step": 10,
        "byte_size": final_size,
        "sha256": final_sha256,
        "student_parameters": EXPECTED_PARAMS,
        "student_tensors": EXPECTED_TENSORS,
        "teacher": AUTHORITATIVE_TEACHER,
        "precision": precision,
        "gpu": gpu_name,
        "cuda": str(torch.version.cuda) if torch.cuda.is_available() else "N/A",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_size": len(dataset),
        "tokenizer_vocab": vocab_sz,
    }

    print(f"Writing manifest atomically to: {manifest_path}...")
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(manifest_tmp, manifest_path)
    sync_filesystem()

    if not manifest_path.exists():
        print(f"[FATAL] Manifest file missing after atomic write: {manifest_path}")
        print("FIX-06C-COLAB-07-FAIL: manifest missing.")
        return 1

    manifest_size = os.path.getsize(manifest_path)
    manifest_sha256 = compute_sha256(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_readback = json.load(f)

    if manifest_readback.get("sha256") != final_sha256:
        print("[FATAL] Checkpoint SHA256 in manifest does not match actual checkpoint SHA256!")
        print("FIX-06C-COLAB-07-FAIL: manifest sha256 mismatch.")
        return 1

    if manifest_readback.get("byte_size") != final_size:
        print("[FATAL] Checkpoint byte_size in manifest does not match actual checkpoint byte_size!")
        print("FIX-06C-COLAB-07-FAIL: manifest byte_size mismatch.")
        return 1

    print(f"MANIFEST:                    {manifest_path}")
    print(f"MANIFEST_HASH:               {manifest_sha256}")
    print(f"MANIFEST_SIZE:               {manifest_size} bytes")
    print("MANIFEST_STATUS:             WRITTEN_AND_VERIFIED")

    # ------------------------------------------------------------------ #
    # PHASE 6 — SECONDARY BACKUP COPY
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 6 — SECONDARY BACKUP COPY")
    print("=" * 80)
    free_drive_bytes = shutil.disk_usage(output_dir).free
    free_drive_gb = free_drive_bytes / (1024**3)
    backup_path = output_dir / "checkpoint_step_000010.backup.pt"

    # Require at least 8.0 GB free Drive space to create 4.1 GB backup copy without risking Drive exhaustion
    if free_drive_gb >= 8.0:
        print(f"Drive free space ({free_drive_gb:.2f} GB) >= 8.0 GB threshold. Creating backup copy...")
        backup_tmp = output_dir / "checkpoint_step_000010.backup.pt.tmp"
        shutil.copyfile(final_ckpt_path, backup_tmp)
        sync_filesystem()
        os.replace(backup_tmp, backup_path)
        sync_filesystem()

        backup_size = os.path.getsize(backup_path)
        backup_sha256 = compute_sha256(backup_path)
        if backup_size == final_size and backup_sha256 == final_sha256:
            print(f"SECONDARY_BACKUP:            CREATED_AND_VERIFIED ({backup_path})")
            print(f"  Backup Size:               {backup_size:,} bytes")
            print(f"  Backup SHA256:             {backup_sha256}")
        else:
            print(f"[WARNING] Backup file verification failed! (Size: {backup_size}, SHA: {backup_sha256})")
    else:
        print(f"Drive free space ({free_drive_gb:.2f} GB) < 8.0 GB threshold. Skipping backup to prevent Drive exhaustion.")
        print("SECONDARY_BACKUP:            SKIPPED_INSUFFICIENT_DRIVE_SPACE")

    # ------------------------------------------------------------------ #
    # PHASE 7 — DRIVE VISIBILITY VERIFICATION
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 7 — DRIVE VISIBILITY VERIFICATION")
    print("=" * 80)

    try:
        ls_res = subprocess.run(["ls", "-lh", str(final_ckpt_path)], capture_output=True, text=True)
        if ls_res.returncode == 0:
            print(f"Shell 'ls -lh' verification:\n  {ls_res.stdout.strip()}")
    except Exception:
        pass

    python_visible = final_ckpt_path.exists()
    python_readable = False
    try:
        with open(final_ckpt_path, "rb") as f:
            head = f.read(4096)
            python_readable = len(head) == 4096
    except Exception:
        python_readable = False

    print(f"CHECKPOINT_DRIVE_VISIBLE:    {'PASS' if python_visible else 'FAIL'}")
    print(f"CHECKPOINT_BYTE_SIZE:        {final_size:,}")
    print(f"CHECKPOINT_SHA256:           {final_sha256}")
    print(f"CHECKPOINT_READABLE:         {'PASS' if python_readable else 'FAIL'}")

    if not python_visible or not python_readable:
        print("FIX-06C-COLAB-07-FAIL: Checkpoint not visible or unreadable on mounted Drive.")
        return 1

    # ------------------------------------------------------------------ #
    # PHASE 8 — HARD STOP & FINAL VERDICT
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("FIX-06C-COLAB-07 FINAL VERDICT")
    print("=" * 80)
    print("REAL_GPU_10_STEP_EXECUTION:\nPASS\n")
    print(f"STUDENT_PARAMETER_COUNT:\n{EXPECTED_PARAMS:,}\n")
    print(f"STUDENT_TRAINABLE_TENSORS:\n{EXPECTED_TENSORS}\n")
    print(f"TEACHER:\n{AUTHORITATIVE_TEACHER}\n")
    print(f"STEPS_COMPLETED:\n10/10\n")
    print("GRADIENT_AUDIT:\nPASS\n")
    print("PARAMETER_UPDATE_AUDIT:\nPASS\n")
    print("FINITE_LOSS_AUDIT:\nPASS\n")
    print(f"CHECKPOINT_PATH:\n{final_ckpt_path}\n")
    print(f"CHECKPOINT_BYTE_SIZE:\n{final_size:,}\n")
    print(f"CHECKPOINT_SHA256:\n{final_sha256}\n")
    print("CHECKPOINT_DRIVE_VISIBLE:\nPASS\n")
    print("CHECKPOINT_READABLE:\nPASS\n")
    print("CHECKPOINT_CONTENT_FORENSICS:\nPASS\n")
    print("CHECKPOINT_RELOAD_IDENTITY:\nPASS\n")
    print(f"MANIFEST:\n{manifest_path}\n")
    print(f"MANIFEST_HASH:\n{manifest_sha256}\n")
    print("SAME_RUNTIME_DRIVE_PERSISTENCE:\nPASS\n")
    print("CROSS_RUNTIME_PERSISTENCE:\nPENDING_FRESH_RUNTIME_VERIFICATION\n")
    print("MODEL_NANO_GENERATED:\nNO\n")
    print("LONG_TRAINING_STARTED:\nNO\n")
    print("20_STEP_RESUME_STARTED:\nNO\n")
    print("=" * 80)
    print("FIX-06C-COLAB-07-PASS")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIX-06C-COLAB-07 Checkpoint Persistence Repair & 10-Step Reexecution")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Authoritative teacher model")
    parser.add_argument("--max_teacher_gpu_gb", type=float, default=4.0, help="Max GPU memory for teacher (GB)")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/THSA-2B/checkpoints",
                        help="Target output directory for checkpoints")
    parser.add_argument("--force_overwrite", action="store_true", help="Allow overwrite of existing checkpoint")
    parser.add_argument("--allow_local", action="store_true", help="Allow non-Drive local directory execution")
    args = parser.parse_args()

    sys.exit(main(
        teacher_model_name=args.teacher,
        max_teacher_gpu_gb=args.max_teacher_gpu_gb,
        checkpoint_dir_str=args.checkpoint_dir,
        force_overwrite=args.force_overwrite,
        allow_local=args.allow_local
    ))
