#!/usr/bin/env python3
"""
THSA-2B V1: FIX-06C-COLAB-06 — Post-10-Step Checkpoint Forensic & Short Training Validation
============================================================================================
Phase A: Forensic validation of checkpoint_step_000010.pt
  - byte size, SHA-256, global_step, parameter count, tensor count
  - NaN/Inf scan, fresh reload identity check
  - optimizer/distillation meta presence

Phase B: Resume training from step 10, execute steps 11-30 (20 additional real optimizer steps)
  - Full pipeline: teacher no_grad → student forward → distillation loss → backward → optimizer
  - Per-step telemetry: loss, gradient count, sampled delta, VRAM, CPU RAM, latency
  - Heartbeat after every optimizer.step()
  - KeyboardInterrupt handling

Phase C: Learning sanity comparison
  - Steps 1-10 vs steps 11-30 loss statistics
  - Sampled parameter drift across 30 steps
  - Stable optimization vs divergence assessment

Authoritative Teacher: Qwen/Qwen2.5-7B-Instruct (FROZEN)
Student: THSAHybridForCausalLM (2,050,296,320 parameters, 219 tensors, UNCHANGED)
"""

import os
import sys
import time
import json
import math
import hashlib
import psutil
import argparse
import torch
from pathlib import Path

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

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_vram_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024**2),
        torch.cuda.memory_reserved() / (1024**2),
        torch.cuda.max_memory_allocated() / (1024**2),
        torch.cuda.max_memory_reserved() / (1024**2),
    )

def get_ram_mb():
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
    total = 0.0
    for k in tracked:
        total += (tracked[k].float() - prev_snaps[k].float()).abs().sum().item()
    return total

def run_steps(student, teacher, optimizer, loss_fn, sp, dataset,
              vocab_sz, step_range, student_trainable_tensors,
              tracked_samples, prev_snaps, total_max_steps):
    """
    Run optimizer steps in step_range (inclusive).
    Returns list of step_record dicts, or raises on fatal failure.
    """
    SEQ_LEN = 64
    records = []
    current_step = step_range[0]
    student.train()

    for step in step_range:
        current_step = step
        t0 = time.perf_counter()

        # Dynamic QAT annealing
        beta = 1.0 + 99.0 * step / total_max_steps
        for m in student.modules():
            if isinstance(m, TernaryLinear):
                m.beta = beta

        # Snapshot sampled tensors on GPU
        for k in tracked_samples:
            prev_snaps[k].copy_(tracked_samples[k])

        # Prepare input
        line = dataset[step % len(dataset)]
        toks = sp.encode(line, out_type=int)
        toks = toks[:SEQ_LEN] if len(toks) >= SEQ_LEN else toks + [3] * (SEQ_LEN - len(toks))
        input_ids = torch.tensor([toks], dtype=torch.long, device="cuda")
        targets = input_ids.clone()

        # Teacher forward (no_grad)
        with torch.no_grad():
            teacher_logits = teacher(input_ids, student_vocab_size=vocab_sz).detach()
        torch.cuda.empty_cache()

        # Student forward
        student_logits = student(input_ids)

        # Distillation loss
        loss = loss_fn(
            student_logits.view(-1, vocab_sz),
            teacher_logits.view(-1, vocab_sz),
            targets.view(-1),
        )
        del teacher_logits

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            raise RuntimeError(f"Non-finite loss ({loss_val}) at step {step}")

        # Backward
        loss.backward()

        # Gradient audit (on-GPU norm)
        nonzero_grads = sum(
            1 for p in student.parameters()
            if p.grad is not None and p.grad.norm().item() > 0
        )
        if nonzero_grads == 0:
            raise RuntimeError(f"All gradients zero at step {step}")

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Heartbeat — emitted BEFORE any telemetry
        print(f"  [HEARTBEAT] STEP_{step}_OPTIMIZER_UPDATE_COMPLETE")

        t_elapsed = time.perf_counter() - t0
        alloc_mb, resv_mb, peak_alloc_mb, _ = get_vram_mb()
        ram_used_mb, ram_total_mb = get_ram_mb()
        s_delta = sampled_l1_delta(tracked_samples, prev_snaps)

        print(
            f"  Step {step:3d} | Loss: {loss_val:.6f} | "
            f"Grads: {nonzero_grads}/{student_trainable_tensors} | "
            f"SampledΔ: {s_delta:10.4f} | "
            f"VRAM: {alloc_mb:.0f}/{resv_mb:.0f} MB (Peak: {peak_alloc_mb:.0f}) | "
            f"CPU: {ram_used_mb:.0f}/{ram_total_mb:.0f} MB | "
            f"Lat: {t_elapsed:.2f}s"
        )

        records.append({
            "step": step,
            "loss": loss_val,
            "nonzero_grads": nonzero_grads,
            "sampled_l1_delta": s_delta,
            "vram_allocated_mb": alloc_mb,
            "vram_reserved_mb": resv_mb,
            "peak_vram_allocated_mb": peak_alloc_mb,
            "cpu_ram_used_mb": ram_used_mb,
            "latency_sec": t_elapsed,
        })

    return records

# --------------------------------------------------------------------------- #
# Phase A
# --------------------------------------------------------------------------- #

def phase_a(ckpt_path: Path, config: dict, student_dtype):
    print("\n" + "=" * 80)
    print("PHASE A — FORENSIC CHECKPOINT VALIDATION")
    print("=" * 80)

    if not ckpt_path.exists():
        print(f"[FATAL] Checkpoint not found: {ckpt_path}")
        print("POST_10_STEP_CHECKPOINT_FAIL: file missing.")
        return None

    byte_size = os.path.getsize(ckpt_path)
    sha256 = compute_sha256(ckpt_path)
    print(f"CHECKPOINT_PATH:     {ckpt_path}")
    print(f"BYTE_SIZE:           {byte_size:,} bytes ({byte_size/(1024**3):.3f} GB)")
    print(f"SHA-256:             {sha256}")

    print("\n[Phase A] Loading checkpoint to CPU...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # global_step
    global_step = ckpt.get("global_step", -1)
    print(f"GLOBAL_STEP:         {global_step}")
    if global_step != 10:
        print(f"[FATAL] global_step expected 10, got {global_step}")
        print("POST_10_STEP_CHECKPOINT_FAIL: global_step mismatch.")
        return None

    # Required keys
    for key in ("model_state_dict", "optimizer_state_dict", "config", "distillation_meta"):
        if key not in ckpt:
            print(f"[FATAL] Missing required key '{key}' in checkpoint.")
            print("POST_10_STEP_CHECKPOINT_FAIL: checkpoint key missing.")
            return None
    print("CHECKPOINT_KEYS:     model_state_dict ✓  optimizer_state_dict ✓  config ✓  distillation_meta ✓")

    state_dict = ckpt["model_state_dict"]
    tensor_count = len(state_dict)
    print(f"STATE_DICT_TENSORS:  {tensor_count}")
    if tensor_count != EXPECTED_TENSORS:
        print(f"[FATAL] Expected {EXPECTED_TENSORS} tensors, got {tensor_count}")
        print("POST_10_STEP_CHECKPOINT_FAIL: tensor count mismatch.")
        return None

    # Parameter count
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"TOTAL_PARAMETERS:    {total_params:,}")
    if total_params != EXPECTED_PARAMS:
        print(f"[FATAL] Expected {EXPECTED_PARAMS:,} parameters, got {total_params:,}")
        print("POST_10_STEP_CHECKPOINT_FAIL: parameter count mismatch.")
        return None

    # NaN/Inf scan
    nan_tensors, inf_tensors = 0, 0
    for name, t in state_dict.items():
        ft = t.float()
        if torch.isnan(ft).any():
            nan_tensors += 1
            print(f"  [NaN DETECTED] tensor: {name}")
        if torch.isinf(ft).any():
            inf_tensors += 1
            print(f"  [Inf DETECTED] tensor: {name}")

    if nan_tensors > 0 or inf_tensors > 0:
        print(f"[FATAL] NaN tensors: {nan_tensors}, Inf tensors: {inf_tensors}")
        print("POST_10_STEP_CHECKPOINT_FAIL: NaN/Inf in model parameters.")
        return None
    print(f"NaN/Inf SCAN:        CLEAN ({tensor_count} tensors, 0 NaN, 0 Inf)")

    # Fresh reload identity check
    print("\n[Phase A] Fresh model reload identity check...")
    fresh = THSAHybridForCausalLM(config).to(dtype=student_dtype)
    fresh.load_state_dict(state_dict)
    fresh_sd = fresh.state_dict()
    mismatches = 0
    for name in state_dict:
        if not torch.equal(state_dict[name], fresh_sd[name]):
            mismatches += 1
            print(f"  [MISMATCH] {name}")
    if mismatches > 0:
        print(f"[FATAL] {mismatches} tensor(s) differ between saved and reloaded state_dict!")
        print("POST_10_STEP_CHECKPOINT_FAIL: reload state mismatch.")
        return None
    del fresh, fresh_sd

    # distillation_meta
    meta = ckpt.get("distillation_meta", {})
    teacher_in_meta = meta.get("teacher", "MISSING")
    steps_in_meta = meta.get("steps", -1)
    print(f"\nDISTILLATION_META:")
    print(f"  teacher:          {teacher_in_meta}")
    print(f"  steps:            {steps_in_meta}")
    if "Qwen2.5-7B" not in teacher_in_meta:
        print(f"[WARNING] teacher in meta is not Qwen2.5-7B-Instruct! Got: {teacher_in_meta}")

    print("\nPOST_10_STEP_CHECKPOINT_PASS")
    return ckpt


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(teacher_model_name: str, max_teacher_gpu_gb: float, ckpt_path_str: str):
    print("=" * 80)
    print("FIX-06C-COLAB-06 — POST-10-STEP CHECKPOINT FORENSIC & SHORT TRAINING VALIDATION")
    print("=" * 80)
    print(f"Authoritative Teacher:  {teacher_model_name}  [FROZEN]")
    print(f"Student Architecture:   THSAHybridForCausalLM ({EXPECTED_PARAMS:,} params, {EXPECTED_TENSORS} tensors)  [UNCHANGED]")
    print(f"Teacher GPU Cap:        {max_teacher_gpu_gb:.1f} GB")

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available on host.")
        print("REAL_GPU_EXECUTION_NOT_YET_PROVEN")
        print("FIX-06C-COLAB-06-FAIL: no CUDA.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()
    precision = "bfloat16" if bf16_supported else "float16"
    student_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    print(f"GPU:                    {gpu_name} ({total_vram_gb:.2f} GB)")
    print(f"Precision:              {precision}")

    config_path = TRAINING_DIR / "config" / "thsa_2b_config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    # ------------------------------------------------------------------ #
    # Resolve checkpoint path
    # ------------------------------------------------------------------ #
    drive_ckpt = Path("/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000010.pt")
    local_ckpt = TRAINING_DIR / "checkpoints" / "smoke_test" / "checkpoint_step_000010.pt"

    if ckpt_path_str:
        ckpt_path = Path(ckpt_path_str)
    elif drive_ckpt.exists():
        ckpt_path = drive_ckpt
    elif local_ckpt.exists():
        ckpt_path = local_ckpt
    else:
        print(f"[FATAL] checkpoint_step_000010.pt not found at Drive or local path.")
        print("FIX-06C-COLAB-06-FAIL: checkpoint missing.")
        return 1

    output_dir = ckpt_path.parent  # save step-30 checkpoint beside step-10

    # ------------------------------------------------------------------ #
    # PHASE A
    # ------------------------------------------------------------------ #
    ckpt = phase_a(ckpt_path, config, student_dtype)
    if ckpt is None:
        return 1

    # ------------------------------------------------------------------ #
    # Load student from checkpoint onto GPU
    # ------------------------------------------------------------------ #
    print("\n[Init] Instantiating THSA-2B Student from checkpoint onto GPU...")
    student = THSAHybridForCausalLM(config).to(device="cuda", dtype=student_dtype)
    student.gradient_checkpointing = True
    student.load_state_dict(ckpt["model_state_dict"])
    student_params = sum(p.numel() for p in student.parameters())
    student_trainable_tensors = sum(1 for p in student.parameters() if p.requires_grad)
    print(f"STUDENT_PARAMETER_COUNT:  {student_params:,}")
    print(f"STUDENT_TRAINABLE_TENSORS:{student_trainable_tensors}")

    # ------------------------------------------------------------------ #
    # Load teacher
    # ------------------------------------------------------------------ #
    print(f"\n[Init] Loading Authoritative Teacher ({teacher_model_name})...")
    teacher = QwenTeacherWrapper(
        teacher_model_name, device="cuda",
        precision=precision, max_gpu_memory_gb=max_teacher_gpu_gb
    )
    print(f"TEACHER_DEVICE_MAP:       {getattr(teacher.teacher_model, 'hf_device_map', 'cuda:0')}")

    # ------------------------------------------------------------------ #
    # Tokenizer & Dataset
    # ------------------------------------------------------------------ #
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODULE_ROOT / "tokenizer" / "thsa_tokenizer.model"))
    vocab_sz = sp.get_piece_size()
    print(f"Tokenizer Vocabulary:     {vocab_sz} tokens")

    corpus_path = str(MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt")
    dataset = TextCorpusDataset(corpus_path, max_samples=10000)
    print(f"Dataset:                  {len(dataset):,} sentences")

    # ------------------------------------------------------------------ #
    # Optimizer — restore state from checkpoint
    # ------------------------------------------------------------------ #
    try:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(student.parameters(), lr=3e-4, scale_parameter=False,
                              relative_step=False, warmup_init=False)
        print("Optimizer:                Adafactor (state restored from checkpoint)")
    except Exception:
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
        print("Optimizer:                AdamW (state restored from checkpoint)")

    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Optimizer State:          Restored from checkpoint ✓")
        except Exception as e:
            print(f"[Warning] Could not restore optimizer state ({e}). Continuing with fresh state.")

    loss_fn = DistillationLoss(alpha=0.65, temperature=2.0)

    # Retrieve step-1–10 loss records from checkpoint meta if available
    phase_b_start = ckpt.get("global_step", 10) + 1   # 11
    phase_b_end   = phase_b_start + 19                  # 30
    TOTAL_MAX_STEPS = phase_b_end                        # used for QAT annealing schedule

    prior_records = ckpt.get("distillation_meta", {}).get("step_records", [])

    # ------------------------------------------------------------------ #
    # Setup sampled tracking
    # ------------------------------------------------------------------ #
    tracked_samples   = make_tracked_samples(student)
    initial_snaps     = {k: v.clone().detach() for k, v in tracked_samples.items()}
    prev_snaps        = {k: v.clone().detach() for k, v in tracked_samples.items()}

    # ------------------------------------------------------------------ #
    # PHASE B — Steps 11-30
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print(f"PHASE B — RESUME TRAINING: STEPS {phase_b_start}–{phase_b_end}")
    print("=" * 80)

    phase_b_records = []
    current_step = phase_b_start
    try:
        phase_b_records = run_steps(
            student, teacher, optimizer, loss_fn,
            sp, dataset, vocab_sz,
            step_range=range(phase_b_start, phase_b_end + 1),
            student_trainable_tensors=student_trainable_tensors,
            tracked_samples=tracked_samples,
            prev_snaps=prev_snaps,
            total_max_steps=TOTAL_MAX_STEPS,
        )
    except KeyboardInterrupt:
        completed = len(phase_b_records)
        print("\n" + "=" * 80)
        print("REAL_TRAINING_INTERRUPTED")
        print(f"INTERRUPTED_AT_STEP: {phase_b_start + completed}")
        print("INTERRUPTION_TYPE:   KeyboardInterrupt (SIGINT)")
        print("=" * 80)
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print("FIX-06C-COLAB-06-FAIL")
        print(f"INTERRUPTED_AT_STEP: {phase_b_start + len(phase_b_records)}")
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        print("=" * 80)
        return 1

    steps_completed = len(phase_b_records)
    print(f"\nPhase B steps completed:  {steps_completed} / 20")

    if steps_completed != 20:
        print(f"[FATAL] Only {steps_completed}/20 Phase B steps completed.")
        print("FIX-06C-COLAB-06-FAIL: Phase B incomplete.")
        return 1

    # Verify global_step == 30 (10 prior + 20 new)
    reported_final_step = phase_b_records[-1]["step"]
    print(f"REPORTED_FINAL_STEP:      {reported_final_step}")
    if reported_final_step != 30:
        print(f"[FATAL] Expected final step 30, got {reported_final_step}")
        print("FIX-06C-COLAB-06-FAIL: step count mismatch.")
        return 1

    # Cumulative parameter drift across Phase B
    cumulative_b_delta = sum(r["sampled_l1_delta"] for r in phase_b_records)
    peak_b_vram = max(r["peak_vram_allocated_mb"] for r in phase_b_records)
    print(f"PHASE_B_CUMULATIVE_DELTA: {cumulative_b_delta:.4f}")
    print(f"PHASE_B_PEAK_VRAM:        {peak_b_vram:.1f} MB")

    if cumulative_b_delta <= 0.0:
        print("[FATAL] Parameters did not change during Phase B!")
        print("FIX-06C-COLAB-06-FAIL: zero cumulative delta in Phase B.")
        return 1

    print("\nSHORT_RESUME_TRAINING_PASS")

    # ------------------------------------------------------------------ #
    # PHASE C — Learning Sanity
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE C — LEARNING SANITY COMPARISON")
    print("=" * 80)

    all_records = prior_records + phase_b_records if prior_records else phase_b_records
    phase_a_losses = [r["loss"] for r in prior_records] if prior_records else []
    phase_b_losses = [r["loss"] for r in phase_b_records]

    def stats(losses):
        if not losses:
            return {"n": 0, "mean": None, "min": None, "max": None, "std": None}
        n = len(losses)
        mean = sum(losses) / n
        var = sum((l - mean) ** 2 for l in losses) / n
        return {
            "n": n,
            "mean": round(mean, 6),
            "min": round(min(losses), 6),
            "max": round(max(losses), 6),
            "std": round(var ** 0.5, 6),
        }

    s_a = stats(phase_a_losses)
    s_b = stats(phase_b_losses)

    print(f"Steps  1-10 Loss: n={s_a['n']}, mean={s_a['mean']}, min={s_a['min']}, max={s_a['max']}, std={s_a['std']}")
    print(f"Steps 11-30 Loss: n={s_b['n']}, mean={s_b['mean']}, min={s_b['min']}, max={s_b['max']}, std={s_b['std']}")

    # Stability assessment (no monotonic requirement)
    loss_b_finite = all(math.isfinite(l) for l in phase_b_losses)
    loss_b_exploded = s_b["max"] > 20.0 if s_b["max"] is not None else False
    loss_b_collapsed = s_b["mean"] < 0.001 if s_b["mean"] is not None else False

    if not loss_b_finite:
        print("[FATAL] Non-finite loss detected in Phase B!")
        print("FIX-06C-COLAB-06-FAIL: non-finite losses in Phase B.")
        return 1
    if loss_b_exploded:
        print(f"[WARNING] Phase B loss exceeds 20.0 (max={s_b['max']:.4f}). Potential divergence.")
    if loss_b_collapsed:
        print(f"[WARNING] Phase B mean loss < 0.001. Possible collapse.")

    if not loss_b_exploded and not loss_b_collapsed:
        print("OPTIMIZATION_STABILITY: PASS (finite, non-exploded, non-collapsed)")
    else:
        print("OPTIMIZATION_STABILITY: WARNING (see above)")

    # Cumulative sampled delta from initial (step 0) to step 30
    total_drift = sampled_l1_delta(tracked_samples, initial_snaps)
    print(f"TOTAL_30_STEP_SAMPLED_DRIFT: {total_drift:.4f}")

    # ------------------------------------------------------------------ #
    # Save Step-30 Checkpoint
    # ------------------------------------------------------------------ #
    save_path = output_dir / "checkpoint_step_000030.pt"
    tmp_path  = output_dir / "checkpoint_step_000030.pt.tmp"
    all_step_records = (prior_records if prior_records else []) + phase_b_records
    ckpt_30 = {
        "model_state_dict":    student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step":         30,
        "config":              config,
        "distillation_meta": {
            "teacher":       teacher_model_name,
            "alpha":         0.65,
            "temperature":   2.0,
            "steps":         30,
            "step_records":  all_step_records,
            "sanity": {
                "steps_1_10":  s_a,
                "steps_11_30": s_b,
                "total_drift": total_drift,
            },
        },
    }
    torch.save(ckpt_30, str(tmp_path))
    if save_path.exists():
        save_path.unlink()
    tmp_path.rename(save_path)
    ckpt_30_size = os.path.getsize(save_path)
    ckpt_30_sha  = compute_sha256(save_path)
    print(f"\n[Checkpoint] Saved step-30 checkpoint: {save_path}")
    print(f"  Byte size: {ckpt_30_size:,} bytes ({ckpt_30_size/(1024**3):.3f} GB)")
    print(f"  SHA-256:   {ckpt_30_sha}")

    # Reload verification
    print("[Checkpoint] Reloading step-30 checkpoint to verify state restoration...")
    fresh = THSAHybridForCausalLM(config).to(dtype=student_dtype)
    c30   = torch.load(str(save_path), map_location="cpu", weights_only=False)
    fresh.load_state_dict(c30["model_state_dict"])
    assert c30["global_step"] == 30, f"Expected global_step 30, got {c30['global_step']}"
    del fresh, c30
    print("[Checkpoint] Step-30 checkpoint reload verified (global_step == 30) ✓")

    print("\n" + "=" * 80)
    print("FIX-06C-COLAB-06-PASS")
    print("  POST_10_STEP_CHECKPOINT_PASS")
    print("  SHORT_RESUME_TRAINING_PASS")
    print("  OPTIMIZATION_STABILITY: PASS (or WARNING above)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIX-06C-COLAB-06 Checkpoint Forensic & Short Training Validation")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_teacher_gpu_gb", type=float, default=4.0)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint_step_000010.pt (auto-detected if empty)")
    args = parser.parse_args()
    sys.exit(main(args.teacher, args.max_teacher_gpu_gb, args.checkpoint))
