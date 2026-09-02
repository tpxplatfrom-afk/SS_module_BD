#!/usr/bin/env python3
"""
THSA-2B V1: Real GPU Smoke Test Script
======================================
Executes a verified 10-step smoke training run on a physical CUDA GPU:
  1. Instantiates production THSA-2B student model on CUDA
  2. Loads teacher model under torch.no_grad()
  3. Executes 10 real forward passes, backward passes, and optimizer updates
  4. Saves and reloads smoke checkpoint
  5. Verifies that student parameters demonstrably updated from initialization
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

from distillation.qwen_teacher_distillation import DistillationTrainer

def run_smoke_test():
    print("=" * 80)
    print("THSA-2B V1: REAL GPU SMOKE TEST (10 STEPS)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[FATAL ERROR] Real GPU smoke test requires a CUDA-enabled GPU.")
        print("COLAB_REAL_GPU_SMOKE_BLOCKED")
        return 1

    config_path = str(TRAINING_DIR / "config" / "thsa_2b_config.json")
    corpus_path = str(MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt")
    smoke_output_dir = str(TRAINING_DIR / "checkpoints" / "smoke_test")

    # Use Qwen2.5-0.5B or 7B depending on available VRAM for quick smoke validation
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    teacher_name = "Qwen/Qwen2.5-7B-Instruct" if vram_gb >= 24.0 else "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[Smoke Test] Detected {vram_gb:.2f} GB VRAM. Using Teacher: {teacher_name}")

    try:
        trainer = DistillationTrainer(
            config_path=config_path,
            teacher_model_name=teacher_name,
            corpus_path=corpus_path,
            output_dir=smoke_output_dir,
            learning_rate=3e-4,
            batch_size=1,
            grad_accum_steps=2,
            max_steps=10,
            checkpoint_interval=5,
            precision="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            device="cuda"
        )
        
        # Capture initial sample parameter before training
        initial_param_sample = trainer.student.lm_head.weight[:5, :5].clone().detach().cpu()

        # Run 10 training steps
        final_ckpt = trainer.run()

        # Verify weights updated
        post_param_sample = trainer.student.lm_head.weight[:5, :5].clone().detach().cpu()
        delta = (post_param_sample - initial_param_sample).abs().sum().item()
        print(f"\n[Validation] Parameter Update L1 Delta: {delta:.6f}")

        if delta == 0.0:
            print("[FATAL ERROR] Model parameters did NOT change after optimizer steps!")
            print("COLAB_REAL_GPU_SMOKE_BLOCKED")
            return 1

        # Checkpoint reload test
        print(f"[Validation] Testing checkpoint reload from {final_ckpt}...")
        ckpt_data = torch.load(final_ckpt, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt_data, "Missing model_state_dict in checkpoint!"
        assert ckpt_data["global_step"] == 10, f"Expected step 10, got {ckpt_data['global_step']}"
        print(f"[Validation] Successfully verified checkpoint {final_ckpt} ({os.path.getsize(final_ckpt):,} bytes).")

        print("=" * 80)
        print("COLAB_REAL_GPU_SMOKE_PASS")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n[Smoke Test Failure] {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        print("COLAB_REAL_GPU_SMOKE_BLOCKED")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(run_smoke_test())
