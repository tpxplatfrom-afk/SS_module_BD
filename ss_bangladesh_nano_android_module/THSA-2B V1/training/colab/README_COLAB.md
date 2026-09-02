# THSA-2B V1 — Google Colab Real 2B Training Guide

This guide provides step-by-step instructions for executing the physical GPU knowledge distillation and Quantization-Aware Training (QAT) run for **THSA-2B V1** ($2,050,296,320$ parameters) on Google Colab with the authoritative **Qwen/Qwen2.5-7B-Instruct** teacher.

---

## 1. Google Colab Hardware & Runtime Requirements

- **Runtime Type:** Python 3 (GPU Accelerated)
- **Hardware Tiers:**
  - **Premium Tier:** NVIDIA A100 GPU ($40\text{ GB} / 80\text{ GB}$ VRAM) — Loads 100% of 7B teacher and 2.05B student into pure GPU VRAM.
  - **Free Tier:** NVIDIA Tesla T4 GPU ($15.0\text{ GB}$ VRAM) — Employs `max_memory={0: "4.0GB", "cpu": "30GB"}` for the 7B teacher to reserve 10.5GB of GPU VRAM exclusively for the 2.05B student and backward activation checkpointing.
- **Google Drive Storage:** Minimum $10\text{ GB}$ free space for storing checkpoint files.

---

## 2. Cell-by-Cell Colab Execution Workflow

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/THSA-2B/checkpoints
!mkdir -p /content/drive/MyDrive/THSA-2B/logs
```

### Cell 2: Clone Repository & Navigate to Directory
```bash
!git clone https://github.com/tpxplatfrom-afk/SS_module_BD.git /content/SS_module_BD
%cd /content/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B\ V1
!git pull origin main
```

### Cell 3: Environment Setup
```bash
!bash training/colab/setup_colab.sh
```

### Cell 4: Run Preflight Check
```bash
!python training/colab/colab_preflight.py
```

### Cell 5: Run Backward Memory & OOM Diagnostic Test
```bash
!python training/colab/real_gpu_backward_memory_test.py --teacher Qwen/Qwen2.5-7B-Instruct --max_teacher_gpu_gb 4.0
```
*Expected Result:* `BACKWARD_MEMORY_REPAIR_PASS`

### Cell 6: Run 1-Step Diagnostic
```bash
!python training/colab/real_gpu_one_step.py --teacher Qwen/Qwen2.5-7B-Instruct
```

### Cell 7: Run 10-Step Smoke Test (Saves Checkpoint to Drive)
```bash
!python training/colab/real_gpu_smoke_test.py --teacher Qwen/Qwen2.5-7B-Instruct
```
*Expected Result:* `COLAB_REAL_GPU_SMOKE_PASS`

---

## 3. What Constitutes Backward Memory Success vs Blocked

- **Success (`BACKWARD_MEMORY_REPAIR_PASS`):**
  1. Student model ($2,050,296,320$ parameters) executes forward pass.
  2. Authoritative 7B teacher generates soft logits under `torch.no_grad()`.
  3. Loss computes in float32 without NaN/Inf.
  4. `loss.backward()` successfully completes activation checkpoint recomputation without CUDA OOM.
  5. Trainable parameters have nonzero gradients and update after `optimizer.step()`.
- **Blocked (`FREE_T4_7B_TEACHER_BACKWARD_MEMORY_BLOCKED`):**
  1. CUDA Out-of-Memory during backward pass on T4.
