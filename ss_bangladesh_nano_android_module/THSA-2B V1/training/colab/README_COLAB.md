# THSA-2B V1 — Google Colab Real 2B Training Guide

This guide provides step-by-step instructions for executing the physical GPU knowledge distillation and Quantization-Aware Training (QAT) run for **THSA-2B V1** ($2,050,296,320$ parameters) on Google Colab.

---

## 1. Google Colab Hardware & Runtime Requirements

- **Runtime Type:** Python 3 (GPU Accelerated)
- **Recommended Hardware Tier:**
  - **Premium (Recommended):** NVIDIA A100 GPU ($40\text{ GB} / 80\text{ GB}$ VRAM) or V100 ($16\text{ GB} / 32\text{ GB}$).
  - **Standard (Supported):** NVIDIA T4 GPU ($15\text{ GB}$ VRAM with batch size 1 and Adafactor memory-factored optimizer).
- **Google Drive Storage:** Minimum $10\text{ GB}$ free space on Google Drive for storing checkpoint files ($4.1\text{ GB}$ per FP16/BF16 checkpoint).

---

## 2. Cell-by-Cell Colab Execution Workflow

### Cell 1: Mount Google Drive for Permanent Checkpoint Storage
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/THSA-2B/checkpoints
!mkdir -p /content/drive/MyDrive/THSA-2B/logs
```

### Cell 2: Clone Repository & Navigate to Module Root
```bash
!git clone https://github.com/tpxplatfrom-afk/SS_module_BD.git /content/SS_module_BD
%cd /content/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B\ V1
```

### Cell 3: Environment Setup & Package Installation
```bash
!bash training/colab/setup_colab.sh
```

### Cell 4: Run Preflight Environment & Architecture Gate
```bash
!python training/colab/colab_preflight.py
```
*Expected Output:* `COLAB_PREFLIGHT_PASS`

### Cell 5: Execute 10-Step Real GPU Smoke Test
```bash
!python training/colab/real_gpu_smoke_test.py
```
*Expected Output:* `COLAB_REAL_GPU_SMOKE_PASS` (Verifies non-zero parameter delta on CUDA GPU).

### Cell 6: Start Full Production Distillation & QAT Training Run
```bash
!python training/distillation/qwen_teacher_distillation.py \
    --config training/config/thsa_2b_config.json \
    --teacher Qwen/Qwen2.5-7B-Instruct \
    --corpus data/processed/clean_pretrain_corpus.txt \
    --output_dir /content/drive/MyDrive/THSA-2B/checkpoints \
    --steps 10000 \
    --batch_size 1 \
    --grad_accum 16 \
    --checkpoint_interval 500 \
    --precision bfloat16 \
    --device cuda
```

### Cell 7: Resume Training After Interruption / Disconnect
If the Colab runtime disconnects, reconnect, re-mount Drive, and run:
```bash
!python training/distillation/qwen_teacher_distillation.py \
    --config training/config/thsa_2b_config.json \
    --teacher Qwen/Qwen2.5-7B-Instruct \
    --corpus data/processed/clean_pretrain_corpus.txt \
    --output_dir /content/drive/MyDrive/THSA-2B/checkpoints \
    --steps 10000 \
    --batch_size 1 \
    --grad_accum 16 \
    --checkpoint_interval 500 \
    --resume auto \
    --precision bfloat16 \
    --device cuda
```

### Cell 8: Verify Final Checkpoint & Generate Manifest
```bash
!python training/colab/verify_checkpoint.py \
    --checkpoint /content/drive/MyDrive/THSA-2B/checkpoints/thsa_2b_trained_final.pt \
    --output /content/drive/MyDrive/THSA-2B/THSA-2B-CHECKPOINT-MANIFEST.json
```
*Expected Output:* `CHECKPOINT_VERIFICATION_PASS`

---

## 3. What Constitutes Training Success vs Blocked

- **Success:**
  1. `thsa_2b_trained_final.pt` is generated in Google Drive.
  2. Total parameter count is **exactly 2,050,296,320** ($219\text{ tensors}$).
  3. Weights differ from initial state (non-zero updates).
  4. Checkpoint manifest passes verification with zero synthetic/dummy tensors.
- **Blocked:**
  1. No GPU available in Colab (`torch.cuda.is_available() == False`).
  2. Insufficient Google Drive storage space.
  3. Teacher download fails without internet access.
