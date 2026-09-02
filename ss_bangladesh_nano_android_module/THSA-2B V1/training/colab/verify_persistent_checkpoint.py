#!/usr/bin/env python3
"""
THSA-2B V1: Fresh-Runtime Persistent Checkpoint Verifier
=========================================================
Executes in a fresh Google Colab runtime after VM restart and Drive remount:
  1. Verifies Google Drive mount at /content/drive
  2. Locates checkpoint_step_000010.pt
  3. Measures byte size & computes SHA-256
  4. Locates and parses checkpoint_step_000010.manifest.json
  5. Verifies manifest byte size and SHA-256 match the physical checkpoint
  6. Loads checkpoint state_dict into CPU memory (zero heavy GPU memory usage)
  7. Verifies global_step == 10
  8. Verifies 219 tensors and 2,050,296,320 parameters
  9. Scans all tensors for NaN / Inf
 10. Emits PERSISTENT_CHECKPOINT_VERIFICATION_PASS or exact failure
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import torch

EXPECTED_PARAMS = 2050296320
EXPECTED_TENSORS = 219
AUTHORITATIVE_TEACHER = "Qwen/Qwen2.5-7B-Instruct"

def compute_sha256(filepath: Path) -> str:
    """Compute streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_fresh_runtime(checkpoint_path_str: str, manifest_path_str: str, check_drive: bool = True):
    print("=" * 80)
    print("THSA-2B V1: FRESH-RUNTIME PERSISTENT CHECKPOINT VERIFICATION")
    print("=" * 80)

    # 1. Drive Mount Verification
    if check_drive:
        drive_root = Path("/content/drive")
        drive_my_drive = Path("/content/drive/MyDrive")
        if not drive_root.exists() or not drive_my_drive.exists():
            print("[FATAL ERROR] Google Drive is not mounted at /content/drive!")
            print("Please run in Colab:")
            print("  from google.colab import drive")
            print("  drive.mount('/content/drive')")
            print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: DRIVE_NOT_MOUNTED")
            return 1
        print("DRIVE_MOUNT:              /content/drive/MyDrive mounted and accessible ✓")

    # 2. Locate Checkpoint
    ckpt_path = Path(checkpoint_path_str)
    if not ckpt_path.exists():
        default_path = Path("/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000010.pt")
        if default_path.exists():
            ckpt_path = default_path
        else:
            print(f"[FATAL ERROR] Checkpoint not found at: {ckpt_path}")
            print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: CHECKPOINT_MISSING")
            return 1

    print(f"CHECKPOINT_PATH:          {ckpt_path}")

    # 3. Checkpoint Size and SHA-256
    byte_size = os.path.getsize(ckpt_path)
    print(f"CHECKPOINT_BYTE_SIZE:     {byte_size:,} bytes ({byte_size/(1024**3):.3f} GB)")
    if byte_size == 0:
        print("[FATAL ERROR] Checkpoint file is empty (0 bytes)!")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: EMPTY_CHECKPOINT")
        return 1

    print("Computing Checkpoint SHA-256 (streaming)...")
    actual_sha256 = compute_sha256(ckpt_path)
    print(f"CHECKPOINT_SHA256:        {actual_sha256}")

    # 4. Manifest Verification
    manifest_path = Path(manifest_path_str) if manifest_path_str else ckpt_path.parent / "checkpoint_step_000010.manifest.json"
    if not manifest_path.exists():
        print(f"[FATAL ERROR] Manifest file missing at: {manifest_path}")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: MANIFEST_MISSING")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_sha = manifest.get("sha256")
    manifest_bytes = manifest.get("byte_size")
    manifest_teacher = manifest.get("teacher")
    manifest_step = manifest.get("global_step")

    print(f"MANIFEST_PATH:            {manifest_path}")
    print(f"MANIFEST_SHA256:          {manifest_sha}")
    print(f"MANIFEST_BYTE_SIZE:       {manifest_bytes:,} bytes")
    print(f"MANIFEST_TEACHER:         {manifest_teacher}")
    print(f"MANIFEST_GLOBAL_STEP:     {manifest_step}")

    if manifest_sha != actual_sha256:
        print(f"[FATAL ERROR] Checkpoint SHA-256 mismatch! Manifest={manifest_sha}, Actual={actual_sha256}")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: MANIFEST_SHA256_MISMATCH")
        return 1

    if manifest_bytes != byte_size:
        print(f"[FATAL ERROR] Checkpoint byte size mismatch! Manifest={manifest_bytes}, Actual={byte_size}")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: MANIFEST_BYTE_SIZE_MISMATCH")
        return 1

    if manifest_step != 10:
        print(f"[FATAL ERROR] Manifest global_step is {manifest_step} (expected 10)")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: MANIFEST_STEP_MISMATCH")
        return 1

    # 5. Load Checkpoint Payload (CPU only)
    print("\nLoading checkpoint payload into CPU memory...")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load checkpoint: {e}")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: CORRUPTED_CHECKPOINT_LOAD")
        return 1

    global_step = ckpt.get("global_step", -1)
    print(f"CHECKPOINT_GLOBAL_STEP:   {global_step}")
    if global_step != 10:
        print(f"[FATAL ERROR] Checkpoint global_step expected 10, got {global_step}")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: GLOBAL_STEP_MISMATCH")
        return 1

    # Check required keys
    for k in ("model_state_dict", "optimizer_state_dict", "config", "distillation_meta"):
        if k not in ckpt:
            print(f"[FATAL ERROR] Checkpoint payload missing key: {k}")
            print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: MISSING_CHECKPOINT_KEYS")
            return 1
    print("CHECKPOINT_KEYS:          model_state_dict ✓  optimizer_state_dict ✓  config ✓  distillation_meta ✓")

    # 6. Parameter & Tensor Count Verification
    state_dict = ckpt["model_state_dict"]
    tensor_count = len(state_dict)
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"STATE_DICT_TENSORS:       {tensor_count} (Expected: {EXPECTED_TENSORS})")
    print(f"TOTAL_PARAMETERS:         {total_params:,} (Expected: {EXPECTED_PARAMS:,})")

    if tensor_count != EXPECTED_TENSORS:
        print(f"[FATAL ERROR] Expected {EXPECTED_TENSORS} tensors, got {tensor_count}")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: TENSOR_COUNT_MISMATCH")
        return 1

    if total_params != EXPECTED_PARAMS:
        print(f"[FATAL ERROR] Expected {EXPECTED_PARAMS:,} parameters, got {total_params:,}")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: PARAMETER_COUNT_MISMATCH")
        return 1

    # 7. NaN / Inf Scan
    nan_tensors, inf_tensors = 0, 0
    for name, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            nan_tensors += 1
            print(f"  [NaN DETECTED] {name}")
        if torch.isinf(tensor).any():
            inf_tensors += 1
            print(f"  [Inf DETECTED] {name}")

    if nan_tensors > 0 or inf_tensors > 0:
        print(f"[FATAL ERROR] Non-finite weights: {nan_tensors} NaN, {inf_tensors} Inf")
        print("\nPERSISTENT_CHECKPOINT_VERIFICATION_FAIL: NON_FINITE_WEIGHTS")
        return 1
    print(f"NaN/Inf SCAN:             CLEAN (219/219 tensors clean, 0 NaN, 0 Inf)")

    # 8. Teacher Metadata
    meta = ckpt.get("distillation_meta", {})
    teacher_in_meta = meta.get("teacher", "UNKNOWN")
    print(f"DISTILLATION_TEACHER:     {teacher_in_meta}")
    if AUTHORITATIVE_TEACHER not in teacher_in_meta:
        print(f"[WARNING] Distillation teacher ({teacher_in_meta}) does not match {AUTHORITATIVE_TEACHER}")

    print("\n" + "=" * 80)
    print("PERSISTENT_CHECKPOINT_VERIFICATION_PASS")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THSA-2B Fresh-Runtime Persistent Checkpoint Verifier")
    parser.add_argument("--checkpoint", type=str,
                        default="/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000010.pt",
                        help="Path to checkpoint_step_000010.pt")
    parser.add_argument("--manifest", type=str, default="",
                        help="Path to checkpoint_step_000010.manifest.json (auto-located if empty)")
    parser.add_argument("--no_drive_check", action="store_true",
                        help="Skip /content/drive mount check (for non-Colab local testing)")
    args = parser.parse_args()

    sys.exit(verify_fresh_runtime(
        checkpoint_path_str=args.checkpoint,
        manifest_path_str=args.manifest,
        check_drive=not args.no_drive_check
    ))
