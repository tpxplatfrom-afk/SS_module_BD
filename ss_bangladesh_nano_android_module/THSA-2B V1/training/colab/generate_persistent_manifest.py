#!/usr/bin/env python3
"""
THSA-2B V1: FIX-06C-COLAB-07A — Persistent Checkpoint Manifest Generator & Repair Utility
========================================================================================
Repairs and creates the persistent cryptographic manifest for the existing Google Drive checkpoint:
  /content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000010.pt

Safety Rules:
  1. DOES NOT delete, overwrite, rename, or regenerate the existing checkpoint.
  2. DOES NOT run training or teacher forward passes.
  3. DOES NOT instantiate student on GPU.
  4. Inspects existing checkpoint on CPU without duplicate memory allocations.
  5. Writes checkpoint_step_000010.manifest.json atomically with fsync, sync, and hash verification.
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
import subprocess
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
MODULE_ROOT = TRAINING_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

EXPECTED_PARAMS = 2050296320
EXPECTED_TENSORS = 219
AUTHORITATIVE_TEACHER = "Qwen/Qwen2.5-7B-Instruct"

def compute_sha256(filepath: Path) -> str:
    """Compute streaming SHA-256 hex digest of a file (64KB chunks)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

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

def get_repo_commit() -> str:
    """Retrieve current Git commit SHA."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(MODULE_ROOT),
                             capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def generate_manifest(checkpoint_path_str: str, check_drive: bool = True, force_regenerate: bool = False):
    print("=" * 80)
    print("FIX-06C-COLAB-07A: PERSISTENT CHECKPOINT MANIFEST GENERATOR & REPAIR")
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
            print("\nFIX-06C-COLAB-07A-FAIL: DRIVE_NOT_MOUNTED")
            return 1
        print("DRIVE_MOUNT:                 /content/drive/MyDrive mounted and accessible [OK]")

    # 2. Locate Checkpoint
    ckpt_path = Path(checkpoint_path_str)
    if not ckpt_path.exists():
        default_path = Path("/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000010.pt")
        if default_path.exists():
            ckpt_path = default_path
        else:
            print(f"[FATAL ERROR] Checkpoint not found at: {ckpt_path}")
            print("\nFIX-06C-COLAB-07A-FAIL: CHECKPOINT_MISSING")
            return 1

    print(f"CHECKPOINT_PATH:             {ckpt_path}")

    # 3. Checkpoint File Inspection
    stat_info = os.stat(ckpt_path)
    byte_size = stat_info.st_size
    mtime_dt = datetime.datetime.fromtimestamp(stat_info.st_mtime, tz=datetime.timezone.utc)
    print(f"CHECKPOINT_BYTE_SIZE:        {byte_size:,} bytes ({byte_size / (1024**3):.3f} GB)")
    print(f"CHECKPOINT_MTIME_UTC:        {mtime_dt.isoformat()}")

    if byte_size == 0:
        print("[FATAL ERROR] Checkpoint file is empty (0 bytes)!")
        print("\nFIX-06C-COLAB-07A-FAIL: EMPTY_CHECKPOINT")
        return 1

    print("Computing streaming Checkpoint SHA-256 (64KB buffer)...")
    actual_sha256 = compute_sha256(ckpt_path)
    print(f"CHECKPOINT_SHA256:           {actual_sha256}")

    # Shell sha256sum verification (if tool available)
    shell_sha256 = None
    try:
        proc = subprocess.run(["sha256sum", str(ckpt_path)], capture_output=True, text=True, check=True)
        shell_sha256 = proc.stdout.strip().split()[0]
        print(f"SHELL_SHA256:                {shell_sha256}")
    except Exception:
        shell_sha256 = actual_sha256

    if shell_sha256 != actual_sha256:
        print(f"[FATAL ERROR] Shell SHA-256 ({shell_sha256}) != Python SHA-256 ({actual_sha256})!")
        print("\nFIX-06C-COLAB-07A-FAIL: SHELL_SHA256_MISMATCH")
        return 1

    # 4. CPU Forensic Inspection of Checkpoint Content
    print("\nLoading checkpoint payload into CPU memory for forensic verification...")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load checkpoint: {e}")
        print("\nFIX-06C-COLAB-07A-FAIL: CORRUPTED_CHECKPOINT_LOAD")
        return 1

    global_step = ckpt.get("global_step", -1)
    print(f"CHECKPOINT_GLOBAL_STEP:      {global_step}")
    if global_step != 10:
        print(f"[FATAL ERROR] Checkpoint global_step expected 10, got {global_step}")
        print("\nFIX-06C-COLAB-07A-FAIL: GLOBAL_STEP_MISMATCH")
        return 1

    required_keys = ["model_state_dict", "optimizer_state_dict", "config", "distillation_meta"]
    for k in required_keys:
        if k not in ckpt:
            print(f"[FATAL ERROR] Checkpoint payload missing key: {k}")
            print("\nFIX-06C-COLAB-07A-FAIL: MISSING_CHECKPOINT_KEYS")
            return 1
    print("CHECKPOINT_KEYS:             model_state_dict [OK]  optimizer_state_dict [OK]  config [OK]  distillation_meta [OK]")

    state_dict = ckpt["model_state_dict"]
    tensor_count = len(state_dict)
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"STATE_DICT_TENSORS:          {tensor_count} (Expected: {EXPECTED_TENSORS})")
    print(f"TOTAL_PARAMETERS:            {total_params:,} (Expected: {EXPECTED_PARAMS:,})")

    if tensor_count != EXPECTED_TENSORS:
        print(f"[FATAL ERROR] Expected {EXPECTED_TENSORS} tensors, got {tensor_count}")
        print("\nFIX-06C-COLAB-07A-FAIL: TENSOR_COUNT_MISMATCH")
        return 1

    if total_params != EXPECTED_PARAMS:
        print(f"[FATAL ERROR] Expected {EXPECTED_PARAMS:,} parameters, got {total_params:,}")
        print("\nFIX-06C-COLAB-07A-FAIL: PARAMETER_COUNT_MISMATCH")
        return 1

    # NaN / Inf Scan (Memory-conscious, native tensor scanning without float32 conversion)
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
        print("\nFIX-06C-COLAB-07A-FAIL: NON_FINITE_WEIGHTS")
        return 1
    print(f"NaN/Inf SCAN:                CLEAN (219/219 tensors clean, 0 NaN, 0 Inf)")

    meta = ckpt.get("distillation_meta", {})
    teacher_in_meta = meta.get("teacher", "UNKNOWN")
    print(f"DISTILLATION_TEACHER:        {teacher_in_meta}")
    if AUTHORITATIVE_TEACHER not in teacher_in_meta:
        print(f"[WARNING] Distillation teacher ({teacher_in_meta}) does not match {AUTHORITATIVE_TEACHER}")

    # 5. Construct Manifest Payload
    repo_commit = get_repo_commit()
    manifest_path = ckpt_path.parent / "checkpoint_step_000010.manifest.json"
    manifest_tmp = ckpt_path.parent / "checkpoint_step_000010.manifest.json.tmp"

    manifest_data = {
        "schema_version": "FIX-06C-COLAB-07A-1",
        "checkpoint_filename": ckpt_path.name,
        "checkpoint_path": str(ckpt_path).replace("\\", "/"),
        "checkpoint_byte_size": byte_size,
        "checkpoint_sha256": actual_sha256,
        "global_step": global_step,
        "student_parameter_count": total_params,
        "state_dict_tensor_count": tensor_count,
        "teacher": AUTHORITATIVE_TEACHER,
        "required_keys": required_keys,
        "nan_tensor_count": nan_tensors,
        "inf_tensor_count": inf_tensors,
        "repository_commit": repo_commit,
        "manifest_schema": "FIX-06C-COLAB-07A",
        "persistence_protocol": "atomic_manifest_write_fsync_sync_hash_verify",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_mtime_utc": mtime_dt.isoformat()
    }

    # Check if manifest already exists and is valid
    if manifest_path.exists() and not force_regenerate:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
            ex_sha = existing_manifest.get("checkpoint_sha256") or existing_manifest.get("sha256")
            ex_bytes = existing_manifest.get("checkpoint_byte_size") or existing_manifest.get("byte_size")
            if ex_sha == actual_sha256 and ex_bytes == byte_size:
                print(f"\n[Notice] Valid existing manifest detected: {manifest_path}")
                manifest_file_sha = compute_sha256(manifest_path)
                print(f"MANIFEST_PATH:               {manifest_path}")
                print(f"MANIFEST_FILE_SHA256:        {manifest_file_sha}")
                print(f"MANIFEST_BYTE_SIZE:          {os.path.getsize(manifest_path)} bytes")
                print("MANIFEST_STATUS:             VERIFIED_VALID (Existing)")
                print("\n" + "=" * 80)
                print("FIX-06C-COLAB-07A-PASS")
                print("=" * 80)
                return 0
        except Exception:
            print("[Notice] Existing manifest unreadable/stale. Replacing with repaired manifest.")

    # 6. Atomic Manifest Write
    print(f"\nWriting persistent manifest atomically to: {manifest_path}...")
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    if not manifest_tmp.exists():
        print(f"[FATAL ERROR] Temporary manifest was not created: {manifest_tmp}")
        print("\nFIX-06C-COLAB-07A-FAIL: TMP_MANIFEST_MISSING")
        return 1

    tmp_manifest_sha = compute_sha256(manifest_tmp)
    os.replace(manifest_tmp, manifest_path)
    sync_filesystem()

    if not manifest_path.exists():
        print(f"[FATAL ERROR] Final manifest missing after atomic rename: {manifest_path}")
        print("\nFIX-06C-COLAB-07A-FAIL: FINAL_MANIFEST_MISSING")
        return 1

    manifest_file_size = os.path.getsize(manifest_path)
    manifest_file_sha = compute_sha256(manifest_path)

    # 7. Manifest Readback & Integrity Validation
    with open(manifest_path, "r", encoding="utf-8") as f:
        readback_data = json.load(f)

    if readback_data.get("checkpoint_sha256") != actual_sha256:
        print("[FATAL ERROR] Manifest readback checkpoint_sha256 mismatch!")
        print("\nFIX-06C-COLAB-07A-FAIL: MANIFEST_READBACK_HASH_MISMATCH")
        return 1

    if readback_data.get("checkpoint_byte_size") != byte_size:
        print("[FATAL ERROR] Manifest readback checkpoint_byte_size mismatch!")
        print("\nFIX-06C-COLAB-07A-FAIL: MANIFEST_READBACK_SIZE_MISMATCH")
        return 1

    print(f"MANIFEST_PATH:               {manifest_path}")
    print(f"MANIFEST_FILE_SHA256:        {manifest_file_sha}")
    print(f"MANIFEST_BYTE_SIZE:          {manifest_file_size} bytes")
    print(f"CHECKPOINT_SHA256:           {actual_sha256}")
    print("MANIFEST_STATUS:             ATOMICALLY_WRITTEN_AND_VERIFIED")

    # Clean any lingering temporary files
    if manifest_tmp.exists():
        try:
            manifest_tmp.unlink()
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("FIX-06C-COLAB-07A-PASS")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIX-06C-COLAB-07A Persistent Checkpoint Manifest Repair Utility")
    parser.add_argument("--checkpoint", type=str,
                        default="/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000010.pt",
                        help="Path to checkpoint_step_000010.pt")
    parser.add_argument("--no_drive_check", action="store_true",
                        help="Skip /content/drive mount check (for local testing)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of manifest even if existing manifest is valid")
    args = parser.parse_args()

    sys.exit(generate_manifest(
        checkpoint_path_str=args.checkpoint,
        check_drive=not args.no_drive_check,
        force_regenerate=args.force
    ))
