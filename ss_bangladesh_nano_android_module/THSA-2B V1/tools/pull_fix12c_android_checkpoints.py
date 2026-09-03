#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX-12C Fast Streaming Checkpoint Puller
========================================
Pulls all intermediate checkpoint binary files from the physical itel A662L phone
using a high-speed streaming tar pipeline over ADB.
"""

import os
import sys
import subprocess
import tarfile
import io
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_BASE = SCRIPT_DIR / "fix12c" / "android"
LOCAL_BASE.mkdir(parents=True, exist_ok=True)

ADB = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"
PKG = "com.aistudio.offlineai.krvq"

def main():
    print("=" * 80)
    print("FIX-12C STREAMING PULL OF ANDROID CHECKPOINTS VIA TAR PIPELINE")
    print("=" * 80)

    tar_path = SCRIPT_DIR / "fix12c" / "android_fix12c_bundle.tar"

    print("Streaming tar from device...")
    cmd = [ADB, "exec-out", "run-as", PKG, "tar", "-c", "-C", f"/data/data/{PKG}/files", "fix12c"]
    with open(tar_path, "wb") as f_out:
        res = subprocess.run(cmd, stdout=f_out, stderr=subprocess.PIPE)

    if res.returncode != 0:
        print(f"Error streaming tar: {res.stderr.decode('utf-8', errors='ignore')}")
        return

    sz = tar_path.stat().st_size
    print(f"Tar bundle received: {sz:,} bytes")

    # Extract tar archive
    print(f"Extracting into {LOCAL_BASE.parent} ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=LOCAL_BASE.parent)

    # If extracted to tools/fix12c/fix12c, move contents to tools/fix12c/android
    extracted_fix12c = LOCAL_BASE.parent / "fix12c"
    # Actually LOCAL_BASE.parent is tools/fix12c, so tar extracted into tools/fix12c/fix12c
    if extracted_fix12c.exists() and extracted_fix12c != LOCAL_BASE:
        # Move all prompt_* subdirs into LOCAL_BASE
        for p_dir in extracted_fix12c.glob("prompt_*"):
            target = LOCAL_BASE / p_dir.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(p_dir), str(target))
        shutil.rmtree(extracted_fix12c, ignore_errors=True)

    # Count extracted files
    total_files = 0
    for p_dir in sorted(LOCAL_BASE.glob("prompt_*")):
        bins = list(p_dir.glob("*.bin"))
        print(f"  {p_dir.name}: {len(bins)} checkpoint files")
        total_files += len(bins)

    print(f"\nSUCCESS: Extracted total {total_files} checkpoint files to {LOCAL_BASE}")

if __name__ == "__main__":
    main()
