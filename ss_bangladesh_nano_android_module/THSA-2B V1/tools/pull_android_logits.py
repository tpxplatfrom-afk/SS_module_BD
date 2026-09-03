#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary Logits & Diagnostics Puller for Physical Android Device
==============================================================
Uses `adb exec-out run-as com.aistudio.offlineai.krvq cat ...` to stream raw bytes
directly to local binary files with zero encoding or conversion corruption.
"""

import os
import sys
import subprocess
import hashlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "fix12b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADB = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"
PKG = "com.aistudio.offlineai.krvq"
REMOTE_DIR = f"/data/data/{PKG}/files"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def pull_file(remote_path, local_path):
    cmd = [ADB, "exec-out", "run-as", PKG, "cat", remote_path]
    with open(local_path, "wb") as f_out:
        res = subprocess.run(cmd, stdout=f_out, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"  FAILED to pull {remote_path}: {res.stderr.decode('utf-8', errors='ignore')}")
        return False
    return True

def main():
    print("=" * 70)
    print("PULLING ANDROID BINARY LOGITS & DIAGNOSTICS FROM PHYSICAL PHONE")
    print("=" * 70)

    # 1. Pull logits for all 5 prompts
    # Prompts in test01:
    # TEST-A: first prompt executed -> generates fix12_logits_p0.bin (last token forward) or p1
    # Let's list files in remote files directory first
    res = subprocess.run([ADB, "exec-out", "run-as", PKG, "ls", "-la", REMOTE_DIR],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Remote files directory contents:")
    print(res.stdout)

    # Search for all fix12_logits_p*.bin files
    pulled_count = 0
    for line in res.stdout.splitlines():
        parts = line.split()
        if not parts: continue
        fname = parts[-1]
        if fname.startswith("fix12_logits_p") and fname.endswith(".bin"):
            rem = f"{REMOTE_DIR}/{fname}"
            loc = OUT_DIR / fname
            if pull_file(rem, loc):
                sz = loc.stat().st_size
                sha = sha256_file(loc)
                print(f"  Pulled {fname}: {sz:,} bytes, SHA256={sha}")
                pulled_count += 1

    # Also pull fix12_diag.bin and fix12_perf.txt
    for diag_name in ["fix12_diag.bin", "fix12_perf.txt"]:
        rem = f"{REMOTE_DIR}/{diag_name}"
        loc = OUT_DIR / diag_name
        if pull_file(rem, loc):
            sz = loc.stat().st_size
            print(f"  Pulled {diag_name}: {sz:,} bytes")

    print(f"\nPull complete: {pulled_count} logit files pulled to {OUT_DIR}")

if __name__ == "__main__":
    main()
