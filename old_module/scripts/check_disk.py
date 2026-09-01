"""
SS Tutor BD - Disk & Storage Health Check Utility
Reports available disk capacity, active model allocations, and storage safety headroom.
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_DIR = PROJECT_ROOT / "models" / "active"


def check_disk_health():
    total, used, free = shutil.disk_usage(PROJECT_ROOT.anchor if PROJECT_ROOT.anchor else ".")
    total_gb = round(total / (1024 ** 3), 2)
    used_gb = round(used / (1024 ** 3), 2)
    free_gb = round(free / (1024 ** 3), 2)
    free_mb = round(free / (1024 ** 2), 2)

    active_model_size_mb = 0.0
    active_files = []
    if ACTIVE_DIR.exists():
        for f in ACTIVE_DIR.iterdir():
            if f.is_file():
                sz = f.stat().st_size / (1024 ** 2)
                active_model_size_mb += sz
                active_files.append((f.name, round(sz, 2)))

    print("=" * 60)
    print("SS TUTOR BD — HOST STORAGE STATUS REPORT")
    print("=" * 60)
    print(f"Target Drive:             {PROJECT_ROOT.anchor}")
    print(f"Total Storage:            {total_gb} GB")
    print(f"Used Storage:             {used_gb} GB")
    print(f"Free Storage:             {free_gb} GB ({free_mb} MB)")
    print(f"Active Model Footprint:   {round(active_model_size_mb, 2)} MB")
    if active_files:
        for name, sz in active_files:
            print(f"  - {name}: {sz} MB")
    else:
        print("  - [No active model weights in models/active/]")

    print("-" * 60)
    if free_mb < 1500:
        print("CRITICAL WARNING: Free disk space is below safe 1500 MB threshold!")
    else:
        print("STORAGE HEALTH: OK (Safe for single-model quantized benchmarking)")
    print("=" * 60)


if __name__ == "__main__":
    check_disk_health()
