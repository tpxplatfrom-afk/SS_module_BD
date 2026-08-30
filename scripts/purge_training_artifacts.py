"""
SS Tutor BD - Training Artifacts Purge Script (Phase 4)
Purges intermediate training checkpoints to enforce the single-model storage policy.
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "sstutor_bengali_70m_edu" / "checkpoints"


def purge_training_checkpoints():
    print("\n--- SS Tutor BD: Purging Training Checkpoints ---")
    if CHECKPOINTS_DIR.exists():
        size_bytes = sum(f.stat().st_size for f in CHECKPOINTS_DIR.rglob('*') if f.is_file())
        size_mb = round(size_bytes / (1024 * 1024), 2)
        shutil.rmtree(CHECKPOINTS_DIR)
        print(f"Purged checkpoints folder: {CHECKPOINTS_DIR} (Freed: {size_mb} MB)")
    else:
        print("No intermediate checkpoints directory found. Clean.")

    # Check disk status
    from models.manager import get_disk_free_mb
    free_mb = get_disk_free_mb(str(PROJECT_ROOT))
    print(f"Current Host Free Disk: {free_mb:.2f} MB")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    purge_training_checkpoints()
