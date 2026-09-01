"""
SS Tutor BD - Model Manager
Handles candidate listing, license gate verification, single-model download, file integrity checks, and weight purging.
"""

import os
import sys
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, List

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"
ACTIVE_DIR = PROJECT_ROOT / "models" / "active"
ACTIVE_META_PATH = ACTIVE_DIR / "active_model.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_registry() -> Dict[str, Any]:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(data: Dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_disk_free_mb(path: Path) -> float:
    """Returns free space in MB for the drive containing path."""
    total, used, free = shutil.disk_usage(path.anchor if path.anchor else ".")
    return free / (1024 * 1024)


def list_candidates() -> List[Dict[str, Any]]:
    registry = load_registry()
    return registry.get("candidates", [])


def get_candidate(candidate_id: str) -> Optional[Dict[str, Any]]:
    candidates = list_candidates()
    for c in candidates:
        if c["id"].upper() == candidate_id.upper():
            return c
    return None


def get_active_model() -> Optional[Dict[str, Any]]:
    if ACTIVE_META_PATH.exists():
        try:
            with open(ACTIVE_META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def verify_license_gate(candidate: Dict[str, Any]) -> bool:
    """Enforces Gate 1: Candidate must have LICENSE_PASSED status."""
    return candidate.get("license_status") == "LICENSE_PASSED"


def verify_gguf_header(filepath: Path) -> bool:
    """Verifies that the file starts with the GGUF magic header (0x47 0x47 0x55 0x46)."""
    if not filepath.exists() or filepath.stat().st_size < 1024:
        return False
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            return magic == b"GGUF"
    except Exception:
        return False


def purge_active_model() -> Dict[str, Any]:
    """Safely removes active model weights and active metadata."""
    purged_files = []
    freed_bytes = 0
    if ACTIVE_DIR.exists():
        for item in ACTIVE_DIR.iterdir():
            if item.is_file():
                freed_bytes += item.stat().st_size
                purged_files.append(item.name)
                item.unlink()
    
    return {
        "status": "PURGED",
        "files_removed": purged_files,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "disk_free_mb": round(get_disk_free_mb(PROJECT_ROOT), 2)
    }


def download_candidate(candidate_id: str, force: bool = False) -> Dict[str, Any]:
    """Downloads a single pre-quantized candidate model after checking license and disk space."""
    candidate = get_candidate(candidate_id)
    if not candidate:
        raise ValueError(f"Candidate '{candidate_id}' not found in registry.")

    # 1. License Gate Check
    if not verify_license_gate(candidate):
        raise PermissionError(
            f"Candidate {candidate_id} failed Gate 1 (License Gate). "
            f"Current status is '{candidate.get('license_status')}'. Only 'LICENSE_PASSED' models can be downloaded."
        )

    # 2. Check active model
    active = get_active_model()
    if active and active.get("id") != candidate["id"]:
        if not force:
            raise RuntimeError(
                f"Another candidate '{active.get('id')}' is currently active. "
                "Purge the active model first or use force=True."
            )
        else:
            purge_active_model()

    # 3. Storage Safety Check
    config = load_config()
    est_size_mb = candidate.get("est_file_size_mb", 500)
    current_free_mb = get_disk_free_mb(PROJECT_ROOT)
    min_required_free = config.get("min_disk_free_mb", 1500)

    if (current_free_mb - est_size_mb) < min_required_free:
        raise RuntimeError(
            f"Storage Safety Alert: Downloading {est_size_mb} MB would leave only "
            f"{round(current_free_mb - est_size_mb, 1)} MB free. "
            f"Minimum safe margin is {min_required_free} MB."
        )

    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    gguf_filename = candidate["gguf_file"]
    target_path = ACTIVE_DIR / gguf_filename

    # 4. Perform Download via huggingface_hub
    if not target_path.exists() or target_path.stat().st_size == 0:
        repo_id = candidate["gguf_repo_id"]
        print(f"[Model Manager] Downloading {gguf_filename} from {repo_id}...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=gguf_filename,
                local_dir=str(ACTIVE_DIR),
                local_dir_use_symlinks=False
            )
            target_path = Path(downloaded_file)
        except Exception as e:
            raise RuntimeError(f"Download failed for {candidate_id}: {str(e)}")

    # 5. Verify GGUF header
    if not verify_gguf_header(target_path):
        target_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file for {candidate_id} is corrupt or not a valid GGUF binary.")

    # 6. Save active metadata
    actual_size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)
    active_meta = {
        "id": candidate["id"],
        "name": candidate["name"],
        "parameters_billion": candidate["parameters_billion"],
        "quantization": "Q4_K_M",
        "file_path": str(target_path.resolve()),
        "file_size_mb": actual_size_mb,
        "license": candidate["license"],
        "tokenizer_repo_id": candidate.get("tokenizer_repo_id")
    }

    with open(ACTIVE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(active_meta, f, indent=2, ensure_ascii=False)

    return {
        "status": "DOWNLOAD_SUCCESS",
        "candidate_id": candidate["id"],
        "file_path": str(target_path),
        "file_size_mb": actual_size_mb,
        "remaining_disk_mb": round(get_disk_free_mb(PROJECT_ROOT), 2)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manager.py [list | status | purge | download <ID>]")
        sys.exit(0)

    action = sys.argv[1].lower()
    if action == "list":
        candidates = list_candidates()
        print(f"{'ID':<10} {'Name':<28} {'Params':<8} {'License':<15} {'License Status':<20} {'Est MB'}")
        print("-" * 90)
        for c in candidates:
            print(f"{c['id']:<10} {c['name']:<28} {c['parameters_billion']:<8} {c['license']:<15} {c['license_status']:<20} {c['est_file_size_mb']}")
    elif action == "status":
        active = get_active_model()
        if active:
            print(f"Active Model: {active['id']} ({active['name']})")
            print(f"File Size: {active['file_size_mb']} MB")
            print(f"Path: {active['file_path']}")
        else:
            print("No model currently active.")
        print(f"Disk Free: {round(get_disk_free_mb(PROJECT_ROOT), 2)} MB")
    elif action == "purge":
        res = purge_active_model()
        print(f"Purged: {res['files_removed']}, Freed: {res['freed_mb']} MB, Disk Free: {res['disk_free_mb']} MB")
    elif action == "download" and len(sys.argv) > 2:
        cid = sys.argv[2]
        res = download_candidate(cid)
        print(f"Downloaded {res['candidate_id']}: {res['file_size_mb']} MB. Disk Free: {res['remaining_disk_mb']} MB")
    else:
        print(f"Unknown action '{action}'")
