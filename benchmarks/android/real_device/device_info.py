"""
SS Tutor BD — Real Device Info Collector (Phase 6)
Discovers connected Android devices via ADB and captures detailed hardware, OS, and memory specs.
"""
import sys
import os
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ADB_PATH = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_adb_cmd(args: list[str], device_id: str | None = None) -> str:
    cmd = [ADB_PATH]
    if device_id:
        cmd.extend(["-s", device_id])
    cmd.extend(args)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.stdout.strip()


def get_connected_devices() -> list[str]:
    output = run_adb_cmd(["devices"])
    devices = []
    for line in output.splitlines():
        line = line.strip()
        if line.endswith("device") and not line.startswith("List"):
            dev_id = line.split()[0]
            devices.append(dev_id)
    return devices


def collect_device_profile(device_id: str | None = None) -> dict:
    devs = get_connected_devices()
    if not devs and not device_id:
        return {"status": "BLOCKED", "error": "No physical Android device connected via ADB"}

    target_id = device_id or devs[0]

    def prop(name: str) -> str:
        return run_adb_cmd(["shell", "getprop", name], target_id)

    def shell(cmd: str) -> str:
        return run_adb_cmd(["shell", cmd], target_id)

    manufacturer = prop("ro.product.manufacturer") or "Unknown"
    model = prop("ro.product.model") or "Unknown"
    android_ver = prop("ro.build.version.release") or "Unknown"
    sdk_level = int(prop("ro.build.version.sdk") or "0")
    abi = prop("ro.product.cpu.abi") or "Unknown"
    cpu_abi2 = prop("ro.product.cpu.abi2") or ""
    board = prop("ro.board.platform") or prop("ro.hardware") or "Unknown"

    # Memory info from /proc/meminfo
    meminfo_raw = shell("cat /proc/meminfo")
    total_ram_kb = 0
    avail_ram_kb = 0
    for line in meminfo_raw.splitlines():
        if line.startswith("MemTotal:"):
            total_ram_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail_ram_kb = int(line.split()[1])

    # Storage info
    df_raw = shell("df -h /data")
    data_size = "Unknown"
    data_avail = "Unknown"
    for line in df_raw.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4:
            data_size = parts[1]
            data_avail = parts[3]
            break

    profile = {
        "status": "VERIFIED",
        "device_id": target_id,
        "manufacturer": manufacturer,
        "model": model,
        "android_version": android_ver,
        "sdk_level": sdk_level,
        "primary_abi": abi,
        "secondary_abi": cpu_abi2,
        "board_platform": board,
        "total_ram_mb": round(total_ram_kb / 1024.0, 2),
        "available_ram_mb": round(avail_ram_kb / 1024.0, 2),
        "total_ram_gb": round(total_ram_kb / (1024.0 * 1024.0), 2),
        "data_storage_total": data_size,
        "data_storage_available": data_avail,
        "is_2gb_ram_class": (1400.0 <= (total_ram_kb / 1024.0) <= 2200.0),
        "target_pss_budget_mb": 150.0,
        "hard_pss_ceiling_mb": 200.0,
        "emergency_pss_ceiling_mb": 250.0
    }

    out_dir = PROJECT_ROOT / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "device_profile.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return profile


if __name__ == "__main__":
    prof = collect_device_profile()
    print("\n============================================================")
    print("  SS TUTOR BD — REAL DEVICE HARDWARE PROFILE")
    print("============================================================")
    print(json.dumps(prof, indent=2, ensure_ascii=False))
    print("============================================================\n")
