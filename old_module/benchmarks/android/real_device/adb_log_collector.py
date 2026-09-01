"""
SS Tutor BD — Real Device ADB Log Collector (Phase 6)
Monitors logcat for fatal exceptions, ANRs, GC pauses, and Android trimMemory events.
"""
import sys
import subprocess
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ADB_PATH = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ADBLogCollector:
    def __init__(self, device_id: str | None = None):
        self.device_id = device_id

    def _adb_cmd(self, args: list[str]) -> str:
        cmd = [ADB_PATH]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return res.stdout.strip()

    def clear_logcat(self):
        self._adb_cmd(["logcat", "-c"])

    def collect_stability_report(self, package_name: str = "bd.sstutor.app") -> dict:
        logcat_raw = self._adb_cmd(["logcat", "-d", "-t", "500"])
        
        crashes = []
        anrs = []
        oom_kills = []
        memory_trims = []

        for line in logcat_raw.splitlines():
            line_str = line.strip()
            if "FATAL EXCEPTION" in line_str or "AndroidRuntime: FATAL" in line_str:
                if package_name in line_str or "sstutor" in line_str.lower():
                    crashes.append(line_str)
            elif "ANR in" in line_str and package_name in line_str:
                anrs.append(line_str)
            elif "LowMemoryKiller" in line_str or "kill" in line_str.lower() and package_name in line_str:
                oom_kills.append(line_str)
            elif "onTrimMemory" in line_str or "TRIM_MEMORY" in line_str:
                memory_trims.append(line_str)

        stability = {
            "status": "VERIFIED_STABLE" if not crashes and not anrs else "UNSTABLE",
            "crash_count": len(crashes),
            "anr_count": len(anrs),
            "oom_kill_count": len(oom_kills),
            "memory_trim_events": len(memory_trims),
            "crashes": crashes,
            "anrs": anrs,
            "zero_crash_gate": len(crashes) == 0,
            "zero_anr_gate": len(anrs) == 0
        }

        out_dir = PROJECT_ROOT / "results" / "phase6" / "stability"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stability_report.json", "w", encoding="utf-8") as f:
            json.dump(stability, f, indent=2, ensure_ascii=False)

        return stability


if __name__ == "__main__":
    collector = ADBLogCollector()
    rep = collector.collect_stability_report()
    print("\n" + "="*60)
    print("  SS TUTOR BD — REAL DEVICE STABILITY REPORT")
    print("="*60)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("="*60 + "\n")
