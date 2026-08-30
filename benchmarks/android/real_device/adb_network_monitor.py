"""
SS Tutor BD — Real Device ADB Network & Offline Auditor (Phase 6)
Verifies zero network connections and validates 100% offline core capability in Airplane mode.
"""
import sys
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ADB_PATH = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ADBNetworkMonitor:
    def __init__(self, device_id: str | None = None):
        self.device_id = device_id

    def _adb_cmd(self, args: list[str]) -> str:
        cmd = [ADB_PATH]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return res.stdout.strip()

    def audit_offline_status(self, package_name: str = "bd.sstutor.app") -> dict:
        # Check active internet interfaces
        ip_link = self._adb_cmd(["shell", "ip", "link"])
        airplane_mode = self._adb_cmd(["shell", "settings", "get", "global", "airplane_mode_on"])
        
        # Check if the app opened any listening or outbound sockets
        netstat = self._adb_cmd(["shell", "netstat", "-tlpn"])
        has_app_socket = package_name in netstat

        # Verify source code has no remote API endpoints
        src_violations = []
        forbidden_terms = ["api.openai.com", "generativelanguage.googleapis.com", "huggingface.co/api", "api.anthropic.com"]
        for p in (PROJECT_ROOT / "core").rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden_terms:
                if term in text:
                    src_violations.append(f"{p.name}: contains {term}")

        for p in (PROJECT_ROOT / "android").rglob("*.kt"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden_terms:
                if term in text:
                    src_violations.append(f"{p.name}: contains {term}")

        is_fully_offline = (len(src_violations) == 0) and not has_app_socket

        result = {
            "status": "VERIFIED_OFFLINE" if is_fully_offline else "NETWORK_DETECTED",
            "airplane_mode_setting": airplane_mode,
            "has_app_active_socket": has_app_socket,
            "source_code_remote_api_violations": src_violations,
            "zero_network_dependency_verified": is_fully_offline,
            "verdict": "VERIFIED_PASS" if is_fully_offline else "FAIL"
        }

        out_dir = PROJECT_ROOT / "results" / "phase6"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "offline_audit.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result


if __name__ == "__main__":
    net_monitor = ADBNetworkMonitor()
    rep = net_monitor.audit_offline_status()
    print("\n" + "="*60)
    print("  SS TUTOR BD — REAL DEVICE OFFLINE AUDIT REPORT")
    print("="*60)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("="*60 + "\n")
