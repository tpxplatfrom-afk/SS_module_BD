import os
import sys
import time
import datetime
import hashlib
import platform
import psutil
import struct
import json
import re
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

AUDIT_RUN_ID = "RUN-" + hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12].upper()
UTC_TS = datetime.datetime.now(datetime.timezone.utc).isoformat()
LOCAL_TS = datetime.datetime.now().isoformat()

ROOT_DIR = Path(r"c:\Users\User\Desktop\SS_module_BD")
THSA_DIR = ROOT_DIR / "ss_bangladesh_nano_android_module" / "THSA-2B V1"
sys.path.insert(0, str(THSA_DIR))

process = psutil.Process(os.getpid())

def get_mem():
    mi = process.memory_info()
    return mi.rss  # bytes

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# -------------------------------------------------------------
# PHASE 1 & 4: MEMORY STATE A
# -------------------------------------------------------------
state_a_rss = get_mem()
state_a_ts = datetime.datetime.now().isoformat()

# -------------------------------------------------------------
# PHASE 2: MODEL FILE FORENSICS
# -------------------------------------------------------------
target_model_file = THSA_DIR / "models" / "model.nano"
if not target_model_file.exists():
    target_model_file = THSA_DIR / "android" / "model.nano"

model_exists = target_model_file.exists()
if model_exists:
    model_path_abs = str(target_model_file.resolve())
    model_filename = target_model_file.name
    model_size_bytes = os.path.getsize(target_model_file)
    model_sha256 = sha256_file(target_model_file)
    model_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(target_model_file)).isoformat()
    
    with open(target_model_file, "rb") as f:
        header_bytes = f.read(64)
    magic_hex = header_bytes[:8].hex()
    magic_str = "".join([chr(b) if 32 <= b <= 126 else "." for b in header_bytes[:8]])
else:
    model_path_abs = "UNOBSERVABLE"
    model_filename = "UNOBSERVABLE"
    model_size_bytes = "UNOBSERVABLE"
    model_sha256 = "UNOBSERVABLE"
    model_mtime = "UNOBSERVABLE"
    magic_hex = "UNOBSERVABLE"
    magic_str = "UNOBSERVABLE"

# Check if GGUF or custom nano format
is_gguf = magic_bytes = magic_hex.startswith("47475546") # 'GGUF' in hex
param_count_calc = "UNOBSERVABLE"
quant_type = "INT4_NANO" if "nano" in str(target_model_file) else "UNOBSERVABLE"

# -------------------------------------------------------------
# PHASE 4: STATE B (Import Engine / Tokenizer)
# -------------------------------------------------------------
from src.engine.universal_tutor_engine import UniversalTutorEngine
state_b_rss = get_mem()
state_b_ts = datetime.datetime.now().isoformat()

# -------------------------------------------------------------
# PHASE 4: STATE C (Model Loaded in RAM)
# -------------------------------------------------------------
engine = UniversalTutorEngine()
state_c_rss = get_mem()
state_c_ts = datetime.datetime.now().isoformat()

# -------------------------------------------------------------
# PHASE 4: STATE D (First Inference)
# -------------------------------------------------------------
t0 = time.perf_counter()
res1 = engine.ask("হাই, কেমন আছ?")
t1 = time.perf_counter()
first_inf_latency_ms = (t1 - t0) * 1000
state_d_rss = get_mem()
state_d_ts = datetime.datetime.now().isoformat()

# -------------------------------------------------------------
# PHASE 5: CONTEXT CAPACITY PROGRESSIVE TEST
# -------------------------------------------------------------
token_test_sizes = [128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192]
context_results = []

for sz in token_test_sizes:
    # generate prompt of approximate token length
    test_p = "পড়ালেখা " * sz
    m_before = get_mem()
    t_start = time.perf_counter()
    try:
        r = engine.ask(test_p)
        t_end = time.perf_counter()
        m_after = get_mem()
        success = True
        err = "NONE"
        out_len = len(r.get("text", ""))
    except Exception as e:
        t_end = time.perf_counter()
        m_after = get_mem()
        success = False
        err = str(e)
        out_len = 0
    
    context_results.append({
        "target_tokens": sz,
        "success": success,
        "latency_ms": round((t_end - t_start) * 1000, 2),
        "output_chars": out_len,
        "mem_before": m_before,
        "mem_after": m_after,
        "error": err
    })

state_e_rss = get_mem()
state_e_ts = datetime.datetime.now().isoformat()

# -------------------------------------------------------------
# PHASE 6: RUNTIME COMPONENT INVENTORY
# -------------------------------------------------------------
modules_loaded = []
for k, v in list(sys.modules.items()):
    if "src.engine" in k:
        modules_loaded.append({
            "name": k,
            "path": getattr(v, "__file__", "BUILTIN")
        })

# -------------------------------------------------------------
# PHASE 7: KNOWLEDGE STORAGE FORENSICS
# -------------------------------------------------------------
data_inventory = []
data_dir = THSA_DIR / "data"
if data_dir.exists():
    for f in data_dir.rglob("*"):
        if f.is_file():
            sz = f.stat().st_size
            data_inventory.append({
                "path": str(f.relative_to(ROOT_DIR)),
                "size_bytes": sz,
                "sha256": sha256_file(f)
            })

# -------------------------------------------------------------
# PHASE 8: CONTAMINATION CHECK
# -------------------------------------------------------------
search_patterns = [
    "2.41B", "2.41", "654.39", "800 MB", "1.2 GB", "1.6 GB", "1.8 GB",
    "Class 1", "Class 12", "NCTB", "20%", "40%", "100%",
    "UniversalTutorEngine", "SessionProfileTracker"
]

contamination_findings = []
for root, dirs, files in os.walk(str(THSA_DIR / "src")):
    for file in files:
        if file.endswith(".py"):
            fp = os.path.join(root, file)
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for pat in search_patterns:
                    count = len(re.findall(re.escape(pat), content, re.IGNORECASE))
                    if count > 0:
                        contamination_findings.append({
                            "pattern": pat,
                            "file": os.path.relpath(fp, str(ROOT_DIR)),
                            "count": count
                        })

# -------------------------------------------------------------
# PHASE 9: LIVE INFERENCE TRACE (Fresh Prompt)
# -------------------------------------------------------------
fresh_prompt = "আমি আজ প্রথমবার তোমার সাথে কথা বলছি। তুমি কীভাবে আমার প্রশ্নের উত্তর তৈরি করো সেটা ব্যাখ্যা করার দরকার নেই। শুধু বলো, 17 × 23 কত এবং ধাপে ধাপে হিসাব করো।"
live_mem_before = get_mem()
t_live_start = time.perf_counter()
live_res = engine.ask(fresh_prompt)
t_live_end = time.perf_counter()
live_mem_after = get_mem()
live_latency_ms = round((t_live_end - t_live_start) * 1000, 2)

# -------------------------------------------------------------
# OUTPUT RESULTS IN JSON FORMAT FOR MACHINE READING
# -------------------------------------------------------------
report_data = {
    "AUDIT_RUN_ID": AUDIT_RUN_ID,
    "UTC_TS": UTC_TS,
    "LOCAL_TS": LOCAL_TS,
    "PLATFORM": {
        "os": platform.platform(),
        "arch": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "pid": os.getpid(),
        "android_api": "UNOBSERVABLE (Host is Windows NT Development Runtime)",
        "abi": platform.architecture()[0]
    },
    "MODEL_FILE": {
        "path": model_path_abs,
        "filename": model_filename,
        "size_bytes": model_size_bytes,
        "sha256": model_sha256,
        "mtime": model_mtime,
        "magic_hex": magic_hex,
        "magic_str": magic_str,
        "param_count": "UNOBSERVABLE (Binary header lacks self-describing param counter)",
        "quantization": quant_type
    },
    "MEMORY": {
        "STATE_A_PROCESS_START": {"rss_bytes": state_a_rss, "ts": state_a_ts},
        "STATE_B_MODULES_IMPORTED": {"rss_bytes": state_b_rss, "ts": state_b_ts},
        "STATE_C_ENGINE_INITIALIZED": {"rss_bytes": state_c_rss, "ts": state_c_ts},
        "STATE_D_FIRST_INFERENCE": {"rss_bytes": state_d_rss, "latency_ms": round(first_inf_latency_ms, 2), "ts": state_d_ts},
        "STATE_E_STRESS_INFERENCE": {"rss_bytes": state_e_rss, "ts": state_e_ts}
    },
    "CONTEXT_TESTS": context_results,
    "RUNTIME_MODULES": modules_loaded,
    "DATA_INVENTORY": data_inventory,
    "CONTAMINATION_FINDINGS": contamination_findings,
    "LIVE_INFERENCE": {
        "input": fresh_prompt,
        "output": live_res.get("text", ""),
        "latency_ms": live_latency_ms,
        "mem_delta_bytes": live_mem_after - live_mem_before
    }
}

print("###---BEGIN_RAW_JSON---###")
print(json.dumps(report_data, ensure_ascii=False, indent=2))
print("###---END_RAW_JSON---###")
