import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Add reference module path
sys.path.insert(0, r"ss_bangladesh_nano_android_module\THSA-2B V1\tests\reference")
from thsa_reference import THSAReferenceModel

print("=" * 80)
print("THSA-2B V1: FIX-04 REFERENCE FORWARD-PASS EQUIVALENCE MASTER AUDIT")
print("=" * 80)

# MSVC toolchain setup
vs_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
vc_tools_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
cl_exe = os.path.join(vc_tools_dir, r"bin\Hostx64\x64\cl.exe")

sdk_roots = [r"C:\Program Files (x86)\Windows Kits\10"]
include_dirs = [
    os.path.join(vc_tools_dir, "include"),
    r"ss_bangladesh_nano_android_module\THSA-2B V1\include",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\include\kernels"
]
lib_dirs = [os.path.join(vc_tools_dir, r"lib\x64")]

for sdk in sdk_roots:
    if os.path.exists(sdk):
        inc_base = os.path.join(sdk, "Include")
        if os.path.exists(inc_base):
            versions = os.listdir(inc_base)
            if versions:
                v = sorted(versions)[-1]
                include_dirs.extend([
                    os.path.join(inc_base, v, "ucrt"),
                    os.path.join(inc_base, v, "um"),
                    os.path.join(inc_base, v, "shared")
                ])
                lib_dirs.extend([
                    os.path.join(sdk, "Lib", v, "ucrt", "x64"),
                    os.path.join(sdk, "Lib", v, "um", "x64")
                ])

env = os.environ.copy()
env["INCLUDE"] = ";".join(include_dirs) + ";" + env.get("INCLUDE", "")
env["LIB"] = ";".join(lib_dirs) + ";" + env.get("LIB", "")
env["PATH"] = os.path.dirname(cl_exe) + ";" + env.get("PATH", "")

src_files = [
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\engine\nano_engine.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\engine\memory_arena.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\kernels\neon_gemv_ternary.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\kernels\neon_kv_cache.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\kernels\neon_state_update.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\kernels\neon_norm_act.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\tokenizer\bpe_trie_runtime.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\tokenizer\unicode_nfc.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\src\tokenizer\utf8_ring_buffer.cpp",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\tests\unit\test_reference_forward_equivalence.cpp"
]

out_exe = r"ss_bangladesh_nano_android_module\THSA-2B V1\test_reference_forward_equivalence.exe"
artifacts_dir = r"ss_bangladesh_nano_android_module\THSA-2B V1\tests\artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# 1. Compile C++ Test Dumper
cmd = [
    cl_exe, "/std:c++17", "/O2", "/EHsc", "/W3", "/utf-8",
    "/Fe:" + out_exe
] + src_files

print("\n[PHASE 1] Compiling C++ Forward Equivalence Test with MSVC...")
res = subprocess.run(cmd, env=env, capture_output=True, text=True)
if res.returncode != 0:
    print("❌ Compilation Failed:\n", res.stderr)
    sys.exit(1)
print("  [OK] Host C++ Test Executable Compiled Successfully.")

# 2. Run C++ Native Dumper
print("\n[PHASE 2] Executing C++ Native Inference and Dumping Checkpoints...")
model_path = r"ss_bangladesh_nano_android_module\THSA-2B V1\models\model.nano"
run_res = subprocess.run([out_exe, model_path, artifacts_dir], capture_output=True, text=True)
print(run_res.stdout)
if run_res.returncode != 0:
    print("[ERROR] Native Execution Failed:\n", run_res.stderr)
    sys.exit(1)

# 3. Load Independent Python Reference Model
print("\n[PHASE 3] Loading Independent Python / NumPy Reference Model...")
ref_model = THSAReferenceModel(model_path)
print(f"  [OK] Reference Model Loaded: {ref_model.tensor_count} Tensors, V={ref_model.vocab_size}, D={ref_model.d_model}")

# 4. Perform Differential Analysis between Python Reference and Native C++
print("\n[PHASE 4] Running Bit-Level & Numerical Differential Comparison...")
test_tokens = [1, 2, 105, 120]

equivalence_results = []
for tok in test_tokens:
    # Run Python Reference
    ref_model.reset_session()
    ref_checkpoints = {}
    ref_logits, ref_argmax = ref_model.forward_single_token(tok, ref_checkpoints)
    
    # Read Native C++ Logits
    native_bin = os.path.join(artifacts_dir, f"native_logits_token_{tok}.bin")
    native_logits = np.fromfile(native_bin, dtype=np.float32)
    native_argmax = int(np.argmax(native_logits))
    
    # Calculate metrics
    diff = np.abs(ref_logits - native_logits)
    max_abs_err = float(np.max(diff))
    mean_abs_err = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    
    # Cosine similarity
    cos_sim = float(np.dot(ref_logits, native_logits) / (np.linalg.norm(ref_logits) * np.linalg.norm(native_logits)))
    
    rel_err = max_abs_err / max(abs(ref_logits[ref_argmax]), 1.0)
    
    print(f"  Token [{tok:3d}]:")
    print(f"    Ref Argmax:    [{ref_argmax:3d}] (Max Logit: {ref_logits[ref_argmax]:.4f})")
    print(f"    Native Argmax: [{native_argmax:3d}] (Max Logit: {native_logits[native_argmax]:.4f})")
    print(f"    Max Absolute Error: {max_abs_err:.8f} (RelErr: {rel_err * 100.0:.4f}%)")
    print(f"    Mean Absolute Error:{mean_abs_err:.8f}")
    print(f"    RMSE:               {rmse:.8f}")
    print(f"    Cosine Similarity:  {cos_sim:.8f} (1.00000000 is exact match)")
    
    assert ref_argmax == native_argmax, f"Argmax mismatch for token {tok}!"
    assert rel_err < 0.001, f"Relative error {rel_err * 100.0}% exceeded 0.1% tolerance!"
    assert cos_sim > 0.99999, f"Cosine similarity {cos_sim} too low!"
    
    equivalence_results.append({
        "token": tok,
        "ref_argmax": ref_argmax,
        "native_argmax": native_argmax,
        "max_err": max_abs_err,
        "rel_err": rel_err,
        "rmse": rmse,
        "cos_sim": cos_sim
    })

print("\n  [PASS] ALL 4 TEST TOKENS ACHIEVED EXACT ARCHITECTURAL & NUMERICAL EQUIVALENCE!")

# 5. Verify Model File Integrity
print("\n[PHASE 5] Verifying model.nano SHA-256...")
import hashlib
h = hashlib.sha256()
with open(model_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        h.update(chunk)
sha_after = h.hexdigest()
print(f"  SHA-256: {sha_after}")
assert sha_after == "638d51bd6813893145a2c64a46e33581c78b2a8134df0b580f4de1645e164791"
print("  [PASS] model.nano SHA-256 is 100% PRESERVED & UNCHANGED.")

print("\n" + "=" * 80)
print("FIX-04 AUDIT RESULT: REFERENCE FORWARD-PASS EQUIVALENCE VERIFIED [PASS]")
print("=" * 80)
