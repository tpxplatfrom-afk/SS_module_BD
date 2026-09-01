import os
import sys
import subprocess
from pathlib import Path

# MSVC directories found
vs_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
vc_tools_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
cl_exe = os.path.join(vc_tools_dir, r"bin\Hostx64\x64\cl.exe")

sdk_roots = [
    r"C:\Program Files (x86)\Windows Kits\10"
]

include_dirs = [
    os.path.join(vc_tools_dir, "include"),
    r"ss_bangladesh_nano_android_module\THSA-2B V1\include",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\include\kernels"
]

lib_dirs = [
    os.path.join(vc_tools_dir, r"lib\x64")
]

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
    r"ss_bangladesh_nano_android_module\THSA-2B V1\tests\unit\test_neural_forward_pass.cpp"
]

out_exe = r"ss_bangladesh_nano_android_module\THSA-2B V1\test_neural_forward_pass.exe"

cmd = [
    cl_exe,
    "/std:c++17",
    "/O2",
    "/EHsc",
    "/W3",
    "/utf-8",
    "/Fe:" + out_exe
] + src_files

print("Compiling test_neural_forward_pass with MSVC...")
res = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
print("Return code:", res.returncode)

if res.returncode == 0:
    print("\nExecuting test_neural_forward_pass.exe...")
    run_res = subprocess.run([out_exe, r"ss_bangladesh_nano_android_module\THSA-2B V1\models\model.nano"], capture_output=True, text=True)
    print("OUTPUT:\n", run_res.stdout)
    print("ERRORS:\n", run_res.stderr)
    print("Exit code:", run_res.returncode)
