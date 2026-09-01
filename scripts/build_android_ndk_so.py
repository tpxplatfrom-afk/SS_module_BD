import os
import sys
import subprocess
from pathlib import Path

ndk_dir = r"C:\Users\User\AppData\Local\Android\Sdk\ndk\28.2.13676358"
clang_exe = os.path.join(ndk_dir, r"toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe")

include_dirs = [
    r"ss_bangladesh_nano_android_module\THSA-2B V1\include",
    r"ss_bangladesh_nano_android_module\THSA-2B V1\include\kernels"
]

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
    r"ss_bangladesh_nano_android_module\THSA-2B V1\jni\nano_engine_jni.cpp"
]

out_so = r"ss_bangladesh_nano_android_module\THSA-2B V1\android\build\libnano_engine.so"
os.makedirs(os.path.dirname(out_so), exist_ok=True)

cmd = [
    clang_exe,
    "--target=aarch64-linux-android24",
    "-shared",
    "-fPIC",
    "-std=c++17",
    "-O3",
    "-march=armv8-a+simd",
    "-Wall",
    "-Wextra",
    "-o", out_so
]
for inc in include_dirs:
    cmd.extend(["-I", inc])
cmd.extend(src_files)

print("Compiling libnano_engine.so for aarch64-linux-android with NDK Clang...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
print("Return code:", res.returncode)

if res.returncode == 0:
    sz = os.path.getsize(out_so)
    print(f"SUCCESS: Generated libnano_engine.so ({sz} bytes)")
