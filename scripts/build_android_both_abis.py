import os
import sys
import subprocess

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

# 1. Build ARM64 (arm64-v8a)
out_dir_arm64 = r"ss_bangladesh_nano_android_module\THSA-2B V1\android\build\arm64-v8a"
os.makedirs(out_dir_arm64, exist_ok=True)
out_so_arm64 = os.path.join(out_dir_arm64, "libnano_engine.so")

cmd_arm64 = [
    clang_exe,
    "--target=aarch64-linux-android24",
    "-shared",
    "-fPIC",
    "-std=c++17",
    "-O3",
    "-march=armv8-a+simd",
    "-Wall",
    "-Wextra",
    "-o", out_so_arm64
]
for inc in include_dirs:
    cmd_arm64.extend(["-I", inc])
cmd_arm64.extend(src_files)

print("Compiling ARM64 (arm64-v8a)...")
res_arm64 = subprocess.run(cmd_arm64, capture_output=True, text=True)
if res_arm64.returncode == 0:
    print(f"  [OK] Generated ARM64: {out_so_arm64} ({os.path.getsize(out_so_arm64)} bytes)")
else:
    print("  [ERROR] ARM64 build failed:\n", res_arm64.stderr)

# 2. Build ARM32 (armeabi-v7a for itel A662L)
out_dir_arm32 = r"ss_bangladesh_nano_android_module\THSA-2B V1\android\build\armeabi-v7a"
os.makedirs(out_dir_arm32, exist_ok=True)
out_so_arm32 = os.path.join(out_dir_arm32, "libnano_engine.so")

cmd_arm32 = [
    clang_exe,
    "--target=armv7a-linux-androideabi24",
    "-shared",
    "-fPIC",
    "-std=c++17",
    "-O3",
    "-march=armv7-a",
    "-mfpu=neon",
    "-mfloat-abi=softfp",
    "-Wall",
    "-Wextra",
    "-o", out_so_arm32
]
for inc in include_dirs:
    cmd_arm32.extend(["-I", inc])
cmd_arm32.extend(src_files)

print("\nCompiling ARM32 (armeabi-v7a for itel A662L)...")
res_arm32 = subprocess.run(cmd_arm32, capture_output=True, text=True)
if res_arm32.returncode == 0:
    print(f"  [OK] Generated ARM32: {out_so_arm32} ({os.path.getsize(out_so_arm32)} bytes)")
else:
    print("  [ERROR] ARM32 build failed:\n", res_arm32.stderr)
