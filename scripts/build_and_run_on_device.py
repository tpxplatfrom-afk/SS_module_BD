import os
import sys
import subprocess
import time

ndk_dir = r"C:\Users\User\AppData\Local\Android\Sdk\ndk\28.2.13676358"
clang_exe = os.path.join(ndk_dir, r"toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe")
adb = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"

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
    r"ss_bangladesh_nano_android_module\THSA-2B V1\tests\android\test_android_e2e.cpp"
]

out_bin = r"ss_bangladesh_nano_android_module\THSA-2B V1\android\build\armeabi-v7a\test_android_e2e"

cmd = [
    clang_exe,
    "--target=armv7a-linux-androideabi24",
    "-static-libstdc++",
    "-std=c++17",
    "-O3",
    "-march=armv7-a",
    "-mfpu=neon",
    "-mfloat-abi=softfp",
    "-Wall",
    "-Wextra",
    "-o", out_bin
]
for inc in include_dirs:
    cmd.extend(["-I", inc])
cmd.extend(src_files)

print("Compiling standalone test_android_e2e with -static-libstdc++...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print("Compilation Error:\n", res.stderr)
    sys.exit(1)
print(f"  [OK] Generated Standalone Executable: {out_bin} ({os.path.getsize(out_bin)} bytes)")

def run_adb(cmd_list):
    for attempt in range(5):
        res = subprocess.run([adb] + cmd_list, capture_output=True, text=True)
        if res.returncode == 0 and "device not found" not in res.stderr and "offline" not in res.stderr:
            return res
        time.sleep(1)
    return res

print("\nPushing standalone binary to /data/local/tmp/test_android_e2e...")
r1 = run_adb(["push", out_bin, "/data/local/tmp/test_android_e2e"])
print("Push output:", r1.stdout.strip())

run_adb(["shell", "chmod 755 /data/local/tmp/test_android_e2e"])

print("\nExecuting /data/local/tmp/test_android_e2e on physical device itel A662L...")
r2 = run_adb(["shell", "/data/local/tmp/test_android_e2e /data/local/tmp/model.nano"])
print("STDOUT:\n", r2.stdout)
if r2.stderr:
    print("STDERR:\n", r2.stderr)
print("Return code:", r2.returncode)
