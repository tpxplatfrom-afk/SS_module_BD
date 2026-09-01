import subprocess
import time
import os

adb = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"

def run_cmd(args):
    return subprocess.run([adb, "-s", "100713836F004822"] + args, capture_output=True, text=True)

print("1. Waiting for device...")
run_cmd(["wait-for-device"])
time.sleep(1)

out_bin = r"ss_bangladesh_nano_android_module\THSA-2B V1\android\build\armeabi-v7a\test_android_e2e"

print("2. Pushing standalone executable...")
p = run_cmd(["push", out_bin, "/data/local/tmp/test_android_e2e"])
print("Push:", p.stdout.strip(), p.stderr.strip())

run_cmd(["shell", "chmod", "755", "/data/local/tmp/test_android_e2e"])

print("3. Executing on physical device itel A662L...")
res = run_cmd(["shell", "/data/local/tmp/test_android_e2e", "/data/local/tmp/model.nano"])

print("\n" + "=" * 80)
print("ON-DEVICE EXECUTION STDOUT:")
print("=" * 80)
print(res.stdout)

if res.stderr:
    print("STDERR:\n", res.stderr)
print("Exit Code:", res.returncode)
