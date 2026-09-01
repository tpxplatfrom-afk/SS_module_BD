import subprocess
import time

adb = r"C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"
subprocess.run([adb, "start-server"], check=True)
time.sleep(1)

def query(cmd):
    p = subprocess.run([adb, "shell", cmd], capture_output=True, text=True)
    return p.stdout.strip()

print("--- PHYSICAL ANDROID DEVICE METRICS ---")
print("Model:         ", query("getprop ro.product.model"))
print("Brand:         ", query("getprop ro.product.brand"))
print("Manufacturer:  ", query("getprop ro.product.manufacturer"))
print("Primary ABI:   ", query("getprop ro.product.cpu.abi"))
print("Supported ABIs:", query("getprop ro.product.cpu.abilist"))
print("Android OS:    ", query("getprop ro.build.version.release"))
print("SDK API Level: ", query("getprop ro.build.version.sdk"))
print("Kernel Arch:   ", query("uname -m"))
print("RAM Info:\n" + query("cat /proc/meminfo | head -n 5"))
