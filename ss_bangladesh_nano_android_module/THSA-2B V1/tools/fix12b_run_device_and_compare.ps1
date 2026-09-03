# FIX-12B Device Execution & Forensic Comparison Orchestrator
$ErrorActionPreference = "Stop"

$ADB = "C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODULE_ROOT = Split-Path -Parent $SCRIPT_DIR
$APP_DIR = Join-Path $MODULE_ROOT "offline-ai_chatbot"
$OUT_DIR = Join-Path $SCRIPT_DIR "fix12b"

New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "FIX-12B ANDROID PHYSICAL DEVICE EXECUTION & LOGITS CAPTURE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check ADB connection
Write-Host "`n[1/5] Checking ADB device connection..." -ForegroundColor Yellow
$devices = & $ADB devices | Select-String -Pattern "device$"
if (-not $devices) {
    Write-Host "ERROR: No physical device detected via ADB!" -ForegroundColor Red
    Write-Host "Please ensure USB debugging is enabled and the phone (itel A662L) is connected via USB." -ForegroundColor Yellow
    exit 1
}
Write-Host "Device detected: $($devices[0].Line)" -ForegroundColor Green

# 2. Install newly built APKs
$appApk = Join-Path $APP_DIR "app\build\outputs\apk\debug\app-debug.apk"
$testApk = Join-Path $APP_DIR "app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk"

Write-Host "`n[2/5] Installing newly built APKs with instrumented native engine..." -ForegroundColor Yellow
Write-Host "Installing app APK ($([math]::Round((Get-Item $appApk).Length / 1MB, 1)) MB)..."
& $ADB install -r $appApk
Write-Host "Installing test APK ($([math]::Round((Get-Item $testApk).Length / 1KB, 1)) KB)..."
& $ADB install -r $testApk

# 3. Clean device logs and previous diagnostic dumps
Write-Host "`n[3/5] Clearing device state and logcat..." -ForegroundColor Yellow
& $ADB logcat -c
& $ADB shell run-as com.aistudio.offlineai.krvq rm -f /data/data/com.aistudio.offlineai.krvq/files/fix12_*.bin 2>$null
& $ADB shell run-as com.aistudio.offlineai.krvq rm -f /data/data/com.aistudio.offlineai.krvq/files/fix12_perf.txt 2>$null

# 4. Run test01_singleTokenForward
Write-Host "`n[4/5] Executing FIX-12B test01 on physical hardware..." -ForegroundColor Yellow
$testClass = "com.example.THSA2BFix12DiagTest#test01_singleTokenForward"
$runner = "com.aistudio.offlineai.krvq.test/androidx.test.runner.AndroidJUnitRunner"
& $ADB shell am instrument -w -r -e class $testClass $runner

# 5. Pull full 65,536 float32 logits and diagnostics from device
Write-Host "`n[5/5] Pulling binary diagnostic files from device..." -ForegroundColor Yellow

for ($i = 0; $i -lt 5; $i++) {
    $remote = "/data/data/com.aistudio.offlineai.krvq/files/fix12_logits_p$i.bin"
    $local = Join-Path $OUT_DIR "android_logits_p$i.bin"
    
    # Try direct pull first, fallback to run-as cat
    & $ADB pull $remote $local 2>$null
    if (-not (Test-Path $local) -or (Get-Item $local).Length -ne 262144) {
        Write-Host "Pulling via run-as cat for prompt $i..."
        & $ADB shell "run-as com.aistudio.offlineai.krvq cat $remote" > $local
    }
    
    if (Test-Path $local) {
        $sz = (Get-Item $local).Length
        Write-Host "  android_logits_p$i.bin: $sz bytes (expected 262144)" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: android_logits_p$i.bin could not be pulled!" -ForegroundColor Red
    }
}

# Pull diag binary and perf text
& $ADB pull /data/data/com.aistudio.offlineai.krvq/files/fix12_diag.bin "$OUT_DIR\fix12_diag.bin" 2>$null
& $ADB pull /data/data/com.aistudio.offlineai.krvq/files/fix12_perf.txt "$OUT_DIR\fix12_perf.txt" 2>$null

Write-Host "`n[6/6] Running full 65,536 logits numerical comparison..." -ForegroundColor Cyan
python "$SCRIPT_DIR\fix12b_phase_efj_full_logits_compare.py"
