# Android Memory Validation Specification (Phase 3C)

**Document Version:** 2.0.0  
**Phase:** 3C — Ultra-Low-Memory Micro-Runtime Validation  
**Target Device Profile:** Low-End Android (2 GB Physical RAM, 16 GB Storage, Android 9–14, ARMv8 64-bit)  
**Production Contract:** Preferred $\le 150–180\text{ MB}$, Absolute Ceiling $\le 200\text{ MB}$, Engineering Safety Margin $\le 220\text{ MB}$  

---

## 1. Android Memory Architecture & PSS vs RSS

On Android, **PSS (Proportional Set Size)** is the authoritative metric used by the Android Low Memory Killer (LMK), while desktop Linux/Windows metrics measure **RSS (Resident Set Size)**.

```
Total Process PSS = Private Dirty + Private Clean + (Shared Dirty / Sharing Processes) + (Shared Clean / Sharing Processes)
```

```
========================================================================
ANDROID PROCESS MEMORY COMPONENT      PHASE 3C BUDGET (MB)
========================================================================
1. Dalvik / ART Heap (UI / App)        20 – 30 MB
2. Native Heap (mmap model / C++)      50 – 80 MB
3. SQLite FTS5 (Knowledge Pack)         5 – 10 MB
4. Graphics / SurfaceFlinger Buffer    15 – 25 MB
5. Stack & Process Overhead             5 – 10 MB
6. Safety Headroom Margin              25 – 35 MB
------------------------------------------------------------------------
TOTAL APPLICATION TARGET PSS          150 – 180 MB
ABSOLUTE PRODUCTION CEILING                200 MB
========================================================================
```

---

## 2. ADB Live Inspection Commands

Execute against connected test device or emulator:

### A. Total Process Memory Dump
```bash
adb shell dumpsys meminfo bd.sstutor.app
```

### B. Track Real-Time PSS & Native Heap During 100-Turn Session
```bash
adb shell "while true; do dumpsys meminfo bd.sstutor.app | grep -E 'TOTAL PSS:|Native Heap|Dalvik Heap'; sleep 3; done"
```

### C. Low-Memory Killer (LMK) Simulation Test
Simulate severe device memory pressure while SS Tutor BD is active in foreground:
```bash
adb shell am send-trim-memory bd.sstutor.app RUNNING_CRITICAL
```
*Expected Behavior:* Application flushes non-essential RAG cache without crash, process PSS drops below 150 MB.

---

## 3. Production Release Gate Matrix

| Test Scenario | Inspection Command | Pass Threshold | Fail Action |
| :--- | :--- | :--- | :--- |
| **Cold Startup** | `dumpsys meminfo` (at launch) | Total PSS $\le 120\text{ MB}$ | Reject binary |
| **First Inference Peak** | `dumpsys meminfo` (turn 1) | Total PSS $\le 180\text{ MB}$ | Flag warning |
| **100-Turn Multi-Turn** | `dumpsys meminfo` (turn 100) | Total PSS $\le 200\text{ MB}$, Growth $\le 5\text{ MB}$ | Disqualify runtime |
| **Background / Trim Test**| `am send-trim-memory` | Survives without crash, PSS $\le 100\text{ MB}$ | Fix lifecycle |
| **Thermal / CPU Throttle**| `dumpsys thermalservice` | No thermal throttling on 4-core Cortex-A53 | Throttle thread count |
