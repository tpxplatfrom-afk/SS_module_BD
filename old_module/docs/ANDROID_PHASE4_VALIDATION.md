# Android Phase 4 Memory & MicroRuntime Validation Specification

**Document Version:** 1.0.0  
**Phase:** 4 — Bengali Micro-Model Training & Android Integration  
**Target Device Baseline:** 2 GB Physical RAM / 16 GB Storage (Android 9.0–14.0, ARMv8 64-bit)  
**Production Contract:** Preferred $\le 150\text{ MB}$, Absolute Hard Ceiling $\le 200\text{ MB}$  

---

## 1. Phase 4 Target Memory Allocation Table

```
========================================================================
PROCESS MEMORY SUBSYSTEM              ALLOCATION TARGET (MB)
========================================================================
1. Dalvik / ART Java Heap (UI)         20 – 30 MB
2. Micro-Model Weights (INT4 Quantized) 34 – 40 MB
3. MicroRuntime & Bounded KV-Cache      20 – 25 MB
4. SQLite FTS5 RAG & Compressor Buffer   5 – 10 MB
5. Deterministic Math & Validators       2 – 5 MB
6. Native Process / Stack / C++ Glue    10 – 15 MB
7. Safety Buffer Margin                 25 – 35 MB
------------------------------------------------------------------------
TOTAL APPLICATION TARGET PSS          116 – 160 MB (Preferred <= 150 MB)
ABSOLUTE PRODUCTION CEILING                200 MB
========================================================================
```

---

## 2. Low-End Device Adaptive Policies (via `core/runtime/device_profile.py`)

| Profile Tier | Available System RAM | Max Context Tokens | Max Output Tokens | RAG Chunk Limit | Model Load Policy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`TIER_ULTRA_LOW`** | $< 800\text{ MB}$ | 256 | 64 | 1 chunk ($\le 30$ words) | Aggressive unload on background |
| **`TIER_LOW`** | $800–2048\text{ MB}$ | 384 | 96 | 2 chunks ($\le 40$ words) | Keep loaded in foreground |
| **`TIER_STANDARD`** | $> 2048\text{ MB}$ | 512 | 128 | 3 chunks ($\le 50$ words) | Full caching enabled |

---

## 3. ADB Memory & Lifecycle Validation Suite

### A. Real-Time Memory Profiling (PSS / Native Heap / Java Heap)
```bash
adb shell dumpsys meminfo bd.sstutor.app | grep -E 'TOTAL PSS:|Native Heap|Dalvik Heap'
```

### B. 100-Turn Monotonic Memory Leak Assertion
```bash
# Must verify <= 0.05 MB growth per turn over 100 consecutive queries
adb shell "for i in $(seq 1 100); do am broadcast -a bd.sstutor.TEST_QUERY --es q '৩/৪ + ৫/৬'; sleep 1; done"
```

### C. Low Memory Pressure & Activity Recreation Test
```bash
# Test background memory trimming
adb shell am send-trim-memory bd.sstutor.app RUNNING_CRITICAL

# Test screen rotation / configuration change without leaking model context
adb shell cmd statusbar expand-notifications
adb shell settings put system user_rotation 1
```
