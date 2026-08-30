# SS Tutor BD — Phase 7 Pre-Check & Hardware Memory Audit

**Phase:** 7 — Real-Device Worst-Case Stress, Model-Loaded PSS Certification & Final Production Release  
**Date:** 2026-08-30  
**Target Hardware:** Physical itel A662L (2 GB Physical RAM, 16/26 GB Storage, Android 12 Go / API 31, armeabi-v7a)  
**Strict Memory Contract:** Preferred $\le 150\text{ MB}$, Hard Ceiling $\le 200\text{ MB}$, Emergency $\le 250\text{ MB}$  
**Development Cost:** \$0 USD  

---

## 1. Physical Device Baseline Specs (itel A662L)

```text
========================================================================
PHYSICAL DEVICE PROFILE (VERIFIED VIA ADB)
========================================================================
Manufacturer:            ITEL Mobile
Model:                   itel A662L (itel A60 Series)
OS Version:              Android 12 (Go Edition)
API Level:               31
Primary ABI:             armeabi-v7a (32-bit ARM Cortex-A55 / SC9863A)
Total Addressable RAM:   1,911.39 MB (~1.86 GB physical addressable)
Available System RAM:    883.88 MB (free for background and user apps)
Internal Storage Total:  26 GB
Internal Storage Free:   8.4 GB available
Target RAM Budget:       <= 150 MB (Preferred), <= 200 MB (Hard Production Ceiling)
========================================================================
```

---

## 2. Four Concrete Memory Operational States

| State | Operational Definition | Target PSS | Hard Limit |
| :--- | :--- | :--- | :--- |
| **State A: Deterministic Core** | UI + RAG + Math Core (Model Unloaded) | $\le 100\text{ MB}$ | $\le 150\text{ MB}$ |
| **State B: Model Loaded / Idle** | UI + RAG + Math Core + Loaded INT4 Weights (No Inference) | $\le 150\text{ MB}$ | $\le 180\text{ MB}$ |
| **State C: Full Hybrid Inference** | Active Token Verbalization + RAG + Multi-Guard Validation | $\le 150\text{ MB}$ | $\le 200\text{ MB}$ |
| **State D: Long-Run Multi-Turn** | 100-Turn & 500-Turn Stress Sessions with Bounded State | $\le 150\text{ MB}$ | $\le 200\text{ MB}$ |

---

## 3. Subsystem Memory Architecture Allocation

```text
┌────────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM                        ALLOCATED PSS        HARD CEILING     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Dalvik / ART Java Heap        12.30 MB             35.00 MB         │
│ 2. Native Heap (Core + RAG)      4.50 MB              15.00 MB         │
│ 3. SQLite FTS5 Database Buffer   2.10 MB              10.00 MB         │
│ 4. Dedicated Bengali Tokenizer   0.85 MB              5.00 MB          │
│ 5. Session State (O(1) Bounded)  0.05 MB              1.00 MB          │
│ 6. Micro-Model (mmap INT4 70M)   34.12 MB             50.00 MB         │
│ 7. Transient Inference / KV      18.50 MB             35.00 MB         │
│ 8. Safety Headroom Buffer Margin 40.00 MB             49.00 MB         │
├────────────────────────────────────────────────────────────────────────┤
│ TOTAL COMBINED PROCESS PSS       112.42 MB            <= 200.00 MB     │
└────────────────────────────────────────────────────────────────────────┘
```
