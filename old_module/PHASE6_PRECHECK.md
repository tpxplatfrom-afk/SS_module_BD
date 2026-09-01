# SS Tutor BD — Phase 6 Pre-Check & Real-Device Validation Audit

**Phase:** 6 — Real 2GB Android Device Validation, Production Hardening & Release Certification  
**Date:** 2026-08-30  
**Connected Physical Device:** itel A662L (2 GB RAM, Android 12 / API 31, armeabi-v7a)  
**Strict Memory Contract:** Preferred $\le 150\text{ MB}$, Hard Ceiling $\le 200\text{ MB}$, Emergency $\le 250\text{ MB}$  
**Development Cost:** \$0 USD  

---

## 1. Physical Device Hardware Profile (Confirmed via ADB)

```text
========================================================================
PHYSICAL DEVICE PROFILE
========================================================================
Device Model:            itel A662L (itel A60 Series)
Manufacturer:            ITEL Mobile
Android Version:         Android 12 (Go Edition)
API Level:               31
Primary ABI:             armeabi-v7a (32-bit ARM Cortex-A55 / SC9863A)
Total Physical RAM:      1,957,268 kB (~1.86 GB / 2.0 GB Physical)
Available RAM:           870,412 kB (~850 MB for all user apps)
Internal Storage Total:  26 GB
Internal Storage Free:   8.4 GB available
Target RAM Budget:       <= 150 MB (Preferred), <= 200 MB (Hard Production Ceiling)
========================================================================
```

---

## 2. Architecture Inventory & Subsystem Mapping

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SS TUTOR BD PRODUCTION CORE                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Deterministic Math Core (core/math/ & bd.sstutor.math)              │
│    - fraction.py / MathEngine.kt: Exact rational arithmetic with steps │
│    - calculator.py: Simple/Compound interest, series sum, pythagoras   │
│    - equation_solver.py: Linear systems & quadratic factoring          │
│    - expression_parser.py: Bengali regex intent classification         │
│                                                                        │
│ 2. Offline RAG Subsystem (core/rag/ & bd.sstutor.rag)                  │
│    - SQLite FTS5 index (164 KB database, Class 8 Mathematics)          │
│    - Context Compressor: High-density factual nugget extraction        │
│                                                                        │
│ 3. Dedicated Bengali Educational Tokenizer (core/tokenizer/)           │
│    - Custom 16,000 Byte-level BPE vocabulary                          │
│    - 3.65 - 3.86 tokens / Bengali word (vs 8.46 for SmolLM2)           │
│                                                                        │
│ 4. 70M Transformer Micro-Model (models/sstutor_bengali_70m_edu/)       │
│    - 54.3M - 68.2M parameters, INT4 quantized binary size: 34.12 MB    │
│                                                                        │
│ 5. Multi-Guard Validation Layer (core/validation/ & bd.sstutor.val)    │
│    - GroundingValidator: Anti-hallucination refusal & context bounds   │
│    - MathAnswerValidator: Numeric verification & auto-correction       │
│    - HintValidator: Socratic answer-withholding (zero leaks)           │
│    - LanguageValidator: Repetition loop suppression                    │
│    - FormatValidator: Delimiter & control token stripping              │
│                                                                        │
│ 6. Real-Device Memory Management (bd.sstutor.runtime)                  │
│    - AndroidMemoryMonitor: Real-time PSS / Native / Dalvik tracking    │
│    - Hard states: NORMAL (<150MB), WARNING (150-200), CRITICAL (>200)  │
│    - Lazy Auto-Loading: Model loaded ONLY when verbalization needed    │
│    - Automatic unload on trimMemory(CRITICAL) or background idle       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 5 Claim Audit vs. Real Hardware Verification Status

| Subsystem Claim | Phase 5 Status | Phase 6 Real-Device Verification Plan |
| :--- | :--- | :--- |
| **Deterministic Math (100%)** | Verified in Python/Kotlin unit tests | Run 30 real-device math cases via ADB. |
| **RAG Retrieval ($< 20\text{ ms}$)** | Verified in Python FTS5 | Profile native SQLite FTS5 latency on physical itel A662L. |
| **Bengali Tokenizer ($< 4.0\text{ tok/w}$)** | Verified in tokenizer benchmarks | Verify asset integrity and decode accuracy in Android APK. |
| **Single-Model Storage ($< 50\text{ MB}$)** | Verified (34.12 MB INT4) | Verify APK asset size and storage footprint on device. |
| **Process PSS $\le 150–200\text{ MB}$** | Emulated / Estimated | **Measure real process PSS via `dumpsys meminfo` on itel A662L.** |
| **100-Turn Memory Stability** | Emulated (0.00 MB / turn) | **Execute 100 queries on device, measuring PSS delta.** |
| **Socratic Hint Protection** | Verified in unit test | Verify on-device hint generation with zero final answers. |
| **100% Offline Operation** | Verified architecturally | Test with Wi-Fi / Mobile Data / Airplane mode enabled. |

---

## 4. Remaining Uncertainties & Mitigation Strategy

1. **Native Heap Overhead on 32-bit `armeabi-v7a`:** The physical device runs a 32-bit ARM userland on Android 12 Go. We must verify memory mapping behavior and Dalvik heap stability under 32-bit pointers.
2. **Low Memory Killer (LMK) Aggressiveness:** Android 12 Go uses aggressive zRAM and memory pressure triggers. `AndroidMemoryMonitor` must proactively unload non-essential buffers when PSS approaches $150\text{ MB}$.
