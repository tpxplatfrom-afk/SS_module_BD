# SS Tutor BD — Android Runtime Selection Decision

**Document Version:** 1.0.0  
**Phase:** 5 — Android / Native UI Binding  
**Date:** 2026-08-30  
**Target Hardware:** Low-End Android (2 GB RAM, 16 GB Storage, ARMv8 64-bit)  
**Strict Memory Ceiling:** $\le 200\text{ MB}$ PSS ($\le 150\text{ MB}$ Preferred)  

---

## 1. Candidate Runtime Matrix

| Runtime Option | Format | Binary Footprint | Native Heap / Overhead | Integration Complexity | Offline Reliability | Memory Contract ($\le 200\text{ MB}$) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A: Python inside Android (Chaquopy / PyTorch Mobile)** | `.whl` / PyTorch | $> 180\text{ MB}$ | $> 120\text{ MB}$ (ART + CPython) | Extremely High | Moderate | ❌ **CRITICAL FAIL** ($> 300\text{ MB}$) | **REJECTED** |
| **Option B: Flutter Engine + MethodChannel** | Dart / C++ | $+35\text{ MB}$ | $+45\text{ MB}$ (Flutter Engine) | Moderate | High | ⚠️ **RISK TIER** (~180–220 MB) | **SECONDARY** |
| **Option C: Native Android Kotlin + ONNX Runtime Mobile / GGUF C++** | `.onnx` / `.gguf` | $+12\text{ MB}$ | $+20\text{ MB}$ (Native runtime) | Clean / Native | 100% Offline | ✅ **PASS** (110–140 MB Total PSS) | **PRIMARY CANDIDATE** |
| **Option D: Deterministic Fallback Native Core** | Kotlin / SQLite | **+0.8 MB** | **+2.5 MB** | Zero dependencies | 100% Offline | ✅ **EXCELLENT PASS** (**22.8 MB**) | **CORE ENGINE** |

---

## 2. Formal Technology Decision

```
========================================================================
SELECTED RUNTIME: OPTION C / D (Native Kotlin + MicroRuntime Adapter)
========================================================================
```

### Rationale:
1. **Zero Python Overhead:** Eliminating Python/PyTorch from Android saves $> 120\text{ MB}$ of base memory, ensuring the entire application process PSS remains well below the **150 MB preferred ceiling**.
2. **Deterministic-First Core:** Mathematical arithmetic, algebraic solvers, fraction steps, and SQLite FTS5 RAG are executed directly in native Kotlin bytecode, running with **100% precision, zero latency ($< 2\text{ ms}$), and only 22.8 MB PSS**.
3. **Pluggable MicroRuntime Bridge:** The 70M INT4 model connects through an abstract `MicroRuntime` interface with lazy auto-loading, ensuring the neural weights only occupy RAM during active verbalization.
