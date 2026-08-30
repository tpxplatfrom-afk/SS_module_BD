# SS Tutor BD — Phase 3C Master Implementation Plan

**Project:** SS Tutor BD — Offline-First Modular AI Education Platform  
**Phase:** 3C — Ultra-Low-Memory Micro-Runtime & Bengali Micro-Model Architecture  
**Target Hardware:** Low-End Android (2 GB Physical RAM, 16 GB Storage, Cortex-A53/A55 CPU)  
**Strict Memory Contract:** Preferred $\le 150–200\text{ MB}$, Absolute Ceiling $\le 200\text{ MB}$, Safety Ceiling $\le 220\text{ MB}$ ($>220\text{ MB}$ = FAIL)  
**Date:** 2026-08-30  
**Budget:** \$0 USD  

---

## 1. Executive Context & Phase 3B Failure Analysis

In Phase 3B, we discovered that standard `llama.cpp` dynamic allocations with general-purpose instruction models exceed real-world Android memory limits:
* `CAND-01` (Qwen2.5-0.5B Q4_K_M): Peak RSS reached **738.07 MB** (3.7× the 200 MB ceiling).
* `CAND-03` (SmolLM2-135M Q4_K_M): Model binary is only **100.57 MB**, and cold startup is **223 MB**, but sustained multi-turn sessions expand to **315.62 MB** due to dynamic context buffer allocations and a 49K vocabulary lacking dedicated Bengali subwords (causing 4–8× byte expansion).

**Phase 3C Core Directive:**  
We do **NOT** relax the memory contract. Instead, we pivot to:
> **Deterministic Computation + Compact Retrieval + Intermediate Structured Representation + Bounded Micro-Runtime Language Generation.**

---

## 2. Strict Production Memory Budget Matrix

```
============================================================
SS TUTOR BD — PHASE 3C PRODUCTION MEMORY ALLOCATION
============================================================
Subsystem Component               Allocated Memory Budget
------------------------------------------------------------
Micro-Model Weights (Static mmap) 50 – 80 MB (Max 100 MB)
Micro-KV Cache (Bounded <=256)    8 – 15 MB
Micro-Runtime Engine Overhead     15 – 25 MB
RAG Engine / SQLite FTS5 Native   5 – 10 MB
Context Compressor & Buffers      3 – 6 MB
Deterministic Math Subsystem      3 – 5 MB
Sanitization & Validation State   2 – 4 MB
Application UI & Dart/Native Host 20 – 30 MB
Headroom / Safety Margin          25 – 35 MB
------------------------------------------------------------
PREFERRED PEAK APPLICATION RSS   150 – 180 MB
ABSOLUTE PRODUCTION CEILING           200 MB
ENGINEERING SAFETY CEILING            220 MB
DISQUALIFYING THRESHOLD              >220 MB (or Sustained >200 MB)
============================================================
```

---

## 3. Subsystem Architecture

```
                    Student Bengali Query
                              │
                    Device Profile Check
             (TIER_ULTRA_LOW / TIER_LOW / TIER_STANDARD)
                              │
                    Query Normalization
                              │
                    Intent Classification
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
   Math Task Detected                     General Concept / Inquiry
         │                                         │
Deterministic Math Engine                 SQLite FTS5 Retrieval
(Exact Calculations & Steps)                       │
         │                                Context Compressor
         │                         (Duplicate & Irrelevant Filter)
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                 Structured Intermediate Task
                          (TutorTask)
                              │
                     Micro-Prompt Protocol
                   ([T], [F], [R], [G], [H], [C])
                              │
                    Micro-Runtime Adapter
             ┌────────────────┼────────────────┐
             │                │                │
        Deterministic     Micro-LLM       ONNX / Compact
          Template       (Bounded 256)      Runtime
             │                │                │
             └────────────────┼────────────────┘
                              │
                   Output Sanitizer Layer
               (Loops, Echoes, Byte Artifacts)
                              │
                 Deterministic Math Validator
               (Check Final Number vs Calculated)
                              │
                    Final Student Response
```

---

## 4. Work Breakdown & Deliverables

| Component | Target File | Responsibility |
| :--- | :--- | :--- |
| **1. Memory Budget Manager** | `core/runtime/memory_budget.py` | Explicit budget allocations, threshold checks, dynamic budget enforcement |
| **2. Device Profiler** | `core/runtime/device_profile.py` | Detects host/Android RAM, CPU, ABI, maps to TIER_ULTRA_LOW / LOW / STANDARD |
| **3. Micro-Runtime Adapter** | `core/runtime/micro_runtime.py` | Abstract inference runtime supporting bounded context, streaming, and deterministic fallbacks |
| **4. RAG Context Compressor** | `core/rag/context_compressor.py` | Extracts concise factual nuggets (<40 words), preserves math formulas, eliminates duplicates |
| **5. Micro-Prompt Protocol** | `core/prompts/micro_protocol.py` | Compact structured tags (`[T]`, `[F]`, `[R]`, `[G]`, `[H]`, `[C]`), target <70 tokens |
| **6. Bengali Tokenizer Benchmark**| `benchmarks/phase3c/bengali_token_efficiency.py` | Evaluates characters/token, words/token, sentence/token, and mixed math tokenization |
| **7. Multi-Turn Session Memory**| `core/runtime/session_manager.py` | Bounded $O(1)$ constant memory session state (no raw history accumulation) |
| **8. Memory Benchmark Engine** | `benchmarks/memory/phase3c_memory_benchmark.py` | Cold start, 10-query warm, 25/50/100-turn multi-turn, 10 load/unload cycles |
| **9. 100-Question Quality Suite**| `benchmarks/phase3c/tutor_100_benchmark.json` | 100 curriculum questions across all 10 NCTB categories |
| **10. Comprehensive Unit Tests** | `tests/test_*.py` | 8 dedicated unit test modules covering all new Phase 3C subsystems |
| **11. Reports & Decision** | `PHASE3C_REPORT.md` & `results/model_decision/model_decision_phase3c.json` | 20-section comprehensive empirical report and formal verdict |

---

## 5. Sequential Acceptance Gates

1. **Gate 1 (License):** Permissive open-source license (`Apache-2.0`, `MIT`, `BSD`).
2. **Gate 2 (Binary Size):** Model file $\le 100\text{ MB}$ preferred, $\le 150\text{ MB}$ max.
3. **Gate 3 (Hard Memory Contract):** Cold peak $\le 200\text{ MB}$, Multi-turn sustained $\le 200\text{ MB}$, Absolute ceiling $\le 220\text{ MB}$.
4. **Gate 4 (Multi-Turn Plateau):** Memory growth over 100 turns $\le 0.05\text{ MB / query}$ (No leak).
5. **Gate 5 (Speed & TTFT):** Throughput $\ge 4.0\text{ tok/s}$, TTFT $\le 2.0\text{ s}$.
6. **Gate 6 (Bengali & Math Quality):** Grounded Bengali $\ge 80\%$, Hybrid Mathematical Correctness $\ge 90\%$.
7. **Gate 7 (Grounding & Anti-Leak):** Textbook factual alignment $\ge 95\%$, Socratic Hint withholding $\ge 95\%$.
8. **Gate 8 (Zero Cost & Disk Safety):** \$0 budget, single active model on disk at all times.
