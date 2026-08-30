# SS Tutor BD — Phase 3B Master Implementation Plan

**Project:** SS Tutor BD  
**Phase:** 3B — Production Memory-Constrained Tutor Engine & Model Re-Evaluation  
**Target Device:** Android, 2 GB RAM / 16 GB Storage  
**Production Memory Target:** 150–200 MB (Hard Ceiling: 250 MB)  
**Date:** 2026-08-30  
**Budget:** \$0 USD  

---

## 1. Context & Executive Rationale

Phase 3A demonstrated that textbook RAG grounding and Socratic prompt scaffolding dramatically improved tutoring adherence (100% grounding, 100% hint compliance). However, `CAND-01` (Qwen2.5-0.5B Q4_K_M) required **738 MB peak RSS**, which is completely unviable for a real 2 GB Android device where OS, system daemons, GPU, and user background tasks leave only ~150–250 MB for the foreground application process.

**The previous 750 MB target is formally retired.** Phase 3B enforces the **150–200 MB production memory contract** and introduces **Hybrid Intelligence**:
* **Deterministic Math Engine:** Handles arithmetic, fractions, equations, percentages, geometry formulas, and unit conversions with 100% precision.
* **Compact Textbook RAG:** Injects concise (<150 word) textbook chunks with SQLite FTS5 (<15 MB index memory).
* **Ultra-Small LLM / Micro-model:** Dedicated exclusively to Bengali phrasing, pedagogical explanations, and Socratic hints.
* **Math & Fact Validator:** Validates model responses against deterministic calculations to prevent hallucinated numbers.

---

## 2. Production Memory Budget Breakdown

```
============================================================
SS TUTOR BD — PHASE 3B PRODUCTION WORKING SET BUDGET
============================================================
Component                         Target Memory Allocation
------------------------------------------------------------
Model Weights + Runtime Engine    70 – 120 MB
KV Cache + Context Buffers        15 – 30 MB (at 1024 context)
Tokenizer Tables                  10 – 20 MB
RAG Engine / SQLite FTS5           5 – 15 MB
Prompt / Sanitization Buffers      5 – 10 MB
Deterministic Math Subsystem       5 – 10 MB
Application Overhead & Runtime    20 – 30 MB
Safety Headroom Margin            20 – 30 MB
------------------------------------------------------------
PREFERRED TOTAL WORKING SET      150 – 200 MB
WARNING THRESHOLD                200 – 250 MB
HARD PRODUCTION CEILING                250 MB
DISQUALIFYING THRESHOLD               >250 MB
============================================================
```

---

## 3. Subsystem Architecture

```
                       Student Question (Bengali)
                                   │
                           Intent Classifier
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
        Math Detected                           General / Concept
              │                                         │
    Deterministic Math Engine                     RAG Retrieval
    (Calculator / Fractions / Equations)         (SQLite FTS5 BM25)
              │                                         │
    Verified Solution/Formula                           │
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                         Context Assembly Stage
                     (Compact Prompts: [TASK], [TEXTBOOK],
                      [VERIFIED_RESULT], [QUESTION], [OUTPUT_RULES])
                                   │
                            Micro-LLM Runtime
                        (LlamaCpp / Micro Runtime)
                                   │
                         Output Sanitization Layer
                      (Control Tokens, Loops, Echoes)
                                   │
                         Deterministic Validator
                  (Check Result Against Calculation)
                                   │
                            Final Response
```

---

## 4. Work Breakdown & Implementation Steps

| Step | Component | Description |
| :--- | :--- | :--- |
| **1** | `benchmarks/memory/memory_benchmark.py` | Granular RSS profiler measuring baseline, load, tokenizer, RAG, inference, peak, and delta |
| **2** | `core/math/` | Deterministic mathematical engine (calculator, fractions, equations, geometry, units, validator) |
| **3** | `core/prompts/compact_tutor_templates.py` | Structured compact prompt protocol with delimiter tags and minimal token overhead |
| **4** | `core/runtime/adapter.py` | Model-agnostic runtime adapter abstracting local GGUF / ONNX / Micro runtimes |
| **5** | `core/tutor_engine.py` | Hybrid orchestrator integrating math parser, RAG, prompt builder, LLM, sanitizer, and validator |
| **6** | `models/registry.json` | Extended model registry with sub-500M ultra-compact candidate models (~100M-360M) |
| **7** | `benchmarks/phase3b/tutor_100_benchmark.json` | Comprehensive 100-question benchmark covering all NCTB Class 8 math topics & modes |
| **8** | `benchmark_runner/phase3b_runner.py` | Evaluation runner testing Mode A (LLM only), Mode B (LLM+RAG), Mode C (LLM+RAG+Math Tools) |
| **9** | `docs/ANDROID_MEMORY_VALIDATION.md` | Android memory validation specification (PSS, native heap, continuous session leak testing) |
| **10** | Unit & Regression Tests | Full test suite across memory, math engine, validator, compact prompts, repetition, and hybrid tutor |
| **11** | `PHASE3B_REPORT.md` & `MODEL_SELECTION_PHASE3B.md` | Comprehensive Phase 3B reporting and model decisions |

---

## 5. Acceptance & Sequential Gates

1. **Gate 1 (License):** Permissive open-source license compatible with Android app redistribution (Apache-2.0, MIT, BSD).
2. **Gate 2 (Binary Size):** Preferred $\le 150\text{ MB}$, Warning $150–200\text{ MB}$.
3. **Gate 3 (Production Memory):** Peak RSS $\le 200\text{ MB}$ preferred, $\le 250\text{ MB}$ hard ceiling. Sustained $>250\text{ MB}$ = FAIL.
4. **Gate 4 (Speed):** Throughput $\ge 4.0\text{ tok/s}$, TTFT $\le 2.0\text{ s}$.
5. **Gate 5 (Bengali Quality):** Grounded Bengali quality $\ge 70\%$ (Target $\ge 80\%$).
6. **Gate 6 (Educational Tutoring):** Socratic hint and step-by-step guidance $\ge 70\%$.
7. **Gate 7 (Hybrid Mathematical Correctness):** Combined LLM + deterministic engine $\ge 90\%$.
8. **Gate 8 (Grounding & Anti-Hallucination):** Textbook factual alignment $\ge 95\%$.
