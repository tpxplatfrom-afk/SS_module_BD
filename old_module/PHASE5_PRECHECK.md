# SS Tutor BD — Phase 5 Pre-Check & Architecture Audit

**Phase:** 5 — Android / Native UI Binding, Offline APK Packaging & Real-Device Validation  
**Date:** 2026-08-30  
**Target Hardware:** Low-End Android (2 GB Physical RAM, 16 GB Storage, Android 9.0–14.0)  
**Strict Memory Contract:** Preferred $\le 150\text{ MB}$, Hard Ceiling $\le 200\text{ MB}$, Emergency $\le 250\text{ MB}$  
**Development Cost:** \$0 USD  

---

## 1. Complete Architecture Inventory (Phases 1–4)

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SS TUTOR BD PRODUCTION CORE                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Deterministic Math Core (core/math/)                                │
│    - fraction.py: Exact rational arithmetic with NCTB steps (100% acc) │
│    - calculator.py: Simple/Compound interest, series sum, pythagoras   │
│    - equation_solver.py: Linear 2x2 systems, quadratic factoring       │
│    - expression_parser.py: Bengali math regex intent extraction        │
│    - validator.py: Exact numeric verification                          │
│                                                                        │
│ 2. Offline RAG Subsystem (core/rag/)                                   │
│    - indexer.py / retriever.py: SQLite FTS5 index (164 KB, 1.39 ms)    │
│    - context_compressor.py: High-density factual nugget extractor      │
│    - Knowledge Pack: Class 8 Math (.ssp package format)                │
│                                                                        │
│ 3. Dedicated Bengali Tokenizer (core/tokenizer/)                       │
│    - Custom 16,000-vocabulary Byte-level BPE tokenizer                 │
│    - 3.65 - 3.86 tokens / Bengali word (vs 8.46 for SmolLM2)           │
│    - Full mathematical symbol fidelity (x², √, π, ≤, ≥, =, +, -, etc)  │
│                                                                        │
│ 4. 70M Transformer Micro-Model (models/sstutor_bengali_70m_edu/)       │
│    - 54.3M - 68.2M parameters, INT4 quantized binary size = 34.12 MB   │
│    - Trained on 13,000 synthetic NCTB curriculum examples              │
│                                                                        │
│ 5. Multi-Guard Validation Layer (core/validation/)                     │
│    - grounding_validator.py: Anti-hallucination factual bounds (95%)   │
│    - math_answer_validator.py: Numerical consistency & auto-correction │
│    - hint_validator.py: Socratic direct answer withholding (100%)      │
│    - language_validator.py: Repetition loop suppression                │
│    - format_validator.py: Delimiter tag & control token stripping      │
│                                                                        │
│ 6. Bounded Session & Memory Architecture (core/runtime/)               │
│    - session_manager.py: Constant O(1) memory state (0.00 MB growth)   │
│    - memory_budget.py: Explicit 150-200 MB subsystem budget allocation │
│    - device_profile.py: ULTRA_LOW / LOW / STANDARD adaptive policies   │
│    - context_budget.py: Strict component token bounds (<= 256 ctx)     │
│    - micro_runtime.py: Abstract MicroRuntimeBase adapter interface     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Android Integration Risks & Mitigation Strategies

| Risk Area | Specific Failure Mode | Phase 5 Mitigation Strategy |
| :--- | :--- | :--- |
| **Python Runtime in APK** | Embedding CPython/PyTorch adds $> 150\text{ MB}$ overhead, violating memory ceiling. | **Zero Python on Android:** Compile core algorithms into native Kotlin/C++ engine. Pure native/ONNX/GGUF runtime bridge. |
| **Out of Memory (OOM) Killers** | Android Low Memory Killer (LMK) kills processes $> 200\text{ MB}$ PSS on 2 GB RAM devices. | **Lazy Auto Loading + AndroidMemoryMonitor:** Model unloaded when idle; emergency trimming on `onTrimMemory(RUNNING_CRITICAL)`. |
| **Unbounded Conversation Context** | Message history accumulation inflates KV-cache. | **Preserve $O(1)$ `SessionState`:** Never send full chat history to inference; only inject compact topic summary ($< 20$ tokens). |
| **Answer Leakage in Socratic Mode** | Model hints directly giving away final numeric result. | **Native `HintValidator`:** Sanitizes output before UI commit; strips forbidden numbers. |
| **Storage Bloat** | Extracting duplicates of model / DB into app sandbox. | **Direct Asset Streaming:** Read SQLite and INT4 model weights directly via `mmap` without duplication. |

---

## 3. Technology Stack Decision (Kotlin Android Native Engine + Modern UI)

* **UI Layer:** Native Kotlin Android / Material Design 3 (Zero Flutter runtime bloat, minimal base APK size $< 15\text{ MB}$, $< 25\text{ MB}$ Java/ART heap).
* **Inference Engine:** Abstract `MicroRuntime` native bridge (ONNX Runtime Mobile / GGUF C++ / Deterministic Fallback).
* **Database:** SQLite FTS5 native Android driver (`android.database.sqlite.SQLiteDatabase`).
* **Packaging:** Single offline APK containing bundled `.ssp` Class 8 Mathematics pack and INT4 model.
