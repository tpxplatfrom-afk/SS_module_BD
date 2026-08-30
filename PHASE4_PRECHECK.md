# SS Tutor BD — Phase 4 Pre-Check & Repository Audit

**Phase:** 4 — Bengali Micro-Model Training, Distillation & Android MicroRuntime Integration  
**Date:** 2026-08-30  
**Target Hardware:** Low-End Android (2 GB RAM, 16 GB Storage)  
**Host Hardware:** Intel Core i5-6500 (4C/4T CPU), Windows 10 x64, 0$ Budget  

---

## 1. Existing Interfaces & Assumptions

| Subsystem | Existing Interface / Component | Phase 4 Compatibility & Role |
| :--- | :--- | :--- |
| **Deterministic Math** | `core/math/` (`FractionHelper`, `MathCalculator`, `EquationSolver`, `UnitConverter`, `ExpressionParser`, `MathValidator`) | **Authoritative (100% Preserved).** The micro-model will NEVER perform raw arithmetic independently; it only verbalizes deterministic results. |
| **RAG Retrieval** | `core/rag/` (`KnowledgeIndexer`, `KnowledgeRetriever`, `ContextCompressor`) | **Preserved.** SQLite FTS5 index (164 KB, 1.39 ms latency) provides factual textbook definitions and formulas. |
| **Micro Prompt Protocol** | `core/prompts/micro_protocol.py` (`[T]`, `[F]`, `[R]`, `[G]`, `[H]`, `[C]`) | **Preserved & Integrated.** Keeps prompt token overhead $< 70$ tokens. |
| **Session Memory** | `core/runtime/session_manager.py` (`SessionState`, `SessionManager`) | **Preserved.** $O(1)$ constant-memory session state (0.00 MB growth over 100 turns). |
| **Memory Budget** | `core/runtime/memory_budget.py` | **Preserved & Enforced.** Strict $\le 200\text{ MB}$ hard ceiling ($\le 150\text{ MB}$ preferred). |
| **Device Profiling** | `core/runtime/device_profile.py` | **Extended.** Maps `ULTRA_LOW` (256 ctx, 64 out), `LOW` (384 ctx, 96 out), and `STANDARD` (512 ctx, 128 out). |
| **Runtime Adapter** | `core/runtime/micro_runtime.py` (`MicroRuntimeBase`, `MicroRuntimeFactory`) | **Extended.** Adds pluggable support for the custom trained Bengali micro-model. |

---

## 2. Incompatible Components & Required Changes

1. **SmolLM2 / Llama-based Tokenizers (Disqualified):**
   * SmolLM2's 49K vocabulary suffered from **8.47 tokens per Bengali word** (byte expansion), breaking context limits.
   * *Required Change:* Train a custom, dedicated **16K–24K Bengali-first BPE tokenizer** (`core/tokenizer/`) optimized for NCTB Bengali, English technical terms, and mathematical symbols ($\le 3.5$ tokens/word).
2. **Generic LLM Prompts & Weight Footprint:**
   * Off-the-shelf 0.5B–1.5B models consume 470 MB–1065 MB on disk and 738 MB–1771 MB in RAM.
   * *Required Change:* Build a specialized **60M–80M parameter Transformer** ($\le 50\text{ MB}$ INT4 binary, $\le 150\text{ MB}$ peak RSS).
3. **Context Budget Enforcement:**
   * *Required Change:* Create `core/runtime/context_budget.py` to enforce strict token boundaries per component (Prompt $\le 70$, Facts $\le 120$, Output $\le 128$).
4. **Validation Layer Expansion:**
   * *Required Change:* Create `core/validation/` containing `GroundingValidator`, `MathAnswerValidator`, `HintValidator`, `LanguageValidator`, and `FormatValidator`.

---

## 3. Training & Data Requirements ($0 Budget, Local CPU Feasible)

* **Tokenizer Training Data:** 25,000+ representative Bengali NCTB curriculum sentences, mathematical expressions, formulas, and conversational tutoring dialogues.
* **Synthetic Training Sets:** Generated via deterministic scripts in `scripts/`:
  1. `generate_math_dataset.py`: 5,000+ arithmetic, fraction, percentage, interest, and algebra verbalization pairs.
  2. `generate_socratic_dataset.py`: 3,000+ hint-generation and answer-withholding pairs.
  3. `generate_grounding_dataset.py`: 3,000+ textbook-grounded Q&A and anti-hallucination refusal pairs ("প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না").
  4. `generate_bengali_variants.py`: Formal, colloquial, student shorthand, and Banglish input variations.
* **Storage Footprint:** Training data $\approx 15–25\text{ MB}$ JSONL. Checkpoints strictly purged after conversion to ensure disk free $> 2.0\text{ GB}$.
