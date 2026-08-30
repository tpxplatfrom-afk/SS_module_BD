# SS Tutor BD — Phase 4 Master Implementation Plan

**Project:** SS Tutor BD — Offline-First Modular AI Education Platform  
**Phase:** 4 — Bengali Micro-Model Training, Distillation & Android MicroRuntime Integration  
**Target Hardware:** Android (2 GB Physical RAM, 16 GB Storage)  
**Strict Memory Contract:** Preferred $\le 150\text{ MB}$, Absolute Ceiling $\le 200\text{ MB}$  
**Target Model Size:** 60M–80M Parameters, $\le 50\text{ MB}$ INT4 Binary  
**Target Bengali Tokenizer:** $\le 4.0\text{ tokens / Bengali word}$ (Stretch $\le 3.0$)  
**Development Cost:** \$0 USD  

---

## 1. Architectural Strategy & Pipeline

```
                    Student Query (Bengali / Banglish)
                                  │
                       Device Capability Check
                                  │
                       Intent Classification
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
    Math Intent Detected                        Textbook Concept Inquiry
          │                                               │
Deterministic Math Engine                       SQLite FTS5 RAG Retrieval
 (Exact 100% Precision)                                   │
          │                                     Context Compressor
          │                              (High-Density Factual Nuggets)
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  │
                     TutorTask Structured IR
                                  │
                     Micro Prompt Protocol
                 ([T], [F], [R], [G], [H], [C])
                                  │
                 Context Budget Manager Enforcer
                  (Prompt <= 70, RAG <= 120 tok)
                                  │
                     Bengali Micro-Model (70M)
               (INT4 Quantized, Custom 16K Vocab)
                                  │
                 Output Validation & Guard Layer
             ┌────────────────────┼────────────────────┐
             │                    │                    │
      Grounding Guard      Math Validator       Hint-Leak Guard
     (Anti-Hallucinate)   (Check Number)       (Answer Withheld)
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                       Student Bengali Response
```

---

## 2. Phase 4 Work Breakdown & 21-Step Execution

1. **Repository Audit & Licensing:** `PHASE4_PRECHECK.md`, `PHASE4_PLAN.md`, `DATA_LICENSE_POLICY.md`, and license records in `results/licenses/phase4/`.
2. **Dedicated Bengali Tokenizer (`core/tokenizer/`):**
   * `tokenizer_trainer.py`: Trains a 16,000 vocabulary Byte-level BPE tokenizer on Bengali NCTB math + science + symbols.
   * `tokenizer_benchmark.py`: Evaluates characters/token, tokens/word, and comparison vs Qwen and SmolLM2.
   * `tokenizer_validator.py`: Asserts mathematical symbol fidelity ($x^2, \sqrt{}, \pi, \le, \ge, =, +, -, \times, \div, \%$).
3. **Synthetic Dataset Generation (`data/phase4/` & `scripts/`):**
   * `scripts/generate_math_dataset.py`: 5,000 arithmetic, fraction, percentage, algebra, and geometry tool-result verbalizations.
   * `scripts/generate_socratic_dataset.py`: 3,000 Socratic hint and guidance pairs.
   * `scripts/generate_grounding_dataset.py`: 3,000 grounded Q&A pairs with refusal of unsupported facts.
   * `scripts/generate_bengali_variants.py`: Formal, colloquial, and student shorthand input variations.
4. **Micro-Model Architecture Experiment & Distillation (`training/`):**
   * Configs: 50M, 70M, 90M Transformer backbones.
   * Distillation documentation: `docs/DISTILLATION.md`.
   * Training configuration: `configs/phase4_training.json`.
5. **Validation Layer & Context Budget (`core/validation/` & `core/runtime/context_budget.py`):**
   * Multi-guard validator checking math numbers, textbook grounding, Socratic hint leaks, and repetition.
   * Hard context token budget enforcement.
6. **Model Export & Storage Cleanup:**
   * `scripts/export_phase4_model.py`: Validates checksums, tokenizers, and INT4 export.
   * `scripts/purge_training_artifacts.py`: Purges intermediate training checkpoints.
7. **Comprehensive 450+ Question Benchmark & A/B Comparison:**
   * 7 evaluation categories in `benchmarks/phase4/` (450+ items).
   * `benchmark_runner/phase4_runner.py`: 4-system comparative runner (System A: Deterministic Core, System B: Model Only, System C: Model + RAG, System D: Full Hybrid).
8. **Android PSS Validation & Final Reporting:**
   * `docs/ANDROID_PHASE4_VALIDATION.md`.
   * `results/model_decision/model_decision_phase4.json`.
   * `PHASE4_REPORT.md` (27 sections).

---

## 3. Strict Mandatory Production Acceptance Gates

| Gate | Criterion | Target Threshold |
| :--- | :--- | :--- |
| **Gate 1** | Bengali Tokenizer Efficiency | $\le 4.0\text{ tokens / Bengali word}$ (Stretch $\le 3.0$) |
| **Gate 2** | Model Parameter Count | 60M–80M parameters (Preferred ~70M) |
| **Gate 3** | Quantized Binary Size | $\le 50\text{ MB}$ (INT4) |
| **Gate 4** | Production Process Peak RAM | Preferred $\le 150\text{ MB}$, Hard Ceiling $\le 200\text{ MB}$ |
| **Gate 5** | Multi-Turn Memory Stability | Growth $\le 0.05\text{ MB / query}$ over 100 turns (Zero Leak) |
| **Gate 6** | Mathematical Accuracy (with tools) | $\ge 98.0\%$ Exact Correctness |
| **Gate 7** | Textbook Grounding Adherence | $\ge 95.0\%$ Grounded Facts |
| **Gate 8** | Socratic Hint Compliance | $\ge 95.0\%$ Direct Answer Withholding |
| **Gate 9** | 450+ Question Benchmark Score | $\ge 80.0 / 100.0$ Weighted Score |
| **Gate 10** | License & Zero Cost | Permissive FOSS, 100% \$0 development cost |
