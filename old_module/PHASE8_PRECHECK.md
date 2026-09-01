# SS Tutor BD — Phase 8 Pre-Check & Core Model Architecture Audit

**Phase:** 8 — Core Model Development, Curriculum Knowledge Architecture & Package-Ready Foundation  
**Date:** 2026-08-30  
**Product Responsibility:** AI Educational Core Module & Model Provider (Not an App Distributor)  
**Budget Constraint:** \$0 USD (Local CPU Development, Zero Paid APIs, Zero Cloud GPU)  

---

## 1. Product Identity & Architectural Role

```text
                         SS TUTOR BD
                        CORE AI MODULE
                              │
             ┌────────────────┼────────────────┐
             │                │                │
         Developer A      Developer B      Developer C
         AI App           AI App           AI App
             │                │                │
             └────────────────┼────────────────┘
                              │
                           Student
                              │
                       Offline Learning
```

* **Application Developers own:** UI, activities, views, user authentication, distribution, end-user product.
* **SS Tutor BD owns:** The Core AI Model, Curriculum Knowledge Base, Bengali Tutoring Capabilities, Deterministic Math Engine, Multi-Guard Validators, Offline Inference Engine, and Developer Module Interfaces.

---

## 2. Baseline Architecture Audit (Phases 1–7)

| Component | Current Implementation | Status | Reusability in Phase 8 |
| :--- | :--- | :--- | :--- |
| **Deterministic Math Core** | `core/math/` (`fraction.py`, `calculator.py`, `equation_solver.py`, `unit_converter.py`, `expression_parser.py`) | 100% Exact | **Fully Reusable & Authoritative** |
| **Dedicated Tokenizer** | `core/tokenizer/` (16,000 Byte-level BPE, 3.65 tok/word) | Validated | **Fully Reusable Foundation** |
| **Micro-Model Baseline** | `models/sstutor_bengali_70m_edu/` (68.2M params, 34.12 MB INT4) | Baseline | **Baseline Reference Model** |
| **Validation Layer** | `core/validation/` (Grounding, Math, Hint, Language, Format) | 5 Guards | **Fully Reusable Guardrails** |
| **RAG Retrieval** | `core/rag/` (SQLite FTS5 + Context Compressor) | Class 8 Math | **Needs Extension across Grades 6–10** |
| **Session Memory** | `core/runtime/session_manager.py` ($O(1)$ constant state) | Validated | **Fully Reusable** |
| **Training Pipeline** | `training/train_micro_model.py` (HuggingFace Trainer CPU) | Functional | **Needs Enhancement for Multi-Class Data** |

---

## 3. Current Model & Dataset Knowledge Reality Check

> [!WARNING]
> **Audit Finding:** The existing Phase 4 dataset comprises **13,000 synthetic examples** exclusively focused on **Class 8 Mathematics** (Fractions, Simple/Compound Interest, Pythagoras Theorem, Arithmetic Series).
> 
> The current model **DOES NOT** yet contain curriculum coverage for:
> * Class 6 (Mathematics, General Science, Bengali)
> * Class 7 (Mathematics, General Science, Bengali)
> * Class 8 (General Science, Bengali, English)
> * Class 9–10 (Higher Mathematics, General Mathematics, Physics, Chemistry, Biology, Bengali)

### Empirical Knowledge Scope Summary
* **Class 6 Coverage:** $0\%$ (Source missing in Phase 4 dataset)
* **Class 7 Coverage:** $0\%$ (Source missing in Phase 4 dataset)
* **Class 8 Mathematics Coverage:** $\sim 85\%$ (Well-represented in 13k dataset)
* **Class 8 Other Subjects:** $0\%$
* **Class 9–10 Coverage:** $0\%$

---

## 4. Technical Debt & Risks Identified

1. **Synthetic QA Pattern Overfitting:** The 13k dataset is heavily structured as single-turn question-to-answer pairs. It lacks conversational follow-ups ("সহজ করে বলো", "আরেকটা উদাহরণ দাও"), step-by-step misconception corrections, and counter-examples.
2. **Monolithic Curriculum Representation:** Class 8 knowledge is embedded directly without a formal ontological hierarchy (`Grade -> Subject -> Chapter -> Topic -> Concept`).
3. **No Package Boundary Abstractions:** Current knowledge ingestion lacks metadata tagging (`CurriculumScope`, deterministic concept IDs) required to enable modular knowledge separation in the future.

---

## 5. Phase 8 Action Strategy

1. Establish a formal **Curriculum Knowledge Schema** with deterministic hierarchical IDs (`g08.math.ch02.topic01.concept03`).
2. Build an automated **Curriculum Coverage Engine** to accurately audit coverage across Class 6–10.
3. Build a **Training Dataset Quality Auditor** to identify imbalances, synthetic repetition, and behavior gaps.
4. Establish a **Real Curriculum Evaluation Framework** testing 13 distinct educational dimensions.
5. Create **Package-Ready Knowledge Boundaries** (`KnowledgeUnit`, `KnowledgePackMetadata`, `CurriculumScope`) without prematurely creating package distribution systems.
6. Provide clear **Model Versioning** and developer-facing integration contracts (`core/tutor_module.py`).
