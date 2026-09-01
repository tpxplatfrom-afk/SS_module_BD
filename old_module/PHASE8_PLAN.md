# SS Tutor BD — Phase 8 Master Implementation Plan

**Project:** SS Tutor BD — Offline-First Bangladesh NCTB AI Tutor Core  
**Phase:** 8 — Core Model Development, Curriculum Knowledge Architecture & Package-Ready Foundation  
**Primary Goal:** Build the full Core Model and Knowledge Foundation for Class 6–10 NCTB without implementing package distribution.  
**Budget Constraint:** \$0 USD  

---

## 1. 15-Step Sequential Execution Roadmap

```text
┌─────────────────────────────────────────────────────────────┐
│               PHASE 8 IMPLEMENTATION ROADMAP                │
├─────────────────────────────────────────────────────────────┤
│ 1. Repository Audit & Product Role Clarification            │
│    - Completed in PHASE8_PRECHECK.md                        │
│                                                             │
│ 2. Curriculum Knowledge Schema (core/curriculum/schema.py)  │
│    - Grade -> Subject -> Book -> Chapter -> Topic -> Concept│
│    - Deterministic Concept IDs (e.g. g08.math.ch02.c01)     │
│                                                             │
│ 3. Package-Ready Boundaries (core/curriculum/boundaries.py) │
│    - KnowledgeUnit, KnowledgePackMetadata, CurriculumScope  │
│                                                             │
│ 4. Curriculum Coverage Engine (core/curriculum/coverage.py) │
│    - Audit Class 6-10 NCTB concepts & example density       │
│    - Output results/phase8/curriculum_coverage.json & .md   │
│                                                             │
│ 5. Dataset Quality Auditor (core/curriculum/dataset_auditor)│
│    - Audit 13k examples for duplicates, balance, behaviors  │
│    - Output results/phase8/dataset_quality.json             │
│                                                             │
│ 6. Educational Persona & Diverse Behaviors Pipeline         │
│    - QA, Step-by-Step, Hints, Misconceptions, Follow-ups    │
│    - Enrich dataset generator with diverse tutoring modes   │
│                                                             │
│ 7. Real Curriculum Evaluator (benchmarks/phase8/eval.py)    │
│    - 13 Core Educational Dimensions evaluation              │
│    - Output results/phase8/baseline_model_evaluation.json   │
│    - Output results/phase8/model_capability_matrix.json     │
│                                                             │
│ 8. Developer Module Contract (core/tutor_module.py)         │
│    - Clean SDK interface: initialize, ask, hint, explain    │
│                                                             │
│ 9. Model Versioning System (docs/MODEL_VERSIONING.md)       │
│    - Core v0.8.0, Tokenizer v0.4.0, Knowledge v0.8.0       │
│                                                             │
│ 10. Training Pipeline Optimization (configs/phase8_train)   │
│     - Reproducible training configuration & checkpointing   │
│                                                             │
│ 11. Experiment Registry (results/phase8/experiment_registry)│
│     - Track baseline vs Phase 8 iterations                  │
│                                                             │
│ 12. Full Documentation Suite                                │
│     - CORE_MODEL_ARCHITECTURE.md                            │
│     - CURRICULUM_KNOWLEDGE_ARCHITECTURE.md                  │
│     - TRAINING_DATA_GUIDELINES.md                           │
│     - FUTURE_PACKAGE_ARCHITECTURE.md                        │
│                                                             │
│ 13. Phase 8 Test Suite & Full Regression                    │
│     - 17 Existing regression tests + Phase 8 unit tests     │
│                                                             │
│ 14. Phase 8 Gate Matrix (results/phase8/gate_matrix.json)   │
│                                                             │
│ 15. Final Report & Answers to Q1-Q10 (PHASE8_REPORT.md)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Invariants in Phase 8

1. **Core Model First, Packages Later:** Phase 8 focuses 100% on the core model and knowledge architecture. No package distribution UI or marketplace will be created.
2. **Deterministic Mathematical Authority:** Exact computations remain strictly delegated to the deterministic math engine (`fraction.py`, `calculator.py`, `equation_solver.py`).
3. **Evidence-Based Reporting:** All metrics must be empirically measured from actual datasets and model outputs; zero fabricated claims.
