# SS Tutor BD — Model & Core Module Versioning Specification

**Document Version:** 1.0.0  
**Phase:** 8 — Core Model Development  

---

## 1. Modular Versioning Schema

To guarantee full reproducibility across development, training, and deployment, SS Tutor BD components are versioned independently:

```text
========================================================================
SS TUTOR BD COMPONENT VERSION REGISTRY (Current Baseline)
========================================================================
Core Module Release:         v0.8.0
Core Model Architecture:     v0.8.0 (70M LLaMA-based Transformer)
Bengali Tokenizer:           v0.4.0 (16,000 Byte-level BPE)
Knowledge Framework:         v0.8.0 (NCTB Class 6–10 Hierarchy)
Active Knowledge Pack:       v0.8.0 (Class 8 Mathematics .ssp)
Training Dataset:            v0.4.0 (13,000 Synthetic Curriculum Pairs)
Inference Runtime Adapter:   v0.6.0 (Native MicroRuntime + Deterministic)
========================================================================
```

---

## 2. Version Tagging Semantics

* **`model_id`:** Unique identifier for model architecture (e.g. `sstutor_bengali_70m_edu`).
* **`model_version`:** SemVer tag indicating weight revision and training iteration (e.g. `v0.8.0`).
* **`tokenizer_version`:** Vocabulary and merge-table release identifier.
* **`knowledge_version`:** Curriculum pack schema and SQLite FTS5 database revision.
* **`dataset_version`:** Training dataset hash and example composition.
