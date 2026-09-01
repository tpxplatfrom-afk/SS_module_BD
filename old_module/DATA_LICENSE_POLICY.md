# SS Tutor BD — Data License & Provenance Policy

**Version:** 1.0.0  
**Phase:** 4 — Micro-Model Training & Distillation  
**Date:** 2026-08-30  

---

## 1. Core Policy Principles

1. **Zero Copyrighted Text Redistribution:**
   * Copyrighted NCTB textbook text is **never** redistributed as part of pre-baked training datasets.
   * User-provided or local curriculum textbook packs remain **external retrieval data** indexed locally on the user device via SQLite FTS5.
2. **Synthetic Data from First Principles:**
   * All training examples (arithmetic step-by-step verbalizations, algebraic derivations, Socratic questioning dialogues, and grounding adherence pairs) are synthetically generated from mathematical definitions, formulas, and general pedagogical templates.
3. **Permissive Open Data Provenance:**
   * Any reference datasets used for vocabulary induction or grammatical grounding must possess verified permissive licenses (`CC-BY-4.0`, `CC0`, `MIT`, `Apache-2.0`, or public domain).
4. **Machine-Readable Audit Trail:**
   * Every dataset artifact in `data/phase4/` records its generator script, parameter configurations, provenance, and license metadata.

---

## 2. Dataset Classification & Provenance Matrix

| Dataset Category | Generation Method | Content Description | License Status |
| :--- | :--- | :--- | :--- |
| `data/phase4/math/` | `scripts/generate_math_dataset.py` | Step-by-step arithmetic, fraction, percentage, interest, and algebra verbalizations | Synthetic / CC0 (FOSS) |
| `data/phase4/socratic/` | `scripts/generate_socratic_dataset.py` | Socratic hint-generation, question scaffolding, and answer withholding | Synthetic / CC0 (FOSS) |
| `data/phase4/grounding/`| `scripts/generate_grounding_dataset.py`| Grounded context-answering and refusal of unmentioned facts | Synthetic / CC0 (FOSS) |
| `data/phase4/bengali/`  | `scripts/generate_bengali_variants.py` | Formal, colloquial, student shorthand, and Banglish variations | Synthetic / CC0 (FOSS) |
| `packs/class8_math/`    | Local SQLite FTS5 Index | Local curriculum chunks for RAG (retrieved on-device) | User/Local Retrieval Only |
