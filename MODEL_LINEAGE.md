# SS Tutor BD — Master Model Lineage Specification

**Document Version:** 2.0.0  
**Phase:** 8.2 — Core Model Master Assembly  

---

## 1. Authoritative Lineage Graph

```text
                     [PyTorch LLaMA Config Definition]
                     File: training/train_micro_model.py:L95-123
                                    │
                                    │ Deterministic Init (Seed 42, sigma=0.02)
                                    ▼
                [SS BANGLADESH CORE MODEL MASTER (ss_bangladesh)]
                Location: models/core/ss_bangladesh/ & ss_bangladesh/
                Parameters: 71,528,256 (71.53M) in 93 Tensors (Untrained Baseline)
                Tokenizer: 16,000 Byte-level BPE (Generic Bengali-First)
                SHA-256 Checksum: bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb
                                    │
           ┌────────────────────────┴────────────────────────┐
           │ COPY / FORK                                     │ COPY / FORK
           ▼                                                 ▼
[SS TUTOR BD SPECIALIZATION]                       [MECHANICS SPECIALIZATION]
Target: Bangladesh High School (NCTB Class 6–10)   Target: Applied Physics & Mechanics
Training Data: data/phase4/ (Class 8 Math)         Training Data: data/mechanics/ (Future)
Knowledge Pack: packs/class8_math/                 Knowledge Pack: packs/mechanics/ (Future)
Active Weights: models/sstutor_bengali_70m_edu/    Status: Planned Future Fork
           │
           ├── INT4 Quantization Export (models/export_int4/, 34.12 MB)
           │
           └── Scaffolding: Exact Solvers (core/math/) + Multi-Guard Validators
                   │
                   ▼
       [SS TUTOR BD DEVELOPER MODULE]
       File: core/tutor_module.py (SSTutorBDModule)
       Status: Integratable by Third-Party Chatbot Developers
```

---

## 2. Invariant Rules of the Lineage

1. **Unidirectional Specialization:** Specialization metadata, training datasets, and learned weights flow strictly downstream from `ss_bangladesh`.
2. **Zero Upstream Contamination:** Downstream training experiments on SS Tutor BD or Mechanics must never overwrite the `ss_bangladesh` baseline weights or manifest.
