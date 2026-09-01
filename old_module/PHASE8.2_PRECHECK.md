# SS Tutor BD — Phase 8.2 Pre-Check: Core Model Master Assembly

**Phase:** 8.2 — Core Model Master Assembly & Baseline Creation  
**Date:** 2026-08-30  
**Product Strategy:** AI Module/Model Provider (Reusable Core Model Master -> Specialized Downstream Modules)  
**Strict Directives:** **Zero Training, Zero Quantization, Zero File Destruction, Controlled Master Assembly Only**  

---

## 1. Product Identity & Core Master Role

Our company is an **AI Model / Module Provider**, not an Android application distributor.

```text
OUR COMPANY (SS Bangladesh)
    │
    ├── builds AI models
    ├── trains AI models
    ├── maintains AI models
    ├── versions AI models
    └── distributes AI modules
             │
             ▼
       SS TUTOR BD MODULE
             │
             ▼
     Android Developers
             │
             ▼
       Their AI / Chatbot App
             │
             ▼
          Students
             │
             ▼
       Offline Tutoring
```

---

## 2. Core Baseline vs. Specialization Audit

| Dimension | Core Model Master (`ss_bangladesh`) | SS Tutor BD Specialization |
| :--- | :--- | :--- |
| **Canonical Role** | Root Reusable Foundation AI | First Domain Specialization |
| **Domain Scope** | Domain-Neutral / General Bengali Base | Bangladesh NCTB High School (Class 6–10) |
| **Training Status** | **Untrained Baseline** (Seed 42 Truncated Normal) | **Supervised Trained** on 13,000 Class 8 Math pairs |
| **Parameters** | 71,528,256 (71.53M with 16K Vocab) | 54,332,352 (Active slice) up to 68.2M |
| **Weights File** | `models/core/ss_bangladesh/model/model.safetensors` | `models/sstutor_bengali_70m_edu/model.safetensors` |
| **Tokenizer** | 16,000 Byte-level BPE (`tokenizer_bengali_16k`) | 16,000 Byte-level BPE (`tokenizer_bengali_16k`) |
| **Curriculum Data** | **None (Zero Curriculum Bias)** | `packs/class8_math/index.db` (SQLite FTS5) |
| **Mathematical Engine** | Interface only | Exact Solvers (`core/math/`) + Multi-Guard Validators |
| **Portability** | Master bundle (`models/core/ss_bangladesh/`) | Exported INT4 module (`models/export_int4/`) |

---

## 3. Risks & Architectural Guards

1. **Risk of Conflating Base with Specialization:** Resolved by maintaining `ss_bangladesh` strictly domain-neutral with zero hardcoded textbook facts or Class 8 Math weights.
2. **Preservation of Existing Assets:** The existing Class 8 Math model and Android integration remain 100% intact and untouched.
3. **Cryptographic Immutability:** SHA-256 integrity checksums ensure accidental drift or mutation in the Core Master is immediately detected.
