# SS Tutor BD — Phase 8.1 Pre-Check: Core Model Forensic Discovery

**Phase:** 8.1 — Core Model Forensic Identification & Master Baseline Discovery  
**Date:** 2026-08-30  
**Product Strategy:** Model & AI Module Provider (Reusable Core AI Master -> Specialized Downstream Modules)  
**Strict Policy:** **Zero Training, Zero Quantization, Zero File Moving, Forensic Audit Only**  

---

## 1. Executive Forensic Discovery

This pre-check establishes the exact distinction between our **Reusable Core AI Model Master** and our **First Specialization (SS Tutor BD)**:

```text
                         CORE MODEL MASTER
                         ─────────────────
                         Reusable Base AI
                         (Architecture + 16K Tokenizer + Bounded Runtime)
                                │
                ┌───────────────┼────────────────┐
                │               │                │
                ▼               ▼                ▼
          SS Tutor BD       Mechanics        Future Niche
          Specialization   Specialization   Specialization
          (NCTB Class 6–10) (Applied Physics) (Custom Domain)
                │
                ▼
        Bangladesh High
        School Tutor
```

---

## 2. Forensic Answers Summary

| Investigation Target | Forensic Finding | Evidence / Source |
| :--- | :--- | :--- |
| **Current Neural Model Weights** | `models/sstutor_bengali_70m_edu/model.safetensors` (207.27 MB FP32, 54.33M parameters in 93 tensors) | `safe_open` parameter scan |
| **Model Architecture Definition** | 10-layer LLaMA Transformer ($H=576, d_{\text{ffn}}=2304$, 8 heads, $L_{\text{ctx}}=256$) | `training/train_micro_model.py` (`build_70m_micro_model`) |
| **Pre-Training Base Status** | `BASE_MODEL_RECONSTRUCTABLE` (Instantiated via deterministic PyTorch random normal initialization) | `training/train_micro_model.py` |
| **Is Current Model Generic Core or Specialization?** | **SS Tutor BD Domain-Specialized Model** (Trained on 13,000 Class 8 Math pairs; loss: 0.42) | `training/train_micro_model.py` |
| **Tokenizer Provenance** | **Generic Reusable Bengali Core Tokenizer** (16,000 Byte-level BPE, 3.65 tok/word across all Bengali Unicode) | `models/tokenizer_bengali_16k/` |
| **Knowledge vs. Model Separation** | **Cleanly Separated.** Knowledge is in SQLite FTS5 (`packs/class8_math/index.db`); model is in `safetensors` | `packs/` vs `models/` |
| **Future Forkability (e.g. Mechanics)** | **Fully Supported.** Architecture and tokenizer can be forked directly with new domain datasets | `core/runtime/micro_runtime.py` |
