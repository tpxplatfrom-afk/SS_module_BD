# SS Bangladesh Core Model Master (`ss_bangladesh`)

**Canonical Identity:** SS Bangladesh Core Model Master  
**Version:** 0.8.2  
**Role:** Root Reusable AI Foundation for Bengali-First Educational Specializations  
**Training Status:** **UNTRAINED / DOMAIN-NEUTRAL BASELINE**  
**Primary Bundle Location:** [`models/core/ss_bangladesh/`](file:///c:/Users/User/Desktop/SS_Tutor_BD/models/core/ss_bangladesh/)  
**Root Bundle Location:** [`ss_bangladesh/`](file:///c:/Users/User/Desktop/SS_Tutor_BD/ss_bangladesh/)  

---

## 1. Executive Definition & Architecture

The **SS Bangladesh Core Model Master** (`ss_bangladesh`) is the authoritative reusable base model from which all future domain-specific educational models are forked and trained.

```text
                    SS BANGLADESH CORE MODEL
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
        SS Tutor BD        Mechanics        Future Niche
      (Class 6–10 NCTB) (Applied Physics) (Custom Domain)
```

---

## 2. Master Specification & Parameter Metrics

* **Architecture Class:** `LlamaForCausalLM` (`transformers.LlamaConfig`)
* **Transformer Layers:** 10
* **Hidden Dimension ($d_{\text{model}}$):** 576
* **Intermediate FFN Dimension ($d_{\text{ffn}}$):** 2,304 (SwiGLU activation)
* **Attention Heads:** 8
* **Key-Value Heads:** 8
* **Context Length ($L_{\text{ctx}}$):** 256 tokens
* **Total Parameters:** **71,528,256 (71.53M)**
* **Tensor Count:** **93 Tensors**
* **DType:** `float32` (Weights: `model/model.safetensors`)
* **Tokenizer:** 16,000 Byte-level BPE (`tokenizer/tokenizer.json`)
* **Initialization:** Seed 42 Truncated Normal distribution ($\mu=0.0, \sigma=0.02$)
* **Safetensors Checksum (SHA-256):** `bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb`

---

## 3. Internal Master Bundle Layout

```text
models/core/ss_bangladesh/
├── model/
│   ├── config.json              (LlamaConfig architecture parameters)
│   ├── generation_config.json   (Sampling and generation settings)
│   └── model.safetensors        (Untrained baseline weights, 71.53M params)
├── tokenizer/
│   ├── tokenizer.json           (16,000 Byte-level BPE vocab & merges)
│   └── tokenizer_config.json    (Special token mappings)
├── config/
│   └── architecture.json        (Canonical architecture and metrics)
├── manifest.json                (Machine-readable master manifest)
├── lineage.json                 (Lineage and downstream specialization map)
├── checksums.sha256             (Cryptographic file integrity hashes)
└── README.md                    (Master usage guide)
```

---

## 4. Specialization Isolation Invariants

1. **Zero Curriculum Hardcoding:** `ss_bangladesh` contains zero textbook facts, formulas, or curriculum chapters.
2. **Untrained Baseline:** Unlike `models/sstutor_bengali_70m_edu/model.safetensors` (which was trained on Class 8 Math), `ss_bangladesh` contains raw, reproducible baseline weights.
3. **Seamless Forkability:** Downstream projects copy `ss_bangladesh`, inject their specific training corpus, and train without modifying the Core Master.
