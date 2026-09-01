# SS Bangladesh Core Model Master (`ss_bangladesh`)

**Canonical Identity:** SS Bangladesh Core Model Master  
**Version:** 0.8.2  
**Role:** Root Reusable AI Foundation for Bengali-First Educational Specializations  
**Training Status:** **UNTRAINED / DOMAIN-NEUTRAL BASELINE**  

---

## 1. Overview & Architectural Role

`ss_bangladesh` is the authoritative master core model from which all downstream specialized models (such as **SS Tutor BD** and **Mechanics**) are forked and trained.

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

## 2. Parameter & Layer Specifications

* **Architecture Class:** `LlamaForCausalLM`
* **Layers:** 10
* **Hidden Dimension ($d_{\text{model}}$):** 576
* **Intermediate Dimension ($d_{\text{ffn}}$):** 2,304 (SwiGLU activation)
* **Attention Heads:** 8
* **Key-Value Heads:** 8
* **Context Length ($L_{\text{ctx}}$):** 256 tokens
* **Total Parameters:** **71,528,256 (71.53M)**
* **Tokenizer:** 16,000 Byte-level BPE (`tokenizer/`)
* **Initialization Seed:** 42 (Truncated Normal, $\sigma = 0.02$)
* **Model Checksum (model.safetensors):** `bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb`

---

## 3. Immutability & Domain Neutrality

* This artifact contains **ZERO curriculum bias or hardcoded textbook facts**.
* Downstream training and domain knowledge packs are strictly isolated in downstream specializations.
