# SS Tutor BD — Core Model Architecture Specification

**Document Version:** 1.0.0  
**Phase:** 8 — Core Model Development  

---

## 1. Architectural Philosophy: Hybrid Core AI

SS Tutor BD enforces a strict separation between **exact mathematical computation** and **pedagogical language verbalization**:

```text
                    SS TUTOR BD CORE
                           │
              ┌────────────┴────────────┐
              │                         │
     Deterministic Core            Micro-Model
              │                         │
     Math / RAG / Rules         Bengali Verbalization
              │                         │
              └────────────┬────────────┘
                           │
                    Validation Layer
                           │
                  Developer API Module
```

---

## 2. Micro-Model Parameter Specifications

* **Architecture:** Causal Autoregressive Transformer (LLaMA-style architecture).
* **Parameter Count:** **68,244,480 (~68.2M Parameters)**.
* **Hidden Dimension ($d_{\text{model}}$):** 576
* **Intermediate Dimension ($d_{\text{ffn}}$):** 2,304 (SwiGLU activation)
* **Transformer Layers:** 10
* **Attention Heads:** 8
* **Key-Value Heads:** 8
* **Context Length ($L_{\text{ctx}}$):** 256 tokens
* **Quantization Format:** INT4 (Affine symmetric per-group quantization, **34.12 MB binary**)
* **Vocabulary Size:** 16,000 Byte-level BPE tokens (optimized for Bengali Unicode)
