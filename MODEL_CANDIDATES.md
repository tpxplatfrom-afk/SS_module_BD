# Candidate Model Registry: SS Tutor BD

**Document Version:** 1.0.0  
**Purpose:** Formal registry and tracking ledger of all candidate open-weight language models investigated for SS Tutor BD.  
**Lifecycle Statuses:**
* `UNKNOWN` — Information pending initial research.
* `REQUIRES VERIFICATION` — Claim identified; primary-source legal/technical verification needed.
* `LICENSE PASSED` — License verified; permits modification, redistribution, commercial use, and offline bundling.
* `TECHNICAL TEST PENDING` — Ready for empirical benchmarking (tokenizer, RAM, latency, reasoning).
* `BENCHMARKED` — Empirical benchmark completed and logged.
* `REJECTED` — Disqualified due to license violation, excessive RAM, or quality failure.
* `SELECTED` — Approved as a production/prototype engine.

---

## 1. Candidate Master Registry Table

| Candidate ID | Model Identifier | Publisher | Params | Context Length | License Type | License Gate | Technical Gate | Final Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAND-01** | `Qwen2.5-0.5B-Instruct` | Alibaba Cloud | 0.49B | 32,768 | Apache 2.0 | **PASSED** | PENDING | `TECHNICAL TEST PENDING` |
| **CAND-02** | `Qwen2.5-1.5B-Instruct` | Alibaba Cloud | 1.54B | 32,768 | Apache 2.0 | **PASSED** | PENDING | `TECHNICAL TEST PENDING` |
| **CAND-03** | `SmolLM2-135M-Instruct` | Hugging Face | 0.13B | 2,048 | Apache 2.0 | **PASSED** | PENDING | `TECHNICAL TEST PENDING` |
| **CAND-04** | `SmolLM2-360M-Instruct` | Hugging Face | 0.36B | 2,048 | Apache 2.0 | **PASSED** | PENDING | `TECHNICAL TEST PENDING` |
| **CAND-05** | `SmolLM2-1.7B-Instruct` | Hugging Face | 1.71B | 8,192 | Apache 2.0 | **PASSED** | PENDING | `TECHNICAL TEST PENDING` |
| **CAND-06** | `Llama-3.2-1B-Instruct` | Meta AI | 1.23B | 131,072 | Llama 3.2 Community | `REQUIRES VERIFICATION` | PENDING | `REQUIRES VERIFICATION` |
| **CAND-07** | `TinyLlama-1.1B-Chat-v1.0` | TinyLlama Project | 1.10B | 2,048 | Apache 2.0 | **PASSED** | PENDING | `TECHNICAL TEST PENDING` |
| **CAND-08** | `gemma-2-2b-it` | Google DeepMind | 2.61B | 8,192 | Gemma Terms of Use | `REQUIRES VERIFICATION` | PENDING | `REQUIRES VERIFICATION` |
| **CAND-09** *(Teacher)* | `Qwen2.5-7B-Instruct` | Alibaba Cloud | 7.61B | 131,072 | Apache 2.0 | **PASSED** | PENDING (Offline) | `TECHNICAL TEST PENDING` |
| **CAND-10** *(Teacher)* | `Llama-3.1-8B-Instruct` | Meta AI | 8.03B | 131,072 | Llama 3.1 Community | `REQUIRES VERIFICATION` | PENDING (Offline) | `REQUIRES VERIFICATION` |

---

## 2. Detailed Candidate Profiles

### Candidate 01: Qwen2.5-0.5B-Instruct
* **Model Name:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Publisher:** Alibaba Cloud / Qwen Team
* **Parameter Count:** 0.49 Billion (~490 Million)
* **Architecture:** Causal LM with RoPE, GQA, RMSNorm, SwiGLU
* **Context Length:** 32,768 tokens (configurable to 2K/4K on mobile)
* **Tokenizer Information:** Tiktoken BPE tokenizer with ~151,643 vocabulary size. Dedicated Indic/Bengali subwords.
* **License:** **Apache 2.0** (Commercial use, modification, redistribution, and offline bundling permitted).
* **Official Model Card:** `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`
* **Quantized Availability:** Pre-quantized GGUF (`Q4_K_M`, `Q3_K_M`, `IQ3_XXS`, `Q8_0`) readily available.
* **GGUF Compatibility:** Fully supported in upstream `llama.cpp`.
* **Known Bengali Capability:** Model card claims strong multilingual support. Exact token expansion ratio on NCTB Bengali: `UNKNOWN — REQUIRES TEST`.
* **Expected Mobile Suitability:** **Very High**. At ~350 MB INT4, easily satisfies the $\le 650\text{ MB}$ RAM budget on 2GB devices.
* **Status:** `TECHNICAL TEST PENDING`

---

### Candidate 02: Qwen2.5-1.5B-Instruct
* **Model Name:** `Qwen/Qwen2.5-1.5B-Instruct`
* **Publisher:** Alibaba Cloud / Qwen Team
* **Parameter Count:** 1.54 Billion
* **Architecture:** Causal LM with RoPE, GQA, RMSNorm, SwiGLU
* **Context Length:** 32,768 tokens
* **Tokenizer Information:** 151,643 vocabulary size.
* **License:** **Apache 2.0**
* **Official Model Card:** `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`
* **Quantized Availability:** Readily available in GGUF (`Q4_K_M`, `Q3_K_M`, `IQ3_XS`).
* **GGUF Compatibility:** Fully supported in upstream `llama.cpp`.
* **Known Bengali Capability:** Vendor reports high multilingual reasoning. Real-world Bengali tutoring quality: `UNKNOWN — REQUIRES TEST`.
* **Expected Mobile Suitability:** **Moderate / Tight for 2GB RAM**. Q4 footprint (~950 MB) risks triggering Android LMK unless quantized to 3-bit (`Q3_K_S` ~650 MB).
* **Status:** `TECHNICAL TEST PENDING`

---

### Candidate 03: SmolLM2-135M-Instruct
* **Model Name:** `HuggingFaceTB/SmolLM2-135M-Instruct`
* **Publisher:** Hugging Face
* **Parameter Count:** 0.135 Billion (135 Million)
* **Architecture:** Compact Llama-derived Transformer
* **Context Length:** 2,048 tokens
* **Tokenizer Information:** 49,152 vocabulary size (Cosmos-derived).
* **License:** **Apache 2.0**
* **Official Model Card:** `https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct`
* **Quantized Availability:** GGUF (`Q4_K_M`, `Q8_0`, `FP16`).
* **GGUF Compatibility:** Native `llama.cpp` support.
* **Known Bengali Capability:** `UNKNOWN — REQUIRES TEST` (High risk of severe subword fragmentation due to 49K vocab).
* **Expected Mobile Suitability:** **Extremely High** (~100 MB RAM footprint).
* **Status:** `TECHNICAL TEST PENDING`

---

### Candidate 04: SmolLM2-360M-Instruct
* **Model Name:** `HuggingFaceTB/SmolLM2-360M-Instruct`
* **Publisher:** Hugging Face
* **Parameter Count:** 0.36 Billion (360 Million)
* **Architecture:** Compact Llama-derived Transformer
* **Context Length:** 2,048 tokens
* **Tokenizer Information:** 49,152 vocabulary size.
* **License:** **Apache 2.0**
* **Official Model Card:** `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct`
* **Quantized Availability:** GGUF (`Q4_K_M`, `Q8_0`).
* **GGUF Compatibility:** Native `llama.cpp` support.
* **Known Bengali Capability:** `UNKNOWN — REQUIRES TEST`.
* **Expected Mobile Suitability:** **Very High** (~230 MB RAM footprint).
* **Status:** `TECHNICAL TEST PENDING`

---

### Candidate 05: SmolLM2-1.7B-Instruct
* **Model Name:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
* **Publisher:** Hugging Face
* **Parameter Count:** 1.71 Billion
* **Architecture:** Llama-derived Transformer with GQA
* **Context Length:** 8,192 tokens
* **Tokenizer Information:** 49,152 vocabulary size.
* **License:** **Apache 2.0**
* **Official Model Card:** `https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct`
* **Quantized Availability:** GGUF (`Q4_K_M`, `Q3_K_M`, `IQ2_M`).
* **GGUF Compatibility:** Native `llama.cpp` support.
* **Known Bengali Capability:** `UNKNOWN — REQUIRES TEST`.
* **Expected Mobile Suitability:** **Low for 2GB RAM** (~1.05 GB INT4); intended for 3GB+ RAM tier devices.
* **Status:** `TECHNICAL TEST PENDING`

---

### Candidate 06: Llama-3.2-1B-Instruct
* **Model Name:** `meta-llama/Llama-3.2-1B-Instruct`
* **Publisher:** Meta AI
* **Parameter Count:** 1.23 Billion
* **Architecture:** Llama 3.2 Transformer (GQA, RoPE)
* **Context Length:** 131,072 tokens
* **Tokenizer Information:** 128,256 vocabulary size.
* **License:** **Llama 3.2 Community License** (Commercial use permitted under 700M monthly active users; requires "Built with Llama" attribution and downstream distribution compliance).
* **Official Model Card:** `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`
* **Quantized Availability:** GGUF formats available.
* **GGUF Compatibility:** Native `llama.cpp` support.
* **Known Bengali Capability:** Bengali is not officially listed in Meta's primary language evaluation table. `UNKNOWN — REQUIRES TEST`.
* **Expected Mobile Suitability:** **Moderate** (~750 MB INT4).
* **Status:** `REQUIRES VERIFICATION` (Must confirm offline embedded redistribution terms).

---

### Candidate 07: TinyLlama-1.1B-Chat-v1.0
* **Model Name:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Publisher:** TinyLlama Project / Singapore University of Technology and Design
* **Parameter Count:** 1.10 Billion
* **Architecture:** Llama-2 Architecture
* **Context Length:** 2,048 tokens
* **Tokenizer Information:** 32,000 vocabulary size.
* **License:** **Apache 2.0**
* **Official Model Card:** `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Quantized Availability:** GGUF (`Q4_K_M`, `Q5_K_M`).
* **GGUF Compatibility:** Native `llama.cpp` support.
* **Known Bengali Capability:** `UNKNOWN — REQUIRES TEST` (Expected high token/word expansion ratio due to 32K vocab).
* **Expected Mobile Suitability:** **Moderate** (~670 MB INT4).
* **Status:** `TECHNICAL TEST PENDING`

---

### Candidate 08: Gemma-2-2B-IT
* **Model Name:** `google/gemma-2-2b-it`
* **Publisher:** Google DeepMind
* **Parameter Count:** 2.61 Billion
* **Architecture:** Gemma-2 Architecture (Sliding Window Attention, Logit Soft-Capping)
* **Context Length:** 8,192 tokens
* **Tokenizer Information:** 256,000 vocabulary size.
* **License:** **Gemma Terms of Use** (Requires primary source review for offline SDK distribution terms).
* **Official Model Card:** `https://huggingface.co/google/gemma-2-2b-it`
* **Quantized Availability:** GGUF (`Q4_K_M`, `Q3_K_M`, `Q2_K`).
* **GGUF Compatibility:** Native `llama.cpp` support.
* **Known Bengali Capability:** Strong multilingual representation reported. `UNKNOWN — REQUIRES TEST`.
* **Expected Mobile Suitability:** **Low for 2GB RAM** (>1.3 GB RAM). Suitable as a Tier C teacher model.
* **Status:** `REQUIRES VERIFICATION`
