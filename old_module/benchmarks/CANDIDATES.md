# Candidate Model Registry

**Document Purpose:** Registry of small open-weight language models investigated for the SS Tutor BD core engine.  
**Selection Status:** No model is selected or disqualified. All candidate entries serve as targets for empirical evaluation.  
**Rule:** Multilingual claims from model cards are NOT accepted as proof of Bengali competence without experimental testing.

---

## 1. Candidate Comparison Matrix

| Candidate ID | Model Identifier | Parameter Count | Native Architecture | Max Context Window | Official License | GGUF Support | Est. Q4 Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAND-01** | Qwen2.5-0.5B-Instruct | 0.49B (~490M) | Transformer (RoPE, GQA) | 32,768 tokens | Apache 2.0 | Native (`llama.cpp`) | ~350 MB |
| **CAND-02** | Qwen2.5-1.5B-Instruct | 1.54B | Transformer (RoPE, GQA) | 32,768 tokens | Apache 2.0 | Native (`llama.cpp`) | ~950 MB |
| **CAND-03** | SmolLM2-135M-Instruct | 135M | Transformer (Llama-like) | 2,048 tokens | Apache 2.0 | Native (`llama.cpp`) | ~100 MB |
| **CAND-04** | SmolLM2-360M-Instruct | 360M | Transformer (Llama-like) | 2,048 tokens | Apache 2.0 | Native (`llama.cpp`) | ~230 MB |
| **CAND-05** | SmolLM2-1.7B-Instruct | 1.71B | Transformer (GQA, RoPE) | 8,192 tokens | Apache 2.0 | Native (`llama.cpp`) | ~1,050 MB |
| **CAND-06** | Llama-3.2-1B-Instruct | 1.23B | Transformer (GQA, RoPE) | 131,072 tokens | Llama 3.2 Community | Native (`llama.cpp`) | ~750 MB |
| **CAND-07** | TinyLlama-1.1B-Chat-v1.0 | 1.10B | Llama-2 Architecture | 2,048 tokens | Apache 2.0 | Native (`llama.cpp`) | ~670 MB |
| **CAND-08** | Gemma-2-2B-IT | 2.61B | Transformer (Sliding Window, GQA) | 8,192 tokens | Gemma Terms of Use | Native (`llama.cpp`) | ~1,600 MB |

---

## 2. Detailed Candidate Profiles

### Candidate 01: Qwen2.5-0.5B-Instruct
* **Model Name:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Parameter Count:** 0.49 Billion
* **Architecture:** Causal Language Model (GQA, RMSNorm, SwiGLU, RoPE)
* **Context Length:** 32,768 tokens (recommended 4K–8K on edge)
* **Tokenizer Information:** Tiktoken-based BPE vocabulary (~151,643 tokens). Known to allocate dedicated subwords for Indic/Bengali scripts.
* **Bengali / Multilingual Support:** Claims multilingual pretraining.  
  *Status:* `UNKNOWN — REQUIRES TEST` (Must verify if 0.5B retains Bengali grammar coherence without code-mixing).
* **Instruction-Tuned Variant:** Yes (`Qwen2.5-0.5B-Instruct` official).
* **GGUF / llama.cpp Compatibility:** Verified native in `llama.cpp` upstream.
* **Android Feasibility:** **High**. At ~350MB INT4, fits well within the 600MB RAM budget.
* **Quantization Options:** `Q4_K_M`, `Q3_K_M`, `IQ3_XXS`, `Q2_K`.
* **License & Redistribution:** **Apache 2.0**. Commercial use and offline bundling permitted.
* **Official Repository:** `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`
* **Known / Suspected Limitations:** Small parameter count may cause arithmetic errors and hallucinations in complex multi-step algebra unless assisted by RAG context.

---

### Candidate 02: Qwen2.5-1.5B-Instruct
* **Model Name:** `Qwen/Qwen2.5-1.5B-Instruct`
* **Parameter Count:** 1.54 Billion
* **Architecture:** Causal Language Model (GQA, RMSNorm, SwiGLU, RoPE)
* **Context Length:** 32,768 tokens
* **Tokenizer Information:** 151,643 tokens BPE vocabulary.
* **Bengali / Multilingual Support:** High multilingual benchmark performance reported by vendor.  
  *Status:* `UNKNOWN — REQUIRES TEST` (Must test Bengali reasoning and token generation speed).
* **Instruction-Tuned Variant:** Yes (`Qwen2.5-1.5B-Instruct`).
* **GGUF / llama.cpp Compatibility:** Verified native.
* **Android Feasibility:** **Borderline for 2GB RAM**. Q4 requires ~950MB RAM + KV cache, which may push beyond safe 2GB RAM limits unless aggressive 3-bit (`Q3_K_S` / `IQ3_XXS` ~650MB) quantization is used.
* **Quantization Options:** `Q4_K_M`, `Q3_K_M`, `IQ3_XS`, `IQ2_M`.
* **License & Redistribution:** **Apache 2.0**.
* **Official Repository:** `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`
* **Known / Suspected Limitations:** May exceed memory limits on 2GB devices when foreground apps or system UI consume memory.

---

### Candidate 03: SmolLM2-135M-Instruct
* **Model Name:** `HuggingFaceTB/SmolLM2-135M-Instruct`
* **Parameter Count:** 135 Million
* **Architecture:** Llama-based compact architecture
* **Context Length:** 2,048 tokens
* **Tokenizer Information:** 49,152 vocab size (derived from Cosmos tokenizer).
* **Bengali / Multilingual Support:** Primarily trained on English/multilingual web data.  
  *Status:* `UNKNOWN — REQUIRES TEST` (High risk of poor Bengali token representation and broken grammar).
* **Instruction-Tuned Variant:** Yes (`SmolLM2-135M-Instruct`).
* **GGUF / llama.cpp Compatibility:** Native support.
* **Android Feasibility:** **Extremely High** (~100MB RAM footprint).
* **Quantization Options:** `Q4_K_M`, `Q8_0`, `FP16`.
* **License & Redistribution:** **Apache 2.0**.
* **Official Repository:** `https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct`
* **Known / Suspected Limitations:** Extreme risk of insufficient reasoning for high-school mathematics and high Bengali tokenization penalty.

---

### Candidate 04: SmolLM2-360M-Instruct
* **Model Name:** `HuggingFaceTB/SmolLM2-360M-Instruct`
* **Parameter Count:** 360 Million
* **Architecture:** Llama-based compact transformer
* **Context Length:** 2,048 tokens
* **Tokenizer Information:** 49,152 vocab size.
* **Bengali / Multilingual Support:** `UNKNOWN — REQUIRES TEST`.
* **Instruction-Tuned Variant:** Yes.
* **GGUF / llama.cpp Compatibility:** Native.
* **Android Feasibility:** **Very High** (~230MB RAM footprint).
* **Quantization Options:** `Q4_K_M`, `Q5_K_M`.
* **License & Redistribution:** **Apache 2.0**.
* **Official Repository:** `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct`
* **Known / Suspected Limitations:** May struggle with complex Bengali sentence structures.

---

### Candidate 05: SmolLM2-1.7B-Instruct
* **Model Name:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
* **Parameter Count:** 1.71 Billion
* **Architecture:** Llama-derived Transformer
* **Context Length:** 8,192 tokens
* **Tokenizer Information:** 49,152 vocab size.
* **Bengali / Multilingual Support:** `UNKNOWN — REQUIRES TEST`.
* **Instruction-Tuned Variant:** Yes.
* **GGUF / llama.cpp Compatibility:** Native.
* **Android Feasibility:** **Moderate / Low for 2GB RAM** (~1.05 GB INT4).
* **Quantization Options:** `Q4_K_M`, `Q3_K_M`, `IQ2_M`.
* **License & Redistribution:** **Apache 2.0**.
* **Official Repository:** `https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct`
* **Known / Suspected Limitations:** Vocabulary size (49K) may lead to high fragmentation of Bengali unicode text compared to Qwen's 151K vocab.

---

### Candidate 06: Llama-3.2-1B-Instruct
* **Model Name:** `meta-llama/Llama-3.2-1B-Instruct`
* **Parameter Count:** 1.23 Billion
* **Architecture:** Meta Llama 3.2 Transformer (GQA, RoPE)
* **Context Length:** 131,072 tokens
* **Tokenizer Information:** 128,256 vocab size (tiktoken).
* **Bengali / Multilingual Support:** Official support lists English, German, French, Italian, Portuguese, Hindi, Spanish, Thai. Bengali is not officially listed in primary evaluation.  
  *Status:* `UNKNOWN — REQUIRES TEST`.
* **Instruction-Tuned Variant:** Yes.
* **GGUF / llama.cpp Compatibility:** Native.
* **Android Feasibility:** **Moderate** (~750MB INT4).
* **Quantization Options:** `Q4_K_M`, `Q3_K_M`.
* **License & Redistribution:** **Llama 3.2 Community License** (Requires compliance with Meta commercial thresholds and redistribution notices).
* **Official Repository:** `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`
* **Known / Suspected Limitations:** License terms require explicit user acceptance and have specific redistribution terms; Bengali capability unverified.

---

### Candidate 07: TinyLlama-1.1B-Chat-v1.0
* **Model Name:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Parameter Count:** 1.10 Billion
* **Architecture:** Llama-2 Architecture (32k vocab)
* **Context Length:** 2,048 tokens
* **Tokenizer Information:** Standard LLaMA tokenizer (32,000 vocab). Severe byte-fallback for Bengali unicode.
* **Bengali / Multilingual Support:** Primarily English pretraining.  
  *Status:* `UNKNOWN — REQUIRES TEST` (Expected high token/word ratio for Bengali).
* **Instruction-Tuned Variant:** Yes.
* **GGUF / llama.cpp Compatibility:** Native.
* **Android Feasibility:** **High** (~670MB INT4).
* **Quantization Options:** `Q4_K_M`, `Q5_K_M`.
* **License & Redistribution:** **Apache 2.0**.
* **Official Repository:** `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Known / Suspected Limitations:** Small vocabulary causes massive token explosion on Bengali text (e.g. 4–8 tokens per word), dramatically slowing inference.

---

### Candidate 08: Gemma-2-2B-IT
* **Model Name:** `google/gemma-2-2b-it`
* **Parameter Count:** 2.61 Billion
* **Architecture:** Gemma-2 Architecture (Sliding Window Attention, Logit Soft-Capping)
* **Context Length:** 8,192 tokens
* **Tokenizer Information:** 256,000 vocab size.
* **Bengali / Multilingual Support:** Trained on rich multilingual data.  
  *Status:* `UNKNOWN — REQUIRES TEST`.
* **Instruction-Tuned Variant:** Yes.
* **GGUF / llama.cpp Compatibility:** Supported.
* **Android Feasibility:** **Low for 2GB RAM** (Model is 2.6B params; even Q3_K_M requires >1.2GB RAM). Fits only 3GB+ RAM devices.
* **Quantization Options:** `Q4_K_M`, `Q3_K_M`, `Q2_K`.
* **License & Redistribution:** **Gemma Terms of Use**.
* **Official Repository:** `https://huggingface.co/google/gemma-2-2b-it`
* **Known / Suspected Limitations:** Exceeds memory safety margin on 2GB RAM devices unless used as a teacher model.
