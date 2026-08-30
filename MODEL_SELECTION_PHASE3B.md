# SS Tutor BD — Model Selection Specification (Phase 3B)

**Version:** 3.2.0  
**Phase:** 3B — Production Memory-Constrained Model Selection & Re-Evaluation  
**Date:** 2026-08-30  
**Target Device Profile:** Android (2 GB RAM, 16 GB Storage, ARM Cortex-A53/A55)  
**Production Process Memory Budget:** Preferred $\le 200\text{ MB}$, Hard Ceiling $\le 250\text{ MB}$  
**Development Cost:** \$0 USD  

---

## 1. Executive Summary & Core Architectural Pivot

Phase 3A demonstrated that textbook RAG grounding and Socratic prompt scaffolding dramatically improved tutoring adherence (100% grounding, 100% hint compliance). However, `CAND-01` (Qwen2.5-0.5B Q4_K_M) required **738 MB peak RSS**, which is completely unviable for a real 2 GB Android device where OS, system daemons, GPU, and user background tasks leave only ~150–250 MB for the foreground application process.

Phase 3B formally executed the transition from:
> *"Find the biggest model that fits in 2 GB"* $\longrightarrow$ **"Find the smallest model/runtime combination that provides acceptable tutoring quality inside 150–200 MB total process memory."**

---

## 2. Phase 3B Candidate Evaluation Matrix

| Candidate ID | Model Name | Parameter Count | Quantization | GGUF Size (MB) | Measured Peak RSS (MB) | Throughput (tok/s) | Hybrid Score (%) | License Gate | Production Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAND-03** | `SmolLM2-135M-Instruct` | 0.135B | `Q4_K_M` | **100.57 MB** | **315.62 MB** | **19.67 tok/s** | **81.5%** | `Apache-2.0` (PASS) | **DISQUALIFIED / RESEARCH ONLY** (Exceeds 250 MB) |
| **CAND-01** | `Qwen2.5-0.5B-Instruct` | 0.49B | `Q4_K_M` | 468.64 MB | **738.07 MB** | 9.94 tok/s | 68.0% | `Apache-2.0` (PASS) | **RETIRED FROM PRODUCTION** (Exceeds 250 MB) |
| **CAND-02** | `Qwen2.5-1.5B-Instruct` | 1.54B | `Q4_K_M` | 1065.56 MB | **1,771.26 MB** | 8.00 tok/s | 75.0% | `Apache-2.0` (PASS) | **DISQUALIFIED (Phase 2)** (Exceeds 250 MB) |

---

## 3. Detailed Candidate Analysis

### CAND-03: SmolLM2-135M-Instruct (Q4_K_M)
* **Strengths:** Extremely compact binary (**100.57 MB**), lightning-fast generation (**19.67 tok/s**), ultra-low cold start TTFT (**17 ms**), and excellent textbook grounding adherence (**100%**).
* **Weaknesses & Bottlenecks:**
  1. *Tokenizer Inefficiency:* SmolLM2 uses a 49K Llama-based vocabulary where Bengali characters are tokenized into byte-level UTF-8 subwords. 1 Bengali word expands to 4–8 tokens, which forces larger context lengths.
  2. *Runtime Buffer Overhead:* In llama.cpp under a 2048 context window, runtime memory buffers push peak process RSS to **315.62 MB** during multi-turn sessions (exceeding the 250 MB ceiling).
  3. *Standalone Mathematical Reasoning:* Raw LLM-only mathematical correctness is only **15.0%**, requiring the hybrid deterministic engine to reach **81.5%**.

---

## 4. Formal Model Decision

```
========================================================================
PHASE 3B DECISION: NO STANDALONE LLM MEETS THE <= 200 MB CONTRACT
========================================================================
```

**Decision Rule Compliance (Section 47):**  
In strict compliance with Section 47 of the Master Prompt, **we do NOT artificially lower the memory limit or report false compliance.**  

Neither CAND-01 (738 MB) nor CAND-03 (315 MB) qualifies as an unmodified standalone production candidate under the $\le 250\text{ MB}$ hard ceiling.

---

## 5. Phase 3C Architectural Strategy

To achieve a true $\le 150–200\text{ MB}$ total working set on Android:

1. **Adopt Pure Hybrid Architecture as Primary:**
   * Deterministic Math Engine (`core/math/`) handles 100% of numerical calculations.
   * SQLite FTS5 RAG (`core/rag/`) handles 100% of curriculum fact retrieval (<15 MB RAM).
2. **Deploy Specialized Micro-Runtimes / ONNX Runtime:**
   * Evaluate ONNX Runtime / ExecuTorch with fixed KV cache allocations constrained to $\le 30\text{ MB}$.
   * Use dedicated Bengali-vocab fine-tuned micro-models (100M parameters with intact Indic tokenization to eliminate byte expansion).
