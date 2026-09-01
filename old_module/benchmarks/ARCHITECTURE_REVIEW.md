# Architectural Review & Assumption Audit

**Document Version:** 1.0.0  
**Target Document Reviewed:** `ARCHITECTURE.md`  
**Purpose:** Systematically audit all technical claims, targets, and assumptions in the initial architecture document to separate verified engineering facts from experimental hypotheses.

---

## 1. Summary of Architectural Audit

The initial `ARCHITECTURE.md` establishes a strong, decoupled foundation (separating core runtime from modular `.ssp` knowledge packs). However, several performance targets, runtime assumptions, and NLP claims must be formally classified as **Experimental Hypotheses** rather than confirmed baselines until proven in benchmarks.

---

## 2. Detailed Assumption Audit & Risk Analysis

### A. Model Parameter Range & 2GB RAM Feasibility
* **Original Claim in Architecture:** Models in the range of $0.5\text{B}\text{–}1.8\text{B}$ parameters can comfortably operate within a $600\text{–}750\text{ MB}$ RAM budget on 2GB Android devices.
* **Audit Finding / Risk:**  
  * While a 0.5B model (e.g., Qwen2.5-0.5B INT4) requires $\approx 350\text{ MB}$ and easily fits within the budget, a **1.7B–1.8B model** (e.g., SmolLM2-1.7B INT4) requires $\ge 1.05\text{ GB}$ of model memory alone.
  * Adding KV cache ($\approx 80\text{ MB}$) and native heap ($\approx 120\text{ MB}$) puts a 1.7B model at $\approx 1.25\text{ GB}$ RSS, which **will trigger Android Low Memory Killer (LMK) termination on a 2GB device**.
* **Required Realignment:** The upper ceiling for 2GB devices must be realistically capped at **$\le 1.0\text{B}$ parameters** (or require 3-bit / 2-bit quantization). Models $> 1.0\text{B}$ must be marked as targeting $\ge 3\text{GB}\text{–}4\text{GB}$ RAM tier devices.

---

### B. Inference Generation Throughput ($4\text{ tokens/sec}$ on Low-End CPU)
* **Original Claim in Architecture:** $\ge 4.0\text{ tokens/sec}$ generation speed on low-end ARM CPUs (e.g., Cortex-A53 @ 1.4GHz).
* **Audit Finding / Risk:**  
  * Cortex-A53 lacks native ARMv8.2-A dot-product instructions. Autoregressive token generation on a 1.5B model on 2 threads of Cortex-A53 may drop to $1.5\text{–}2.5\text{ tok/s}$.
  * A 0.5B model may achieve $4\text{–}7\text{ tok/s}$, but this requires empirical benchmarking on actual hardware.
* **Status:** Must remain classified as `TARGET — REQUIRING BENCHMARK VALIDATION`.

---

### C. Bengali Tokenization Efficiency
* **Original Claim in Architecture:** Small models have adequate Bengali tokenization efficiency ($\le 1.80\text{ tokens/word}$).
* **Audit Finding / Risk:**  
  * Tokenizers with small vocabularies (e.g., 32K in TinyLlama/LLaMA-2 or 49K in SmolLM2) often fragment single Bengali words into 3 to 6 subword tokens or byte tokens.
  * Qwen2.5 features a 151K vocabulary which may perform significantly better, but exact tokens-per-word ratios across NCTB textbook corpora have not yet been measured.
* **Status:** Must remain classified as `UNKNOWN — PENDING TOKENIZER BENCHMARK`.

---

### D. SQLite FTS5 Bengali Stemming & Search Capabilities
* **Original Claim in Architecture:** SQLite FTS5 provides fast, stemmed Bengali full-text search with $< 15\text{ ms}$ latency.
* **Audit Finding / Risk:**  
  * Standard SQLite FTS5 ships with simple ASCII / Porter tokenizers designed for English. It does **not** include native Bengali morphological stemming or sandhi/compound-word splitting out of the box.
  * Without a custom Bengali tokenizer or unicode word boundary normalizer, FTS5 queries matching inflected Bengali words (e.g., `"সূচকের"`, `"সূচককে"`, `"সূচকগুলো"`) will fail to match the base lemma `"সূচক"`.
* **Required Realignment:** The architecture must incorporate a lightweight Bengali character/suffix normalizer module before feeding queries into SQLite FTS5.

---

### E. Micro-Embedding Semantic Search Feasibility
* **Original Claim in Architecture:** An optional $\le 20\text{M}$ parameter micro-embedding model running in $< 25\text{ MB}$ RAM will serve as a Tier-3 fallback.
* **Audit Finding / Risk:**  
  * Highly compressed multilingual embedding models under 20M parameters rarely have strong representation for Bengali academic semantics.
  * Running an additional neural embedding model alongside the generation SLM adds memory contention.
* **Status:** Must remain classified as `OPEN RESEARCH QUESTION — DEFERRED UNTIL TIER 1/2 VALIDATED`.

---

### F. Legacy 32-Bit ARM (ARMv7-a) Support
* **Original Claim in Architecture:** SDK will support both 32-bit (armeabi-v7a) and 64-bit (arm64-v8a).
* **Audit Finding / Risk:**  
  * 32-bit processes have a virtual address space limit of 2GB–3GB. Memory mapping a large model file alongside heap buffers can lead to virtual address fragmentation on ARMv7.
  * Modern `llama.cpp` and PyTorch ExecuTorch optimizations heavily favor ARM64-v8a (64-bit NEON and FP16 vector extensions).
* **Required Realignment:** `arm64-v8a` must be the primary Tier-1 supported architecture. `armeabi-v7a` should be treated as best-effort Tier-2 legacy support.

---

### G. Battery & Thermal Impact Targets
* **Original Claim in Architecture:** $< 3\%$ battery drain per 30-minute session.
* **Audit Finding / Risk:**  
  * Continuous multi-turn inference keeping 2–4 CPU cores pinned at 100% load on a 3000mAh battery can draw significant power ($> 5\text{–}8\%$ per 30m).
* **Required Realignment:** Battery drain is highly dependent on student query frequency. Realistic benchmarking must measure *Joules per 100 generated tokens* rather than arbitrary session time.

---

## 3. Recommended Adjustments for Future Architecture Revisions

1. **Explicit Parameter Tiering:**
   * **Tier A (Budget 2GB RAM Devices):** $0.3\text{B}\text{–}0.6\text{B}$ parameters (e.g., Qwen2.5-0.5B, SmolLM2-360M) quantized to INT4.
   * **Tier B (Standard 3GB–4GB RAM Devices):** $1.0\text{B}\text{–}1.5\text{B}$ parameters (e.g., Qwen2.5-1.5B, Llama-3.2-1B).
2. **Explicit Bengali Normalization Layer:** Insert a deterministic Bengali lemmatizer / suffix-stripping preprocessing step ahead of SQLite FTS5 indexing.
3. **Strict Gate 1 Thresholds:** Any model requiring $> 750\text{ MB}$ total operational RAM must be automatically disqualified from the 2GB device target.
