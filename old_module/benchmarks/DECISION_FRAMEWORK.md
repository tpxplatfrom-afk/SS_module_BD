# Model Selection & Decision Framework: SS Tutor BD

**Document Version:** 1.0.0  
**Status:** **PROPOSED CRITERIA** (Pending Experimental Validation)  
**Core Principle:** Model selection is a multi-dimensional optimization problem under strict edge hardware constraints. No candidate will be selected on raw parameter count or synthetic benchmark hype alone.

---

## 1. Two-Stage Decision Protocol

To ensure rigorous evaluation without premature optimization, candidates pass through two sequential evaluation gates:

```
                  All Registered Candidates (CAND-01 .. CAND-08)
                                        │
                                        ▼
                  +───────────────────────────────────────────+
                  |  GATE 1: Hard Disqualification Criteria   |
                  |  (Binary Pass / Fail Constraints)         |
                  +───────────────────────────────────────────+
                                        │
                           Pass ────────┴──────── Fail ──> Disqualified
                          /
                         v
                  +───────────────────────────────────────────+
                  |  GATE 2: Multi-Criteria Weighted Scoring   |
                  |  (Composite Fitness Function)             |
                  +───────────────────────────────────────────+
                                        │
                                        v
                          Candidate Ranking & Trade-off
                              Architectural Decision
```

---

## 2. Gate 1: Hard Disqualification Criteria (Pass/Fail)

A candidate is **immediately disqualified** if it violates any of the following non-negotiable boundaries:

1. **Permissive Redistribution License:** Must possess an Apache-2.0, MIT, or compatible commercial/offline-redistributable open-weight license without restrictive revenue caps or proprietary runtime lock-in.
2. **2GB Android Memory Ceiling:** Peak resident memory (Model + Runtime Heap + 2K KV Cache) must not exceed **$750\text{ MB}$** under any standard quantization format (`Q4_K_M`, `Q3_K_M`, `IQ3_XXS`). Models exceeding $750\text{ MB}$ will trigger the Android Low Memory Killer (LMK) on 2GB devices.
3. **Bengali Script Generation Minimum:** The model must be capable of generating valid Bengali unicode without catastrophic character corruption or byte-level fallback loops.
4. **Mobile Runtime Support:** Must compile and execute deterministically on ARM64-v8a and ARMv7-a via an open-source runtime (`llama.cpp`, ExecuTorch, or ONNX Runtime).

---

## 3. Gate 2: Weighted Multi-Criteria Evaluation Matrix

*(Note: All weights are marked **`PROPOSED`** and subject to calibration after Phase 1 data collection).*

| Dimension | Proposed Weight | Primary Evaluation Focus | Justification |
| :--- | :--- | :--- | :--- |
| **1. Bengali Linguistic Quality** | **20%** (`PROPOSED`) | Grammatical coherence, vocabulary depth, natural dialogue flow in Bengali | The tutor's primary interaction medium is Bengali; broken grammar alienates students. |
| **2. Bengali Token Efficiency** | **15%** (`PROPOSED`) | Token-to-word expansion ratio ($\le 1.80\text{ tok/word}$) | High token expansion multiplies prompt prefill latency and cuts effective context window. |
| **3. Memory Footprint (RAM)** | **15%** (`PROPOSED`) | Peak RSS under 2048-token context | Vital for survival on 2GB physical RAM devices. |
| **4. Instruction & Constraint Following** | **10%** (`PROPOSED`) | Strict negative constraint adherence (withholding final answers when asked for hints, following JSON/bullet formats) | Crucial for pedagogical control and preventing answer dumping. |
| **5. Mathematical Reasoning (with RAG)** | **10%** (`PROPOSED`) | Step correctness and formula execution when grounded in retrieved textbook context | The model must correctly explain mathematical derivations without computational drift. |
| **6. Pedagogical Tutor Behavior** | **10%** (`PROPOSED`) | Socratic scaffolding, empathy, error diagnosis, intuitive analogy generation | Differentiates a tutor from a static search engine. |
| **7. Inference Generation Speed** | **10%** (`PROPOSED`) | Autoregressive generation throughput on low-end ARM CPU | Speed below $4\text{ tok/s}$ results in unacceptable user wait times. |
| **8. Quantization Robustness** | **5%** (`PROPOSED`) | Perplexity retention when stepping from Q8 $\rightarrow$ Q4 $\rightarrow$ Q3 | Determines how low we can compress weights without language degradation. |
| **9. Model Storage Footprint** | **5%** (`PROPOSED`) | File size on device disk ($\le 450\text{ MB}$) | Crucial for 16GB total storage devices. |

---

## 4. Architectural Trade-Off Philosophies

### Trade-Off Scenario A: Small Model (0.5B) vs Larger Model (1.5B)
* **The Dilemma:** A 0.5B model easily satisfies RAM ($\approx 350\text{MB}$) and speed ($\approx 15\text{ tok/s}$), but has weaker standalone mathematical derivation. A 1.5B model has superior intrinsic math capabilities, but pushes RAM ($\approx 950\text{MB}$) to the threshold of Android LMK termination.
* **Architectural Principle:** **Prioritize Grounded Retrieval over Memorized Neural Reasoning.** If a 0.5B model can reliably follow step-by-step derivation injected from the `.ssp` knowledge pack (Tier-1 RAG), we favor the 0.5B model for its guaranteed stability on 2GB devices.

### Trade-Off Scenario B: Low Token-to-Word Ratio vs High Token Expansion
* **The Dilemma:** Model A has a 151K vocabulary with $1.35\text{ tokens/word}$ for Bengali. Model B has a 32K vocabulary with $4.5\text{ tokens/word}$ for Bengali.
* **Architectural Principle:** Model B is effectively generating $3.3\times$ more tokens for the exact same sentence, rendering it $3.3\times$ slower in practical perceived reading speed, even if its raw token/second counter is identical. Token efficiency is a primary driver of perceived user latency.
