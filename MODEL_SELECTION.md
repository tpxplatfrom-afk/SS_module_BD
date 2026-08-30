# Model Selection & Evaluation Strategy: SS Tutor BD

**Document Version:** 1.0.0  
**Status:** Approved Research Framework  
**Target Device Environment:** Android (~2GB RAM, ~16GB Internal Storage, ARM Cortex-A53/A55 CPU)  
**Execution Mode:** CPU-First, 100% Offline-Capable Inference  
**Linguistic & Domain Focus:** Bengali-First Interaction for Bangladesh High School (NCTB Class 6–10)  
**Budget Constraint:** $0 USD (Zero-Cost Open Source & Local Tooling)

---

## 1. Executive Summary & Objective

The objective of this framework is to systematically identify, audit, evaluate, and select a small open-weight language model that can serve as the primary reasoning and dialogue engine (**SS Core**) for **SS Tutor BD**.

The model must operate within severe mobile hardware constraints while delivering pedagogically sound, step-by-step educational explanations in natural Bengali without cloud dependencies.

---

## 2. Core Optimization Principle: Quality Over Raw Size

A critical architectural pitfall in edge AI is assuming that the smallest model is automatically the best model. 

> **Core Optimization Formula:**  
> $$\text{Candidate Fitness} = \text{Educational Quality} \times \text{Bengali Fluency} \times \text{Mathematical Reasoning} \times \text{Mobile Efficiency} \times \text{License Freedom}$$

```
+-------------------------------------------------------------------------+
|                  THE PARAMETER TRADEOFF REALITY                         |
|                                                                         |
|  0.5B Model (e.g. ~250MB INT4)          1.0B-1.5B Model (~600-900MB INT4)|
|  - Ultra-fast on low-end CPU            - Moderate CPU speed            |
|  - Minimal RAM footprint                - Higher RAM consumption        |
|  - RISK: Weak multi-step math reasoning - ADVANTAGE: Superior reasoning |
|  - RISK: Potential Bengali grammar loss - ADVANTAGE: Richer explanation |
|                                                                         |
|  CONCLUSION: Smallest is NOT automatically best. We must find the       |
|  SMALLEST model that passes the MINIMUM EDUCATIONAL QUALITY THRESHOLD.  |
+-------------------------------------------------------------------------+
```

---

## 3. Candidate Categorization & Search Tiers

To structure the search space, candidate models are grouped into four operational tiers:

```
                                CANDIDATE TIERS
                                       │
         ┌─────────────────┬───────────┴───────────┬─────────────────┐
         ▼                 ▼                       ▼                 ▼
   [ TIER A ]        [ TIER B ]              [ TIER C ]        [ TIER D ]
  Primary SLMs      Ultra-Low-RAM          Teacher Models    Multimodal Vision
  (0.3B - 1.5B)     (100M - 500M)           (7B - 14B)       (SmolVLM, Moondream)
  Main Candidates   Extreme Fallback       Cloud/Offline      Future Roadmap
```

### Tier A: Primary Candidates (0.3B – 1.5B Parameters)
* Models that could potentially become the main SS Tutor model deployed to standard budget Android devices ($\approx 2\text{GB}\text{–}3\text{GB}$ RAM).
* *Initial Candidates:* Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, Llama-3.2-1B-Instruct, TinyLlama-1.1B-Chat.

### Tier B: Ultra-Low-RAM Candidates (100M – 500M Parameters)
* Highly compact models investigated for extreme legacy hardware or ultra-constrained devices.
* *Initial Candidates:* SmolLM2-135M-Instruct, SmolLM2-360M-Instruct.

### Tier C: Teacher & Distillation Models (7B – 14B Parameters)
* Larger open-weight models evaluated strictly for **offline/cloud dataset synthesis and knowledge distillation** on free compute tiers (Colab/Kaggle).
* **Explicit Rule:** Tier C models will **never** be deployed to 2GB Android devices.
* *Candidates:* Qwen2.5-7B/14B-Instruct, Llama-3.1-8B-Instruct.

### Tier D: Vision & Multimodal Models (Future Roadmap)
* Lightweight vision-language models for textbook geometry diagram understanding.
* *Candidates:* SmolVLM, Moondream2. (Deferred to Phase 5+; excluded from initial model selection).

---

## 4. License-First Audit Framework

Before spending bandwidth, disk space, or compute on benchmarking, every candidate must undergo an upfront **License & Legal Audit**.

```
                        Candidate Model Card
                                 │
                                 ▼
                 +───────────────────────────────+
                 |  Primary-Source License Audit |
                 +───────────────────────────────+
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    ▼                            ▼                            ▼
Modification OK?         Redistribution OK?           Commercial Use OK?
    │                            │                            │
    └────────────────────────────┼────────────────────────────┘
                                 │
                      Pass ──────┴────── Fail ──> DISQUALIFIED
                     /
                    v
    +─────────────────────────────────────────────+
    |  Attribution & Offline Bundling Review     |
    +─────────────────────────────────────────────+
```

### Formal License Audit Matrix

| Candidate ID | Candidate Model | Parameters | Declared License | Modification Allowed | Redistribution Allowed | Commercial Use Allowed | Offline Bundling Allowed | Attribution Burden | Branding Restrictions | Primary Source | License Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAND-01** | Qwen2.5-0.5B-Instruct | 0.49B | Apache 2.0 | Yes | Yes | Yes | Yes | Standard Notice | None beyond Apache standard | Official HF Repo / LICENSE | `LICENSE PASSED` |
| **CAND-02** | Qwen2.5-1.5B-Instruct | 1.54B | Apache 2.0 | Yes | Yes | Yes | Yes | Standard Notice | None beyond Apache standard | Official HF Repo / LICENSE | `LICENSE PASSED` |
| **CAND-03** | SmolLM2-135M-Instruct | 0.13B | Apache 2.0 | Yes | Yes | Yes | Yes | Standard Notice | None | Official HF Repo / LICENSE | `LICENSE PASSED` |
| **CAND-04** | SmolLM2-360M-Instruct | 0.36B | Apache 2.0 | Yes | Yes | Yes | Yes | Standard Notice | None | Official HF Repo / LICENSE | `LICENSE PASSED` |
| **CAND-05** | SmolLM2-1.7B-Instruct | 1.71B | Apache 2.0 | Yes | Yes | Yes | Yes | Standard Notice | None | Official HF Repo / LICENSE | `LICENSE PASSED` |
| **CAND-06** | Llama-3.2-1B-Instruct | 1.23B | Llama 3.2 Community | Conditional | Conditional | Conditional ($\le 700\text{M}$ MAU) | Requires Notice | Must include "Built with Llama" | Meta Llama 3.2 License Policy | `REQUIRES PRIMARY-SOURCE VERIFICATION` |
| **CAND-07** | TinyLlama-1.1B-Chat | 1.10B | Apache 2.0 | Yes | Yes | Yes | Yes | Standard Notice | None | Official GitHub / LICENSE | `LICENSE PASSED` |
| **CAND-08** | Gemma-2-2B-IT | 2.61B | Gemma Terms of Use | Conditional | Conditional | Allowed with restrictions | Allowed | Required attribution | Google Gemma Brand Guidelines | `REQUIRES PRIMARY-SOURCE VERIFICATION` |

> [!IMPORTANT]
> **Audit Rule:** Any model with ambiguous or unverified licensing is marked `REQUIRES PRIMARY-SOURCE VERIFICATION` until its primary license text is directly audited.

---

## 5. Ownership, Branding & Adaptation Boundaries

We establish a clear legal demarcation across the project's intellectual property:

```
+-------------------------------------------------------------------------+
|                      OWNERSHIP & ATTRIBUTION MODEL                      |
|                                                                         |
|  1. SS Tutor BD Codebase & Engine                                       |
|     - Original open-source engineering under Apache 2.0 License.        |
|                                                                         |
|  2. Third-Party Base Model                                              |
|     - Third-party weights remain property of original publishers.       |
|     - SS Tutor BD makes ZERO claim of ownership over upstream weights.  |
|                                                                         |
|  3. SS Tutor Adapted Model                                              |
|     - Fine-tuned adapters / merged weights distributed under upstream   |
|       license terms with full attribution.                              |
|                                                                         |
|  4. Education Knowledge Packs (.ssp)                                    |
|     - Derived from official Bangladesh NCTB curriculum under national   |
|       educational open-access / fair-use provisions.                    |
|                                                                         |
|  5. Training & Evaluation Datasets                                      |
|     - Original pedagogical datasets created specifically for this SDK.  |
+-------------------------------------------------------------------------+
```

---

## 6. Technical Evaluation Scorecard (100-Point Framework)

Each model surviving the License Gate will be scored out of 100 points across six objective categories:

```
                       COMPOSITE SCORECARD (100 PTS)
                                     │
         ┌──────────────┬────────────┼────────────┬──────────────┐
         ▼              ▼            ▼            ▼              ▼
    [ Bengali ]    [ Reasoning ]  [ Mobile ]  [ Follow/Ctx ] [ Grounding/License ]
      20 Pts          25 Pts       20 Pts        15 Pts           20 Pts
```

| Dimension | Max Points | Evaluation Focus & Measurement Criteria |
| :--- | :--- | :--- |
| **1. Bengali Linguistic Quality** | **20** | Fluency, natural grammar, accurate mathematical/scientific terminology in Bengali, resistance to English language drift. |
| **2. Educational Reasoning** | **25** | Multi-step mathematical derivation, conceptual clarity, step-by-step scaffolding, error explanation. |
| **3. Mobile Resource Efficiency** | **20** | Quantized model size ($\le 450\text{MB}$), peak RAM footprint ($\le 650\text{MB}$), time-to-first-token ($< 1.8\text{s}$), generation throughput ($\ge 4\text{ tok/s}$). |
| **4. Instruction & Constraint Following** | **15** | Negative constraint adherence (withholding final answers when hints requested), JSON/Markdown formatting, multi-turn consistency. |
| **5. Knowledge Grounding & Anti-Hallucination** | **10** | Accuracy in synthesizing injected textbook context; refusing to invent non-existent chapters or theorems. |
| **6. License & Redistribution Freedom** | **10** | Minimal attribution burden, unrestricted offline bundling, commercial redistribution freedom. |
| **TOTAL** | **100** | **Minimum Passing Score for Primary Consideration: 70 / 100** |

---

## 7. Standardized Benchmark Dataset Specification (~100 Cases)

To provide fair, reproducible evaluation across all candidates, a structured 100-item test suite is specified:

```
+-------------------------------------------------------------------------+
|                  100-ITEM BENCHMARK TEST SUITE (TARGET)                 |
|                                                                         |
|  1. Bengali Language & Dialogue Tests:           20 Prompts             |
|     - Formal Bengali, colloquial student queries, code-mixed phrasing   |
|                                                                         |
|  2. Mathematics Reasoning (NCTB Class 6-10):     30 Prompts             |
|     - Arithmetic (5), Algebra (8), Geometry (7), Word Problems (10)     |
|                                                                         |
|  3. General Science & Concepts:                  20 Prompts             |
|     - Physics (Newton's laws, energy), Chemistry, Biology               |
|                                                                         |
|  4. Educational & Tutor Behavior:                20 Prompts             |
|     - "Explain don't just solve", progressive hints, student errors     |
|                                                                         |
|  5. Knowledge Grounding & Anti-Hallucination:    10 Prompts             |
|     - Non-existent chapters, missing variables, false premises          |
+-------------------------------------------------------------------------+
```

---

## 8. First Product Prototype Target: Class 8 Mathematics

To prevent scope explosion and maintain clear engineering milestones, the initial end-to-end proof-of-concept (POC) target is locked:

```
+─────────────────────────────────────────────────────────────────────────+
|                 INITIAL PROOF-OF-CONCEPT (POC) TARGET                   |
|                                                                         |
|                   SS Tutor BD ──> Class 8 ──> Mathematics               |
+─────────────────────────────────────────────────────────────────────────+
```

### POC Verification Criteria:
1. Deterministic coordinate matching for Class 8 Math chapters (e.g. Chapter 4: Algebraic Formulae).
2. Natural Bengali explanation of core formulas ($(a+b)^2$, $(a-b)^2$).
3. Multi-turn Socratic hint progression without leaking final answers.
4. Execution in $\le 650\text{ MB}$ RAM on CPU runtime.
5. Successful operation with WiFi / Cellular data completely disabled.

---

## 9. Model Download & Local Storage Policy

Due to host workstation storage constraints (~4.96 GB free on C:), strict storage discipline is mandatory:

```
+-------------------------------------------------------------------------+
|                      MODEL DOWNLOAD DISCIPLINE RULES                    |
|                                                                         |
|  Rule 1: Never download full-precision (FP16/FP32) multi-gigabyte models|
|  Rule 2: Download only ONE pre-quantized (GGUF INT4/INT3) candidate     |
|          at a time for active testing.                                  |
|  Rule 3: Perform license audit BEFORE any download starts.              |
|  Rule 4: Purge cached weights of rejected candidates before fetching   |
|          subsequent models.                                             |
+-------------------------------------------------------------------------+
```

---

## 10. Mobile Runtime & Quantization Test Plan

### 10.1 Runtime Hierarchy
1. **Primary Experimental Runtime:** `llama.cpp` (GGUF format) — Highest CPU optimization and mature Android NDK bindings.
2. **Alternative Runtime 1:** PyTorch ExecuTorch (`.pte` format) — Evaluated if ARM NEON kernels prove superior.
3. **Alternative Runtime 2:** ONNX Runtime Mobile — Evaluated for multi-platform consistency.

### 10.2 Quantization Stepping Matrix
Candidates will be evaluated across quantization levels to determine the optimal quality-to-RAM sweet spot:

$$\text{FP16 (Reference)} \longrightarrow \text{Q4\_K\_M (Target)} \longrightarrow \text{Q3\_K\_M (Low-RAM)} \longrightarrow \text{IQ2\_M (Extreme)}$$

We will measure the exact point where Bengali grammar or mathematical reasoning degrades significantly.

---

## 11. Final Model Selection Protocol (Sequential Gates)

A candidate must successfully clear six sequential gates to become the default SS Core model:

```
[ Candidate Model ]
         │
         ▼
[ 1. LICENSE GATE ] ──────── Fail ──> Disqualify
         │ Pass
         ▼
[ 2. BENGALI LINGUISTIC GATE ] ── Fail ──> Disqualify (Broken script / High token ratio)
         │ Pass
         ▼
[ 3. EDUCATIONAL REASONING GATE ] ── Fail ──> Disqualify (Hallucinated math)
         │ Pass
         ▼
[ 4. MOBILE MEMORY GATE ] ── Fail ──> Disqualify (Peak RSS > 750MB)
         │ Pass
         ▼
[ 5. MOBILE SPEED GATE ] ── Fail ──> Disqualify (Speed < 4.0 tok/s on ARM CPU)
         │ Pass
         ▼
[ 6. REAL SS TUTOR BENCHMARK ] ──> Score >= 70/100 ──> PRIMARY CANDIDATE
```
