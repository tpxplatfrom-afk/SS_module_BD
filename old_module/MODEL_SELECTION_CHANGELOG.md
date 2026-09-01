# Model Selection Changelog & Rationale: SS Tutor BD

**Document Version:** 1.0.0  
**Date:** 2026-08-30  
**Phase:** Pre-Benchmarking Research & Framework Definition

---

## 1. Rationale for Dedicated Model Selection Framework

### 1.1 Why the Model-Selection Phase Was Introduced
* **Problem:** In low-resource edge AI development, teams often make the catastrophic mistake of selecting a popular foundation model (e.g. Llama-3 8B or Gemma-2 9B) and attempting to shoehorn it onto budget hardware, or conversely selecting an ultra-tiny 100M model that lacks the capacity for coherent algebraic reasoning.
* **Solution:** A dedicated research framework ensures that candidate models are systematically evaluated against empirical criteria (Bengali token efficiency, RAM headroom, arithmetic precision, and licensing compliance) before committing to a development pipeline.

---

## 2. Key Decisions & Strategic Rationale

### 2.1 Why License Audit (Gate 0) Precedes Technical Benchmarking
* **Rationale:** A model with state-of-the-art Bengali fluency is completely useless for SS Tutor BD if its license prohibits offline embedding, commercial redistribution in an SDK, or downstream application packaging without restrictive per-seat licensing.
* Conducting the legal audit first eliminates invalid candidates with zero wasted compute or bandwidth.

### 2.2 Why 0.3B–1.5B Parameters Was Chosen as the Primary Search Range (Tier A)
* **The Lower Bound (0.3B):** Models under 300M parameters generally lack the transformer depth required for multi-step reasoning, Socratic dialogue management, and structured Bengali grammar.
* **The Upper Bound (1.5B):** On a 2GB physical RAM Android device, total available foreground memory is constrained to $\le 600\text{–}750\text{ MB}$. A 1.5B model quantized to INT4 / INT3 represents the absolute theoretical ceiling that can execute without triggering Android's Low Memory Killer (LMK).

### 2.3 Why the Initial Proof-of-Concept (POC) Targets Class 8 Mathematics
* **Rationale:** Class 8 Mathematics represents the critical inflection point in the Bangladesh NCTB curriculum where students transition from basic arithmetic to abstract algebraic identities ($(a+b)^2$, $(a-b)^2$), linear equations, and geometric proofs.
* If the engine can successfully parse, retrieve, scaffold, and explain Class 8 Math problems in Bengali within memory limits, the architecture is guaranteed to scale smoothly to Classes 6, 7, 9, and 10.

### 2.4 Why Only One Model Should Be Downloaded at a Time
* **Workstation Constraint:** The host development machine has approximately **4.96 GB of free disk space on Drive C:**.
* Downloading multiple multi-gigabyte models simultaneously risks depleting host storage, corrupting virtual environments, and failing builds.
* Strict download hygiene (downloading one pre-quantized candidate, benchmarking it, logging metrics, and cleaning cache if rejected) ensures zero-budget reproducibility.

### 2.5 Why Model Selection Remains Strictly Experimental
* **Status:** No model is selected or crowned as the winner.
* Every model card claim regarding "multilingual support" is treated as an unverified hypothesis until measured against our 100-item Bengali educational benchmark.
