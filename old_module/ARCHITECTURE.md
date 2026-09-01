# Architecture Specification: SS Tutor BD

**Document Version:** 1.1.0  
**Status:** Confirmed Architecture Baseline (Revision v1.1)  
**Target Domain:** Bangladesh National Curriculum (NCTB) Class 6–10  
**Target Environment:** Offline-First Android Devices (Down to ~2GB RAM / ~16GB Storage)  
**Budget Constraint:** $0 USD (Zero-Budget Open Source Infrastructure)

---

## 1. Project Overview & Platform Identity

**SS Tutor BD** is a developer-friendly, offline-first **Modular AI Education Platform** and embeddable SDK tailored specifically for the Bangladesh National Curriculum and Textbook Board (NCTB) curriculum (Classes 6 to 10).

SS Tutor BD is **not** a collection of independent AI models for each class or subject. It is designed as **one unified core engine** paired with **modular, swappable curriculum knowledge packs**.

```
+-----------------------------------------------------------------------------------+
|                                   SS Tutor BD                                     |
|                                                                                   |
|  +-------------------------------------+  +------------------------------------+  |
|  |               SS Core               |  |     Education Knowledge Packs      |  |
|  | - Small Language Model (SLM)        |  | - Class 6 Pack (Math, Sci, ...)   |  |
|  | - Offline Inference Runtime (C++)   |  | - Class 7 Pack (Math, Sci, ...)   |  |
|  | - Pedagogical Tutor State Machine   |  | - Class 8 Pack (Math, Sci, ...)   |  |
|  | - Context & Prompt Orchestration    |  | - Class 9-10 Pack (Math, Sci, ...) |  |
|  | - Deterministic & Hybrid Retrieval  |  |   * Granular Subject & Class Packs |  |
|  +-------------------------------------+  +------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Product Vision & Success Criteria

### 2.1 Core Vision
To democratize personalized AI tutoring across Bangladesh by providing a lightweight, plug-and-play developer module that brings high-quality, culturally attuned, curriculum-aligned, Bengali-native tutoring to every student, irrespective of internet connectivity or device cost.

### 2.2 Product Success Dimensions
The platform is considered successful only if it reasonably satisfies all three dimensions:

```
                          PRODUCT SUCCESS CRITERIA
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
  [ DEVELOPER FRIENDLY ]      [ USER FRIENDLY ]          [ TEACHING FRIENDLY ]
  - Simple 1-module API      - Simple class picker       - Step-by-step scaffolding
  - Modular dependencies     - Small downloads (20-50MB) - Socratic hints (no answer leak)
  - Strict offline operation - Fast startup (<2.5s)      - Concept analogies in Bengali
  - Zero-budget licensing    - 100% Bengali interaction  - Curriculum-grounded accuracy
```

1. **Developer-Friendly:**
   * Single unified entry-point API (`SS.init()`, `SS.install("class-8")`, `SS.ask(...)`).
   * Developers do not need to understand GGUF, quantization, vector embeddings, JNI, or tokenizer internals.
   * Modular build and asset distribution options.
2. **User-Friendly (Student-Focused):**
   * Intuitive class and subject selection.
   * Compact downloads (preferred target ~20–50 MB per module).
   * Total offline capability with zero recurring internet dependency.
   * Responsive Bengali conversational feedback on budget hardware.
3. **Teaching-Friendly (Pedagogical Excellence):**
   * Scaffolding over raw answer dumping (Socratic tutoring).
   * Progressive hint generation.
   * Error diagnosis and misconception remediation.
   * Foundation for teacher assistance and classroom content generation.

---

## 3. Goals & Non-Goals

### Goals
* **Single Reusable Engine [CONFIRMED]:** One core model powers all classes and subjects by default.
* **Decoupled Architecture [CONFIRMED]:** Knowledge modules and model weights are independently versioned, distributed, and updated.
* **Strict Offline-First Operation [CONFIRMED]:** 100% of standard retrieval, reasoning, hint generation, and Bengali dialogue executes locally on-device without cloud connectivity.
* **Low-End Android Feasibility [CONFIRMED TARGET]:** Target operational viability on devices with ~2GB RAM and ~16GB storage.
* **Modular Granularity [CONFIRMED]:** Developers can bundle a single subject, a single class, or the entire high-school curriculum.
* **Zero Infrastructure Cost [CONFIRMED]:** Built entirely on free/open-source tools, local compute, and free artifact tiers.

### Non-Goals
* **Not a Monolithic Consumer App:** We are building an SDK and engine module, not an end-user social media or gamified consumer app.
* **No Cloud-Scale Microservices:** No hosted vector databases, paid inference APIs, or subscription tiers required for core operation.
* **Not Training Foundation Models from Scratch:** We will not train multi-billion parameter models from scratch ($0 budget constraint).
* **Not a Direct Homework Solver / Answer Dumping Bot:** The system intentionally scaffolds learning rather than outputting immediate unguided solutions.
* **Not Baking Curriculum into Model Weights:** Models are not expected to memorize textbook exercises; factual grounding is retrieved from knowledge packs.

---

## 4. One Core, Multiple Knowledge Packs Architecture

```
                                  SS CORE
                      (Small Language Model + Runtime)
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
     [ Class 6 Pack ]          [ Class 8 Pack ]        [ Class 9-10 Pack ]
      ├── Mathematics           ├── Mathematics          ├── Mathematics
      ├── Science               ├── Science              ├── General Science
      └── ...                   └── English              └── ...
```

* **Default Architectural Baseline [CONFIRMED]:** **One reusable core model is the default architecture.** Multiple specialized models remain possible if empirical benchmarking proves that a single model cannot satisfy quality, memory, speed, or subject-specific requirements.

---

## 5. Model Module vs Knowledge Module Distinction

To prevent architectural coupling, model assets and educational data are strictly separated:

```
+------------------------------------------+  +------------------------------------------+
|               MODEL MODULE               |  |             KNOWLEDGE MODULE             |
|                                          |  |                                          |
| - Model Weights (Quantized GGUF/PTE)     |  | - Curriculum Hierarchy (Class/Ch/Sec)    |
| - Tokenizer Vocabulary & Chat Templates  |  | - Structured Textbook Exercise Content   |
| - Runtime Compatibility Metadata         |  | - Step-by-Step Derivations & Hints       |
| - Quantization Parameters & Imatrix      |  | - Formulas (LaTeX) & Diagram References  |
| - Engine Execution Configurations        |  | - SQLite FTS5 Search Indices             |
+------------------------------------------+  +------------------------------------------+
```

---

## 6. Granular Knowledge Pack Architecture & Size Philosophy

### 6.1 Granularity Levels
The packaging system supports flexible granularity:
1. **Single Subject Pack:** (e.g., `nctb_class9_math.ssp`) — For apps focusing on a specific subject.
2. **Single Class Complete Pack:** (e.g., `nctb_class8_all.ssp`) — Bundling Math, Science, English for one grade.
3. **High School Multi-Class Bundle:** (e.g., `nctb_highschool_bundle.ssp`) — Comprehensive Class 6–10 package.

### 6.2 Package Size Philosophy [PROPOSED TARGET]
* **Preferred Target:** **Approximately 20–50 MB per class/subject module** where practically achievable.
* **Guiding Rule:** This is a *preferred target*, not a destructive constraint. Educational completeness, mathematical accuracy, and retrieval quality will **never** be compromised solely to meet an arbitrary size limit.
* **Optimization Strategies Under Investigation:**
  * Zstandard / Deflate text compression inside SQLite containers.
  * Structural deduplication of common formulas, theorems, and definitions.
  * Compact binary indexing structures.
  * Verified actual sizes will be measured after real NCTB content ingestion.

---

## 7. End-User & Student Experience Flow

The student experience in a host application is frictionless and offline-first:

```
[ Install Host Application ]
             │
             ▼
[ SS Tutor Core Initialized ]
             │
             ▼
[ Select Class & Subjects ] ────── (e.g. Class 8: Math [✓], Science [✓], English [✓])
             │
             ▼
[ Download / Install Pack ] ────── (One-time online download or pre-bundled in APK/assets)
             │
             ▼
[ Installation & Checksum Verified ]
             │
             ▼
=========================================================
  INTERNET = OFF  ──>  100% OFFLINE TUTORING ACTIVE
=========================================================
  - Student asks textbook questions in Bengali
  - Engine matches chapter/exercise locally
  - Step-by-step Socratic hints & explanations rendered
```

### Required Lifecycle Features [CONFIRMED]:
* Seamless pack installation, verification (SHA256), dynamic loading, removal, and multi-pack coexistence.

---

## 8. Teacher-Friendly Capabilities (Future Horizon)

While the initial focus is student tutoring, the same SS Core engine is architected to support future **Teacher Assistant Modes**:

* **Concept Explanation Generation:** Formulating real-world Bengali analogies for classroom lectures.
* **Lesson Material Drafting:** Generating structured summaries of textbook chapters.
* **Practice Question & Quiz Generation:** Synthesizing formative assessment quizzes with answer keys.
* **Misconception Diagnosis:** Highlighting common student pitfalls in specific math/science topics.
* **Differentiated Revision Exercises:** Producing tiered exercises (Basic, Intermediate, Advanced).

*(Note: These are future API extensions, not current implementation prerequisites).*

---

## 9. Model Strategy, Evaluation Framework & License-First Selection

### 9.1 License-First Model Selection Gate [CONFIRMED]
No model will be benchmarked or selected on technical performance alone. Every candidate must first pass a strict **License Review Gate**:

```
                       Candidate Open-Weight Model
                                    │
                                    ▼
                    +───────────────────────────────+
                    |  GATE 0: License & IP Audit   |
                    +───────────────────────────────+
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
[ Modification Allowed? ]  [ Redistribution Allowed? ]  [ Commercial / Offline OK? ]
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    │
                         Pass ──────┴────── Fail ──> DISQUALIFIED
                        /
                       v
       +─────────────────────────────────────────────+
       |  GATE 1: Technical & Mobile Benchmarking    |
       |  (RAM < 750MB, Bengali Tokenizer, Speed)    |
       +─────────────────────────────────────────────+
```

### 9.2 Candidate Registry Summary

| Candidate ID | Model Identifier | Parameter Count | License Type | License Gate Status |
| :--- | :--- | :--- | :--- | :--- |
| **CAND-01** | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | Apache 2.0 | **PASSED GATE 0** |
| **CAND-02** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | Apache 2.0 | **PASSED GATE 0** |
| **CAND-03** | `HuggingFaceTB/SmolLM2-135M-Instruct` | 0.13B | Apache 2.0 | **PASSED GATE 0** |
| **CAND-04** | `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.36B | Apache 2.0 | **PASSED GATE 0** |
| **CAND-05** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.71B | Apache 2.0 | **PASSED GATE 0** |
| **CAND-06** | `meta-llama/Llama-3.2-1B-Instruct` | 1.23B | Llama 3.2 Community | `REQUIRES PRIMARY-SOURCE VERIFICATION` (Redistribution/notice terms) |
| **CAND-07** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.10B | Apache 2.0 | **PASSED GATE 0** |
| **CAND-08** | `google/gemma-2-2b-it` | 2.61B | Gemma Terms | `REQUIRES PRIMARY-SOURCE VERIFICATION` (High RAM, teacher-only) |

---

## 10. Ownership, Branding & Attribution Model

To maintain strict legal clarity and open-source integrity, we enforce clear boundaries between project components:

```
+-----------------------------------------------------------------------------------+
|                            OWNERSHIP & LICENSING MODEL                            |
|                                                                                   |
|  1. SS Tutor BD Codebase & Engine                                                 |
|     - Open source under Apache 2.0 License (Our original engineering).            |
|                                                                                   |
|  2. Base Foundation Model                                                         |
|     - Third-party open weights subject to upstream license (e.g. Apache 2.0).    |
|     - We do NOT claim authorship or ownership of the base model.                  |
|                                                                                   |
|  3. SS Tutor Adapted Model                                                        |
|     - Base model fine-tuned on our Bengali pedagogical dataset.                   |
|     - Distributed under upstream license terms with clear attribution.            |
|                                                                                   |
|  4. Education Knowledge Packs (.ssp)                                              |
|     - Derived from official NCTB curriculum under educational fair-use/public     |
|       domain guidelines. Separately tracked with full provenance metadata.        |
|                                                                                   |
|  5. Training & Evaluation Datasets                                                |
|     - Synthesized / curated Bengali tutoring datasets with distinct provenance.   |
+-----------------------------------------------------------------------------------+
```

---

## 11. Low-End Mobile Constraints & Memory Reality

Target Device Baseline: ~2.0 GB Physical RAM, ~16 GB Storage, ARM Cortex-A53/A55 CPU.

### Memory Tiers & Operational Limits:
* **Total Android OS & System Overhead:** ~1000–1250 MB.
* **Safe Foreground Process Memory Budget:** **$\le 600\text{–}750\text{ MB}$ peak RSS**.
* **Clean Mapped Memory (`mmap`):** Model weights mapped from storage; can be paged out under pressure.
* **Dirty Working RAM (Native Heap):** Intermediate tensor allocations ($\le 150\text{ MB}$).
* **KV Cache:** 2048-token context cache ($\le 80\text{ MB}$ with FP8/Q4 cache).
* **Host App Runtime (ART Heap):** UI and application state ($\le 100\text{ MB}$).
* **CPU Core Throttling:** Inference restricted to **2 worker threads** to prevent thermal throttling and battery collapse.

---

## 12. RAG & Tiered Hybrid Retrieval Architecture

To avoid vector database memory overhead ($>200\text{ MB}$), retrieval uses a lightweight multi-tier design:

```
                              Student Bengali Query
                                       │
                                       ▼
                     +───────────────────────────────────+
                     |     1. Deterministic Parser       |  < 2ms, 0MB RAM
                     |   (Regex / Bengali Entity Rules)  |
                     +───────────────────────────────────+
                                       │
                         Matched? ─────┼───── Not Matched
                        /                              \
                       v                                v
         +───────────────────────────+    +───────────────────────────+
         | Direct DB Lookup (O(1))   |    | 2. SQLite FTS5 Match with |  < 15ms, < 4MB RAM
         | Exact Chapter / Exercise  |    |  Bengali Normalizer/Stem  |
         +───────────────────────────+    +───────────────────────────+
                       │                                │
                       │                  High Score? ──┼── Low / Ambiguous
                       │                 /                                 \
                       │                v                                   v
                       │  +───────────────────────────+    +────────────────────────────────+
                       │  | Direct Context Injection  |    | 3. Micro-Embedding Semantic    |
                       │  |   (Zero Extra RAM/Compute)|    |    Search (Optional Fallback)  |
                       │  +───────────────────────────+    +────────────────────────────────+
                       │                │                                   │
                       └────────────────┼───────────────────────────────────┘
                                        │
                                        v
                       +─────────────────────────────────+
                       | Context Synthesizer & Tokenizer |
                       +─────────────────────────────────+
                                        │
                                        v
                       +─────────────────────────────────+
                       |    SS Core Local SLM Engine     |
                       +─────────────────────────────────+
```

---

## 13. Independent Model and Curriculum Update Strategy

A curriculum or textbook edition change **never** requires retraining the AI model:

```
[ Scenario A: Textbook Curriculum Update ]
  SS Core Engine v1.0.0 (Unchanged)
  SS Model Binary v1.0.0 (Unchanged)
  Knowledge Pack: `nctb_class9_math_2024.ssp` ──> `nctb_class9_math_2026.ssp` (Updated)

[ Scenario B: AI Model Engine Upgrade ]
  SS Core Engine v1.0.0 ──> v1.1.0 (Updated)
  SS Model Binary v1.0.0 ──> v1.1.0 (Updated)
  Knowledge Packs (Unchanged, fully backward-compatible)
```

---

## 14. Packaging & Distribution Strategy

Under our $0 budget constraint, packaging and distribution utilize free open infrastructure:

* **Android SDK Engine:** Published as an Android Archive (`.aar`) via Maven Central / JitPack or GitHub Releases.
* **Quantized Model Weights:** Hosted on Hugging Face Hub (free model hosting tier) for on-demand in-app download or direct asset bundling.
* **Knowledge Packs (`.ssp`):** Distributed via GitHub Releases / Hugging Face Datasets or bundled in the host app's `assets/` directory.
* **Developer Flexibility:** Developers can bundle assets locally for zero-network deployments or download modules dynamically on first launch.

---

## 15. Conceptual Developer SDK Experience

*(Note: Conceptual design only — not to be implemented yet)*

```kotlin
// 1. Initialize SS Tutor BD Engine
val tutor = SSTutor.init(context)

// 2. Install / Load Modular Knowledge Pack
tutor.installPack("nctb_class8_math.ssp")

// 3. Simple Unified Query API
tutor.ask(
    query = "Class 8 এর গণিত বইয়ের ৪র্থ অধ্যায়ের অনুশীলনী ৪.১ এর ৩ নম্বর অংকটি বুঝিয়ে দাও",
    callback = object : TutorCallback {
        override fun onToken(token: String) { print(token) }
        override fun onHint(hint: String) { showHintButton(hint) }
        override fun onComplete(response: TutorResponse) { /* Finished */ }
        override fun onError(error: TutorError) { /* Handle error */ }
    }
)
```

---

## 16. Architecture Decision Log (ADR)

* **ADR-001: Separation of Knowledge from Model Weights**
  * *Status:* **CONFIRMED**
  * *Decision:* Textbook curriculum is stored in modular `.ssp` SQLite containers and retrieved at runtime, NOT memorized in neural weights.
* **ADR-002: Zero-Budget Development Constraint**
  * *Status:* **CONFIRMED**
  * *Decision:* Only FOSS tools, local execution, and free-tier compute/distribution platforms will be used.
* **ADR-003: Deterministic & Hybrid Retrieval over Pure Vector DB**
  * *Status:* **CONFIRMED**
  * *Decision:* Retrieval relies primarily on deterministic coordinate matching and SQLite FTS5 with Bengali normalization, avoiding memory-heavy vector index overhead.
* **ADR-004: Model Selection Strategy**
  * *Status:* **PROPOSED / PENDING EXPERIMENTATION**
  * *Decision:* Final model selection is determined via the Stage 1 Benchmarking Framework; no model is locked prematurely.
* **ADR-005: Socratic Scaffolding State Machine**
  * *Status:* **PROPOSED**
  * *Decision:* The core engine defaults to progressive hints and step-by-step pedagogical guidance rather than raw answer dumps.
* **ADR-006: Modular Knowledge Pack Architecture**
  * *Status:* **CONFIRMED**
  * *Decision:* Knowledge packs support flexible granularity (single subject, single class, full high-school bundle) and can be installed, updated, or removed independently.
* **ADR-007: One Reusable Core Model by Default**
  * *Status:* **CONFIRMED**
  * *Decision:* One reusable core model is the default architecture. Multiple specialized models remain possible if empirical benchmarking proves that a single model cannot satisfy quality, memory, speed, or subject-specific requirements.
* **ADR-008: License-First Model Selection**
  * *Status:* **CONFIRMED**
  * *Decision:* Candidate models must pass a strict license audit (permitting modification, redistribution, commercial use, and offline bundling) before technical benchmarking.
* **ADR-009: Knowledge Pack Size 20–50 MB as Preferred Target**
  * *Status:* **PROPOSED TARGET**
  * *Decision:* 20–50 MB per class/subject module is adopted as a preferred optimization target without sacrificing educational completeness.
* **ADR-010: Independent Model and Knowledge Versioning**
  * *Status:* **CONFIRMED**
  * *Decision:* Engine/Model versions and Curriculum/Knowledge pack versions are strictly decoupled, enabling textbook updates without model retraining.

---

## 17. Open Questions & Experimental Verification Items

1. **[OPEN QUESTION / EXPERIMENT 1] Bengali Tokenizer Expansion Ratio:**  
   *Measure exact subword splitting on NCTB text across candidate tokenizers (Qwen 151k, SmolLM2 49k, Llama 128k, TinyLlama 32k).*
2. **[OPEN QUESTION / EXPERIMENT 2] Sub-1B Model Pedagogical Reasoning:**  
   *Determine whether a 0.5B model can follow Socratic scaffolding when guided by Tier-1 RAG context, or if a 1.0B–1.5B model is strictly necessary.*
3. **[OPEN QUESTION / EXPERIMENT 3] Mobile Runtime Throughput:**  
   *Benchmark `llama.cpp` CPU inference throughput on budget ARM hardware under 2-thread throttling.*
4. **[OPEN QUESTION / EXPERIMENT 4] FTS5 Bengali Normalization:**  
   *Validate lightweight Bengali character/suffix stripping for inflected search queries.*
