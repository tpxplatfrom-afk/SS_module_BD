# Architecture Changelog: SS Tutor BD

This document tracks revisions, rationale, confirmed decisions, and open uncertainties across versions of `ARCHITECTURE.md`.

---

## [1.1.0] - 2026-08-30

### Summary of Changes
Revision v1.1.0 formalizes the confirmed product direction of **SS Tutor BD** as a modular offline AI education platform, establishes a license-first evaluation gate, defines multi-tiered knowledge pack granularity, introduces product success dimensions, and adds ADRs 006 through 010.

---

### Detailed Changes from v1.0.0-DRAFT

1. **Platform Identity Clarification (Section 1 & 4):**
   * *Change:* Explicitly defined SS Tutor BD as **One Reusable Core Model + Modular Knowledge Packs**, rejecting monolithic model duplication across classes.
   * *Rationale:* Prevent unnecessary neural weight overhead and preserve maintainability.
2. **Product Success Dimensions (Section 2):**
   * *Change:* Added a dedicated framework evaluating success across three pillars: **Developer-Friendly**, **User-Friendly**, and **Teaching-Friendly**.
   * *Rationale:* Ensure technical elegance does not compromise student experience or pedagogical rigor.
3. **Knowledge Pack Granularity & Size Philosophy (Section 6):**
   * *Change:* Formalized support for single subject packs, single class complete packs, and full high-school bundles. Added **20–50 MB per module** as a *preferred optimization target* rather than a rigid constraint.
   * *Rationale:* Provide developer flexibility while preventing educational degradation.
4. **License-First Model Selection Gate (Section 9):**
   * *Change:* Introduced **Gate 0: License & IP Audit** prior to technical benchmarking.
   * *Rationale:* Avoid wasting engineering resources benchmarking models with restrictive commercial terms, naming prohibitions, or non-redistributable licenses.
5. **Ownership & Attribution Model (Section 10):**
   * *Change:* Explicitly demarcated SS Tutor BD Code, Base Foundation Model, SS Tutor Adapted Model, Knowledge Packs, and Datasets.
   * *Rationale:* Protect open-source integrity and ensure strict legal compliance.
6. **Teacher-Friendly Horizon (Section 8):**
   * *Change:* Documented future teacher-assistant capabilities (lesson drafting, quiz generation, error diagnosis) as future SDK extensions.
   * *Rationale:* Establish future roadmap without bloating initial implementation requirements.
7. **Packaging & Distribution Options (Section 14):**
   * *Change:* Outlined distribution strategies (Android AAR, Hugging Face, GitHub Releases, Maven) operating within the $0 budget constraint.
   * *Rationale:* Clarify developer delivery paths.
8. **ADR Additions (Section 16):**
   * *Change:* Formally added ADR-006 (Modular Knowledge Packs), ADR-007 (One Reusable Model), ADR-008 (License-First Selection), ADR-009 (20–50MB Target Size), and ADR-010 (Independent Versioning).

---

### Confirmed Decisions in v1.1.0
* **ADR-001:** Strict separation of knowledge data from model weights.
* **ADR-002:** $0 development budget constraint (FOSS tooling only).
* **ADR-003:** Tiered hybrid retrieval (Deterministic regex + SQLite FTS5) over heavy vector DBs.
* **ADR-006:** Modular knowledge pack architecture with multi-level granularity.
* **ADR-007:** One reusable core model by default across all subjects and classes.
* **ADR-008:** License-first model selection gate.
* **ADR-010:** Independent versioning and distribution for model binaries and knowledge packs.

---

### Proposed Decisions in v1.1.0
* **ADR-004:** Candidate model selection framework and evaluation weights.
* **ADR-005:** Socratic scaffolding state machine behavioral constraints.
* **ADR-009:** 20–50 MB preferred package size per subject/class module.

---

### Remaining Uncertainties & Experimental Items
1. **Bengali Tokenization Efficiency:** Exact subword expansion ratios for Bengali unicode across candidate tokenizers.
2. **Sub-1B Model Pedagogical Reasoning:** Whether a 0.5B model can follow Socratic scaffolding when guided by Tier-1 RAG context, or if a 1.0B–1.5B model is required.
3. **Mobile Runtime Throughput on Budget ARM:** Real-world generation speed on low-end Cortex-A53/A55 CPUs under memory and thermal limits.
4. **SQLite FTS5 Bengali Normalization:** Performance and accuracy of deterministic Bengali suffix stripping.
