# SS Tutor BD — Third-Party License Inventory

**Document Version:** 1.0.0  
**Phase:** 5 — Release Packaging  
**Date:** 2026-08-30  

> [!IMPORTANT]
> This document lists all third-party software, libraries, and data sources used in
> production builds of SS Tutor BD. All components are compatible with the $0 development budget constraint.

---

## 1. Runtime Libraries & SDKs

| Component | Version | License | Usage |
| :--- | :--- | :--- | :--- |
| **Android SDK** | API 24–34 | Apache-2.0 | Android platform runtime |
| **AndroidX Core KTX** | 1.12.0 | Apache-2.0 | Kotlin Android extensions |
| **AndroidX AppCompat** | 1.6.1 | Apache-2.0 | Backward-compatible Activity |
| **Material Design 3** | 1.11.0 | Apache-2.0 | UI components |
| **AndroidX ConstraintLayout** | 2.1.4 | Apache-2.0 | Flexible UI layouts |
| **AndroidX Lifecycle ViewModel** | 2.7.0 | Apache-2.0 | MVVM lifecycle handling |
| **AndroidX RecyclerView** | 1.3.2 | Apache-2.0 | Chat message list |

---

## 2. Database & Storage

| Component | License | Usage |
| :--- | :--- | :--- |
| **SQLite** (bundled in Android) | Public Domain | FTS5 knowledge pack storage |
| **SQLite FTS5** | Public Domain | Offline curriculum full-text search |

---

## 3. Neural Model & Tokenizer

| Component | License | Usage |
| :--- | :--- | :--- |
| **SS Tutor Bengali 70M Edu** (Custom-trained) | MIT / CC0 | Pedagogical language generation |
| **Custom 16K Bengali BPE Tokenizer** (Custom-trained) | CC0-1.0 | Bengali tokenization |
| **Training Data Synthetic JSONL** (13,000 examples) | CC0-1.0 / Custom | Model training corpus |
| **HuggingFace Transformers** (Dev only) | Apache-2.0 | Training scaffolding (not in APK) |
| **PyTorch** (Dev only, CPU) | BSD-3-Clause | Training backend (not in APK) |
| **Sentencepiece / tokenizers** (Dev only) | Apache-2.0 | Tokenizer training (not in APK) |

---

## 4. Python Development Tools (NOT in Production APK)

| Tool | License | Purpose |
| :--- | :--- | :--- |
| **Python 3.14** | PSF License (permissive) | Development scripting |
| **PyTorch CPU** | BSD-3-Clause | Local model training |
| **Transformers** | Apache-2.0 | Training scaffolding |
| **Pytest** | MIT | Unit testing |
| **SQLite3** (stdlib) | Public Domain | RAG index construction |

---

## 5. Knowledge Pack Provenance

| Source | License | Status |
| :--- | :--- | :--- |
| **NCTB Class 8 Mathematics Curriculum** (paraphrased synthetic summaries) | Bangladesh public curriculum (educational use) | Reformulated as CC0 synthetic JSON |
| **Synthetic Q&A Pairs** (13,000 generated examples) | CC0-1.0 | Generated locally by SS Tutor BD data pipeline |

> [!NOTE]
> The production knowledge pack (`.ssp` format) contains **only synthetic paraphrased content** derived from the NCTB curriculum framework. No verbatim copyrighted text is included.

---

## 6. Incompatible Dependency Policy

The following are explicitly **PROHIBITED** in the production APK:

* OpenAI API clients or keys
* Anthropic / Claude SDK
* Google Cloud AI Platform SDK
* Any paid inference backend
* Python runtime (CPython, Chaquopy)
* PyTorch Mobile embedded runtime (unless memory-validated)
* Training-only artifacts (optimizer states, checkpoints)

---

## 7. License Compatibility Matrix

| License | APK Distribution | Dev Use | Note |
| :--- | :--- | :--- | :--- |
| Apache-2.0 | ✅ Yes | ✅ Yes | Attribution required |
| MIT | ✅ Yes | ✅ Yes | Attribution required |
| BSD-3-Clause | ✅ Yes | ✅ Yes | Attribution required |
| CC0-1.0 | ✅ Yes | ✅ Yes | Public domain; no restrictions |
| Public Domain | ✅ Yes | ✅ Yes | No restrictions |
| GPL-2.0 / GPL-3.0 | ❌ NO (copyleft) | Dev only | Must not be bundled in APK |
