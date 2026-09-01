# SS Tutor BD — Phase 2 Pre-Execution Repository Audit

**Audit Date:** 2026-08-30  
**Baseline Architecture:** SS Tutor BD v1.1.0  
**Auditor:** Primary Autonomous Coding Agent  

---

## 1. What Phase 1 Actually Implemented

A comprehensive line-by-line inspection of the repository confirms that Phase 1 delivered:

1. **Architecture & Design Documentation:**
   - [`ARCHITECTURE.md`](ARCHITECTURE.md): Complete v1.1.0 specification including ADRs 001–010, product success criteria, and ownership boundaries.
   - [`ARCHITECTURE_CHANGELOG.md`](ARCHITECTURE_CHANGELOG.md): Detailed revision history.
   - [`MODEL_SELECTION.md`](MODEL_SELECTION.md) & [`MODEL_CANDIDATES.md`](MODEL_CANDIDATES.md): 100-point evaluation matrix, sequential gates (Gates 1–6), candidate tiers A–D.
2. **Environment & Host Audit:**
   - [`benchmarks/ENVIRONMENT.md`](benchmarks/ENVIRONMENT.md): Host specifications (Windows 10, Intel i5-6500 4C/4T @ 3.20GHz, 8GB RAM, Python 3.14.0, Android NDKs).
3. **Configuration & Model Registry:**
   - [`config/settings.json`](config/settings.json): Global limits (`min_disk_free_mb: 1500`, `max_model_size_mb: 1200`), gate thresholds.
   - [`models/registry.json`](models/registry.json): 10 registered candidates (`CAND-01` through `CAND-10`).
   - [`models/manager.py`](models/manager.py): Download manager with Gate 1 license check, storage safety guard, GGUF magic byte validation (`b'GGUF'`), and clean weight purge mechanism.
4. **Benchmarking Dataset (100 NCTB Items):**
   - `benchmarks/bengali/bn_prompts.json` (20 items: `BN-001`..`BN-020`)
   - `benchmarks/mathematics/math_prompts.json` (30 items: `MATH-001`..`MATH-030`)
   - `benchmarks/science/sci_prompts.json` (20 items: `SCI-001`..`SCI-020`)
   - `benchmarks/pedagogy/ped_prompts.json` (20 items: `PED-001`..`PED-020`)
   - `benchmarks/grounding/ground_prompts.json` (10 items: `GROUND-001`..`GROUND-010`)
   - `benchmarks/tokenizer/` (12-sample NCTB compression benchmark comparing Qwen2.5, TinyLlama, SmolLM2).
5. **Inference Runtime Engine:**
   - [`runtimes/base.py`](runtimes/base.py): Abstract `ModelRuntime` interface and `GenerationResult` dataclass.
   - [`runtimes/llama_cpp_runtime.py`](runtimes/llama_cpp_runtime.py): CPU-driven GGUF runtime powered by `llama-cpp-python` (with ChatML formatting and `create_chat_completion`).
   - [`runtimes/mock_runtime.py`](runtimes/mock_runtime.py): Instant deterministic dry-run harness.
6. **Scoring, Reporting & CLI:**
   - [`benchmark_runner/scoring.py`](benchmark_runner/scoring.py): 100-point scorecard calculation, sequential gate evaluation, and repetition loop detector.
   - [`benchmark_runner/reporter.py`](benchmark_runner/reporter.py): Machine JSON generator (`results/raw/`), Markdown report generator (`results/reports/`), failure case extractor (`results/failures/`).
   - [`benchmark_runner/cli.py`](benchmark_runner/cli.py): Unified CLI for candidate management, tokenization, dry-run, and real benchmarks.
7. **Empirical Benchmarking of CAND-01:**
   - `CAND-01` (`Qwen2.5-0.5B-Instruct` Q4_K_M) was downloaded, benchmarked across all 100 items, scored (**50.5/100**, Failed Gate 2 & Gate 6), and purged cleanly.

---

## 2. Inconsistencies Between Phase 1 Report and Repository

1. **Gate Threshold Calibration in settings.json vs scoring.py:**
   - `config/settings.json` specified `gate_2_bengali_min_score: 12.0` and `gate_3_reasoning_min_score: 15.0`.
   - `scoring.py` had preliminary hardcoded checks `gate_2: >= 10.0` and `gate_3: >= 12.0`.
   - *Resolution:* `scoring.py` will be dynamically wired to load exact thresholds directly from `config/settings.json` to prevent drift.
2. **CAND-01 Download File Size:**
   - Estimated file size in `registry.json` was 398 MB; actual GGUF binary was 468.64 MB.
   - *Resolution:* Adjust CAND-02 estimated size in registry if needed (e.g. Qwen2.5-1.5B Q4_K_M is ~986 MB).

---

## 3. Existing Technical Debt

1. **System Prompt Hardcoding in Runner:**
   - `runner.py` had an inline system prompt. System prompts must be modularized under `core/prompts/` for Phase 2 scaffolding and systematic parameter testing.
2. **Output Sanitization Missing in Runtime Loop:**
   - Raw outputs from models with internal artifacts (such as `</tool_call>`) passed directly to scoring without pre-filtering.
   - Phase 2 requires an explicit output sanitization pipeline.
3. **Lack of Automated Failure Diagnostic Suite:**
   - While 100 items test general NCTB curriculum, Phase 2 needs a dedicated diagnostic suite specifically probing Bengali conjuncts, vowel signs (কার/ফলা), negative constraints, and prompt echo.

---

## 4. Components Safe to Reuse Without Rewriting

| Component | Path | Reason for Reuse |
| :--- | :--- | :--- |
| **Model Manager** | `models/manager.py` | Verified disk margin checks, GGUF magic byte validation, single-model purge. |
| **Model Registry** | `models/registry.json` | Candidate profiles for CAND-01 through CAND-10 are already well structured. |
| **100-Item Dataset** | `benchmarks/*/` | Complete, high-quality 100-item NCTB evaluation dataset. |
| **GGUF Runtime** | `runtimes/llama_cpp_runtime.py` | `llama-cpp-python` backend verified and running at >20 tok/s on CPU. |
| **Mock Runtime** | `runtimes/mock_runtime.py` | Fast dry-run validation harness. |
| **Disk Utilities** | `scripts/check_disk.py` | Accurate host storage monitor. |

---

## 5. Components Requiring Modification / Extension for Phase 2

1. **`core/prompts/` (NEW):** Centralized tutor prompt scaffolding with Socratic hints, math step-by-step guidance, and negative constraint enforcement.
2. **`core/sanitization/` (NEW):** Dedicated output sanitization layer for control token filtering and anti-repetition guards.
3. **`core/rag/` (NEW):** Lightweight, offline-first SQLite/FTS5 indexing and semantic chunking engine.
4. **`benchmarks/phase2_diagnostics/` (NEW):** Targeted diagnostic test suite for Bengali orthography, negative constraints, and Socratic recovery.
5. **`benchmark_runner/scoring.py` (UPDATE):** Bind gate thresholds dynamically to `config/settings.json`.

---

## 6. Phase 2 Risk Assessment & Mitigation

| Identified Risk | Severity | Mitigation Strategy |
| :--- | :--- | :--- |
| **CAND-02 RAM Spikes on 2GB Target** | High | CAND-02 (1.54B) in Q4_K_M takes ~986 MB disk. On 2GB Android, peak RSS must stay <= 750 MB. We will benchmark memory RSS with strict KV-cache allocation (`n_ctx=2048`) and evaluate if Q3_K_M or IQ3_M is necessary. |
| **Host Disk Space Exhaustion** | Critical | Host drive C: currently has 4.47 GB free. Single-model storage policy is strictly enforced. No second model binary will be downloaded until CAND-01 / active dir is confirmed purged. |
| **Bengali Degeneration in 1.5B Base** | Medium | If CAND-02 also shows repetition loops, output sanitization, stop sequences (`<|im_end|>`, `\n\n`), repetition penalties, and RAG context grounding will be tested in Phase 2 parameter experiments. |

---

## 7. Audit Conclusion

The Phase 1 foundation is sound, operational, and clean. No rewrite is necessary. We are cleared to proceed to **STEP 3 (CAND-02 Registration & Evaluation)**.
