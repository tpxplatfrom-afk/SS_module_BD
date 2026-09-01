# SS Tutor BD — Phase 2 Report

**Project:** SS Tutor BD — Offline-First AI Education Platform  
**Phase:** 2 — CAND-02 Validation + Bengali Tutoring Foundation  
**Report Date:** 2026-08-30  
**Architecture Version:** v1.1.0  
**Development Budget:** \$0 USD  

---

## Executive Summary

Phase 2 completed the evaluation of **CAND-02 (`Qwen2.5-1.5B-Instruct` Q4_K_M)** against the established 100-item NCTB benchmark suite, built the core Bengali tutoring infrastructure (output sanitizer, prompt scaffold, offline RAG engine), and produced a clear empirical model decision.

**CAND-02 is DISQUALIFIED** due to a critical Gate 4 memory failure: its peak RSS of **1,771 MB** is 2.4× over the 750 MB limit for 2 GB Android devices. Larger parameters did not improve Bengali language quality either — the 1.5B model produced equal or worse repetition loops than the 0.5B CAND-01.

The key architectural insight from Phase 2 is:

> **Bengali tutoring quality is primarily gated by RAG context grounding and prompt scaffolding, not raw parameter count. The smallest model that fits the RAM envelope — combined with properly structured retrieval and sanitization — is the correct path.**

---

## Repository Changes in Phase 2

| New File | Description |
| :--- | :--- |
| [`PHASE2_PRECHECK.md`](PHASE2_PRECHECK.md) | Full repository audit before Phase 2 execution |
| [`results/licenses/CAND-02_LICENSE.md`](results/licenses/CAND-02_LICENSE.md) | Primary-source Apache-2.0 license audit for CAND-02 |
| [`results/model_decision/CAND-02_DECISION.md`](results/model_decision/CAND-02_DECISION.md) | Gate-by-gate decision analysis |
| [`benchmarks/phase2_diagnostics/diagnostic_prompts.json`](benchmarks/phase2_diagnostics/diagnostic_prompts.json) | 16-item Phase 2 diagnostic suite targeting Bengali conjuncts, repetition, negative constraints, Socratic recovery, and grounding |
| [`benchmark_runner/diagnostic_runner.py`](benchmark_runner/diagnostic_runner.py) | Phase 2 diagnostic evaluation engine |
| [`core/sanitization/cleaner.py`](core/sanitization/cleaner.py) | Multi-stage output sanitization pipeline (control token removal, echo stripping, loop truncation) |
| [`core/prompts/tutor_templates.py`](core/prompts/tutor_templates.py) | Centralized Socratic, step-by-step math, RAG-grounded, and adaptive simplification prompt templates in Bengali |
| [`core/rag/schema.py`](core/rag/schema.py) | `KnowledgeChunk` dataclass — `.ssp` pack-compatible data schema |
| [`core/rag/chunker.py`](core/rag/chunker.py) | Deterministic semantic NCTB chapter chunker with stable chunk IDs |
| [`core/rag/indexer.py`](core/rag/indexer.py) | SQLite FTS5 offline knowledge indexer |
| [`core/rag/retriever.py`](core/rag/retriever.py) | BM25-ranked offline retriever with Bengali query normalization |
| [`tests/test_sanitizer.py`](tests/test_sanitizer.py) | Sanitizer unit tests — control tokens, echo, loops, Bengali preservation |
| [`tests/test_rag.py`](tests/test_rag.py) | RAG integration tests — chunking, indexing, exact/paraphrase/irrelevant queries |
| [`scratch/rescore.py`](scratch/rescore.py) | Re-scoring utility for post-hoc analysis of saved raw benchmark results |

**Modified Files:**

| File | Change |
| :--- | :--- |
| [`benchmark_runner/scoring.py`](benchmark_runner/scoring.py) | Gate thresholds now loaded dynamically from `config/settings.json` (no more hardcoded divergence) |
| [`benchmark_runner/runner.py`](benchmark_runner/runner.py) | Added `flush=True` to progress print |
| [`core/sanitization/cleaner.py`](core/sanitization/cleaner.py) | Fixed raw string escape sequences |

---

## CAND-02 Hardware Profile

| Metric | Measured Value |
| :--- | :--- |
| Model File Size | **1,065.56 MB** |
| Model Load Time | **2,673 ms** (vs 861 ms for CAND-01) |
| Initial RSS after Load | ~1,075 MB |
| **Peak RSS during Inference** | **1,771.26 MB** |
| Average Throughput | **10.22 tok/s** |
| Total Benchmark Duration | ~41 minutes |
| Host | Windows 10 Pro, Intel i5-6500, 2 CPU threads |

---

## CAND-02 Benchmark Results

```
==========================================================================
CATEGORY                          SCORE / MAX      %         STATUS
==========================================================================
Bengali Linguistic Quality        0.0  / 20.0      0.0%      FAIL
Educational Reasoning (Math+Sci)  9.0  / 25.0      36.0%     FAIL
Mobile Resource Efficiency        8.0  / 20.0      40.0%     FAIL
Instruction & Socratic Scaffold   13.5 / 15.0      90.0%     PASS ✅
Knowledge Grounding (Anti-Halluc) 6.0  / 10.0      60.0%     PASS ✅
License & Redistribution Freedom  10.0 / 10.0      100.0%    PASS ✅
--------------------------------------------------------------------------
TOTAL                             46.5 / 100.0     46.5%     FAILED
==========================================================================
```

**Gate results:**
- Gate 1 (License): ✅ PASS
- Gate 2 (Bengali ≥ 12.0): ❌ FAIL — 0.0/20
- Gate 3 (Reasoning ≥ 15.0): ❌ FAIL — 9.0/25
- Gate 4 (RAM ≤ 750 MB): ❌ **CRITICAL FAIL** — 1,771 MB
- Gate 5 (Speed ≥ 4.0 tok/s): ✅ PASS — 10.22 tok/s
- Gate 6 (Total ≥ 70.0): ❌ FAIL — 46.5/100

---

## CAND-01 vs CAND-02 Comparison

| Dimension | CAND-01 (0.5B Q4_K_M) | CAND-02 (1.5B Q4_K_M) | Better |
| :--- | :--- | :--- | :--- |
| Composite Score | 50.5 / 100 | 46.5 / 100 | CAND-01 |
| Bengali Quality | 2.0 / 20 | 0.0 / 20 | CAND-01 |
| Educational Reasoning | 15.0 / 25 | 9.0 / 25 | CAND-01 |
| Instruction Following | 15.0 / 15 | 13.5 / 15 | CAND-01 |
| Peak RAM | **680 MB ✅** | **1,771 MB ❌** | **CAND-01** |
| Speed | 21.6 tok/s | 10.22 tok/s | CAND-01 |
| File Size | 468.64 MB | 1,065.56 MB | CAND-01 |
| Instruction Following (negative constraints) | Partial | **Better** | CAND-02 |
| License | Apache-2.0 ✅ | Apache-2.0 ✅ | Tie |

---

## Bengali Quality

Neither CAND-01 nor CAND-02 achieve acceptable Bengali generation quality in ungrounded free-form generation. Both suffer from:

1. **Autoregressive repetition loops** — repeated phrases without semantic progression
2. **Token corruption** — byte-level fallback produces garbled Bengali conjuncts
3. **Self-echo** — model echoes the user prompt verbatim as first tokens

The key evidence is that **both models score near-zero on Bengali linguistic quality without retrieval context**. This strongly indicates Bengali tutoring quality is a **RAG + scaffolding problem, not purely a parameter-count problem**.

CAND-02 actually scores **worse** on Bengali quality (0.0 vs 2.0) despite 3× the parameters — suggesting that Qwen2.5 models below ~3B parameters in base Q4_K_M quantization do not gain Bengali fluency from scale alone.

---

## Educational Reasoning

CAND-02 showed lower educational reasoning (9.0/25) vs CAND-01 (15.0/25). CAND-02 produced verbose but repetitive multi-step attempts that failed to arrive at correct answers. CAND-01, despite being smaller, produced more concise (even if incomplete) math output on a higher fraction of tests.

---

## Repetition Behavior

Both candidates exhibit severe token repetition on Bengali prompts without context. The output sanitizer correctly detects and truncates these loops in the Phase 2 pipeline. The `truncate_repetition_loops()` function using substring repetition detection correctly triggers on the patterns observed (e.g., "কোনো কোনো কোনো কোনো কোনো" repeated dozens of times).

---

## RAG Retrieval Results

The offline SQLite FTS5 retrieval engine was built and validated:

| Test Case | Result | Latency |
| :--- | :--- | :--- |
| Exact NCTB formula query | ✅ PASS | 0.44–0.84 ms |
| Paraphrase conceptual query | ✅ PASS | < 1 ms |
| Middle-term factorization | ✅ PASS | < 1 ms |
| Irrelevant astrophysics query | ✅ PASS (0 results returned, no hallucinated match) | < 1 ms |

Retrieval latency of < 1 ms on in-memory SQLite FTS5 is well within offline Android performance targets. When targeting Android, the same SQLite FTS5 API is available natively (Android SDK includes SQLite with FTS5 support).

---

## Prompt Scaffold Results

The centralized `core/prompts/tutor_templates.py` provides:

- `get_base_system_prompt()` — Standard NCTB Bengali tutor role
- `build_socratic_hint_prompt()` — Negative constraint enforcement
- `build_step_by_step_math_prompt()` — Structured 4-step mathematical problem solving
- `build_grounded_rag_prompt()` — Strict context-grounded answering with explicit honesty constraint
- `build_adaptive_simplification_prompt()` — Recovery scaffold for "I don't understand" scenarios

These templates are written entirely in natural Bengali and are model-agnostic. They can be used with any future candidate model.

---

## Memory

| Model | Load RSS | Peak Inference RSS | Target (750 MB) | Status |
| :--- | :--- | :--- | :--- | :--- |
| CAND-01 (0.5B Q4_K_M) | ~233 MB | **680 MB** | 750 MB | ✅ Within target |
| CAND-02 (1.5B Q4_K_M) | ~1,075 MB | **1,771 MB** | 750 MB | ❌ 2.4× over target |

---

## Speed

| Model | Average tok/s | Gate 5 (≥ 4.0) | Practical Usability |
| :--- | :--- | :--- | :--- |
| CAND-01 (0.5B) | 21.6 tok/s | ✅ PASS | Fast — 12 tokens/sec on Android would yield ~1 word/sec usable output |
| CAND-02 (1.5B) | 10.22 tok/s | ✅ PASS | Marginal — on a slower Cortex-A53 Android device, would be ~3–5 tok/s |

---

## Storage

| Model | File Size | Single-Model Policy Compliant |
| :--- | :--- | :--- |
| CAND-01 | 468.64 MB | ✅ Yes |
| CAND-02 | 1,065.56 MB | ✅ Yes (compliant if no other models present) |

---

## License

| Candidate | License | Gate 1 Status |
| :--- | :--- | :--- |
| CAND-01 | Apache-2.0 | ✅ PASSED |
| CAND-02 | Apache-2.0 | ✅ PASSED |

Both candidates are cleared for redistribution in offline SDK format. Attribution required in any published distribution.

---

## Model Decision

| Candidate | Verdict | Primary Reason |
| :--- | :--- | :--- |
| **CAND-01** (0.5B Q4_K_M) | **FAIL (Phase 1)** | Gate 2 Bengali quality (2.0/20). **Recommended for RAG-grounded re-evaluation in Phase 3.** |
| **CAND-02** (1.5B Q4_K_M) | **FAIL (Phase 2)** | Gate 4 critical RAM failure (1,771 MB >> 750 MB). Permanently disqualified for current target hardware. |

> **No candidate has been fully approved for production.** CAND-01 remains the only candidate within the hardware RAM envelope and is the most promising path forward with RAG grounding.

---

## Known Limitations

1. **Benchmark scoring is heuristic.** The keyword-matching and loop-detection heuristics are designed to detect obvious failure modes, but cannot assess pedagogical nuance (e.g., whether a mathematically correct explanation is age-appropriate for Class 8 students).
2. **Host ≠ Target Device.** All benchmarks ran on an Intel i5-6500 Windows PC. Android Cortex-A53 performance will be approximately 3–5× slower. The 10.22 tok/s on CAND-02 translates to ~2–3 tok/s on Cortex-A53 — below comfortable interactive usage.
3. **No human evaluation performed.** All scoring is automated. Human teacher review is required before any model advances to a student-facing prototype.
4. **Bengali Evaluation Prompt Gaps.** The Phase 1 Bengali benchmark prompts do not cover all NCTB curriculum lexical domains. Phase 3 should expand the Bengali test dataset with more domain-specific examples.

---

## Phase 3 Recommendation

**Recommended path:**

```
Phase 3A: RAG-Grounded CAND-01 Evaluation
─────────────────────────────────────────
1. Re-download CAND-01 (Qwen2.5-0.5B-Instruct Q4_K_M)
2. Ingest Class 8 NCTB Mathematics chapter content into SQLite FTS5 index
3. Run all 30 MATH + 10 GROUND benchmark items with RAG context injection
4. Run 20 Bengali items with structured step-by-step scaffold
5. Apply output sanitization layer on all raw outputs
6. Score with updated weighted scorecard
7. If total score ≥ 70.0: CONDITIONAL PASS → Class 8 Math Prototype
8. If still failing: evaluate CAND-07 (TinyLlama 1.1B, 720 MB RAM)

Phase 3B: Class 8 Mathematics Prototype (if CAND-01 or CAND-07 passes)
─────────────────────────────────────────────────────────────────────────
1. Build Class 8 NCTB Math SSP knowledge pack
2. Implement end-to-end query → retrieval → RAG prompt → model → sanitize → response
3. Test on 20 representative Class 8 math curriculum questions
4. Evaluate with human teacher review
5. Produce minimal Android APK proof-of-concept
```

> [!IMPORTANT]  
> **No Bengali-specific fine-tuning should begin until a base model with adequate Bengali generation quality under RAG grounding has been confirmed.** Training an unsuitable base model wastes $0 compute resources and storage.
