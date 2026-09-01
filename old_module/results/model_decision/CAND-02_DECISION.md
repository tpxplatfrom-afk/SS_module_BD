# SS Tutor BD — CAND-02 Model Decision Report

**Candidate ID:** `CAND-02`  
**Model Name:** `Qwen2.5-1.5B-Instruct`  
**Quantization Tested:** `Q4_K_M`  
**Evaluation Date:** 2026-08-30  
**Decision Author:** SS Tutor BD Phase 2 Benchmarking Agent  

---

## FINAL VERDICT: ❌ FAIL

**CAND-02 in Q4_K_M quantization is DISQUALIFIED from Phase 2 selection as the SS Core model.**

---

## 1. Gate-by-Gate Results

| Gate | Criterion | Result | Threshold | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | License Compliance | `Apache-2.0` | `LICENSE_PASSED` | ✅ **PASS** |
| **Gate 2** | Bengali Linguistic Quality | **0.0 / 20.0** | ≥ 12.0 | ❌ **FAIL** |
| **Gate 3** | Educational Reasoning | **9.0 / 25.0** | ≥ 15.0 | ❌ **FAIL** |
| **Gate 4** | Peak Memory (RSS) | **1,771.26 MB** | ≤ 750 MB | ❌ **CRITICAL FAIL** |
| **Gate 5** | Generation Throughput | **10.22 tok/s** | ≥ 4.0 tok/s | ✅ **PASS** |
| **Gate 6** | Composite Total Score | **46.5 / 100** | ≥ 70.0 | ❌ **FAIL** |

---

## 2. Critical Failure Analysis

### Gate 4 CRITICAL FAIL — Peak Memory: 1,771.26 MB

This is the hardest failure. **CAND-02 Q4_K_M cannot run on the target 2 GB RAM Android device.**

- Model file: **1,065.56 MB** on disk
- Peak RSS during inference: **1,771.26 MB**
- Target device RAM envelope: **2,048 MB total, ~600–750 MB available for foreground inference process**
- Memory headroom violation: **+1,021 MB over target**

This is not a close miss. CAND-02 Q4_K_M requires approximately 2.4× the available inference RAM on the target device. Even with aggressive OS compression, this model cannot run reliably on 2 GB Android devices.

### Gate 2 FAIL — Bengali Linguistic Quality: 0.0/20

Despite having 3× more parameters than CAND-01, CAND-02's raw Q4_K_M base shows **no improvement** in Bengali generation quality. The model produces degenerative token repetition on nearly all 20 Bengali linguistic test cases:

*Example (BN-002 — "Explain Exponents in Bengali"):*
> ভাই, সহজ ভাষা সূচক (Exponent) একটি অংক যা বুঝা যায় যে একটি কোনো ক্ষেত্রে কোনো কোনো কোনো ক্ষেত্রে কোনো কোনো কোনো কোনো ক্ষেত্রে কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো...

Root cause: The **base Qwen2.5-1.5B-Instruct model is not Bengali-instruction-tuned**. Without RAG context grounding or Bengali-specific fine-tuning, the model generates repetition loops on Bengali prompts regardless of parameter count.

### Gate 3 FAIL — Educational Reasoning: 9.0/25

Mathematical step-by-step reasoning showed partial performance: the model attempted structured multi-step work but degraded into repetitive filler text before completing answers. Sample output for MATH-001 (fraction addition):
> আমার জন্য সব ধাপ বাংলা দেখানো হবে... ভগ্নাংশের ভগ্নাংশের ভগ্নাংশ একে যোগ করব... [truncated]

---

## 3. What Unexpectedly Performed Well

| Dimension | Score | Observation |
| :--- | :--- | :--- |
| **Instruction & Socratic Scaffolding** | **13.5 / 15** | The model **correctly refused to give direct answers** when instructed with Socratic constraints, showing improved negative constraint following vs CAND-01 |
| **Grounding & Anti-Hallucination** | **6.0 / 10** | When provided explicit textbook context, grounding improved over CAND-01 |
| **Generation Speed** | **10.22 tok/s** | 2× slower than CAND-01's 21.6 tok/s but still 2.5× above the minimum 4.0 tok/s target |

---

## 4. CAND-01 vs CAND-02 Comparison

| Dimension | CAND-01 (0.5B Q4_K_M) | CAND-02 (1.5B Q4_K_M) | Winner |
| :--- | :--- | :--- | :--- |
| Composite Score | 50.5 / 100 | 46.5 / 100 | CAND-01 |
| Bengali Quality | 2.0 / 20 | 0.0 / 20 | CAND-01 |
| Educational Reasoning | 15.0 / 25 | 9.0 / 25 | CAND-01 |
| Instruction Following | 15.0 / 15 | 13.5 / 15 | CAND-01 |
| Peak RAM (RSS) | **680.11 MB** ✅ | **1,771.26 MB** ❌ | **CAND-01 by far** |
| Speed (tok/s) | 21.6 | 10.22 | CAND-01 |
| File Size | 468.64 MB | 1,065.56 MB | CAND-01 |
| License | Apache-2.0 ✅ | Apache-2.0 ✅ | Tie |

> **Key insight:** Larger parameter count alone does NOT improve Bengali tutoring quality without Bengali-specific fine-tuning or retrieval grounding. CAND-02's extra parameters are consumed by model weight overhead without yielding Bengali language gains.

---

## 5. Recommended Next Steps

### Immediate — Q3_K_M / IQ3_M Quantization Trial of CAND-02

> **NOT yet confirmed viable.** This requires another download and benchmark.

Theoretical projection for CAND-02 Q3_K_M:
- Expected file size: ~820 MB (vs 1,065 MB for Q4_K_M)
- Expected peak RSS: ~1,350–1,450 MB
- Still likely **too large** for 2 GB Android target

### Primary Recommendation — RAG-Grounded Evaluation with CAND-01 (0.5B)

CAND-01's 680 MB RSS **fits** the target RAM envelope. Its 50.5/100 score failed primarily due to **ungrounded free-form Bengali generation**. The hypothesis is:

> **With structured RAG context grounding + scaffolded prompts + output sanitization, CAND-01 may achieve acceptable tutoring quality for NCTB Class 8 Mathematics on the target hardware.**

Phase 2 should evaluate CAND-01 with:
1. `build_grounded_rag_prompt()` textbook context injection
2. Socratic scaffolding via `build_step_by_step_math_prompt()`
3. Output sanitization via `core/sanitization/cleaner.py`

### Alternative — Evaluate a Q4_K_M Quantization of a Different 1B Model

Consider `TinyLlama-1.1B` (CAND-07, ~720 MB RAM, Apache-2.0) as an alternative 1B-class candidate fitting the RAM budget — though tokenizer benchmark showed poor Bengali compression (32K vocab).

---

## 6. Decision Summary

```
==========================================================================
CANDIDATE:     CAND-02 (Qwen2.5-1.5B-Instruct Q4_K_M)
DECISION:      FAIL — DISQUALIFIED
PRIMARY CAUSE: Gate 4 CRITICAL FAIL: Peak RSS 1,771.26 MB >> 750 MB limit
               Gate 2 FAIL: Bengali Quality 0.0/20
               Gate 3 FAIL: Educational Reasoning 9.0/25
NEXT ACTION:   Proceed with RAG-grounded evaluation of CAND-01 (0.5B)
               OR evaluate CAND-02 with more aggressive quantization
               (Q3_K_M or IQ3_M) in a future benchmark cycle
==========================================================================
```
