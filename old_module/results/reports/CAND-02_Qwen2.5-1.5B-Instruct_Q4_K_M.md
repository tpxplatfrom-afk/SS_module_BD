# SS Tutor BD — Benchmark Report: Qwen2.5-1.5B-Instruct

**Candidate ID:** `CAND-02`  
**Model Name:** `Qwen2.5-1.5B-Instruct`  
**Publisher:** Alibaba Cloud  
**Parameter Count:** 1.54B  
**Quantization Tested:** `Q4_K_M`  
**Evaluation Date:** 2026-08-30 12:02 UTC  
**Overall Benchmark Status:** **DID NOT PASS ALL GATES (Total Score: 46.5 / 100)**

---

## 1. Executive Summary & Recommendation

| Assessment Dimension | Result | Target Benchmark Status |
| :--- | :--- | :--- |
| **Composite Score** | **46.5 / 100** | FAIL (< 70) |
| **License Compliance** | `LICENSE_PASSED` | PASS |
| **Peak Memory (RSS)** | **1771.26 MB** | FAIL (> 750 MB) |
| **Generation Speed** | **10.22 tok/s** | PASS (>= 4.0 tok/s) |
| **Bengali Linguistic Score** | **0.0 / 20** | NEEDS IMPROVEMENT |
| **Educational Reasoning Score** | **9.0 / 25** | NEEDS IMPROVEMENT |

---

## 2. 100-Point Scorecard Breakdown

```
========================================================================================
CATEGORY                                SCORE / MAX        PERCENTAGE    STATUS
========================================================================================
1. Bengali Linguistic Quality           0.0    / 20.0        0.0   %     FAIL
2. Educational Reasoning (Math/Sci)     9.0    / 25.0        36.0  %     FAIL
3. Mobile Resource Efficiency           8.0    / 20.0        40.0  %     FAIL
4. Instruction & Socratic Scaffolding   13.5   / 15.0        90.0  %     PASS
5. Knowledge Grounding & Anti-Halluc.   6.0    / 10.0        60.0  %     PASS
6. License & Redistribution Freedom     10.0   / 10.0        100.0%     PASS
----------------------------------------------------------------------------------------
TOTAL COMPOSITE SCORE                   46.5   / 100.0       46.5%      FAILED
========================================================================================
```

---

## 3. Sequential Gate Validation

* **Gate 1 (License Gate):** `PASS` (Declared: `Apache-2.0`)
* **Gate 2 (Bengali Linguistic Gate):** `FAIL` (Score: 0.0 / 20, Min: 10.0)
* **Gate 3 (Educational Reasoning Gate):** `FAIL` (Score: 9.0 / 25, Min: 12.0)
* **Gate 4 (Mobile Memory Gate):** `FAIL` (Peak RSS: 1771.26 MB, Cap: 750 MB)
* **Gate 5 (Mobile Speed Gate):** `PASS` (Throughput: 10.22 tok/s, Min: 4.0 tok/s)
* **Gate 6 (Composite Threshold Gate):** `FAIL` (Total: 46.5 / 100, Min: 70.0)

---

## 4. Resource & Latency Measurements

* **Model File Size:** `1065.56 MB`
* **Model Load Time:** `2673.14 ms`
* **Time-to-First-Token (TTFT):** `750.0 ms`
* **Average Generation Throughput:** `10.22 tokens/second`
* **Peak Resident Set Size (RSS):** `1771.26 MB`
* **Host Platform:** Windows 10 Pro (x64) — Intel i5-6500 @ 3.20GHz

---

## 5. Sample Evaluation Outputs

### Sample A: Bengali Explanation (BN-002)
> *Prompt:* `ভাইয়া, আমার সূচকের অংকগুলো বুঝতে খুব সমস্যা হচ্ছে। সহজ ভাষায় সূচক (Exponent) কী একটু বুঝিয়ে বলবেন?`  
> *Model Output:*  
> ভাই, সহজ ভাষা সূচক (Exponent) একটি অংক যা বুঝা যায় যে একটি কোনো ক্ষেত্রে কোনো কোনো কোনো ক্ষেত্রে কোনো কোনো কোনো কোনো ক্ষেত্রে কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কোনো কো

### Sample B: Mathematics Step-by-Step (MATH-001)
> *Prompt:* `3/4 + 5/6 এর যোগফল নির্ণয় করো এবং অপ্রকৃত ভগ্নাংশ থেকে মিশ্র ভগ্নাংশে রূপান্তর করো।`  
> *Model Output:*  
> আমার জন্য সব ধাপ বাংলা দেখানো হবে।

1. প্রথম ধাপ: যোগফল নির্ণেয় করুন। এই পদক্ষেপে আমরা দুটি ভগ্নাংশ যোগ করব। এর মাধ্যমে আমরা দুটি ভগ্নাংশের ভগ্নাংশ এবং ভগ্নাংশের ভগ্নাংশের ভগ্নাংশ একে যোগ করব। এর মাধ্যমে আমরা একটি ভগ্নাংশ পাব।

2. দুটি ভগ্নাংশ যো

### Sample C: Negative Constraint / Socratic Hint (PED-001)
> *Prompt:* `প্রশ্ন: x^2 - 5x + 6 = 0 এর সমাধান কী? আমাকে কিন্তু সরাসরি উত্তর বলবে না, শুধু ইঙ্গিত দাও।`  
> *Model Output:*  
> আমার জন্য এই প্রশ্নটি সরাসরি উত্তর না বলব। আমি প্রথম দরকারি ইঙ্গিত দেখবে। আমার জন্য এই প্রশ্নটি সমাধান করার জন্য প্রথম দরকারি ইঙ্গিত দেখবে।

এই প্রশ্নটি একটি বহুপদী সমীকরণ হলে আমরা কিছু ইঙ্গিত দেখব। এই সমীকরণে আমরা প্রথম দরকারি ইঙ্গিত দেখবে।

এই প্রশ্নট

---

## 6. Failure Analysis Summary

Total Failed / Sub-optimal Test Cases Recorded: **0**  
Detailed failure dump saved to: `results/failures/CAND-02_failures.json`

---

## 7. Next Architectural Recommendation

Based on this evaluation:
* If the candidate passed all gates, it qualifies for **Phase 2 (Class 8 Math Retrieval & Prototype)**.
* If memory or speed gates failed, evaluate **more aggressive quantization (Q3_K_M / IQ2_M)** or step down to an alternative candidate.
