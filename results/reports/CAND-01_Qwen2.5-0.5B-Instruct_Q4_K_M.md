# SS Tutor BD — Benchmark Report: Qwen2.5-0.5B-Instruct

**Candidate ID:** `CAND-01`  
**Model Name:** `Qwen2.5-0.5B-Instruct`  
**Publisher:** Alibaba Cloud  
**Parameter Count:** 0.49B  
**Quantization Tested:** `Q4_K_M`  
**Evaluation Date:** 2026-08-30 12:02 UTC  
**Overall Benchmark Status:** **DID NOT PASS ALL GATES (Total Score: 50.5 / 100)**

---

## 1. Executive Summary & Recommendation

| Assessment Dimension | Result | Target Benchmark Status |
| :--- | :--- | :--- |
| **Composite Score** | **50.5 / 100** | FAIL (< 70) |
| **License Compliance** | `LICENSE_PASSED` | PASS |
| **Peak Memory (RSS)** | **680.11 MB** | PASS (<= 750 MB) |
| **Generation Speed** | **21.6 tok/s** | PASS (>= 4.0 tok/s) |
| **Bengali Linguistic Score** | **1.0 / 20** | NEEDS IMPROVEMENT |
| **Educational Reasoning Score** | **10.0 / 25** | NEEDS IMPROVEMENT |

---

## 2. 100-Point Scorecard Breakdown

```
========================================================================================
CATEGORY                                SCORE / MAX        PERCENTAGE    STATUS
========================================================================================
1. Bengali Linguistic Quality           1.0    / 20.0        5.0   %     FAIL
2. Educational Reasoning (Math/Sci)     10.0   / 25.0        40.0  %     FAIL
3. Mobile Resource Efficiency           15.0   / 20.0        75.0  %     PASS
4. Instruction & Socratic Scaffolding   7.5    / 15.0        50.0  %     FAIL
5. Knowledge Grounding & Anti-Halluc.   7.0    / 10.0        70.0  %     PASS
6. License & Redistribution Freedom     10.0   / 10.0        100.0%     PASS
----------------------------------------------------------------------------------------
TOTAL COMPOSITE SCORE                   50.5   / 100.0       50.5%      FAILED
========================================================================================
```

---

## 3. Sequential Gate Validation

* **Gate 1 (License Gate):** `PASS` (Declared: `Apache-2.0`)
* **Gate 2 (Bengali Linguistic Gate):** `FAIL` (Score: 1.0 / 20, Min: 10.0)
* **Gate 3 (Educational Reasoning Gate):** `FAIL` (Score: 10.0 / 25, Min: 12.0)
* **Gate 4 (Mobile Memory Gate):** `PASS` (Peak RSS: 680.11 MB, Cap: 750 MB)
* **Gate 5 (Mobile Speed Gate):** `PASS` (Throughput: 21.6 tok/s, Min: 4.0 tok/s)
* **Gate 6 (Composite Threshold Gate):** `FAIL` (Total: 50.5 / 100, Min: 70.0)

---

## 4. Resource & Latency Measurements

* **Model File Size:** `468.64 MB`
* **Model Load Time:** `860.57 ms`
* **Time-to-First-Token (TTFT):** `750.0 ms`
* **Average Generation Throughput:** `21.6 tokens/second`
* **Peak Resident Set Size (RSS):** `680.11 MB`
* **Host Platform:** Windows 10 Pro (x64) — Intel i5-6500 @ 3.20GHz

---

## 5. Sample Evaluation Outputs

### Sample A: Bengali Explanation (BN-002)
> *Prompt:* `ভাইয়া, আমার সূচকের অংকগুলো বুঝতে খুব সমস্যা হচ্ছে। সহজ ভাষায় সূচক (Exponent) কী একটু বুঝিয়ে বলবেন?`  
> *Model Output:*  
> ভাইabwe, আমার সূচকের অংকগুলো বুঝতে খুব সমস্যা হচ্ছে। সহজ ভাষাabwe (Exponent) কী একটু বুঝিabwebih বলবেন তা আমার সূচকের অংকগুলো বুঝতে খুব সমস্যা হচ্ছে। সহজ ভাষাabwe কী একটু বুঝিbih বলবেন তা আমার সূচকের অংকগুলো বুঝতে খুব সমস্যা হচ্ছে।

### Sample B: Mathematics Step-by-Step (MATH-001)
> *Prompt:* `3/4 + 5/6 এর যোগফল নির্ণয় করো এবং অপ্রকৃত ভগ্নাংশ থেকে মিশ্র ভগ্নাংশে রূপান্তর করো।`  
> *Model Output:*  
> প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংলা/csv দেখার জন্য প্রতিটি ধাপ বাংল

### Sample C: Negative Constraint / Socratic Hint (PED-001)
> *Prompt:* `প্রশ্ন: x^2 - 5x + 6 = 0 এর সমাধান কী? আমাকে কিন্তু সরাসরি উত্তর বলবে না, শুধু ইঙ্গিত দাও।`  
> *Model Output:*  
> প্রশ্ন: x^2 - 5x + 6 = 0 এর সমাধান কী?

আমাকে কিন্তু সরাসরি উত্তর বলবে না, শুধু সমাধান করার জন্য প্রথম দরকারি ইঙ্গিত (Hint 1) দাও।

এখন আমি একটি সমাধান করার জন্য ইঙ্গিত করার জন্য প্রথম দরকারি ইঙ্গিত করার জন্য প্রথম দরকারি ইঙ্গিত করার জন্য প্রথম দরকারি ইঙ্গিত করার

---

## 6. Failure Analysis Summary

Total Failed / Sub-optimal Test Cases Recorded: **0**  
Detailed failure dump saved to: `results/failures/CAND-01_failures.json`

---

## 7. Next Architectural Recommendation

Based on this evaluation:
* If the candidate passed all gates, it qualifies for **Phase 2 (Class 8 Math Retrieval & Prototype)**.
* If memory or speed gates failed, evaluate **more aggressive quantization (Q3_K_M / IQ2_M)** or step down to an alternative candidate.
