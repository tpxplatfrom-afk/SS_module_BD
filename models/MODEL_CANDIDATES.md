# SS Tutor BD — Model Candidates Specification (Phase 3B Revision)

**Version:** 3.2.0  
**Phase:** 3B — Production Memory-Constrained Model Selection  
**Production RAM Target:** 150–200 MB (Hard Ceiling: 250 MB)  
**Host Environment:** Windows 10 Pro (x64), CPU Inference, $0 Development Cost  
**Target Deployment:** Android (2 GB RAM, 16 GB Storage)  

---

## 1. Candidate Classification & Memory Tiers

Candidates are categorized under the **Phase 3B Production Memory Contract**:

| Candidate ID | Model Name | Parameter Count | Quantization | GGUF Size (MB) | Est. Peak RAM (MB) | License | Tier Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAND-03** | `SmolLM2-135M-Instruct` | 0.135B | `Q4_K_M` | ~105 MB | **~150–180 MB** | Apache-2.0 | **PRIMARY PRODUCTION CANDIDATE** (Fits <=200 MB) |
| **CAND-04** | `SmolLM2-360M-Instruct` | 0.36B | `Q4_K_M` | ~240 MB | **~240–280 MB** | Apache-2.0 | **WARNING TIER CANDIDATE** |
| **CAND-01** | `Qwen2.5-0.5B-Instruct` | 0.49B | `Q4_K_M` | 468.64 MB | **738.07 MB** | Apache-2.0 | **RETIRED FROM PRODUCTION** (Exceeds 250 MB ceiling) |
| **CAND-02** | `Qwen2.5-1.5B-Instruct` | 1.54B | `Q4_K_M` | 1065.56 MB | **1,771.26 MB** | Apache-2.0 | **DISQUALIFIED (Phase 2)** |
| **CAND-07** | `TinyLlama-1.1B-Chat` | 1.10B | `Q4_K_M` | 668 MB | **~720 MB** | Apache-2.0 | **DISQUALIFIED** (Exceeds 250 MB ceiling) |

---

## 2. Gate Requirements for Phase 3B

* **Gate 1 (License):** Permissive (`Apache-2.0`, `MIT`, `BSD`).
* **Gate 2 (Binary Size):** Preferred $\le 150\text{ MB}$, Warning $150–200\text{ MB}$.
* **Gate 3 (Production Memory):** Peak RSS $\le 200\text{ MB}$ preferred, $\le 250\text{ MB}$ hard ceiling. Sustained $>250\text{ MB}$ = **FAIL**.
* **Gate 4 (Speed):** $\ge 4.0\text{ tok/s}$.
* **Gate 5 (Grounded Bengali Quality):** $\ge 70\%$.
* **Gate 6 (Educational Tutoring):** $\ge 70\%$.
* **Gate 7 (Hybrid Mathematical Correctness):** $\ge 90\%$.
* **Gate 8 (Textbook Grounding):** $\ge 95\%$.
