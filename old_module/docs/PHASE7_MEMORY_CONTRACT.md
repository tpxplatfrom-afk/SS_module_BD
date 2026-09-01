# Phase 7 Production Memory Contract Specification

**Document Version:** 1.0.0  
**Phase:** 7 — Production Certification  
**Target Hardware:** itel A662L (2 GB Physical RAM / Android 12 Go / ARMv7a)  

---

## 1. Immutable State Ceilings

| Operational State | Subsystem Components | Preferred PSS | Hard Production Ceiling | Emergency Limit |
| :--- | :--- | :--- | :--- | :--- |
| **State A: Cold Launch** | App + UI + RAG + Math (Model Unloaded) | $\le 100\text{ MB}$ | $\le 150\text{ MB}$ | $200\text{ MB}$ |
| **State B: Model Loaded / Idle** | App + UI + RAG + Math + INT4 Model Weights (No Inference) | $\le 150\text{ MB}$ | $\le 180\text{ MB}$ | $220\text{ MB}$ |
| **State C: Full Hybrid Inference** | App + UI + RAG + Math + Model + KV Cache + Validators | $\le 150\text{ MB}$ | **$\le 200\text{ MB}$** | **$250\text{ MB}$** |
| **State D: 100-Turn Session** | Hybrid Engine over 100 Consecutive Turns | $\le 150\text{ MB}$ | **$\le 200\text{ MB}$** | **$250\text{ MB}$** |
| **State E: 500-Turn Stress** | Continuous Stress Queries (Maximum Context) | $\le 150\text{ MB}$ | **$\le 200\text{ MB}$** | **$250\text{ MB}$** |

---

## 2. Hard Invariants

1. **The 200 MB Production Contract is Non-Negotiable:** If total process PSS exceeds $200\text{ MB}$ under any steady-state condition, the system fails production certification.
2. **Deterministic Priority:** When a math query or pure textbook lookup is received, the neural micro-model is never loaded, maintaining cold process PSS ($22.85\text{ MB}$).
3. **Emergency Eviction:** If Android signals `TRIM_MEMORY_RUNNING_CRITICAL`, the neural model is immediately evicted and the system falls back to the deterministic core.
