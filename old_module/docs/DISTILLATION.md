# SS Tutor BD — Phase 4 Distillation & Training Architecture

**Document Version:** 1.0.0  
**Phase:** 4 — Bengali Micro-Model Training & Distillation  
**Date:** 2026-08-30  
**Cost:** \$0 USD (Fully Local, CPU-Compatible)  

---

## 1. Distillation & Training Philosophy

Phase 3C proved that general-purpose 7B/1B models cannot run within a 200 MB Android memory ceiling.

Phase 4 introduces a **curriculum-distilled educational micro-model (~70M parameters)** trained with a dedicated **16,000-vocabulary Bengali tokenizer**.

### Dual-Role Architecture:
```text
  Offline Training Phase (CPU / Local / $0)
  ──────────────────────────────────────────
  Curriculum First Principles / Synthetic Generators
                      │
                      ▼
  High-Density Structured JSONL Training Sets (13,000 Examples)
  [Math Verbalization | Socratic Hints | Grounding | Linguistic Variants]
                      │
                      ▼
  Custom 16K Bengali Tokenizer + 70M Compact Transformer
                      │
                      ▼
  Supervised Educational Fine-Tuning & Knowledge Distillation
                      │
                      ▼
  INT4 Quantization & Micro-Runtime Export (34.1 MB Binary)
```

---

## 2. Model Architecture Candidates Matrix

| Candidate | Layers | Hidden Dim | Attention Heads | FFN Dim | Parameter Count | INT4 Binary Size | Estimated Peak RSS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A (50M)** | 8 | 512 | 8 | 2048 | 49.8M | 25.4 MB | ~75–95 MB |
| **Candidate B (70M - Primary)** | **10** | **576** | **8** | **2304** | **68.2M** | **34.1 MB** | **~90–120 MB** |
| **Candidate C (90M)** | 12 | 640 | 8 | 2560 | 89.6M | 44.8 MB | ~125–155 MB |

---

## 3. Training Hyperparameters & Bounded Context

* **Sequence Length:** 256 tokens (Strictly bounded to match `TIER_LOW` mobile policy).
* **Batch Size:** 8 (gradient accumulation = 4).
* **Learning Rate:** $3 \times 10^{-4}$ with Cosine Annealing.
* **Loss Function:** Cross-Entropy with label smoothing (0.1).
* **Data Mix Ratio:**
  * Math Tool Verbalization: 40%
  * Grounded Q&A & Anti-Hallucination: 25%
  * Socratic Hint Scaffolding: 20%
  * Linguistic & Banglish Robustness: 15%
