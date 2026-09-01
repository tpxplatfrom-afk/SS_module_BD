# SS Tutor BD — Model, Knowledge & Tool Architectural Boundaries

**Document Version:** 1.0.0  
**Phase:** 8.1 — Forensic Discovery  

---

## 1. Multi-Layer Subsystem Boundary

To ensure the Core Model remains modular, reusable, and forkable for other domains (e.g. Mechanics), we strictly demarcate four distinct architectural subsystems:

```text
                               SS TUTOR BD
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
[NEURAL MODEL LAYER]      [KNOWLEDGE & RAG LAYER]      [DETERMINISTIC TOOLS LAYER]
- LLaMA 70M Architecture  - NCTB Curriculum Schema     - Exact Math Solvers (fraction,
- 54.3M-68.2M Weights     - SQLite FTS5 Index            calculator, series, pythagoras)
- 16K BPE Tokenizer       - Chapter Markdown Files     - 5 Multi-Guard Validators
- RoPE + SiLU + RMSNorm   - CurriculumScope Boundaries - Context Compressor
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    │
                                    ▼
                         [RUNTIME ADAPTER LAYER]
                         - Native MicroRuntime Engine
                         - O(1) Session Memory Manager
                         - <= 200MB PSS Memory Budget Guard
                         - SSTutorBDModule API Contract
```

---

## 2. Forking Feasibility for Future Specialized Domains (e.g. Mechanics AI Module)

When forking this Core Model for a new specialization:

| Component | In Core Master | Action for Mechanics Specialization |
| :--- | :--- | :--- |
| **Transformer Architecture** | 10-layer, 576-dim LLaMA | **Reuse 100%** as-is |
| **Bengali Tokenizer** | 16,000 Byte-level BPE | **Reuse 100%** as-is |
| **Inference Runtime** | `micro_runtime.py` | **Reuse 100%** as-is |
| **Session Manager** | $O(1)$ Bounded State | **Reuse 100%** as-is |
| **Training Dataset** | `data/phase4/` | **Replace** with Mechanics dataset (`data/mechanics/`) |
| **Knowledge Pack / RAG** | `packs/class8_math/` | **Replace** with Mechanics pack (`packs/mechanics/`) |
| **Deterministic Tools** | Math Arithmetic | **Extend** with Physics/Mechanics solvers (Torque, Friction, Kinematics) |
| **Domain Weights** | `model.safetensors` | **Retrain / Distill** to produce `models/mechanics_70m_edu/` |
