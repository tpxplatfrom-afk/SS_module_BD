# SS Tutor BD — Offline-First Architecture Specification

**Document Version:** 1.0.0  
**Phase:** 5 — Android / Native UI Binding  
**Zero-Network Guarantee:** 100% Core Functionality in Airplane Mode  

---

## 1. Zero-Network Subsystem Design

Every core layer in SS Tutor BD operates with zero external network connectivity:
1. **Mathematical Intent & Calculation:** 100% local rule-based parsing and exact arithmetic algorithms.
2. **Curriculum Knowledge Retrieval:** Packaged SQLite FTS5 database (`.ssp` format) accessed locally via native SQLite file descriptors.
3. **Dedicated Bengali Tokenizer:** Byte-level BPE vocabulary loaded directly from local JSON assets without Hugging Face API calls.
4. **Micro-Model Inference:** Local INT4 weights mapped directly via `mmap` into the process address space.
5. **Multi-Guard Validation:** Local heuristic and pattern-matching rules without cloud verification.

```text
Student Device (Airplane Mode ON, WiFi OFF)
─────────────────────────────────────────────
[Student Question] ──▶ [Local Native Router]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [Local Math Core]           [Local SQLite FTS5 RAG]
               │                             │
               └──────────────┬──────────────┘
                              ▼
                [Local MicroRuntime / INT4]
                              │
                [Local Validation Guards]
                              │
               [Instant Bengali Response]
```
