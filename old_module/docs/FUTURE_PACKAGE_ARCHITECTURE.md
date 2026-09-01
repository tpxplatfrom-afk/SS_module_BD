# SS Tutor BD — Future Package Architecture Direction

**Document Version:** 1.0.0  
**Phase:** 8 — Architectural Foundation  

> [!IMPORTANT]
> **Package distribution is NOT implemented in Phase 8.**
> This document specifies the future-facing architectural direction to ensure current knowledge and model foundations remain package-friendly from day one.

---

## 1. Future Package Derivation Concept

When the full Core Model achieves maturity across Class 6–10, modular packages can be extracted using the `CurriculumScope` abstraction without rewriting the inference runtime:

```text
                    SS TUTOR BD CORE MODEL
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    [FUTURE PACK]       [FUTURE PACK]       [FUTURE PACK]
     SS Tutor Full     Class 6–10 Math      Class 9–10 Science
    (All Subjects)     (Grades 6,7,8,9,10)   (Physics/Chem/Bio)
```

---

## 2. Package Separation Abstraction

1. **`KnowledgePack (.ssp)`:** A self-contained SQLite FTS5 database containing curated textbook facts, formulas, and topic nuggets for a specific grade/subject scope.
2. **`Model Runtime Module`:** The universal, model-agnostic inference engine that consumes `.ssp` knowledge packs dynamically.
