# SS Tutor BD — Phase 7 Known Limitations & Scope Boundaries

**Document Version:** 1.0.0  
**Phase:** 7 — Production Certification  

---

## 1. Supported Curriculum Boundaries

* **Curriculum Focus:** Bangladesh National Curriculum (NCTB) Class 8 Mathematics.
* **Knowledge Pack Format:** `.ssp` package (SQLite FTS5 full-text search index).
* **Future Class Support:** Class 6, 7, 9, and 10 can be integrated by adding modular `.ssp` knowledge packs without modifying the underlying inference engine or runtime contract.

---

## 2. Intentional Memory & Algorithmic Guardrails

1. **Context Window Constraint:** Maximum prompt context is limited to 256 tokens to guarantee the strict $\le 200\text{ MB}$ PSS ceiling on 2 GB RAM phones.
2. **Deterministic Precedence:** Mathematical queries are always dispatched to authoritative deterministic algorithms (`fraction.py`, `calculator.py`, `equation_solver.py`), preventing hallucinated numerical outputs.
3. **Out-of-Scope Refusal:** Inquiries outside of the active curriculum trigger a polite anti-hallucination refusal ("প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না").
