# SS Tutor BD — Known Limitations & Scope Boundaries

**Document Version:** 1.0.0  
**Phase:** 6 — Release Certification  

---

## 1. Supported Curriculum Scope

* **Primary Subject:** Class 8 Mathematics (NCTB Bangladesh Curriculum).
* **Supported Domains:** Fractions, Simple/Compound Interest, Pythagoras Theorem, Arithmetic Series, Linear Systems, Quadratic Factoring, Circle Geometry, Unit Conversions.
* **Knowledge Pack Format:** `.ssp` package (SQLite FTS5 full-text search).

---

## 2. Intentional Architectural Boundaries

1. **Deterministic Authority:** The neural micro-model is strictly a pedagogical language verbalizer and is not authorized to generate standalone mathematical answers. If mathematical input is detected, the deterministic math core overrides model output.
2. **Anti-Hallucination Polite Refusal:** Inquiries beyond the packaged NCTB textbook topics receive a deterministic polite refusal ("প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না").
3. **Bounded Multi-Turn Session:** In order to guarantee the strict $\le 200\text{ MB}$ PSS ceiling on 2 GB RAM phones, conversational history is bounded to compact topic metadata rather than unbounded chat transcript tokens.
