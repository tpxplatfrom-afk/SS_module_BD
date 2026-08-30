# SS Tutor BD — Curriculum Knowledge Architecture

**Document Version:** 1.0.0  
**Phase:** 8 — Core Model Development  

---

## 1. Ontological Hierarchy

Curriculum knowledge is structured hierarchically with deterministic identifiers:

```text
Grade (Class 6, 7, 8, 9, 10)
  └── Subject (Mathematics, Science, Bengali, English)
       └── Book (NCTB Official Framework)
            └── Chapter (e.g. Chapter 2: Profit & Loss)
                 └── Topic (e.g. Simple Interest vs Compound Interest)
                      └── Concept (e.g. Formula I = Prn)
                           ├── Definition
                           ├── Mathematical Formula
                           ├── Pedagogical Explanation
                           ├── Worked Example
                           ├── Exercise Problem
                           ├── Common Misconceptions
                           └── Socratic Hints
```

---

## 2. Deterministic Identifier Syntax

Concept identifiers follow a standard notation:
```text
g{GRADE:02d}.{SUBJECT_ABBR}.ch{CHAPTER:02d}.t{TOPIC:02d}.c{CONCEPT:02d}
```

**Examples:**
* `g08.math.ch02.t01.c01` — Class 8 Mathematics, Chapter 2 (Profit), Topic 1, Concept 1 (Simple Interest).
* `g08.math.ch08.t02.c03` — Class 8 Mathematics, Chapter 8 (Geometry), Topic 2, Concept 3 (Pythagoras Theorem).
* `g09.math.ch09.t01.c02` — Class 9 Mathematics, Chapter 9 (Trigonometry), Topic 1, Concept 2 (Sin/Cos/Tan Ratios).
