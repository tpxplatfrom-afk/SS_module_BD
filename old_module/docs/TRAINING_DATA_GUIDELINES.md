# SS Tutor BD — Training Data Engineering Guidelines

**Document Version:** 1.0.0  
**Phase:** 8 — Core Model Development  

---

## 1. Educational Tutoring Modes

Training datasets must reflect diverse educational dialogue modes rather than monolithic synthetic QA:

1. **`[T] explain` — Pedagogical Explanation:** Simple, student-friendly Bengali explanation of theoretical principles.
2. **`[T] solve` — Step-by-Step Problem Solving:** Detailed step-by-step intermediate reduction with reasoning.
3. **`[T] hint` — Socratic Hinting:** Guides student intuition without leaking direct answers.
4. **`[T] misconception` — Misconception Correction:** Detects erroneous student assumptions and clarifies why they are incorrect.
5. **`[T] followup` — Conversational Clarification:** Responds to student follow-ups ("সহজ করে বলো", "আরেকটা উদাহরণ দাও").
6. **`[T] grounded` — Polite Anti-Hallucination Refusal:** Explicit refusal on ungrounded out-of-scope queries ("প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না").

---

## 2. Bengali Quality & Formatting Standards

* **Language Purity:** Authentic, natural Bengali grammar with appropriate educational terminology.
* **Numeral Consistency:** Bengali numerals (`১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯, ০`) with Arabic numeral fallback where appropriate.
* **Typo Tolerance:** Robustness against colloquial Bengali queries (`"ভাই বুঝি নাই"`, `"সহজ করে বলেন"`).
