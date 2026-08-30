# SS Tutor BD — Class 8 Mathematics Content Provenance & Specification

**Curriculum Level:** Bangladesh NCTB Class 8 (অষ্টম শ্রেণি)  
**Subject:** Mathematics (গণিত)  
**Document ID:** `SSP-NCTB-CL8-MATH-PROVENANCE`  
**Date:** 2026-08-30  
**License / Terms:** Public Domain Educational Curriculum Content (NCTB Open Curriculum Reference Structure)  

---

## 1. Content Provenance & Scope

The content for this prototype is adapted from the official National Curriculum and Textbook Board (NCTB) Bangladesh Class 8 Mathematics curriculum framework.

* **Curriculum Authority:** National Curriculum and Textbook Board (NCTB), Bangladesh (জাতীয় শিক্ষাক্রম ও পাঠ্যপুস্তক বোর্ড, বাংলাদেশ)
* **Book Reference:** অষ্টম শ্রেণির গণিত পাঠ্যপুস্তক (Class 8 Mathematics Textbook)
* **Language:** Bengali (প্রমিত বাংলা) with standard mathematical notations and English terminologies.
* **Retrieval & Ingestion Date:** 2026-08-30
* **Transformation Performed:**
  1. Systematic structuring into semantic Markdown chapters with explicit Section (`##`), Formula (`###`), Worked Example (`উদাহরণ`), and Exercise (`অনুশীলনী`) demarcations.
  2. Formatting of equations into standardized LaTeX-compatible expressions ($I = Prn$, $(a+b)^2$, etc.).
  3. Semantic metadata tagging: `class=8`, `subject=mathematics`, `pack_id=ssp-nctb-cl8-math-v1`.

---

## 2. Selected Chapter Modules

| Chapter ID | Chapter Title (Bengali) | Chapter Topic / Domain | Key Formulas & Concepts Included |
| :--- | :--- | :--- | :--- |
| **CH-01** | প্যাটার্ন (Patterns) | Arithmetic Sequences | $k$-তম পদ, ক্রমিক সংখ্যার যোগফল $S_n = \frac{n(n+1)}{2}$, ৩-ক্রমের ম্যাজিক বর্গ |
| **CH-02** | মুনাফা (Profit & Interest) | Financial Arithmetic | সরল মুনাফা $I = Prn$, চক্রবৃদ্ধি মূলধন $C = P(1+r)^n$, চক্রবৃদ্ধি মুনাফা $C - P$ |
| **CH-03** | পরিমাপ (Measurement) | Applied Measurement | ক্ষেত্রফল, ঘনফল, তরল পরিমাপ, মেট্রিক ও ব্রিটিশ একক রূপান্তর |
| **CH-04** | বীজগণিতীয় সূত্রাবলী ও প্রয়োগ | Algebra & Polynomials | $(a \pm b)^2$, $(a \pm b)^3$, $a^2 - b^2$, অনুসিদ্ধান্ত, মধ্যপদ বিভাজন (Middle-term) |
| **CH-05** | বীজগণিতীয় ভগ্নাংশ | Algebraic Fractions | সাধারণ হরবিশিষ্টকরণ, ভগ্নাংশের যোগ-বিয়োগ, গুণ, ভাগ ও লঘিষ্ঠকরণ |
| **CH-06** | সরল সহসমীকরণ | Linear Equations | প্রতিস্থাপন পদ্ধতি (Substitution), অপনয়ন পদ্ধতি (Elimination) |
| **CH-08** | চতুর্ভুজ ও জ্যামিতিক পরিমাপ | Geometry | চতুর্ভুজের বৈশিষ্ট্য, পিথাগোরাসের উপপাদ্য $c^2 = a^2 + b^2$, বৃত্তের পরিধি $2\pi r$ |

---

## 3. Data Integrity & Verification

All structured chapters are verified for:
* Correct Bengali Unicode character representation (\u0980-\u09FF).
* Complete worked solutions with stepwise reasoning.
* Deterministic chunk ID mapping (`ssp-nctb-cl8-math-v1-chXX-secYY-cZZZ`).
