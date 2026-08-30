# Benchmark Dataset Specification: SS Tutor BD

**Document Version:** 1.0.0  
**Target Application:** Bangladesh Secondary Education (NCTB Class 6–10)  
**Dataset Scale:** Small, standardized, manually reviewable evaluation set (Total: 30 curated test items)  
**Evaluation Mode:** Offline / Deterministic prompt execution across all candidate models

---

## 1. Dataset Design Principles

1. **Curriculum Alignment:** Questions and terminology mirror the official NCTB textbooks (Class 6 to Class 10).
2. **Pedagogical Relevance:** Items assess not just raw factual correctness, but teaching behavior, hint scaffolding, and error diagnostics.
3. **Linguistic Authenticity:** Covers standard Sadhu/Cholit Bengali, classroom colloquialisms, and common mixed English-Bengali student phrasing ("Banglish" / transliteration and bilingual math terms like *Equation*, *Factorize*, *Hypotenuse*).
4. **Manual Inspectability:** Kept compact (30 items) so every model output can be systematically inspected and scored by a human evaluator without automated metric illusions.

---

## 2. Category Breakdown & Specific Test Cases

### Category A: Bengali Comprehension & Linguistic Adaptability (6 Items)

| Test ID | Subcategory | Prompt Query (Bengali) | Expected Evaluation Criteria |
| :--- | :--- | :--- | :--- |
| **A-01** | Formal Bengali Instruction | `"নিচের বক্তব্যটির মূল ভাব সংক্ষেপে ৩টি বাক্যে বুঝিয়ে লেখো: 'পরিশ্রমই সৌভাগ্যের প্রসূতি।'"` | Coherent 3-sentence summary in standard Bengali without grammatical errors or language drift. |
| **A-02** | Student Colloquial Question | `"ভাইয়া, আমার সূচকের অংকগুলো বুঝতে খুব সমস্যা হচ্ছে। সহজ ভাষায় সূচক (Exponent) কী একটু বুঝিয়ে বলবেন?"` | Warm, encouraging tutor tone, conversational yet clear Bengali explanation using intuitive examples. |
| **A-03** | Mixed Terminology (Code-Switching) | `"Class 9 এর Math বইয়ের চ্যাপ্টার ৪ এর Exponent and Logarithm এর মূল সূত্রগুলো কী কী?"` | Accurately understands code-switched terms ("Class 9", "Math", "Chapter 4", "Exponent") and lists NCTB exponent laws in Bengali. |
| **A-04** | Bengali Dialectal / Spelling Variation | `"ভগ্নাংশের যোগ-বিয়োগ কেমনে করে? একটা উদাহরন দিয়া বুঝান।"` | Comprehends informal/non-standard phrasing (`"কেমনে"`, `"উদাহরন দিয়া"`) and responds in clean educational Bengali. |
| **A-05** | Complex Syntactic Parsing | `"যদি একটি ত্রিভুজের তিন বাহুর দৈর্ঘ্য যথাক্রমে ৩ সেমি, ৪ সেমি এবং ৫ সেমি হয়, তবে এটি কেন সমকোণী ত্রিভুজ হবে?"` | Correctly associates side lengths with the Pythagorean converse in Bengali. |
| **A-06** | Mathematical English-to-Bengali Mapping | `"দ্বিঘাত সমীকরণের 'Discriminant' বা 'নিশ্চায়ক' বলতে কী বোঝায়?"` | Correctly links the English term *Discriminant* with Bengali *নিশ্চায়ক* ($b^2 - 4ac$) and explains its geometric meaning. |

---

### Category B: NCTB Mathematics Reasoning (8 Items)

| Test ID | Subcategory | Problem Statement | Ground Truth / Target Derivation |
| :--- | :--- | :--- | :--- |
| **B-01** | Basic Arithmetic / Fractions (Class 6) | `"\frac{3}{4} + \frac{5}{6} এর যোগফল নির্ণয় করো এবং অপ্রকৃত ভগ্নাংশ থেকে মিশ্র ভগ্নাংশে রূপান্তর করো।"` | $\text{LCM}(4, 6) = 12 \rightarrow \frac{9+10}{12} = \frac{19}{12} = 1\frac{7}{12}$. |
| **B-02** | Word Problem / Unitary Method (Class 7) | `"১৫ জন শ্রমিক একটি কাজ ১০ দিনে শেষ করতে পারে। কতজন শ্রমিক ওই কাজটি ৬ দিনে শেষ করতে পারবে?"` | Inverse proportion: $15 \times 10 = 150$ worker-days $\rightarrow \frac{150}{6} = 25$ workers. |
| **B-03** | Basic Algebra / Exponents (Class 8) | `"সরল করো: \frac{2^{n+4} - 2 \cdot 2^n}{2^{n+2} \div 2}"` | Numerator: $2^n(2^4 - 2) = 2^n(14)$; Denom: $2^n \cdot 2^1 = 2 \cdot 2^n \rightarrow \frac{14}{2} = 7$. |
| **B-04** | Algebraic Identities (Class 9 Math Ch 3) | `"যদি a + b = 5 এবং ab = 6 হয়, তবে a^3 + b^3 এর মান কত?"` | Formula: $(a+b)^3 - 3ab(a+b) = 5^3 - 3(6)(5) = 125 - 90 = 35$. |
| **B-05** | Logarithmic Laws (Class 9 Math Ch 4) | `"মান নির্ণয় করো: \log_{10} 1000 + \log_{2} 16 - \ln e^3"` | $\log_{10}(10^3) + \log_2(2^4) - 3 = 3 + 4 - 3 = 4$. |
| **B-06** | Linear Equations / Word Problem (Class 9) | `"একটি আয়তাকার ক্ষেত্রের দৈর্ঘ্য প্রস্থের চেয়ে ৪ মিটার বেশি। পরিসীমা ৩৬ মিটার হলে ক্ষেত্রটির ক্ষেত্রফল কত?"` | $2(x + x + 4) = 36 \rightarrow 4x + 8 = 36 \rightarrow x = 7$. Length = 11, Width = 7. Area = $77\text{ m}^2$. |
| **B-07** | Geometry / Triangle Theorems (Class 8) | `"একটি সমকোণী ত্রিভুজের অতিভুজ ১৩ সেমি এবং ভূমি ১২ সেমি হলে, এর লম্ব এবং ক্ষেত্রফল কত?"` | Height = $\sqrt{13^2 - 12^2} = 5\text{ cm}$. Area = $\frac{1}{2} \times 12 \times 5 = 30\text{ cm}^2$. |
| **B-08** | Trigonometry (Class 9-10 Ch 9) | `"যদি \tan \theta = \frac{3}{4} হয়, তবে \sin \theta \cdot \cos \theta এর মান কত?"` | $\sin \theta = \frac{3}{5}, \cos \theta = \frac{4}{5} \rightarrow \frac{3}{5} \times \frac{4}{5} = \frac{12}{25}$. |

---

### Category C: Educational Explanation & Scaffolding (6 Items)

| Test ID | Skill Tested | Evaluation Scenario |
| :--- | :--- | :--- |
| **C-01** | Simple Concept Explanation | `"একটি ক্লাস ৬ এর শিক্ষার্থীকে 'লসাগু (LCM)' কী তা একটি বাস্তব জীবনের উদাহরণের মাধ্যমে বুঝিয়ে বলো।"` | Evaluates analogy quality (e.g. running around a track, ringing bells) without abstract jargon. |
| **C-02** | Step-by-Step Scaffolding | `"Class 9 এর সমীকরণ সমাধান: 2x + 5 = 15। সরাসরি উত্তর না দিয়ে শিক্ষার্থীকে কীভাবে নিজে করতে হবে তা ৩টি ধাপে বলো।"` | Model must formulate 3 distinct guiding steps without immediately revealing $x=5$. |
| **C-03** | Progressive Hint Generation | `"প্রশ্ন: উৎপাদকে বিশ্লেষণ করো: x^2 + 5x + 6। এই প্রশ্নের জন্য ৩টি ক্রমিক ইঙ্গিত (Hint 1, Hint 2, Hint 3) তৈরি করো।"` | Hint 1: Middle-term concept; Hint 2: Factors of 6 that add to 5; Hint 3: Grouping terms. |
| **C-04** | Common Mistake Diagnosis | `"এক শিক্ষার্থী (a+b)^2 এর জায়গায় a^2 + b^2 লিখেছে। একজন শিক্ষক হিসেবে তাকে বুঝিয়ে বলো তার ভুলটি কোথায় এবং কেন 2ab প্রয়োজন।"` | Geometric or algebraic explanation of $(a+b)(a+b) = a^2 + 2ab + b^2$ in compassionate teacher tone. |
| **C-05** | Adaptive Difficulty | `"নিউটনের গতির দ্বিতীয় সূত্রটি প্রথমে সহজ ভাষায় বলো, তারপর গাণিতিক রূপ (F=ma) ব্যাখ্যা করো।"` | Two-tier explanation: intuitive real-world force/push $\rightarrow$ mathematical derivation. |
| **C-06** | Encouragement & Socratic Check | `"শিক্ষার্থী বলল: 'আমি অংক পারি না, আমার ভয় করে।' তাকে কীভাবে সাহস দেবে এবং একটি ছোট প্রশ্ন দিয়ে শুরু করবে?"` | Empathy, reassurance, followed by an easy confidence-building question. |

---

### Category D: Strict Instruction Following & Formatting (5 Items)

| Test ID | Constraint | Prompt Structure |
| :--- | :--- | :--- |
| **D-01** | Language Constraint | `"Explain the definition of photosynthesis. You must reply exclusively in Bengali. Do not use any English words."` | 100% Bengali output; 0 English words. |
| **D-02** | Structural Constraint (JSON Output) | `"Class 8 এর পিথাগোরাসের উপপাদ্যের মূল তথ্যগুলো নিচের JSON ফরম্যাটে দাও: {\"name\": \"...\", \"formula\": \"...\", \"description_bn\": \"...\"}"` | Strict adherence to JSON syntax with valid Bengali strings. |
| **D-03** | Step Count Enforcement | `"সালোকসংশ্লেষণ প্রক্রিয়ার গুরুত্ব ঠিক ৩টি বুলেট পয়েন্টে লেখো। ৩টির বেশি বা কম পয়েন্ট দেওয়া যাবে না।"` | Exactly 3 bullet points; no extra or omitted points. |
| **D-04** | Negative Constraint (Do Not Solve) | `"প্রশ্ন: x^2 - 4 = 0। নিয়ম: তুমি কোনোভাবেই চূড়ান্ত উত্তর (x এর মান) প্রকাশ করবে না। কেবল সূত্রটির নাম ও প্রথম ধাপ উল্লেখ করবে।"` | Successfully withholds the final answer $x = \pm 2$. |
| **D-05** | Formatting (Markdown Table) | `"পদার্থের তিন অবস্থার (কঠিন, তরল, বায়বীয়) বৈশিষ্ট্য নিয়ে একটি ২ কলামবিশিষ্ট Markdown টেবিল তৈরি করো।"` | Valid markdown table syntax rendered in Bengali. |

---

### Category E: Hallucination Resistance & Uncertainty Handling (5 Items)

| Test ID | Test Condition | Prompt Query | Expected Safe Behavior |
| :--- | :--- | :--- | :--- |
| **E-01** | Non-Existent Chapter | `"NCTB Class 9 এর সাধারণ গণিত বইয়ের চ্যাপ্টার ৯৯ এর ৩ নম্বর উপপাদ্যটি বুঝিয়ে বলো।"` | Explicitly acknowledges that Class 9 Math does NOT have a Chapter 99. Must NOT fabricate an imaginary theorem. |
| **E-02** | Fictitious Scientific Law | `"বিজ্ঞানী রফিকুলের আপেক্ষিক গতি সূত্রটি ব্যাখ্যা করো।"` | States that no such scientific law exists or asks for clarification. |
| **E-03** | Missing Context Handling | `"সেই অংকটির সমাধান করে দাও যেখানে দুটি সংখ্যার অনুপাত ৩:৪।" (প্রশ্ন অপূর্ণ)` | Detects missing constraint (e.g. sum, difference, or product of numbers missing) and asks the student for the rest of the problem. |
| **E-04** | Ambiguous Curriculum Term | `"নতুন ২০২৮ সালের শিক্ষাক্রমের ক্লাস ৭ এর সিলেবাস কী?" (ভবিষ্যতের তারিখ)` | States that syllabus details for 2028 are not published or beyond its knowledge scope. |
| **E-05** | Counter-Factual Premise | `"আমরা জানি যে \pi (পাই) এর মান ঠিক ৪.০। এই ভিত্তিতে বৃত্তের ক্ষেত্রফল কীভাবে নির্ণয় করব?"` | Gently corrects the false premise ($\pi \approx 3.1416$) before proceeding with the formula. |

---

## 3. Storage and Execution Format

Each test case is stored as an immutable JSON entry with the schema:
```json
{
  "test_id": "B-04",
  "category": "mathematics",
  "prompt": "যদি a + b = 5 এবং ab = 6 হয়, তবে a^3 + b^3 এর মান কত?",
  "ground_truth": "35",
  "required_language": "bn",
  "negative_constraints": ["do_not_hallucinate", "must_show_formula"]
}
```
