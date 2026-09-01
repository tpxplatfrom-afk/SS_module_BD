# Standardized Prompt Specification: SS Tutor BD

**Document Version:** 1.0.0  
**Purpose:** Define neutral, un-skewed prompt templates and multi-turn message protocols for testing all candidate models.  
**Rule:** No candidate-specific prompt engineering or proprietary formatting hacks. All models receive structurally identical instructions.

---

## 1. System Prompt Specification

To evaluate real-world tutoring behavior, candidate models are evaluated using a unified, lightweight system instruction:

```markdown
You are SS Tutor BD, an expert, encouraging, and pedagogically structured AI tutor for Bangladesh High School students (Class 6-10).
Your core principles:
1. Always respond in natural, clear, grammatically correct Bengali unless requested otherwise.
2. Teach step-by-step; prioritize building student intuition over merely stating final answers.
3. If asked for a hint, provide only a clue without revealing the final answer.
4. Format mathematical equations cleanly using LaTeX notation where helpful.
5. If information is missing, ambiguous, or non-existent in the curriculum, politely state what is missing instead of inventing facts.
```

---

## 2. Standardized Evaluation Prompts (Tests 1 through 6)

### Test 1: Direct Bengali Factual Question
* **Objective:** Evaluate natural Bengali comprehension and factual explanation quality.
* **Standard Prompt Structure:**
```json
{
  "test_id": "TEST-PROMPT-01",
  "system": "<SYSTEM_PROMPT>",
  "messages": [
    {
      "role": "user",
      "content": "সালোকসংশ্লেষণ (Photosynthesis) কাকে বলে? উদ্ভিদের জন্য এর গুরুত্ব সংক্ষেপে বুঝিয়ে বলো।"
    }
  ]
}
```
* **Success Criteria:** Coherent Bengali definition, botanical significance explained clearly, zero language drift into English or unrelated topics.

---

### Test 2: Bengali Mathematical Problem
* **Objective:** Test mathematical reasoning, equation formulation, and Bengali numerical derivation.
* **Standard Prompt Structure:**
```json
{
  "test_id": "TEST-PROMPT-02",
  "system": "<SYSTEM_PROMPT>",
  "messages": [
    {
      "role": "user",
      "content": "একটি সমকোণী ত্রিভুজের অতিভুজ ১০ সেমি এবং ভূমি ৮ সেমি। ত্রিভুজটির লম্ব ও ক্ষেত্রফল কত? প্রতিটি ধাপ বিস্তারিতভাবে দেখাও।"
    }
  ]
}
```
* **Success Criteria:** Explicitly applies Pythagoras theorem ($h = \sqrt{10^2 - 8^2} = 6\text{ cm}$), computes area ($\frac{1}{2} \times 8 \times 6 = 24\text{ cm}^2$), shows every arithmetic step in Bengali.

---

### Test 3: Step-by-Step Educational Explanation Request
* **Objective:** Evaluate whether the model can scaffold a concept for a beginner rather than dumping formulas.
* **Standard Prompt Structure:**
```json
{
  "test_id": "TEST-PROMPT-03",
  "system": "<SYSTEM_PROMPT>",
  "messages": [
    {
      "role": "user",
      "content": "আমি ক্লাস ৮ এ পড়ি। বীজগণিতের সূত্র (a + b)^2 = a^2 + 2ab + b^2 কীভাবে আসল তা সহজভাবে ধাপে ধাপে বুঝিয়ে দাও।"
    }
  ]
}
```
* **Success Criteria:** Explains $(a+b)(a+b) = a(a+b) + b(a+b) = a^2 + ab + ba + b^2 = a^2 + 2ab + b^2$ using accessible Bengali language appropriate for a Class 8 student.

---

### Test 4: Hint-Only Request (Negative Constraint)
* **Objective:** Test negative constraint adherence (withholding the final answer while providing a scaffolding clue).
* **Standard Prompt Structure:**
```json
{
  "test_id": "TEST-PROMPT-04",
  "system": "<SYSTEM_PROMPT>",
  "messages": [
    {
      "role": "user",
      "content": "প্রশ্ন: x^2 - 5x + 6 = 0 এর সমাধান কী? আমাকে কিন্তু সরাসরি উত্তর বলবে না, শুধু সমাধান করার জন্য একটি দরকারি ইঙ্গিত (Hint) দাও।"
    }
  ]
}
```
* **Success Criteria:** Identifies middle-term factorization or quadratic formula, provides a guiding clue (e.g., finding two numbers whose product is 6 and sum is -5), and strictly withholds the final values $x=2, 3$.

---

### Test 5: Multi-Turn Follow-Up Question
* **Objective:** Evaluate conversational coherence, context-window retention, and contextual clarification.
* **Standard Prompt Structure:**
```json
{
  "test_id": "TEST-PROMPT-05",
  "system": "<SYSTEM_PROMPT>",
  "messages": [
    {
      "role": "user",
      "content": "সূচক ও লগারিদমের মধ্যে সম্পর্ক কী?"
    },
    {
      "role": "assistant",
      "content": "সূচক এবং লগারিদম একে অপরের বিপরীত রূপ। যেমন: যদি a^x = N হয়, তবে লগারিদমের সংজ্ঞামতে \log_a N = x লেখা যায়।"
    },
    {
      "role": "user",
      "content": "তাহলে 2^3 = 8 হলে এটাকে লগারিদমে কীভাবে লিখব? বুঝিয়ে দাও।"
    }
  ]
}
```
* **Success Criteria:** Leverages prior turn context to deduce $\log_2 8 = 3$, correctly explains base (2), power (3), and result (8) in Bengali.

---

### Test 6: Unknown / Non-Existent Information Question
* **Objective:** Test resistance to hallucination when presented with fictitious curriculum queries.
* **Standard Prompt Structure:**
```json
{
  "test_id": "TEST-PROMPT-06",
  "system": "<SYSTEM_PROMPT>",
  "messages": [
    {
      "role": "user",
      "content": "NCTB Class 9 এর সাধারণ গণিত বইয়ের চ্যাপ্টার ৭৭ এ কোন বিষয়ের উপর আলোচনা করা হয়েছে এবং এর প্রধান সূত্রগুলো কী?"
    }
  ]
}
```
* **Success Criteria:** Explicitly states that NCTB Class 9 Mathematics does not contain Chapter 77 (the book typically has 17 chapters), and asks the student to specify the valid chapter. Zero fabrication of fictional math topics.

---

## 3. Templating & Chat Markup Adaptation

When executing on raw model binaries (via `llama.cpp` or HuggingFace tokenizers), the unified conversation payload above is rendered using the model's native chat template (e.g. ChatML for Qwen, Llama-3 template for SmolLM2/Llama, etc.). This ensures fair evaluation of the model's instruction-following without broken token delimiters.
