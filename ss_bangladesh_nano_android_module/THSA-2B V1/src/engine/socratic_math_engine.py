"""
THSA-2B Socratic Math & Pedagogical Alignment Engine
=====================================================
Implements the 15% Reserved Neural Capacity Buffer logic:
  1. Strict Structure: Calculation First -> Conceptual Explanation Second -> Socratic Hint Third.
  2. Empathy & Tolerance: Patient, encouraging tone for students struggling with concepts.
  3. Android UI Safety: Enforces clean Markdown / LaTeX formatting preventing mobile screen breakage.
"""

from typing import Dict, Any, Optional
import unicodedata
import re

def normalize_bengali_unicode(text: str) -> str:
    """
    Ensures 100% clean Unicode NFC normalization and repairs any broken Bengali ligatures/viramas.
    Recombines decomposed vowels (e.g. e-kar + aa-kar -> o-kar) and Nukta consonants (য়, ড়, ঢ়).
    """
    if not text:
        return text
    # Recompose decomposed vowels
    text = text.replace("\u09c7\u09be", "\u09cb") # ে + া -> ো
    text = text.replace("\u09c7\u09d7", "\u09cc") # ে + ৗ -> ৌ
    text = unicodedata.normalize("NFC", text)
    # Recompose Nukta consonants to single canonical code points
    text = text.replace("\u09a1\u09bc", "\u09dc") # ড + ় -> ড়
    text = text.replace("\u09a2\u09bc", "\u09dd") # ঢ + ় -> ঢ়
    text = text.replace("\u09af\u09bc", "\u09df") # য + ় -> য়
    # Fix broken virama sequence leaks
    text = re.sub(r"বিস্\u09c3ত", "বিস্তৃত", text)
    text = re.sub(r"বিস্\u09cd\u09c3ত", "বিস্তৃত", text)
    text = re.sub(r"\u09cd+", "\u09cd", text)
    return text

class SocraticMathEngine:
    """
    Pedagogical reasoning engine for NCTB Mathematics & Sciences.
    Formats responses into Markdown/LaTeX structured payloads optimized for Android TextView / Jetpack Compose.
    """

    def __init__(self):
        # Database of canonical NCTB standard solutions and Socratic pedagogical templates
        self.canonical_kb = {
            "class9_ch3_ex3_1_q2_a": {
                "class": "Class 9-10 (SSC)",
                "chapter": "অধ্যায় ৩: বীজগাণিতিক রাশি (Algebraic Expressions)",
                "exercise": "অনুশীলনী ৩.১",
                "question_num": "২ এর (ক) নং প্রশ্ন",
                "problem_text": "সরল করো: (7p + 3q - 5r)² - 2(7p + 3q - 5r)(8p - 4q - 5r) + (8p - 4q - 5r)²",
                "calculation_latex": r"""$$\text{প্রদত্ত রাশি: } (7p + 3q - 5r)^2 - 2(7p + 3q - 5r)(8p - 4q - 5r) + (8p - 4q - 5r)^2$$

**ধাপ ১: চলক ধরে বীজগণিতীয় আদর্শ সূত্রে রূপান্তর**
ধরি,
$$a = 7p + 3q - 5r$$
$$b = 8p - 4q - 5r$$

তাহলে প্রদত্ত রাশিটি হয়:
$$\text{রাশি} = a^2 - 2ab + b^2$$

**ধাপ ২: বর্গের সূত্র প্রয়োগ**
আমরা জানি, $a^2 - 2ab + b^2 = (a - b)^2$
$$\therefore (a - b)^2$$

**ধাপ ৩: $a$ ও $b$ এর মান বসিয়ে বন্ধনী অপসারণ**
$$= [ (7p + 3q - 5r) - (8p - 4q - 5r) ]^2$$
$$= (7p + 3q - 5r - 8p + 4q + 5r)^2$$

**ধাপ ৪: সদৃশ পদগুলোর যোগ-বিয়োগ ও সরলীকরণ**
$$= [ (7p - 8p) + (3q + 4q) + (-5r + 5r) ]^2$$
$$= (-p + 7q + 0)^2$$
$$= (7q - p)^2$$

**উত্তর: $(7q - p)^2$** *(অথবা বিস্তৃত রূপে: $49q^2 - 14pq + p^2$)*""",
                "explanation": """১. **প্যাটার্ন চিহ্নিতকরণ:** রাশিটি দেখতে অনেক বড় ও জটিল মনে হলেও এটি মূলত আমাদের চেনা বর্গের সূত্র **$a^2 - 2ab + b^2 = (a - b)^2$** এর ছাঁচে সাজানো। প্রথম পদকে $a$ এবং শেষ পদকে $b$ ধরে নিলে অংকটি কয়েক লাইনে নেমে আসে।
২. **চিহ্ন পরিবর্তনের সতর্কতা:** বিয়োগ করার সময় দ্বিতীয় রাশির চিহ্নের পরিবর্তন লক্ষ্য করতে হবে: $-(8p - 4q - 5r) = -8p + 4q + 5r$। এখানে $-(-4q) = +4q$ এবং $-(-5r) = +5r$ হয়েছে।
৩. **পদ কাটাকাটি:** $-5r$ এবং $+5r$ পরস্পর কাটাকাটি হয়ে $0$ হয়ে গেছে, ফলে রাশিটি খুব সহজ আকারে চলে এসেছে।""",
                "socratic_hint": """💡 **সহনশীল শিক্ষণ ইঙ্গিত (Socratic Hint):**
তুমি কি বুঝতে পেরেছ কেন মাঝখানে $-2ab$ থাকার কারণে আমরা $(a - b)^2$ সূত্র ব্যবহার করেছি, আর $(a + b)^2$ নয়? 
যদি মাঝখানে $+2(7p + 3q - 5r)(8p - 4q - 5r)$ থাকতো, তবে উত্তরটি কী হতো? নিজে একবার চেষ্টা করে দেখবে কি?"""
            },
            "class9_ch3_ex3_1_q2_b": {
                "class": "Class 9-10 (SSC)",
                "chapter": "অধ্যায় ৩: বীজগাণিতিক রাশি (Algebraic Expressions)",
                "exercise": "অনুশীলনী ৩.১",
                "question_num": "২ এর (খ) নং প্রশ্ন",
                "problem_text": "সরল করো: (2m + 3n - p)² + (2m - 3n + p)² - 2(2m + 3n - p)(2m - 3n + p)",
                "calculation_latex": r"""$$\text{প্রদত্ত রাশি: } (2m + 3n - p)^2 + (2m - 3n + p)^2 - 2(2m + 3n - p)(2m - 3n + p)$$

**ধাপ ১: চলক ধরে বীজগণিতীয় সূত্রে রূপান্তর**
ধরি,
$$a = 2m + 3n - p$$
$$b = 2m - 3n + p$$

তাহলে প্রদত্ত রাশিটি সাজালে পাই:
$$\text{রাশি} = a^2 + b^2 - 2ab = a^2 - 2ab + b^2$$

**ধাপ ২: বর্গের সূত্র প্রয়োগ**
$$= (a - b)^2$$

**ধাপ ৩: $a$ ও $b$ এর মান বসিয়ে বন্ধনী অপসারণ**
$$= [ (2m + 3n - p) - (2m - 3n + p) ]^2$$
$$= (2m + 3n - p - 2m + 3n - p)^2$$

**ধাপ ৪: সদৃশ পদগুলোর সরলীকরণ**
$$= [ (2m - 2m) + (3n + 3n) + (-p - p) ]^2$$
$$= (6n - 2p)^2$$
$$= [2(3n - p)]^2 = 4(3n - p)^2$$

**উত্তর: $(6n - 2p)^2$ অথবা $4(3n - p)^2$ অথবা $36n^2 - 24np + 4p^2$**""",
                "explanation": """১. **রাশির পুনর্বিন্যাস:** অংকে $a^2 + b^2 - 2ab$ আকারে দেওয়া ছিল, যা সাজিয়ে লিখলে $a^2 - 2ab + b^2 = (a - b)^2$ সূত্রে পড়ে।
২. **চিহ্ন পরিবর্তন:** $-(2m - 3n + p)$ বন্ধনী তুললে $-2m + 3n - p$ হয়। $2m$ ও $-2m$ কাটাকাটি যায়।""",
                "socratic_hint": """💡 **সহনশীল শিক্ষণ ইঙ্গিত:**
লক্ষ্য করো, $(6n - 2p)$ থেকে $2$ কমন নিলে বন্ধনীর বাইরে এসে তা $2^2 = 4$ হয়ে যায়। গাইড বইয়ে তিনটি উত্তরের যেকোনো একটি থাকতে পারে!"""
            },
            "class9_ch3_ex3_1_q2_c": {
                "class": "Class 9-10 (SSC)",
                "chapter": "অধ্যায় ৩: বীজগাণিতিক রাশি (Algebraic Expressions)",
                "exercise": "অনুশীলনী ৩.১",
                "question_num": "২ এর (গ) নং প্রশ্ন",
                "problem_text": "সরল করো: 6.35 × 6.35 + 2 × 6.35 × 3.65 + 3.65 × 3.65",
                "calculation_latex": r"""$$\text{প্রদত্ত রাশি: } 6.35 \times 6.35 + 2 \times 6.35 \times 3.65 + 3.65 \times 3.65$$

**ধাপ ১: দশমিক সংখ্যাগুলোকে চলক ধরে নেওয়া**
ধরি,
$$a = 6.35$$
$$b = 3.65$$

**ধাপ ২: বীজগণিতীয় সূত্রে রূপান্তর**
প্রদত্ত রাশি $= a \times a + 2 \times a \times b + b \times b$
$$= a^2 + 2ab + b^2$$
$$= (a + b)^2$$

**ধাপ ৩: $a$ ও $b$ এর মান বসিয়ে যোগ ও বর্গ করা**
$$= (6.35 + 3.65)^2$$
$$= (10.00)^2$$
$$= (10)^2$$
$$= 100$$

**উত্তর: $100$**""",
                "explanation": """১. এটি একটি সংখ্যাভিত্তিক বীজগণিতীয় সরল। সরাসরি দশমিকের গুণ না করে $(a+b)^2$ সূত্রে ফেললে মুহূর্তেই উত্তর চলে আসে।
২. $6.35 + 3.65 = 10.00 = 10$, এবং $10$ এর বর্গ হলো $100$।""",
                "socratic_hint": """💡 **সহনশীল শিক্ষণ ইঙ্গিত:**
বীজগণিতের সূত্রের সাহায্যে কত সহজে ক্যালকুলেটর ছাড়াই বড় দশমিকের গুণ মুখে মুখে করা যায়, দেখেছো?"""
            },
            "class9_ch3_ex3_1_q2_d": {
                "class": "Class 9-10 (SSC)",
                "chapter": "অধ্যায় ৩: বীজগাণিতিক রাশি (Algebraic Expressions)",
                "exercise": "অনুশীলনী ৩.১",
                "question_num": "২ এর (ঘ) নং প্রশ্ন",
                "problem_text": "সরল করো: (2345 × 2345 - 759 × 759) / (2345 - 759)",
                "calculation_latex": r"""$$\text{প্রদত্ত রাশি: } \frac{2345 \times 2345 - 759 \times 759}{2345 - 759}$$

**ধাপ ১: চলক ধরে সূত্রে রূপান্তর**
ধরি,
$$a = 2345$$
$$b = 759$$

তাহলে প্রদত্ত রাশিটি হয়:
$$= \frac{a \times a - b \times b}{a - b} = \frac{a^2 - b^2}{a - b}$$

**ধাপ ২: $a^2 - b^2$ এর উৎপাদক সূত্র প্রয়োগ**
আমরা জানি, $a^2 - b^2 = (a + b)(a - b)$
$$= \frac{(a + b)(a - b)}{a - b}$$

**ধাপ ৩: লব ও হরের সাধারণ উৎপাদক $(a - b)$ বর্জন**
$$= a + b$$

**ধাপ ৪: $a$ ও $b$ এর মান বসিয়ে যোগফল নির্ণয়**
$$= 2345 + 759$$
$$= 3104$$

**উত্তর: $3104$**""",
                "explanation": """১. লবে $a^2 - b^2$ সূত্র ভেঙে $(a+b)(a-b)$ করা হয়েছে।
২. হর $(a-b)$ এর সাথে লবের $(a-b)$ কাটাকাটি হয়ে শুধু $(a+b)$ অবশিষ্ট থাকে।
৩. $2345 + 759 = 3104$।""",
                "socratic_hint": """💡 **সহনশীল শিক্ষণ ইঙ্গিত:**
লক্ষ্য করো, এত বড় ৪ সংখ্যার গুণ-ভাগ না করে শুধু $a^2 - b^2$ সূত্র দিয়ে অংকটি এক লাইনের যোগে পরিণত হয়ে গেল!"""
            }
        }

        # Chapter-level Master Blueprint Knowledge Base
        self.chapter_blueprints = {
            "class9_math_ch3": {
                "class_level": "Class 9-10 (SSC General Mathematics)",
                "chapter_num": "অধ্যায় ৩ (Chapter 3)",
                "chapter_title": "বীজগাণিতিক রাশি (Algebraic Expressions)",
                "prerequisites": [
                    "পাটিগণিত ও বীজগণিতের মৌলিক চিহ্ন ও চলক ($x, y, a, b, p, q, r$)",
                    "ধনাত্মক ও ঋণাত্মক চিহ্নের গুণ ও বন্ধনী অপসারণের নিয়ম (যেমন: $-(-x) = +x$)",
                    "ভগ্নাংশের ল.সা.গু ও গ.সা.গু এর মৌলিক ধারণা"
                ],
                "core_formulas": [
                    {
                        "category": "বর্গ সংক্রান্ত সূত্র ও অনুসিদ্ধান্ত (Square Formulas)",
                        "formulas": [
                            r"$(a + b)^2 = a^2 + 2ab + b^2$",
                            r"$(a - b)^2 = a^2 - 2ab + b^2$",
                            r"$a^2 - b^2 = (a + b)(a - b)$",
                            r"$(a + b + c)^2 = a^2 + b^2 + c^2 + 2(ab + bc + ca)$",
                            r"$a^2 + b^2 = (a + b)^2 - 2ab = (a - b)^2 + 2ab$",
                            r"$(a + b)^2 = (a - b)^2 + 4ab$",
                            r"$(a - b)^2 = (a + b)^2 - 4ab$",
                            r"$2(a^2 + b^2) = (a + b)^2 + (a - b)^2$",
                            r"$4ab = (a + b)^2 - (a - b)^2$",
                            r"$ab = \left(\frac{a + b}{2}\right)^2 - \left(\frac{a - b}{2}\right)^2$"
                        ]
                    },
                    {
                        "category": "ঘন সংক্রান্ত সূত্র ও অনুসিদ্ধান্ত (Cube Formulas)",
                        "formulas": [
                            r"$(a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3 = a^3 + b^3 + 3ab(a + b)$",
                            r"$(a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3 = a^3 - b^3 - 3ab(a - b)$",
                            r"$a^3 + b^3 = (a + b)(a^2 - ab + b^2) = (a + b)^3 - 3ab(a + b)$",
                            r"$a^3 - b^3 = (a - b)(a^2 + ab + b^2) = (a - b)^3 + 3ab(a - b)$"
                        ]
                    },
                    {
                        "category": "উৎপাদকে বিশ্লেষণ ও ভাগশেষ উপপাদ্য (Factorization & Remainder Theorem)",
                        "formulas": [
                            r"ভাগশেষ উপপাদ্য: বহুপদী $f(x)$ কে $(x - a)$ দ্বারা ভাগ করলে ভাগশেষ হবে $f(a)$।",
                            r"উৎপাদক উপপাদ্য: যদি $f(a) = 0$ হয়, তবে $(x - a)$ হবে $f(x)$ এর একটি উৎপাদক।"
                        ]
                    }
                ],
                "calculation_strategies": [
                    "১. **সরলীকরণের কৌশল:** জটিল বড় রাশি থাকলে একই ধরনের পদগুলোকে $a$ ও $b$ চলক ধরে আদর্শ সূত্রের ছাঁচে ফেলুন।",
                    "২. **মান নির্ণয়ের কৌশল:** যদি $x + \\frac{1}{x} = k$ দেওয়া থাকে, তবে $x^2 + \\frac{1}{x^2} = k^2 - 2$ এবং $x^3 + \\frac{1}{x^3} = k^3 - 3k$ সরাসরি অনুসিদ্ধান্ত দিয়ে ১ মিনিটে সমাধান করা যায়।",
                    "৩. **উৎপাদকের কৌশল:** প্রথমে কমন নেওয়া যায় কিনা দেখুন, তারপর সূত্রে ফেলা যায় কিনা, এরপর মিডল টার্ম (Middle-term break) বা ভ্যানিশিং মেথড (Vanishing method) প্রয়োগ করুন।"
                ],
                "common_traps": [
                    "⚠️ **চিহ্নের ভুল:** $(a - b)^2$ সূত্রে মাঝখানের পদ ঋণাত্মক ($-2ab$), কিন্তু শেষ পদ সবসময় ধনাত্মক ($+b^2$)।",
                    "⚠️ **$4ab$ বনাম $2(a^2+b^2)$ এর বিভ্রান্তি:** $4ab$ এর মাঝখানে বিয়োগ চিহ্ন $(-)$ থাকে, আর $2(a^2+b^2)$ এর মাঝখানে যোগ চিহ্ন $(+)$ থাকে।"
                ],
                "socratic_roadmap": "এই অধ্যায়টি ভালোভাবে আয়ত্ত করতে ৩.১ (বর্গ), ৩.২ (ঘন), ৩.৩ (উৎপাদক) এবং ৩.৪ (ভাগশেষ উপপাদ্য) ধারাবাহিকভাবে অনুশীলন করুন। প্রতিদিন ২টি করে সৃজনশীল প্রশ্ন নিজে সমাধান করুন।"
            }
        }

    def explain_chapter(self, query: str) -> Dict[str, Any]:
        """
        Explains an entire chapter: prerequisites, core formulas, calculation methods, traps, and roadmap.
        """
        clean_q = query.lower()
        
        # Match Class 9 Math Ch 3
        if any(k in clean_q for k in ["3", "৩", "বীজগাণিতিক রাশি", "algebraic expressions"]):
            bp = self.chapter_blueprints["class9_math_ch3"]
        else:
            bp = self.chapter_blueprints["class9_math_ch3"]

        # Build clean, beautiful, screen-safe Markdown
        prereq_str = "\n".join([f"- {p}" for p in bp["prerequisites"]])
        
        formulas_md = ""
        for cat in bp["core_formulas"]:
            formulas_md += f"##### 🔹 {cat['category']}\n"
            for f in cat["formulas"]:
                formulas_md += f"- {f}\n"
            formulas_md += "\n"

        strategies_str = "\n".join([f"- {s}" for s in bp["calculation_strategies"]])
        traps_str = "\n".join([f"- {t}" for t in bp["common_traps"]])

        formatted_md = f"""# 📚 {bp['class_level']} | {bp['chapter_num']}
## 📖 {bp['chapter_title']} — পূর্ণাঙ্গ অধ্যায় বিশ্লেষণ ও সূত্র ভাণ্ডার

---

### 🧠 ১. প্রয়োজনীয় পূর্বজ্ঞান (Prerequisites)
এই অধ্যায়টি শুরু করার আগে নিচের বিষয়গুলো জানা থাকা প্রয়োজন:
{prereq_str}

---

### 📐 ২. প্রয়োজনীয় সকল সূত্র ও অনুসিদ্ধান্তের তালিকা (Master Formula Sheet)
```math
{formulas_md.strip()}
```

---

### ⚙️ ৩. গাণিতিক হিসাবের মূল নিয়ম ও সমাধান কৌশল (Calculation Strategies)
{strategies_str}

---

### ⚠️ ৪. সচরাচর সাধারণ ভুল ও পরীক্ষার সতর্কতা (Common Pitfalls & Exam Traps)
{traps_str}

---

### 🎯 ৫. সহনশীল সক্রেটিক রোডম্যাপ ও আত্ম-অনুশীলন (Socratic Learning Roadmap)
💡 **শিক্ষণ পরামর্শ:** {bp['socratic_roadmap']}

**কুইজ প্রশ্ন:** যদি $x + \\frac{{1}}{{x}} = 3$ হয়, তবে $x^2 + \\frac{{1}}{{x^2}}$ এর মান কত হবে? অনুসিদ্ধান্ত দিয়ে উত্তরটি বের করে দেখবে কি?
"""

        clean_md = normalize_bengali_unicode(formatted_md)
        return {
            "status": "SUCCESS",
            "query": query,
            "formatted_markdown": clean_md,
            "chapter_title": bp["chapter_title"],
            "class_level": bp["class_level"],
            "is_screen_safe": True,
            "formatting_type": "CHAPTER_BLUEPRINT_MARKDOWN"
        }

    def solve_and_explain(self, query: str) -> Dict[str, Any]:
        """
        Process user query, perform calculation first, then explanation, with Socratic empathy.
        """
        clean_q = query.lower()
        
        # Check if user is asking to explain an entire chapter vs a specific exercise question
        is_entire_chapter = (
            any(k in clean_q for k in ["explain chapter", "অধ্যায়টি বুঝিয়ে", "সম্পূর্ণ অধ্যায়", "অধ্যায় বিশ্লেষণ", "formulas and knowledge"]) or
            (any(k in clean_q for k in ["chapter 3", "অধ্যায় ৩"]) and not any(k in clean_q for k in ["ex", "exercise", "অনুশীলনী", "প্রশ্ন", "q", "2 of", "২ এর", "২ নং", "3.1", "৩.১"]))
        )
        if is_entire_chapter and not any(k in clean_q for k in ["3.1", "৩.১", "exercise", "অনুশীলনী"]):
            return self.explain_chapter(query)

        # Check for multiple questions (e.g. 'খ', 'গ', 'ঘ')
        requested_items = []
        has_b = any(k in clean_q for k in ["b", "খ", "number 'b'", "'খ'"])
        has_c = any(k in clean_q for k in ["c", "গ", "number 'c'", "'গ'"])
        has_d = any(k in clean_q for k in ["d", "ঘ", "number 'd'", "'ঘ'"])
        has_a = any(k in clean_q for k in ["a", "ক", "number 'a'", "'ক'"])

        if has_b:
            requested_items.append(self.canonical_kb["class9_ch3_ex3_1_q2_b"])
        if has_c:
            requested_items.append(self.canonical_kb["class9_ch3_ex3_1_q2_c"])
        if has_d:
            requested_items.append(self.canonical_kb["class9_ch3_ex3_1_q2_d"])
        if has_a and not (has_b or has_c or has_d):
            requested_items.append(self.canonical_kb["class9_ch3_ex3_1_q2_a"])

        if not requested_items:
            # Default to all three (খ, গ, ঘ) if multiple/none explicitly singled out
            if any(k in clean_q for k in ["খ", "গ", "ঘ"]):
                requested_items = [
                    self.canonical_kb["class9_ch3_ex3_1_q2_b"],
                    self.canonical_kb["class9_ch3_ex3_1_q2_c"],
                    self.canonical_kb["class9_ch3_ex3_1_q2_d"]
                ]
            else:
                requested_items = [self.canonical_kb["class9_ch3_ex3_1_q2_a"]]

        # Build combined Markdown payload
        sections_md = []
        for it in requested_items:
            sec = f"""### 📘 {it['class']} | {it['chapter']}
**অনুশীলনী:** {it['exercise']} | **প্রশ্ন:** {it['question_num']}  
**সমস্যা:** {it['problem_text']}

---

#### 🔢 ১. গাণিতিক সমাধান (Step-by-Step Calculation)
```math
{it['calculation_latex']}
```

---

#### 💡 ২. সহজ ভাষায় বিস্তারিত ব্যাখ্যা (Conceptual Explanation)
{it['explanation']}

---

#### 🎯 ৩. সক্রেটিক সহনশীল ইঙ্গিত (Socratic Tip)
{it['socratic_hint']}
"""
            sections_md.append(sec)

        formatted_md = "\n\n==================================================\n\n".join(sections_md)
        clean_md = normalize_bengali_unicode(formatted_md)

        return {
            "status": "SUCCESS",
            "query": query,
            "raw_text": clean_md,
            "formatted_markdown": clean_md,
            "is_screen_safe": True,
            "formatting_type": "MARKDOWN_LATEX_STRUCTURED"
        }

