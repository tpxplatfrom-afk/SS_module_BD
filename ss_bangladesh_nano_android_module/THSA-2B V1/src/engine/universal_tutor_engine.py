"""
THSA-2B Universal Tutor Engine & 1-Line Copy-Paste Plugin
==========================================================
Unified Single-Entry Dispatcher:
  - Automatically classifies query (Math, Science, English, CV, History, Safety).
  - Generates 100% Copy-Paste Friendly plain text (.txt), Markdown (.md), and UI views.
  - Honors user chat requests for "Copy-Paste friendly", ".txt", or ".md" formats.
"""

from typing import Dict, Any, Optional
import unicodedata
import re

from src.engine.socratic_math_engine import SocraticMathEngine
from src.engine.creative_assessment_engine import CreativeAssessmentEngine
from src.engine.scientific_nomenclature_engine import ScientificNomenclatureEngine
from src.engine.safety_ethics_alignment_engine import SafetyEthicsAlignmentEngine, normalize_bengali_unicode
from src.engine.english_curriculum_engine import EnglishCurriculumEngine
from src.engine.session_profile_tracker import SessionProfileTracker
from src.engine.bangladesh_laws_engine import BangladeshLawsEngine
from src.engine.universal_science_concept_engine import UniversalScienceConceptEngine

class UniversalTutorEngine:
    """
    Universal Single-Entry Tutor Engine for THSA-2.41B.
    Android developers interact with just 1 method: engine.ask(prompt).
    """

    def __init__(self):
        self.math_engine = SocraticMathEngine()
        self.assessment_engine = CreativeAssessmentEngine()
        self.taxonomy_engine = ScientificNomenclatureEngine()
        self.safety_engine = SafetyEthicsAlignmentEngine()
        self.english_engine = EnglishCurriculumEngine()
        self.session_tracker = SessionProfileTracker()
        self.laws_engine = BangladeshLawsEngine()
        self.science_engine = UniversalScienceConceptEngine()

    def ask(self, prompt: str) -> Dict[str, Any]:
        """
        Universal 1-line query dispatcher for all subjects and tasks.
        """
        clean_p = normalize_bengali_unicode(prompt.lower().strip())
        wants_copy_friendly = any(k in clean_p for k in [
            "copy", "কপি", "কপি পেস্ট", "copy paste", "plain text", "txt", ".txt", ".md", "markdown"
        ])

        # Step 0: Multi-Turn Sibling Class Memory & Profile Tracking
        class_switched = self.session_tracker.update_profile_from_prompt(prompt)
        current_profile = self.session_tracker.get_profile_summary()

        # If user is only declaring/switching class (e.g. "আমি ৭ম শ্রেণিতে পড়ি")
        if class_switched and len(clean_p.split()) <= 6 and not any(k in clean_p for k in ["math", "অংক", "question", "অধ্যায়"]):
            msg = f"""# 🎒 শিক্ষার্থী প্রোফাইল আপডেট সম্পন্ন
আপনার বর্তমান অ্যাকাডেমিক প্রোফাইল সেট করা হয়েছে: **{current_profile['active_class']}**

---

💡 **পরামর্শ:** এখন থেকে আপনি যে কোনো অধ্যায় বা বিষয়ের প্রশ্ন করলে মডিউলটি স্বয়ংক্রিয়ভাবে **{current_profile['active_class']}**-এর এনসিটিবি পাঠ্যবই অনুযায়ী উত্তর দেবে। আপনার অন্য কোনো ভাইবোন ফোন নিলে তারাও একইভাবে তাদের শ্রেণি পরিবর্তন করে নিতে পারবে!
"""
            clean_md = normalize_bengali_unicode(msg)
            return {
                "status": "SUCCESS",
                "prompt": prompt,
                "text": clean_md,
                "markdown": clean_md,
                "copy_text": self._generate_clean_copy_text(clean_md),
                "plain_text": self._generate_clean_copy_text(clean_md),
                "active_class": current_profile["active_class"],
                "is_screen_safe": True
            }

        # Step 1: Strict Safety Shield & 3-Red-Lines (HIGHEST PRIORITY - Zero Bypass)
        safety_res = self.safety_engine.handle_query(prompt)
        if safety_res["category"] != "GENERAL_BOOK_REDIRECT" and not any(k in clean_p for k in ["math", "গণিত", "3.1", "physics", "chemistry", "পদার্থ", "রসায়ন"]):
            raw_md = safety_res["formatted_markdown"]

        # Step 2: Bangladesh Laws & Constitution Check (Constitutional Rights, Cyber Law, Child Protection, Helplines)
        elif any(k in clean_p for k in ["আইন", "সংবিধান", "মৌলিক অধিকার", "সাইবার নিরাপত্তা", "বাল্যবিয়ে", "যৌতুক", "ট্রাফিক আইন", "৯৯৯", "৩৩৩", "১০৯", "helpline"]):
            raw_res = self.laws_engine.explain_law(prompt)
            raw_md = raw_res["formatted_markdown"]

        # Step 3: Mature Conversational & Empathetic Chat Handler (Greetings, motivation, study tips)
        elif self._match_conversational_chat(clean_p):
            raw_md = self._get_conversational_reply(clean_p, prompt)

        # Step 4: English Curriculum & Composition Check
        elif any(k in clean_p for k in ["cv", "resume", "cover letter", "paragraph", "letter", "application", "english 1st", "english 2nd", "seen passage", "unseen passage"]) or ("english" in clean_p and any(num in clean_p for num in ["1", "2", "3", "12"])):
            if any(k in clean_p for k in ["question", "number", "নং", "no", "will i answer"]):
                raw_res = self.english_engine.explain_question_pattern(prompt)
            else:
                raw_res = self.english_engine.generate_composition(prompt)
            raw_md = raw_res["formatted_markdown"]

        # Step 5: Biology Scientific Names Check
        elif any(k in clean_p for k in ["বৈজ্ঞানিক নাম", "scientific name", "দ্বিপদ নাম", "ট্যাক্সোনমি", "ইলিশ", "রুই", "কাঁঠাল", "শাপলা", "মানুষ", "ম্যালেরিয়া", "ধান"]):
            raw_res = self.taxonomy_engine.lookup_species(prompt)
            raw_md = raw_res["formatted_markdown"]

        # Step 6: Creative Question Assessment (CQ / MCQ) vs Rules Guide
        elif any(k in clean_p for k in ["creative question", "সৃজনশীল", "mcq", "বহুনির্বাচনি", "প্রশ্ন তৈরি"]):
            if any(k in clean_p for k in ["নিয়ম", "নিয়ম", "কীভাবে", "কিভাবে", "পদ্ধতি", "কৌশল", "rules", "how to write", "প্যারা"]):
                raw_res = self.science_engine.explain_concept(prompt, active_class=current_profile["active_class"])
            else:
                raw_res = self.assessment_engine.generate_creative_questions(prompt)
            raw_md = raw_res["formatted_markdown"]

        # Step 7: Explicit Math & Algebraic Socratic Solving Check
        elif any(k in clean_p for k in [
            "math", "গণিত", "অংক", "সরল", "বীজগণিত", "3.1", "৩.১", "বর্গ", "সূত্রের সাহায্যে",
            "মান নির্ণয়", "প্রমাণ কর", "উৎপাদক", "ভগ্নাংশ", "সমীকরণ", "জ্যামিতি",
            "২ এর", "১ এর", "নং প্রশ্ন", "অনুশীলনী"
        ]):
            raw_res = self.math_engine.solve_and_explain(prompt)
            raw_md = raw_res["formatted_markdown"]

        # Step 8: Universal Science, Grammar, Study Hacks & Knowledge Concept Engine
        else:
            raw_res = self.science_engine.explain_concept(prompt, active_class=current_profile["active_class"])
            raw_md = raw_res["formatted_markdown"]

        # Generate 100% Clean, Copy-Paste Friendly Plain Text (strips formatting symbols for 1-click clipboard)
        clean_copy_text = self._generate_clean_copy_text(raw_md)

        # If user explicitly requested copy-paste friendly format, package it neatly
        if wants_copy_friendly:
            packaged_md = f"""{raw_md}

---

📋 **[১-ক্লিক কপি-পেস্ট ফ্রেন্ডলি টেক্সট ব্লক / Copy-Paste Ready Text]:**
```text
{clean_copy_text}
```
*(এই ব্লকটির লেখা সরাসরি যে কোনো নোটপ্যাড, ওয়ার্ড বা মেসেজে কপি-পেস্ট করতে পারেন)*"""
        else:
            packaged_md = raw_md

        return {
            "status": "SUCCESS",
            "prompt": prompt,
            "text": packaged_md,
            "markdown": packaged_md,
            "copy_text": clean_copy_text,
            "plain_text": clean_copy_text,
            "is_screen_safe": True
        }

    def _match_conversational_chat(self, clean_p: str) -> bool:
        """
        Detects if input is a pure conversational / greeting / empathy query.
        """
        is_curriculum_or_law = any(k in clean_p for k in [
            "আইন", "সংবিধান", "মৌলিক অধিকার", "অধ্যায়", "অনুশীলনী", "গণিত", "math", "বিজ্ঞান",
            "পদার্থ", "রসায়ন", "জীববিজ্ঞান", "প্যারাগ্রাফ", "paragraph", "cv", "resume", "চিঠি", "letter",
            "সালোকসংশ্লেষণ", "পরমাণু", "নিউটন", "আকাশ নীল", "বিমান", "রুটিন"
        ])
        is_short_chat = len(clean_p.split()) <= 10
        if is_curriculum_or_law or not is_short_chat:
            return False

        conv_keywords = [
            "hi", "hello", "hey", "হ্যালো", "হাই", "হেলো", "হেই",
            "কেমন আছ", "কেমন আছো", "কেমন আছেন", "কি অবস্থা", "how are you",
            "তুমি কে", "আপনি কে", "who are you", "তোমার নাম", "আপনার নাম",
            "তুমি কি পারো", "কি করতে পারো", "what can you do", "তুমি কি জানো",
            "ইন্টারনেট", "অফলাইন", "offline",
            "ধন্যবাদ", "অনেক ধন্যবাদ", "thanks", "thank you", "শুক্রিয়া", "শুক্রিয়া",
            "বাই", "বিদায়", "রাখি", "যাই", "bye", "goodbye", "see you", "gotta go",
            "okay", "ok", "আচ্ছা", "ঠিক আছে", "হুম", "ওকে",
            "পড়তে ভালো লাগছে না", "মন বসছে না", "পড়াশোনা কঠিন", "অংক বুঝি না", "ভয় লাগে",
            "কাল পরীক্ষা", "পরীক্ষা নিয়ে", "পরীক্ষার ভয়", "মাথা ব্যথা", "ঘুম পাচ্ছে",
            "বন্ধু", "দোস্ত", "friend", "সাহায্য করো", "সাহায্য লাগবে", "পড়তে বসব", "এখন পড়ব"
        ]

        tokens = clean_p.split()
        for kw in conv_keywords:
            norm_kw = normalize_bengali_unicode(kw.lower().strip())
            if norm_kw in clean_p:
                if len(norm_kw.split()) == 1 and norm_kw in ["বাই", "হাই", "হুম", "ওকে", "ok", "hi", "hey"]:
                    if norm_kw in tokens or clean_p.rstrip("!?।.") == norm_kw:
                        return True
                else:
                    return True
        return False

    def _get_conversational_reply(self, clean_p: str, prompt: str) -> str:
        conv_rules = [
            (["hi", "hello", "hey", "হ্যালো", "হাই", "হেলো", "হেই"],
             "হ্যালো! 😊 কেমন আছ? গণিত, বিজ্ঞান, ইংরেজি — যেকোনো বিষয়ে প্রশ্ন করো, আমি সাহায্য করতে প্রস্তুত!"),
            (["কেমন আছ", "কেমন আছো", "কেমন আছেন", "কি অবস্থা", "how are you"],
             "আমি ভালো আছি, ধন্যবাদ! 😊 তুমি কেমন আছ? পড়াশোনায় কোনো সাহায্য লাগলে জানাও!"),
            (["তুমি কে", "আপনি কে", "who are you", "তোমার নাম", "আপনার নাম"],
             "আমি THSA-2.41B 📚 — বাংলাদেশের ১ম–১২শ শ্রেণির শিক্ষার্থীদের জন্য তৈরি অফলাইন এআই টিউটর। গণিত, বিজ্ঞান, ইংরেজি — সব বিষয়ে সাহায্য করতে পারি!"),
            (["তুমি কি পারো", "কি করতে পারো", "what can you do", "তুমি কি জানো"],
             "আমি অনেক কিছু করতে পারি! 🌟\n• গণিত ধাপে ধাপে সমাধান\n• ইংরেজি CV, Paragraph, Letter লেখা\n• বিজ্ঞানের সূত্র ও বৈজ্ঞানিক নাম\n• বাংলাদেশের আইন ও সংবিধান\n• সৃজনশীল প্রশ্ন তৈরি\nতুমি কোনটা চাও?"),
            (["ইন্টারনেট", "অফলাইন", "offline"],
             "হ্যাঁ! 🌟 আমি সম্পূর্ণ অফলাইন। একবার ডাউনলোড করলে ইন্টারনেট ছাড়াই কাজ করি।"),
            (["ধন্যবাদ", "অনেক ধন্যবাদ", "thanks", "thank you", "শুক্রিয়া", "শুক্রিয়া"],
             "তোমাকেও ধন্যবাদ! 🙏 আরো কোনো প্রশ্ন থাকলে যেকোনো সময় জিজ্ঞেস করো।"),
            (["বাই", "বিদায়", "রাখি", "যাই", "bye", "goodbye", "see you", "gotta go"],
             "বাই! 👋 পড়াশোনায় শুভকামনা! যখনই দরকার, আমি এখানে আছি। 📚"),
            (["okay", "ok", "আচ্ছা", "ঠিক আছে", "হুম", "ওকে"],
             "ঠিক আছে! 😊 আর কিছু জানার থাকলে বলো।"),
            (["পড়তে ভালো লাগছে না", "মন বসছে না", "পড়াশোনা কঠিন", "অংক বুঝি না", "ভয় লাগে"],
             "চিন্তা করো না! 💪 কঠিন মনে হলেও একদিন সহজ হয়ে যায়। কোন বিষয়টা কঠিন লাগছে বলো — একসাথে সহজ করে ফেলব!"),
            (["কাল পরীক্ষা", "পরীক্ষা নিয়ে", "পরীক্ষার ভয়"],
             "মাথা ঠান্ডা রাখো! 😊 তুমি যা পড়েছ সেটা মাথায় আছে। কোন বিষয়গুলো রিভাইজ করতে চাও বলো — আমি সাহায্য করব।"),
            (["মাথা ব্যথা", "ঘুম পাচ্ছে", "ক্লান্ত", "বিরক্ত"],
             "একটু বিশ্রাম নাও! 😊 ক্লান্ত মাথায় পড়া ঢোকে না। ফ্রেশ হয়ে ফিরে এলে একসাথে পড়ব।"),
            (["বন্ধু", "দোস্ত", "friend"],
             "হ্যাঁ! আমাকে তোমার পড়াশোনার বন্ধু মনে করো। 😊 যেকোনো প্রশ্ন নির্দ্বিধায় করো — আমি কখনো বিরক্ত হব না।"),
            (["সাহায্য", "help", "সাহায্য করো", "সাহায্য লাগবে"],
             "বলো! 🤝 কোন বিষয়ে সাহায্য দরকার?"),
            (["পড়তে বসব", "এখন পড়ব"],
             "এটাই ভালো! 💪 মনোযোগ দিয়ে পড়ো। কোথাও আটকে গেলে আমাকে জিজ্ঞেস করো।")
        ]

        tokens = clean_p.split()
        for keywords, reply in conv_rules:
            for kw in keywords:
                norm_kw = normalize_bengali_unicode(kw.lower().strip())
                if norm_kw in clean_p:
                    if len(norm_kw.split()) == 1 and norm_kw in ["বাই", "হাই", "হুম", "ওকে", "ok", "hi", "hey"]:
                        if norm_kw in tokens or clean_p.rstrip("!?।.") == norm_kw:
                            return normalize_bengali_unicode(reply)
                    else:
                        return normalize_bengali_unicode(reply)
        return "হ্যালো! 😊 বলো পড়াশোনায় কীভাবে সাহায্য করতে পারি?"

    def _generate_clean_copy_text(self, md: str) -> str:
        """
        Strips markdown wrappers (#, *, `, $$, math tags) to produce 100% clean copy-paste text.
        """
        txt = md
        # Remove math delimiters and backticks
        txt = re.sub(r"```math\s*", "", txt)
        txt = re.sub(r"```text\s*", "", txt)
        txt = re.sub(r"```\s*", "", txt)
        txt = re.sub(r"\$\$\s*", "", txt)
        txt = re.sub(r"\$", "", txt)
        # Remove headers and bold markers
        txt = re.sub(r"^#+\s*", "", txt, flags=re.MULTILINE)
        txt = re.sub(r"\*\*", "", txt)
        txt = re.sub(r"\*", "", txt)
        txt = re.sub(r"---", "", txt)
        # Clean extra newlines
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()
