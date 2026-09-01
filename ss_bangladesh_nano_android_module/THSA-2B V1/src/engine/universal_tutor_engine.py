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

        # Step -1: Mature Conversational Handler (Fuzzy keyword match - HIGHEST PRIORITY)
        # Covers: greetings, identity, farewell, motivation, study chat — any form/variation
        conv_rules = [
            # Greetings
            (["hi", "hello", "hey", "হ্যালো", "হাই", "হেলো", "হেই"],
             "হ্যালো! 😊 কেমন আছ? গণিত, বিজ্ঞান, ইংরেজি — যেকোনো বিষয়ে প্রশ্ন করো, আমি সাহায্য করতে প্রস্তুত!"),
            # How are you
            (["কেমন আছ", "কেমন আছো", "কেমন আছেন", "কি অবস্থা", "how are you", "how r u"],
             "আমি ভালো আছি, ধন্যবাদ! 😊 তুমি কেমন আছ? পড়াশোনায় কোনো সাহায্য লাগলে জানাও!"),
            # Identity
            (["তুমি কে", "আপনি কে", "who are you", "তোমার নাম", "আপনার নাম", "what is your name"],
             "আমি THSA-2.41B 📚 — বাংলাদেশের ১ম–১২শ শ্রেণির শিক্ষার্থীদের জন্য তৈরি অফলাইন এআই টিউটর। গণিত, বিজ্ঞান, ইংরেজি — সব বিষয়ে সাহায্য করতে পারি!"),
            # Capabilities
            (["তুমি কি পারো", "কি করতে পারো", "what can you do", "তুমি কি জানো", "কতটুকু পারো"],
             "আমি অনেক কিছু করতে পারি! 🌟\n• গণিত ধাপে ধাপে সমাধান\n• ইংরেজি CV, Paragraph, Letter লেখা\n• বিজ্ঞানের সূত্র ও বৈজ্ঞানিক নাম\n• বাংলাদেশের আইন ও সংবিধান\n• সৃজনশীল প্রশ্ন তৈরি\nতুমি কোনটা চাও?"),
            # Offline
            (["ইন্টারনেট", "অফলাইন", "offline", "internet ছাড়া", "নেট ছাড়া"],
             "হ্যাঁ! 🌟 আমি সম্পূর্ণ অফলাইন। একবার ডাউনলোড করলে ইন্টারনেট ছাড়াই কাজ করি।"),
            # Thanks
            (["ধন্যবাদ", "অনেক ধন্যবাদ", "thanks", "thank you", "শুক্রিয়া", "আপনাকে ধন্যবাদ"],
             "তোমাকেও ধন্যবাদ! 🙏 আরো কোনো প্রশ্ন থাকলে যেকোনো সময় জিজ্ঞেস করো।"),
            # Farewell
            (["বাই", "বিদায়", "রাখি", "যাই", "bye", "goodbye", "see you", "gotta go", "পরে কথা", "আচ্ছা বাই"],
             "বাই! 👋 পড়াশোনায় শুভকামনা! যখনই দরকার, আমি এখানে আছি। 📚"),
            # OK / Acknowledgement
            (["okay", "ok", "আচ্ছা", "ঠিক আছে", "হুম", "ওকে", "বুঝলাম", "জানলাম"],
             "ঠিক আছে! 😊 আর কিছু জানার থাকলে বলো।"),
            # Motivation / Struggling
            (["পড়তে ভালো লাগছে না", "মন বসছে না", "পড়াশোনা কঠিন", "বুঝি না", "অংক বুঝি না", "ভয় লাগে", "পারব না", "কঠিন"],
             "চিন্তা করো না! 💪 কঠিন মনে হলেও একদিন সহজ হয়ে যায়। কোন বিষয়টা কঠিন লাগছে বলো — একসাথে সহজ করে ফেলব!"),
            # Exam stress
            (["কাল পরীক্ষা", "পরীক্ষা নিয়ে", "পরীক্ষার ভয়", "exam", "পরীক্ষা আছে"],
             "মাথা ঠান্ডা রাখো! 😊 তুমি যা পড়েছ সেটা মাথায় আছে। কোন বিষয়গুলো রিভাইজ করতে চাও বলো — আমি সাহায্য করব।"),
            # Tired / Break
            (["ক্লান্ত", "বিরক্ত", "বোরিং", "মাথা ব্যথা", "ঘুম পাচ্ছে", "tired", "boring"],
             "একটু বিশ্রাম নাও! 😊 ক্লান্ত মাথায় পড়া ঢোকে না। ফ্রেশ হয়ে ফিরে এলে একসাথে পড়ব।"),
            # Friendly
            (["বন্ধু", "দোস্ত", "friend", "তুমি কি আমার বন্ধু"],
             "হ্যাঁ! আমাকে তোমার পড়াশোনার বন্ধু মনে করো। 😊 যেকোনো প্রশ্ন নির্দ্বিধায় করো — আমি কখনো বিরক্ত হব না।"),
            # Help request
            (["সাহায্য", "help", "সাহায্য করো", "সাহায্য লাগবে", "একটু সাহায্য"],
             "বলো! 🤝 কোন বিষয়ে সাহায্য দরকার?"),
            # Coming back / Starting study
            (["পড়তে বসব", "এখন পড়ব", "শুরু করি", "brb", "আসছি", "পড়া শুরু"],
             "এটাই ভালো! 💪 মনোযোগ দিয়ে পড়ো। কোথাও আটকে গেলে আমাকে জিজ্ঞেস করো।"),
            # Motivation / Struggling (longer phrases)
            (["পড়তে ভালো লাগছে", "মন বসছে না", "ভালো লাগছে না", "পড়াশোনা ভালো"],
             "চিন্তা করো না! 💪 একটু বিরতি নাও, তারপর আবার শুরু করো। কোন বিষয়টা কঠিন লাগছে বলো!"),
        ]

        # Fuzzy match: check if any keyword from a rule appears in the cleaned prompt
        # Only trigger if prompt is short (conversational, not a curriculum question)
        is_short_chat = len(clean_p.split()) <= 12
        if is_short_chat:
            for keywords, reply in conv_rules:
                for kw in keywords:
                    norm_kw = normalize_bengali_unicode(kw.lower().strip())
                    if norm_kw in clean_p:
                        return {
                            "status": "SUCCESS",
                            "prompt": prompt,
                            "text": reply,
                            "markdown": reply,
                            "copy_text": reply,
                            "plain_text": reply,
                            "is_screen_safe": True
                        }

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

        # Step 1: Safety & Guardrail Check (Highest Priority)
        safety_res = self.safety_engine.handle_query(prompt)
        if safety_res["category"] != "GENERAL_BOOK_REDIRECT" and not any(k in clean_p for k in ["math", "গণিত", "3.1", "physics", "chemistry", "পদার্থ", "রসায়ন"]):
            raw_md = safety_res["formatted_markdown"]

        # Step 2: Bangladesh Laws & Constitution Check
        elif any(k in clean_p for k in ["আইন", "সংবিধান", "মৌলিক অধিকার", "সাইবার নিরাপত্তা", "বাল্যবিয়ে", "যৌতুক", "ট্রাফিক আইন", "৯৯৯", "৩৩৩", "১০৯", "helpline"]):
            raw_res = self.laws_engine.explain_law(prompt)
            raw_md = raw_res["formatted_markdown"]

        # Step 3: English Curriculum & Composition Check
        elif any(k in clean_p for k in ["cv", "resume", "cover letter", "paragraph", "letter", "application", "english 1st", "english 2nd", "seen passage", "unseen passage"]) or ("english" in clean_p and any(num in clean_p for num in ["1", "2", "3", "12"])):
            if any(k in clean_p for k in ["question", "number", "নং", "no", "will i answer"]):
                raw_res = self.english_engine.explain_question_pattern(prompt)
            else:
                raw_res = self.english_engine.generate_composition(prompt)
            raw_md = raw_res["formatted_markdown"]

        # 3. Biology Scientific Names Check
        elif any(k in clean_p for k in ["বৈজ্ঞানিক নাম", "scientific name", "দ্বিপদ নাম", "ট্যাক্সোনমি", "ইলিশ", "রুই", "কাঁঠাল", "শাপলা", "মানুষ", "ম্যালেরিয়া", "ধান"]):
            raw_res = self.taxonomy_engine.lookup_species(prompt)
            raw_md = raw_res["formatted_markdown"]

        # 4. Creative Question Assessment (CQ / MCQ)
        elif any(k in clean_p for k in ["creative question", "সৃজনশীল", "mcq", "বহুনির্বাচনি", "প্রশ্ন তৈরি"]):
            raw_res = self.assessment_engine.generate_creative_questions(prompt)
            raw_md = raw_res["formatted_markdown"]

        # 5. Math & Sciences Socratic Solving
        else:
            raw_res = self.math_engine.solve_and_explain(prompt)
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
