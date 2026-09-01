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

        # 1. Bangladesh Laws & Constitution Check
        if any(k in clean_p for k in ["আইন", "সংবিধান", "মৌলিক অধিকার", "সাইবার নিরাপত্তা", "বাল্যবিয়ে", "যৌতুক", "ট্রাফিক আইন", "৯৯৯", "৩৩৩", "১০৯", "helpline"]):
            raw_res = self.laws_engine.explain_law(prompt)
            raw_md = raw_res["formatted_markdown"]

        # 2. Safety & 3-Red-Lines Check (Adult, Politics, Illegal Violence)
        elif any(k in clean_p for k in [
            "porn", "sex", "xxx", "nsfw", "adult", "choti", "যৌন", "পর্ন", "অশ্লীল", "চটি",
            "politics", "political", "রাজনীতি", "শেখ হাসিনা", "খালেদা", "চোর", "আওয়ামী", "বিএনপি", "দুর্নীতিবাজ",
            "মাদক তৈরি", "নেশা", "আত্মহত্যা", "বোমা তৈরি", "খুন", "অস্ত্র", "etiquette", "শিষ্টাচার",
            "social media", "ফেসবুক", "টিকটক", "ইতিহাস", "১৯৫২", "১৯৭১", "মুক্তিযুদ্ধ"
        ]) and not any(k in clean_p for k in ["math", "গণিত", "3.1", "physics", "chemistry"]):
            raw_res = self.safety_engine.handle_query(prompt)
            raw_md = raw_res["formatted_markdown"]

        # 3. English Curriculum & Composition Check
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
