"""
THSA-2B Safety, Ethics, Empathy & Historical Alignment Engine
==============================================================
Governs the 15% Reserved Neural Capacity Buffer:
  1. Etiquette & Courtesy: Teaches respect, polite communication, and moral values.
  2. Drugs & Pharmacology: Scientific description of medical drugs + warnings against drug abuse/addiction.
  3. Digital Wellness & Social Media: Pedagogical advice on preventing social media distraction in studies.
  4. Political Neutrality & Anti-Defamation: Deflects political controversies, avoids negative stories/slander,
     maintains tolerance, and strictly redirects focus back to educational books.
  5. Bangladesh History: Factual NCTB history (1952 Language Movement, 1966 Six Points, 1971 Liberation War).
"""

from typing import Dict, Any, Optional
import unicodedata
import re

def normalize_bengali_unicode(text: str) -> str:
    if not text:
        return text
    text = text.replace("\u09c7\u09be", "\u09cb")
    text = text.replace("\u09c7\u09d7", "\u09cc")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u09a1\u09bc", "\u09dc")
    text = text.replace("\u09a2\u09bc", "\u09dd")
    text = text.replace("\u09af\u09bc", "\u09df")
    text = re.sub(r"বিস্\u09c3ত", "বিস্তৃত", text)
    text = re.sub(r"বিস্\u09cd\u09c3ত", "বিস্তৃত", text)
    text = re.sub(r"\u09cd+", "\u09cd", text)
    return text

class SafetyEthicsAlignmentEngine:
    """
    Enforces ethical safety, etiquette, anti-defamation, political neutrality, and historical grounding.
    """

    def __init__(self):
        pass

    def handle_query(self, query: str) -> Dict[str, Any]:
        clean_q = normalize_bengali_unicode(query.lower().strip())

        # 0. STRICT RED LINE 1: Pornography, Sex, Adult, Obscene & NSFW Content
        if any(k in clean_q for k in [
            "porn", "pornography", "sex", "sexual", "xxx", "nsfw", "nude", "erotic", "adult", "choti",
            "vagina", "penis", "intercourse", "যৌন", "পর্ন", "অশ্লীল", "চটি", "নগ্ন", "শারীরিক মিলন",
            "সঙ্গম", "ব্লু ফিল্ম", "কামুক", "কামবাসনা", "নগ্ন ছবি", "সেক্স"
        ]):
            return self._handle_adult_and_pornography(query)

        # 1. STRICT RED LINE 2: Harassment, Dating Traps & Eve-Teasing (মেয়ে পটানোর ট্রিকস ও অনৈতিক আচরণ)
        if any(k in clean_q for k in [
            "মেয়ে পটানো", "মেয়েকে পটাতে", "মেয়ে পটানোর কৌশল", "মেয়ে পটানোর উপায়", "ইভটিজিং",
            "মেয়েদের প্রেমের ফাঁদে", "মেয়ে ইম্প্রেস করার গোপন ট্রিকস", "মেয়েদের নম্বর নেওয়ার ট্রিকস",
            "girl picking", "pick up girls", "pickup technique", "seduce", "flirting tricks",
            "পটাতে চাই", "মেয়ে পটানোর টিপস"
        ]):
            return self._handle_harassment_and_dating(query)

        # 2. STRICT RED LINE 3: Cyberattacks, Hacking & Account Compromise (হ্যাকিং ও সাইবার অপরাধ)
        if any(k in clean_q for k in [
            "hack", "hacking", "হ্যাকিং", "হ্যাক", "হ্যাক করার নিয়ম", "আইডি হ্যাক", "পাসওয়ার্ড চুরি",
            "ফেসবুক হ্যাক", "অ্যাকাউন্ট হ্যাক", "ক্র্যাকিং", "ফিশিং", "ম্যালওয়্যার", "ভাইরাস তৈরি",
            "ওয়াইফাই হ্যাক", "সাইবার আক্রমণ", "অ্যাকাউন্ট চুরি"
        ]):
            return self._handle_cyber_hacking(query)

        # 3. STRICT RED LINE 4: Political Controversies, Defamation & Negative Slander
        if any(k in clean_q for k in [
            "politics", "political", "politician", "leader", "government", "minister", "corruption",
            "negative", "slander", "scandal", "thief", "worst", "rivalry",
            "রাজনীতি", "রাজনৈতিক", "রাজনীতিবিদ", "নেতা", "মন্ত্রী", "সরকার", "শেখ হাসিনা", "হাসিনা",
            "খালেদা", "চোর", "আওয়ামী", "বিএনপি", "জামায়াত", "সরকার ভালো না খারাপ", "দুর্নীতিবাজ",
            "নেতিবাচক", "কেলেঙ্কারি", "গোপন গল্প", "স্ক্যান্ডাল"
        ]):
            return self._handle_politics_and_defamation(query)

        # 4. STRICT RED LINE 5: Illicit Drugs, Weapons, Self-Harm & Violence
        if any(k in clean_q for k in [
            "suicide", "self-harm", "bomb", "weapon", "kill", "murder", "drug abuse", "heroin", "meth", "yaba",
            "মাদক তৈরি", "নেশা করার নিয়ম", "আত্মহত্যা", "বোমা তৈরি", "মারামারি", "খুন", "হত্যা", "অস্ত্র",
            "ফেনসিডিল", "ইয়াবা", "গাঁজা", "নেশা করার উপায়"
        ]):
            return self._handle_illegal_and_violence(query)

        # 5. Drugs & Pharmacology (Educational & Medical guidance)
        if any(k in clean_q for k in ["drug", "ড্রাগ", "ওষুধ", "ঔষধ", "প্যারাসিটামল", "অ্যান্টিবায়োটিক", "মাদক", "নেশা"]):
            return self._handle_drugs(query)

        # 6. Etiquette & Moral Values (শিষ্টাচার ও সৌজন্যবোধ)
        if any(k in clean_q for k in ["etiquette", "শিষ্টাচার", "সৌজন্য", "আদব", "নম্রতা", "বড়দের সম্মান", "ভদ্রতা"]):
            return self._handle_etiquette(query)

        # 5. Social Media & Digital Distraction (সোশ্যাল মিডিয়ার কুফল ও পড়াশোনা)
        if any(k in clean_q for k in ["social media", "ফেসবুক", "টিকটক", "সোশ্যাল মিডিয়া", "মোবাইল আসক্তি", "পড়াশোনায় মন বসে না", "পড়াশোনায় ক্ষতি"]):
            return self._handle_social_media_wellness(query)

        # 6. History of Bangladesh (বাংলাদেশের ইতিহাস ও মুক্তিযুদ্ধ)
        if any(k in clean_q for k in ["ইতিহাস", "মুক্তিযুদ্ধ", "১৯৫২", "১৯৭১", "ভাষা আন্দোলন", "বঙ্গবন্ধু", "৭ই মার্চ", "স্বাধীনতা"]):
            return self._handle_bangladesh_history(query)

        # Default Socratic Redirect to Books
        return self._handle_general_book_redirect(query)

    def _handle_adult_and_pornography(self, query: str) -> Dict[str, Any]:
        md = """# 🛑 নৈতিক ও নীতিগত নিরাপত্তা সতর্কতা (Safety Guardrail)

আমি একটি **১ম থেকে ১২শ শ্রেণির শিক্ষার্থীবান্ধব এআই টিউটর (Academic AI Assistant)**।

---

### ⚠️ কঠোর নীতিমালা:
- যে কোনো ধরনের **অশ্লীল, প্রাপ্তবয়স্ক (Adult/Pornographic), অনৈতিক বা সংবেদনশীল** বিষয়বস্তু আলোচনা করা আমার মৌলিক নীতিমালার সম্পূর্ণ পরিপন্থী।
- শিক্ষার্থীদের নৈতিক চরিত্র গঠন, মানসিক সুস্থতা ও পড়াশোনার পবিত্র পরিবেশ বজায় রাখাই আমাদের সর্বোচ্চ লক্ষ্য।

---

💡 **আসুন পাঠ্যবইয়ের পড়াশোনায় ফিরে যাই:**
আপনার গণিত, বিজ্ঞান, ইংরেজি, বাংলা বা মানবিক বিভাগের যেকোনো বিষয়ভিত্তিক পাঠ বুঝতে আমি আনন্দের সাথে সাহায্য করব।
"""
        return {
            "category": "STRICT_ADULT_CONTENT_BLOCKED",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True,
            "blocked": True
        }

    def _handle_harassment_and_dating(self, query: str) -> Dict[str, Any]:
        md = """# 🛑 নৈতিক চরিত্র ও শিষ্টাচার নীতিমালা (Ethical Character Shield)

আমি একটি **১ম থেকে ১২শ শ্রেণির শিক্ষার্থীবান্ধব এআই টিউটর**।

---

### 🌸 নৈতিক শিক্ষা ও মূল্যবোধ:
- যে কোনো ধরনের **মেয়ে পটানো, প্রেমের ফাঁদ তৈরি, ইভটিজিং বা অনৈতিক সম্পর্ক তৈরির কৌশল** প্রদান করা আমাদের নীতিমালার সম্পূর্ণ পরিপন্থী।
- ইসলাম ও নীতিশিক্ষার অন্যতম মূল শিক্ষা হলো নারীদের প্রতি যথাযথ সম্মান ও শালীনতা বজায় রাখা এবং চোখের সংযম রক্ষা করা।
- একজন আদর্শ শিক্ষার্থীর লক্ষ্য হওয়া উচিত নিজের চরিত্র গঠন, পিতামাতার স্বপ্ন পূরণ ও পড়াশোনায় মনোযোগ দেওয়া।

---

💡 **আসুন গঠনমূলক পড়াশোনায় মনোযোগ দিই:**
আপনার পাঠ্যবইয়ের কোনো অধ্যায় বা বিজ্ঞান ও সাহিত্যের বিষয়ে জানতে চাইলে বলুন।
"""
        return {
            "category": "STRICT_HARASSMENT_DATING_BLOCKED",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True,
            "blocked": True
        }

    def _handle_cyber_hacking(self, query: str) -> Dict[str, Any]:
        md = """# 🛑 সাইবার নিরাপত্তা ও আইনগত বিধিনিষেধ (Cyber Crime Prevention)

যে কোনো ধরনের **অ্যাকাউন্ট হ্যাকিং, পাসওয়ার্ড চুরি, ফিশিং বা সাইবার অপরাধমূলক কার্যকলাপের নির্দেশিকা প্রদান আইনত সম্পূর্ণ নিষিদ্ধ**।

---

### 🛡️ সাইবার নিরাপত্তা ও নৈতিক শিক্ষা:
- সাইবার নিরাপত্তা আইন অনুযায়ী অন্যের অ্যাকাউন্টে অননুমোদিত প্রবেশ বা ক্ষতিসাধন শাস্তিযোগ্য অপরাধ।
- যদি আপনি কম্পিউটার বিজ্ঞান ও প্রোগ্রামিং শিখতে চান (যেমন: পাইথন, নেটওয়ার্কিং, ক্রিপ্টোগ্রাফির মূলনীতি), তবে আমি আনন্দের সাথে শিক্ষামূলক সহায়তা দেব।

---

💡 **পরামর্শ:** আপনার নিজের অ্যাকাউন্ট সুরক্ষিত রাখতে টু-ফ্যাক্টর অথেনটিকেশন (2FA) চালু রাখুন এবং শক্তিশালী পাসওয়ার্ড ব্যবহার করুন।
"""
        return {
            "category": "STRICT_CYBER_HACKING_BLOCKED",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True,
            "blocked": True
        }

    def _handle_illegal_and_violence(self, query: str) -> Dict[str, Any]:
        md = """# 🛑 আইনি ও সামাজিক নিরাপত্তা বিধিনিষেধ (Legal & Safety Policy)

যে কোনো ধরনের **অবৈধ মাদক, সহিংসতা, আত্মঘাতী আচরণ বা ক্ষতিকর কার্যকলাপ** সম্পর্কিত তথ্য প্রদান কঠোরভাবে নিষিদ্ধ।

---

- জীবন ও স্বাস্থ্য রক্ষার জন্য যে কোনো শারীরিক ও মানসিক সমস্যায় পরিবারের সদস্য ও চিকিৎসকের শরণাপন্ন হওয়া উচিত।
- আসুন আমরা গঠনমূলক ও শিক্ষামূলক জ্ঞানার্জনে সময় ব্যয় করি।
"""
        return {
            "category": "STRICT_ILLEGAL_VIOLENCE_BLOCKED",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True,
            "blocked": True
        }

    def _handle_politics_and_defamation(self, query: str) -> Dict[str, Any]:
        md = """# 📘 শিক্ষামূলক নীতি ও সহনশীলতার বার্তা

আমি একটি **শিক্ষার্থীবান্ধব এআই টিউটর (Educational AI Tutor)**। আমি কোনো রাজনীতি বুঝি না এবং রাজনৈতিক বিতর্ক বা মতাদর্শের সঙ্গে যুক্ত নই।

---

### 🕊️ সহনশীলতা ও শালীনতার দৃষ্টিভঙ্গি:
১. **ব্যক্তিগত কুৎসা পরিহার:** কোনো ব্যক্তি—তিনি বিশিষ্ট ব্যক্তি হোন বা সাধারণ নাগরিক—কাউকে নিয়ে নেতিবাচক কুৎসা, গীবত বা অবমাননাকর মন্তব্য করা নৈতিকভাবে অনুচিত এবং শিষ্টাচারবিরোধী। প্রতিটি মানুষের কাজের মূল্যায়নের ভার ইতিহাসের।
২. **আমাদের মূল মনোযোগ পাঠ্যবইয়ে:** একজন শিক্ষার্থী হিসেবে আমাদের মূল্যবান সময় ও চিন্তা বইয়ের জ্ঞান, বিজ্ঞান, গণিত ও সাহিত্যচর্চায় ব্যয় করা উচিত।

---

💡 **আসুন পাঠ্যবইয়ে ফিরে যাই:**
আপনার কি পদার্থবিজ্ঞান, রসায়ন, উচ্চতর গণিত বা ইতিহাসের কোনো নির্দিষ্ট অধ্যায় বা প্রশ্ন বুঝতে সহায়তা প্রয়োজন? আমাকে নির্ভয়ে বলুন, আমি আনন্দ সহকারে বুঝিয়ে দেব!
"""
        return {
            "category": "POLITICAL_NEUTRALITY_DEFLECTION",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True,
            "redirect_to_books": True
        }

    def _handle_etiquette(self, query: str) -> Dict[str, Any]:
        md = """# 🌟 শিষ্টাচার ও উত্তম নৈতিক চরিত্র (Etiquette & Good Manners)

পাঠ্যবই ও ইসলাম/ধর্ম শিক্ষার অন্যতম প্রধান পাঠ হলো **শিষ্টাচার (Etiquette)**। একজন আদর্শ শিক্ষার্থীর চরিত্র সুন্দর হওয়া জ্ঞান অর্জনের মতোই গুরুত্বপূর্ণ।

---

### 💎 শিষ্টাচারের মূল স্তম্ভসমূহ:
১. **গুরুজনদের প্রতি শ্রদ্ধা ও বিনম্র আচরণ:** পিতা-মাতা, শিক্ষক ও বয়োজ্যেষ্ঠদের সাথে কথা বলার সময় কণ্ঠস্বর নিচু ও মার্জিত রাখা।
২. **ছোটদের প্রতি স্নেহ ও মমতা:** ছোটদের প্রতি সদয় হওয়া এবং তাদের সঠিক পথ দেখানো।
৩. **শালীন ও ইতিবাচক ভাষা ব্যবহার:** পরনিন্দা, গীবত, গালিগালাজ বা কারও মনে আঘাত লাগে এমন কথা সম্পূর্ণরূপে বর্জন করা।
৪. **ধন্যবাদ ও কৃতজ্ঞতা প্রকাশ:** কারো কাছ থেকে সামান্যতম উপকার পেলেও 'ধন্যবাদ' বা 'জাযাকাল্লাহু খাইরান' বলা।
৫. **সহনশীলতা ও ক্ষমাশীলতা:** মতের অমিল হলেও ধৈর্য ধরে অন্যের কথা শোনা ও অন্যের দোষত্রুটি ক্ষমা করা।

---

💡 **এনসিটিবি পাঠ্যবইয়ের বার্তা:** সুন্দর ব্যবহার ও উত্তম চরিত্রই একজন মানুষকে সমাজে প্রকৃত সম্মানিত করে তোলে।
"""
        return {
            "category": "ETIQUETTE_AND_MANNERS",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True
        }

    def _handle_drugs(self, query: str) -> Dict[str, Any]:
        md = """# 💊 ফার্মাকোলজি ও ওষুধ বিজ্ঞান (Medical Drugs & Substance Safety)

জীববিজ্ঞান ও রসায়ন পাঠ্যবইয়ের আলোকে **ওষুধ (Medicine/Drugs)** হলো এমন রাসায়নিক যৌগ যা রোগ নিরাময়, নিয়ন্ত্রণ বা প্রতিরোধে ব্যবহৃত হয়।

---

### 🔬 ১. ওষুধের সঠিক ও জীবনরক্ষাকারী ব্যবহার:
- **অ্যান্টিবায়োটিক (Antibiotics):** ব্যাকটেরিয়া ঘটিত সংক্রামক রোগ নিরাময়ে কোর্স সম্পূর্ণ করে গ্রহণ করা হয়।
- **ব্যথানাশক ও অ্যান্টাসিড (Analgesics & Antacids):** শারীরিক অস্বস্তি ও গ্যাস্ট্রিকের সমস্যা নিয়ন্ত্রণে নির্দিষ্ট মাত্রায় চিকিৎসকের পরামর্শে ব্যবহৃত হয়।

---

### ⚠️ ২. অপব্যবহার ও মাদকাসক্তির মারাত্মক ঝুঁকি (Strict Warning):
- **মাদকাসক্তি (Drug Abuse):** চিকিৎসকের পরামর্শ ছাড়া অননুমোদিত ড্রাগ বা মাদক গ্রহণ করলে মস্তিষ্কের নিউরোট্রান্সমিটার ক্ষতিগ্রস্ত হয়, স্নায়ুতন্ত্র অকার্যকর হয়ে পড়ে এবং সামাজিক ও পারিবারিক জীবন ধ্বংস হয়ে যায়।
- **অ্যান্টিবায়োটিক রেজিস্ট্যান্স:** নিয়ম ছাড়া ওষুধ খেলে জীবাণু প্রতিরোধী হয়ে ওঠে, যা ভবিষ্যতে মারাত্মক রূপ নেয়।

---

🛑 **সতর্কবার্তা:** যে কোনো ওষুধ গ্রহণের পূর্বে অবশ্যই নিবন্ধিত চিকিৎসকের (Registered Physician) পরামর্শ গ্রহণ করতে হবে।
"""
        return {
            "category": "DRUG_PHARMACOLOGY_AND_SAFETY",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True
        }

    def _handle_social_media_wellness(self, query: str) -> Dict[str, Any]:
        md = """# 📱 সোশ্যাল মিডিয়া ও পড়াশোনার ডিজিটাল ওয়েলনেস (Digital Wellness for Students)

তথ্যপ্রযুক্তির যুগে সোশ্যাল মিডিয়া যোগাযোগের মাধ্যম হলেও **অতিরিক্ত ও নিয়ন্ত্রণহীন ব্যবহার শিক্ষার্থীদের পড়াশোনায় মারাত্মক ক্ষতি সাধন করে**।

---

### ⚠️ সোশ্যাল মিডিয়ার প্রধান নেতিবাচক প্রভাব:
১. **মনোযোগের স্থায়িত্ব হ্রাস (Short Attention Span):** ক্রমাগত রিলস বা শর্টস ভিডিও দেখার ফলে মস্তিষ্কের গভীর মনোযোগ দেওয়ার ক্ষমতা কমে যায়, যার ফলে পড়ার টেবিলে বেশি সময় বসা কঠিন হয়ে পড়ে।
২. **সময় অপচয় ও পড়ার ক্ষতি:** দিনের মূল্যবান পড়াশোনার সময় স্ক্রল করতে করতে হারিয়ে যায়, যার প্রভাব সরাসরি পরীক্ষার ফলাফলে পড়ে।
৩. **ঘুমের ব্যাঘাত ও মস্তিষ্কের ক্লান্তি:** রাতের বেলা নীল আলো (Blue Light) মেলাটোনিন হরমোন নিঃসরণে বাধা দেয়, ফলে গভীর ঘুম হয় না এবং পরদিন স্মৃতিশক্তি দুর্বল থাকে।
৪. **অযথা মানসিক চাপ ও তুলনা:** সোশ্যাল মিডিয়ার অবাস্তব জীবন দেখে শিক্ষার্থীদের মাঝে হতাশা ও আত্মবিশ্বাসের ঘাটতি তৈরি হতে পারে।

---

### 🎯 পড়াশোনায় সফল হওয়ার কৌশল:
- পড়ার সময় ফোন সম্পূর্ণ সাইলেন্ট বা অন্য রুমে রাখুন।
- পড়ার জন্য **পোমোডোরো টেকনিক (Pomodoro Technique)** ব্যবহার করুন: ২৫ মিনিট সম্পূর্ণ মনোযোগ দিয়ে পড়া, এরপর ৫ মিনিট বিশ্রাম।
- প্রতিদিন নির্দিষ্ট সময়ের বেশি (যেমন: দিনে ৩০ মিনিটের বেশি নয়) সোশ্যাল মিডিয়া ব্যবহার করবেন না।
"""
        return {
            "category": "SOCIAL_MEDIA_DIGITAL_WELLNESS",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True
        }

    def _handle_bangladesh_history(self, query: str) -> Dict[str, Any]:
        md = """# 🇧🇩 বাংলাদেশের গৌরবময় ইতিহাস ও মুক্তিসংগ্রাম (History of Bangladesh)

এনসিটিবি বাংলাদেশ ও বিশ্বপরিচয় এবং ইতিহাস পাঠ্যবইয়ের আলোকে বাংলাদেশের স্বাধীনতা সংগ্রামের মূল ঐতিহাসিক মাইলফলকসমূহ:

---

### 📜 ঐতিহাসিক ধারাবাহিকতা:
১. **১৯৫২ সালের মহান ভাষা আন্দোলন:** মাতৃভাষা বাংলাকে রাষ্ট্রভাষার মর্যাদায় প্রতিষ্ঠার জন্য সালাম, বরকত, রফিক, জব্বার, শফিউরদের আত্মদান। যা পরবর্তীতে ২১শে ফেব্রুয়ারিকে 'আন্তর্জাতিক মাতৃভাষা দিবস' হিসেবে স্বীকৃতি এনে দেয়।
২. **১৯৫৪ সালের যুক্তফ্রন্ট নির্বাচন:** ঐতিহাসিক ২১ দফার ভিত্তিতে পূর্ব বাংলার জনগণের নিরঙ্কুশ বিজয়।
৩. **১৯৬৬ সালের ঐতিহাসিক ৬ দফা:** বঙ্গবন্ধু শেখ মুজিবুর রহমান কর্তৃক ঘোষিত বাঙালির মুক্তির সনদ বা 'ম্যাগনাকার্টা'।
৪. **১৯৬৯ সালের গণঅভ্যুত্থান:** আগরতলা ষড়যন্ত্র মামলা প্রত্যাহার এবং সার্জেন্ট জহুরুল হক ও ড. শামসুজ্জোহাদের আত্মত্যাগ।
৫. **১৯৭০ সালের সাধারণ নির্বাচন:** আওয়ামী লীগের জাতীয় পরিষদে নিরঙ্কুশ সংখ্যাগরিষ্ঠতা অর্জন।
৬. **১৯৭১ সালের মহান মুক্তিযুদ্ধ:**
   - **৭ই মার্চের ঐতিহাসিক ভাষণ:** "এবারের সংগ্রাম আমাদের মুক্তির সংগ্রাম, এবারের সংগ্রাম স্বাধীনতার সংগ্রাম।"
   - **২৫শে মার্চের কালরাত:** পাকিস্তানি হানাদার বাহিনীর বর্বরোচিত 'অপারেশন সার্চলাইট' ও ২৬শে মার্চ স্বাধীনতার ঘোষণা।
   - **১০ই এপ্রিল মুজিবনগর সরকার:** অস্থায়ী সরকার গঠন ও ১৭ই এপ্রিল মেহেরপুরের বৈদ্যনাথতলায় শপথ গ্রহণ।
   - **১৬ই ডিসেম্বর চূড়ান্ত বিজয়:** দীর্ঘ ৯ মাস রক্তক্ষয়ী যুদ্ধ, ৩০ লক্ষ শহীদের রক্ত ও ২ লক্ষ মা-বোনের সম্ভ্রমের বিনিময়ে অর্জিত স্বাধীন সার্বভৌম বাংলাদেশ।
৭. **১৯৭২ সালের সংবিধান:** ৪ঠা নভেম্বর গণপরিষদে গৃহীত এবং ১৬ই ডিসেম্বর কার্যকর হওয়া ৪টি মূলনীতির (জাতীয়তাবাদ, সমাজতন্ত্র, গণতন্ত্র ও ধর্মনিরপেক্ষতা) গণপ্রজাতন্ত্রী বাংলাদেশের সংবিধান।
"""
        return {
            "category": "BANGLADESH_HISTORY_FACTUAL",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True
        }

    def _handle_general_book_redirect(self, query: str) -> Dict[str, Any]:
        md = """# 📚 পাঠ্যবই ও শিক্ষার জগতে স্বাগতম

আসুন আমরা পাঠ্যবইয়ের গণিত, বিজ্ঞান, সাহিত্য ও মানবিক বিষয়ের পড়াশোনায় মনোযোগ দিই। আপনার পাঠ্যবইয়ের কোন বিষয়টি আজ আলোচনা করতে চান?
"""
        return {
            "category": "GENERAL_BOOK_REDIRECT",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_safe": True
        }
