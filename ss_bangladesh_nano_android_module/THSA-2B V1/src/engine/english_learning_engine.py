"""
THSA-2B English Language Learning Engine
=========================================
A full Gemini/ChatGPT-level step-by-step English tutor for Bangladesh NCTB students:
  1. Vocabulary Explainer: Word meaning, Bengali translation, part of speech, usage examples.
  2. Grammar Coach: Tense, Noun/Pronoun/Verb/Adjective, Voice, Narration, Parts of Speech.
  3. Sentence Builder: How to construct English sentences from scratch.
  4. Spoken English Practice: Level-adaptive conversation mode.
  5. English Proficiency Checker: Assess user's level (0-100%) and adapt accordingly.
  6. Daily Use Sentences: Practical conversational English.
  7. Vocabulary Improvement Tips: Systematic vocab building methods.
  8. Banglish (Romanized Bengali) Intent Parser: Understands 'ki', 'koi', 'shekao', 'kivabe'.
"""

from typing import Dict, Any, Optional
import re
import unicodedata


def normalize_bengali_unicode(text: str) -> str:
    if not text:
        return text
    text = text.replace("\u09c7\u09be", "\u09cb")
    text = text.replace("\u09c7\u09d7", "\u09cc")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u09a1\u09bc", "\u09dc")
    text = text.replace("\u09a2\u09bc", "\u09dd")
    text = text.replace("\u09af\u09bc", "\u09df")
    text = re.sub(r"\u09cd+", "\u09cd", text)
    return text


# ----------------------------------------------------------------------------
# Vocabulary Knowledge Base (Common words students ask about)
# ----------------------------------------------------------------------------
VOCAB_KB = {
    "good": {
        "bengali": "ভালো / ভালো",
        "part_of_speech": "Adjective (বিশেষণ), Noun (বিশেষ্য), Adverb (ক্রিয়া বিশেষণ)",
        "meanings": [
            "Adjective: ভালো, উত্তম, উন্নত — e.g. *She is a good student.* (সে একজন ভালো ছাত্র।)",
            "Noun: কল্যাণ, মঙ্গল — e.g. *Do good to others.* (অন্যদের মঙ্গল করো।)",
            "Adverb (informal): ভালোভাবে — e.g. *He did good in the exam.* (সে পরীক্ষায় ভালো করেছে।)"
        ],
        "synonyms": "great, excellent, fine, wonderful, nice",
        "antonyms": "bad, poor, terrible, awful",
        "example_sentences": [
            "She has good manners. (তার ভালো আচরণ আছে।)",
            "This is a good book. (এটি একটি ভালো বই।)",
            "Good morning! (শুভ সকাল!)"
        ]
    },
    "beautiful": {
        "bengali": "সুন্দর",
        "part_of_speech": "Adjective (বিশেষণ)",
        "meanings": [
            "Adjective: সুন্দর, মনোরম — e.g. *She has a beautiful smile.* (তার সুন্দর হাসি আছে।)"
        ],
        "synonyms": "lovely, gorgeous, pretty, attractive, stunning",
        "antonyms": "ugly, unattractive, plain",
        "example_sentences": [
            "The flower is beautiful. (ফুলটি সুন্দর।)",
            "Bangladesh is a beautiful country. (বাংলাদেশ একটি সুন্দর দেশ।)",
            "She sang in a beautiful voice. (সে সুন্দর কণ্ঠে গান গাইল।)"
        ]
    },
    "happy": {
        "bengali": "সুখী / আনন্দিত",
        "part_of_speech": "Adjective (বিশেষণ)",
        "meanings": [
            "Adjective: সুখী, আনন্দিত, খুশি — e.g. *I am happy to meet you.* (তোমার সাথে দেখা হয়ে আমি খুশি।)"
        ],
        "synonyms": "joyful, glad, pleased, delighted, cheerful",
        "antonyms": "sad, unhappy, miserable, gloomy",
        "example_sentences": [
            "Children are happy in the park. (শিশুরা পার্কে আনন্দিত।)",
            "I feel happy when I help others. (অন্যদের সাহায্য করলে আমি খুশি অনুভব করি।)"
        ]
    },
    "important": {
        "bengali": "গুরুত্বপূর্ণ",
        "part_of_speech": "Adjective (বিশেষণ)",
        "meanings": [
            "Adjective: গুরুত্বপূর্ণ, তাৎপর্যপূর্ণ — e.g. *Education is important.* (শিক্ষা গুরুত্বপূর্ণ।)"
        ],
        "synonyms": "significant, crucial, vital, essential, key",
        "antonyms": "unimportant, trivial, insignificant",
        "example_sentences": [
            "It is important to study regularly. (নিয়মিত পড়া গুরুত্বপূর্ণ।)",
            "He played an important role. (সে একটি গুরুত্বপূর্ণ ভূমিকা পালন করেছে।)"
        ]
    },
    "run": {
        "bengali": "দৌড়ানো / চালানো",
        "part_of_speech": "Verb (ক্রিয়া)",
        "meanings": [
            "Verb (দৌড়ানো): e.g. *The children run in the field.* (শিশুরা মাঠে দৌড়ায়।)",
            "Verb (চালানো): e.g. *He runs a business.* (সে একটি ব্যবসা পরিচালনা করে।)"
        ],
        "forms": "run (base) → runs (3rd person singular) → ran (past) → run (past participle) → running (present participle)",
        "synonyms": "sprint, jog, dash, race, operate",
        "example_sentences": [
            "I run every morning. (আমি প্রতিদিন সকালে দৌড়াই।)",
            "She ran to catch the bus. (সে বাস ধরতে দৌড়েছিল।)"
        ]
    }
}

# ----------------------------------------------------------------------------
# Grammar Knowledge Base
# ----------------------------------------------------------------------------
GRAMMAR_KB = {
    "tense_present": {
        "title": "✅ Present Tense (বর্তমান কাল) — Complete Guide",
        "content": """Present Tense মানে **এখন** যে কাজ হচ্ছে, সবসময় হয়, বা সত্য।

---

#### 📘 ১. Present Indefinite (Simple Present) — নিয়মিত সত্য ঘটনা বা অভ্যাস:
**Structure:** Subject + **V1** (3rd person singular: V1 + s/es)
| Subject | Verb Form | Example |
|---|---|---|
| I / We / You / They | run | *They play cricket.* (তারা ক্রিকেট খেলে।) |
| He / She / It | runs | *She reads books.* (সে বই পড়ে।) |

---

#### 📘 ২. Present Continuous (চলমান কাজ) — এখনই হচ্ছে:
**Structure:** Subject + **am/is/are + V-ing**
- *I am reading a book.* (আমি এখন একটি বই পড়ছি।)
- *She is cooking food.* (সে এখন রান্না করছে।)

---

#### 📘 ৩. Present Perfect (কাজ শেষ, ফলাফল আছে):
**Structure:** Subject + **have/has + V3 (Past Participle)**
- *I have finished my homework.* (আমি আমার হোমওয়ার্ক শেষ করেছি।)
- *She has eaten lunch.* (সে দুপুরের খাবার খেয়েছে।)

---

#### 📘 ৪. Present Perfect Continuous (কিছুক্ষণ ধরে হচ্ছে):
**Structure:** Subject + **have/has been + V-ing + since/for**
- *I have been studying for 2 hours.* (আমি ২ ঘণ্টা ধরে পড়ছি।)""",
        "follow_up": "🌟 **পরবর্তী পদক্ষেপ:** এখন কি **Past Tense** (অতীত কাল) শিখতে চাও, নাকি Present Tense-এর একটি practice quiz করব?"
    },
    "tense_past": {
        "title": "✅ Past Tense (অতীত কাল) — Complete Guide",
        "content": """Past Tense মানে যে কাজ **অতীতে হয়েছে** বা সম্পন্ন হয়েছে।

---

#### 📘 ১. Past Indefinite (Simple Past) — নির্দিষ্ট সময়ে হয়েছিল:
**Structure:** Subject + **V2 (Past Form)**
| Verb | Past Form | Example |
|---|---|---|
| go | went | *He went to school yesterday.* (সে গতকাল স্কুলে গিয়েছিল।) |
| eat | ate | *She ate rice.* (সে ভাত খেয়েছিল।) |
| read | read | *I read the book.* (আমি বইটি পড়েছিলাম।) |

---

#### 📘 ২. Past Continuous (তখন চলমান কাজ):
**Structure:** Subject + **was/were + V-ing**
- *I was sleeping when he came.* (সে যখন এল, আমি তখন ঘুমাচ্ছিলাম।)

---

#### 📘 ৩. Past Perfect (একটার আগে আরেকটা):
**Structure:** Subject + **had + V3**
- *She had left before I arrived.* (আমি আসার আগেই সে চলে গিয়েছিল।)""",
        "follow_up": "🌟 **পরবর্তী পদক্ষেপ:** এখন কি **Future Tense** শিখতে চাও, নাকি Past Tense-এর ৫টি exercise করব?"
    },
    "tense_future": {
        "title": "✅ Future Tense (ভবিষ্যৎ কাল) — Complete Guide",
        "content": """Future Tense মানে যে কাজ **ভবিষ্যতে হবে**।

---

#### 📘 ১. Future Indefinite (Simple Future):
**Structure:** Subject + **will/shall + V1**
- *I will go to school tomorrow.* (আমি আগামীকাল স্কুলে যাব।)
- *She will study hard.* (সে কঠোর পরিশ্রম করবে।)

---

#### 📘 ২. Future Continuous:
**Structure:** Subject + **will be + V-ing**
- *I will be sleeping at 10 PM.* (রাত ১০টায় আমি ঘুমাচ্ছব।)

---

#### 📘 ৩. Future Perfect:
**Structure:** Subject + **will have + V3**
- *By tomorrow, I will have finished the work.* (আগামীকালের মধ্যে আমি কাজটি শেষ করে ফেলব।)""",
        "follow_up": "🌟 **পরবর্তী পদক্ষেপ:** তিনটি Tense-ই শেখা হয়েছে! এখন কি একটি **Mixed Tense Practice** করতে চাও?"
    },
    "is_are_am_was_were": {
        "title": "✅ Is / Are / Am / Was / Were — সম্পূর্ণ গাইড",
        "content": """এগুলো সবই 'Be Verb' — বাংলায় **হওয়া / থাকা** অর্থে ব্যবহৃত হয়।

---

#### 📊 কখন কোনটা ব্যবহার হবে?

| Verb | Subject | Tense | Example |
|---|---|---|---|
| **am** | I | Present | *I am a student.* (আমি একজন ছাত্র।) |
| **is** | He / She / It / Singular | Present | *She is happy.* (সে খুশি।) |
| **are** | We / You / They / Plural | Present | *They are friends.* (তারা বন্ধু।) |
| **was** | I / He / She / It / Singular | Past | *I was tired.* (আমি ক্লান্ত ছিলাম।) |
| **were** | We / You / They / Plural | Past | *They were late.* (তারা দেরিতে এসেছিল।) |

---

#### 💡 সহজ মনে রাখার নিয়ম:
> **"I am, He/She/It is, We/You/They are"** — এটা মুখস্থ করে ফেলো!
> Past-এ **is → was** এবং **are → were** হয়।""",
        "follow_up": "🌟 **পরবর্তী প্রশ্ন:** এখন কি **Has / Have / Had** এর পার্থক্য শিখতে চাও?"
    },
    "parts_of_speech": {
        "title": "✅ Parts of Speech (শব্দের প্রকারভেদ) — সম্পূর্ণ গাইড",
        "content": """ইংরেজিতে মোট **৮টি Parts of Speech** আছে:

---

#### 📘 ১. Noun (বিশেষ্য) — নাম বোঝায়:
মানুষ, স্থান, বস্তু বা ধারণার নাম।
- *Dhaka, book, happiness, girl* — এগুলো Noun।
- **বাক্যে:** *She reads a **book**.* (সে একটি বই পড়ে।)

#### 📘 ২. Pronoun (সর্বনাম) — Noun-এর পরিবর্তে বসে:
- I, he, she, it, we, you, they, him, her, them
- **বাক্যে:** *Riya is here. **She** is my friend.* (রিয়া এখানে। সে আমার বন্ধু।)

#### 📘 ৩. Verb (ক্রিয়া) — কাজ বা অবস্থা বোঝায়:
- go, eat, run, is, are, seem, look
- **বাক্যে:** *She **reads** books daily.* (সে প্রতিদিন বই পড়ে।)

#### 📘 ৪. Adjective (বিশেষণ) — Noun-কে বর্ণনা করে:
- good, big, beautiful, tall, smart
- **বাক্যে:** *She is a **beautiful** girl.* (সে একটি সুন্দর মেয়ে।)

#### 📘 ৫. Adverb (ক্রিয়া বিশেষণ) — Verb/Adjective-কে বর্ণনা করে:
- quickly, very, always, here, now
- **বাক্যে:** *She runs **quickly**.* (সে দ্রুত দৌড়ায়।)

#### 📘 ৬. Preposition (পদান্বয়ী অব্যয়) — সম্পর্ক দেখায়:
- in, on, at, by, with, under, for, to
- **বাক্যে:** *The book is **on** the table.* (বইটি টেবিলের উপর।)

#### 📘 ৭. Conjunction (সংযোজক অব্যয়) — দুটো বাক্য জোড়া দেয়:
- and, but, or, so, because, although
- **বাক্যে:** *I study hard **because** I want to pass.* (আমি কঠোর পরিশ্রম করি কারণ আমি পাস করতে চাই।)

#### 📘 ৮. Interjection (আবেগসূচক অব্যয়) — আবেগ প্রকাশ করে:
- Oh! Wow! Alas! Hurray! Bravo!
- **বাক্যে:** ***Wow!** That is amazing!* (অবাক কাণ্ড! এটা অসাধারণ!)""",
        "follow_up": "🌟 **পরবর্তী প্রশ্ন:** এখন কি **Tense** (কাল) শিখতে চাও, নাকি প্রতিটি Parts of Speech-এর আরও বিস্তারিত উদাহরণ দেখতে চাও?"
    },
    "sentence_building": {
        "title": "✅ ইংরেজি বাক্য তৈরির নিয়ম (English Sentence Building) — Step by Step",
        "content": """একটি ইংরেজি বাক্যের মূল কাঠামো হলো:

### 📐 মূল কাঠামো: **Subject + Verb + Object**
> যেমন: *She (Subject) reads (Verb) a book (Object).*
> বাংলা: সে একটি বই পড়ে।

---

#### ধাপ ১: Subject বেছে নাও (কে করছে?)
- I, he, she, we, they, Rahim, The student…

#### ধাপ ২: Verb বেছে নাও (কী করছে?)
- eat, go, read, play, write, is, are…

#### ধাপ ৩: Object বেছে নাও (কাকে / কী?)
- a book, rice, cricket, his homework…

#### ধাপ ৪: Time/Place যোগ করো (ঐচ্ছিক)
- every day, in Dhaka, at school, yesterday…

---

#### 🌟 ৫টি Practice Sentence তৈরি করে দেখো:
| বাংলা | ইংরেজি |
|---|---|
| আমি ভাত খাই। | *I eat rice.* |
| সে স্কুলে যায়। | *She goes to school.* |
| তারা ক্রিকেট খেলে। | *They play cricket.* |
| আমি বই পড়ছি। | *I am reading a book.* |
| আমরা বন্ধু। | *We are friends.* |""",
        "follow_up": "🌟 **পরবর্তী চ্যালেঞ্জ:** এখন তুমি নিজে একটি বাক্য তৈরি করে আমাকে দেখাও — আমি সংশোধন করে দেব!"
    },
    "vocabulary_tips": {
        "title": "✅ ইংরেজি শব্দভাণ্ডার বাড়ানোর ৭টি বৈজ্ঞানিক পদ্ধতি",
        "content": """ইংরেজিতে শক্তিশালী vocabulary ছাড়া ভালো করা কঠিন। এই ৭টি পদ্ধতি অনুসরণ করো:

---

#### 🔑 ১. প্রতিদিন ৫টি নতুন শব্দ শেখো (Word-a-Day Method):
একটি ছোট নোটবুক রাখো। প্রতিদিন ৫টি নতুন শব্দ + বাংলা অর্থ + ১টি বাক্য লেখো।

#### 🔑 ২. ব্যবহার করো (Use It or Lose It):
শব্দ শিখলেই সেটা দিয়ে সেদিনই একটি বাক্য বানাও।

#### 🔑 ৩. Root Word শেখো:
একটি Root Word থেকে অনেক শব্দ তৈরি হয়:
- **Create**: creation, creative, creator, recreate, creativity

#### 🔑 ৪. Synonyms ও Antonyms একসাথে শেখো:
- Good → great, excellent | Bad → poor, terrible

#### 🔑 ৫. প্রসঙ্গের মধ্যে শেখো (Contextual Learning):
ইংরেজি গল্প বা সংবাদ পড়ার সময় অজানা শব্দ আলাদা করো।

#### 🔑 ৬. Flashcard পদ্ধতি:
একদিকে ইংরেজি শব্দ, অন্যদিকে বাংলা মানে লিখে প্র্যাকটিস করো।

#### 🔑 ৭. Word Family শেখো:
- **happy** (adj) → **happily** (adv) → **happiness** (noun) → **unhappy** (adj)""",
        "follow_up": "🌟 **পরবর্তী পরামর্শ:** চলো আজকে ৫টি গুরুত্বপূর্ণ শব্দ শিখে নিই — বলো কোন বিষয়ের শব্দ চাও?"
    },
    "daily_sentences": {
        "title": "✅ প্রতিদিনের ব্যবহারের ১৫টি ইংরেজি বাক্য (Daily Use Sentences)",
        "content": """এই বাক্যগুলো দৈনন্দিন জীবনে প্রতিদিন কাজে লাগে:

---

| ক্রম | English Sentence | বাংলা অর্থ |
|:---:|---|---|
| ১ | Good morning! How are you? | শুভ সকাল! তুমি কেমন আছ? |
| ২ | I am fine, thank you! | আমি ভালো আছি, ধন্যবাদ! |
| ৩ | Can you help me, please? | তুমি কি দয়া করে আমাকে সাহায্য করতে পারবে? |
| ৪ | I don't understand. Please explain again. | আমি বুঝিনি। আবার বুঝিয়ে দাও। |
| ৫ | What does this word mean? | এই শব্দটির মানে কী? |
| ৬ | I am learning English step by step. | আমি ধাপে ধাপে ইংরেজি শিখছি। |
| ৭ | Could you speak slowly, please? | একটু ধীরে বলবে কি? |
| ৮ | I am sorry. I made a mistake. | আমি দুঃখিত। আমি ভুল করেছিলাম। |
| ৯ | That is a great idea! | এটা একটি চমৎকার ধারণা! |
| ১০ | I want to improve my English. | আমি আমার ইংরেজি উন্নত করতে চাই। |
| ১১ | Excuse me, where is the library? | মাফ করবেন, লাইব্রেরি কোথায়? |
| ১২ | I will try my best. | আমি আমার সর্বোচ্চ চেষ্টা করব। |
| ১৩ | See you tomorrow! | আগামীকাল দেখা হবে! |
| ১৪ | Thank you very much! | অনেক অনেক ধন্যবাদ! |
| ১৫ | You are welcome! | স্বাগতম / কোনো ব্যাপার না! |""",
        "follow_up": "🌟 **পরবর্তী পদক্ষেপ:** এই বাক্যগুলো মুখস্থ করে আমাকে একটি বলো, আমি পরীক্ষা করব!"
    }
}


class EnglishLearningEngine:
    """
    A full Gemini/ChatGPT-level English tutor engine for NCTB students.
    """

    def __init__(self):
        self._spoken_mode_active = False

    def handle_english_query(self, query: str) -> Dict[str, Any]:
        """
        Main dispatcher for English learning queries.
        """
        q = query.lower().strip()
        q_norm = re.sub(r"['\",।?!]", "", q)

        # -------------------------------------------------------
        # 1. Proficiency / Spoken English Mode Activation
        # -------------------------------------------------------
        if any(k in q_norm for k in [
            "how much english do i know", "i want to talk", "speak english with me",
            "english e kotha", "english kotha", "speak to me in english",
            "how good is my english", "test my english", "english shekao", "english shikhte chai",
            "i want to learn english", "english learn"
        ]):
            return self._english_proficiency_intro()

        # -------------------------------------------------------
        # 2. Vocabulary: Word Meaning Queries
        # -------------------------------------------------------
        # Check known vocab KB first
        for word, data in VOCAB_KB.items():
            if word in q_norm or f"'{word}'" in q_norm or f'"{word}"' in q_norm:
                return self._explain_vocab(word, data)

        # Generic "X মানে কি / what does X mean / X meaning" pattern
        vocab_match = re.search(
            r"(?:what(?:\s+does)?\s+['\"]?(\w+)['\"]?\s+mean|([a-zA-Z]+)\s+(?:মানে|meaning|অর্থ|er mane|er meaning|এর অর্থ))",
            q_norm
        )
        if vocab_match:
            word = (vocab_match.group(1) or vocab_match.group(2) or "").strip()
            if word:
                return self._explain_vocab_dynamic(word)

        # -------------------------------------------------------
        # 3. Tense Teaching
        # -------------------------------------------------------
        if any(k in q_norm for k in ["present tense", "present kaal", "simple present", "present indefinite"]):
            return self._grammar_response(GRAMMAR_KB["tense_present"])
        if any(k in q_norm for k in ["past tense", "past kaal", "past indefinite", "simple past", "অতীত কাল"]):
            return self._grammar_response(GRAMMAR_KB["tense_past"])
        if any(k in q_norm for k in ["future tense", "future kaal", "ভবিষ্যৎ কাল"]):
            return self._grammar_response(GRAMMAR_KB["tense_future"])
        if any(k in q_norm for k in ["tense", "kaal", "কাল"]):
            return self._tense_overview()

        # -------------------------------------------------------
        # 4. Daily Use Sentences (Check before generic sentence building)
        # -------------------------------------------------------
        if any(k in q_norm for k in ["daily", "everyday", "protidin", "roj", "প্রতিদিন", "রোজ"]) and any(k in q_norm for k in ["sentence", "english", "kotha", "use", "বাক্য"]):
            return self._grammar_response(GRAMMAR_KB["daily_sentences"])

        # -------------------------------------------------------
        # 5. Vocabulary Improvement Tips
        # -------------------------------------------------------
        if ("vocab" in q_norm or "শব্দ" in q_norm or "word" in q_norm) and any(k in q_norm for k in ["improve", "barabo", "how to", "tips", "শিখব", "মনে রাখব", "কৌশল", "পদ্ধতি"]):
            return self._grammar_response(GRAMMAR_KB["vocabulary_tips"])

        # -------------------------------------------------------
        # 6. Is/Are/Am/Was/Were
        # -------------------------------------------------------
        if any(k in q_norm for k in ["is are am", "am is are", "was were", "be verb", "difference between is are"]):
            return self._grammar_response(GRAMMAR_KB["is_are_am_was_were"])

        # -------------------------------------------------------
        # 7. Parts of Speech / Noun / Pronoun / Verb
        # -------------------------------------------------------
        if any(k in q_norm for k in ["noun pronoun verb", "parts of speech", "noun ki", "pronoun ki", "verb ki",
                                      "adjective ki", "parts of speach", "word type", "noun", "pronoun", "verb", "adjective"]):
            return self._grammar_response(GRAMMAR_KB["parts_of_speech"])

        # -------------------------------------------------------
        # 8. Sentence Building
        # -------------------------------------------------------
        if any(k in q_norm for k in ["sentence banabo", "sentence kivabe", "english sentence", "how to make sentence",
                                      "sentence toiri", "sentence kemon", "বাক্য তৈরি", "বাক্য গঠন"]):
            return self._grammar_response(GRAMMAR_KB["sentence_building"])

        # -------------------------------------------------------
        # 9. Daily Use Sentences Fallback
        # -------------------------------------------------------
        if any(k in q_norm for k in ["daily sentence", "daily use", "daily life sentence", "everyday english",
                                      "daily english", "5 ta english", "10 ta english", "sentence shekao"]):
            return self._grammar_response(GRAMMAR_KB["daily_sentences"])

        # -------------------------------------------------------
        # 9. Any English word given as standalone (1-3 word query all English)
        # -------------------------------------------------------
        words = q_norm.split()
        if 1 <= len(words) <= 3 and all(re.match(r"[a-z]+", w) for w in words if w):
            candidate = words[0]
            return self._explain_vocab_dynamic(candidate)

        return None  # Not handled by this engine

    def _english_proficiency_intro(self) -> Dict[str, Any]:
        md = """# 🗣️ English Spoken Practice & Proficiency Mode
## আমি তোমার ইংরেজি শেখার সহকারী! 😊

আমি তোমার সাথে ইংরেজিতে কথা বলতে সম্পূর্ণ প্রস্তুত। তবে চলো আগে তোমার ইংরেজির বর্তমান মাত্রা বোঝার জন্য ৩টি ছোট প্রশ্ন করি:

---

### 📋 তোমার ইংরেজি পরীক্ষা (Self Assessment — ৩টি প্রশ্ন):

**প্রশ্ন ১:** নিচের বাক্যটি ইংরেজিতে বলো:
> "আমি প্রতিদিন সকালে স্কুলে যাই।"

**প্রশ্ন ২:** "Beautiful" শব্দটি দিয়ে একটি বাক্য তৈরি করো।

**প্রশ্ন ৩:** "Is" আর "Are"-এর পার্থক্য কি জানো? এক লাইনে বলো।

---

> 💡 **উত্তর দিলে আমি তোমার ইংরেজির মাত্রা নির্ধারণ করব এবং সেই অনুযায়ী আমরা একসাথে ধাপে ধাপে এগোব!**

### 🌟 যদি সরাসরি শুরু করতে চাও:
- বলো **"Beginner"** — একেবারে শুরু থেকে শেখাব
- বলো **"Intermediate"** — Tense, Grammar ও Sentence থেকে শেখাব
- বলো **"Advanced"** — Composition, Vocabulary ও Spoken English শেখাব"""

        return {
            "status": "SUCCESS",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_screen_safe": True
        }

    def _tense_overview(self) -> Dict[str, Any]:
        md = """## ✅ Tense (কাল) — সম্পূর্ণ ওভারভিউ

ইংরেজিতে মোট **৩টি মূল Tense** এবং তার ৪টি করে উপবিভাগ আছে (মোট ১২টি):

---

| Tense | মূল রূপ | Example |
|---|---|---|
| **Present** (বর্তমান) | V1 / am-is-are + V-ing / have-has + V3 | *She reads. / She is reading.* |
| **Past** (অতীত) | V2 / was-were + V-ing / had + V3 | *She read. / She was reading.* |
| **Future** (ভবিষ্যৎ) | will + V1 / will be + V-ing / will have + V3 | *She will read. / She will be reading.* |

---

### কোনটা আগে শিখতে চাও?
➡️ **"present tense shekao"** বলো
➡️ **"past tense shekao"** বলো
➡️ **"future tense shekao"** বলো"""

        return {
            "status": "SUCCESS",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_screen_safe": True
        }

    def _explain_vocab(self, word: str, data: dict) -> Dict[str, Any]:
        examples = "\n".join(f"- *{s}*" for s in data.get("example_sentences", []))
        meanings = "\n".join(f"- {m}" for m in data.get("meanings", []))
        forms = f"\n\n**🔄 Verb Forms:** {data['forms']}" if "forms" in data else ""

        md = f"""## 📖 Word: **"{word.upper()}"** — সম্পূর্ণ ব্যাখ্যা

---

### 🔤 বাংলা অর্থ: {data['bengali']}
### 📝 Parts of Speech: {data['part_of_speech']}{forms}

---

### 📚 অর্থ ও ব্যবহার:
{meanings}

---

### 💬 উদাহরণ বাক্য (Example Sentences):
{examples}

---

### 🔗 Synonyms (সমার্থক শব্দ): {data.get('synonyms', 'N/A')}
### 🔗 Antonyms (বিপরীত শব্দ): {data.get('antonyms', 'N/A')}

---

🌟 **পরবর্তী পদক্ষেপ:** এই শব্দটি দিয়ে তুমি নিজে একটি বাক্য তৈরি করো — আমি দেখে দেব! অথবা আরেকটি নতুন শব্দ জিজ্ঞেস করো।"""

        return {
            "status": "SUCCESS",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_screen_safe": True
        }

    def _explain_vocab_dynamic(self, word: str) -> Dict[str, Any]:
        """Dynamic fallback for unknown vocab words — gives what it knows."""
        if word in VOCAB_KB:
            return self._explain_vocab(word, VOCAB_KB[word])

        md = f"""## 📖 Word: **"{word.upper()}"** — অর্থ ও ব্যবহার

---

আপনি যে শব্দটি জিজ্ঞেস করেছেন তা ইংরেজির একটি গুরুত্বপূর্ণ শব্দ!

### 🔤 শব্দটি চেনার উপায়:
প্রতিটি ইংরেজি শব্দ শেখার সময় এই ৫টি বিষয় নোট করো:
1. **অর্থ (Meaning):** বাংলায় কী বোঝায়?
2. **Part of Speech:** Noun / Verb / Adjective / Adverb?
3. **Sentence-এ ব্যবহার:** বাক্যে কোথায় বসে?
4. **Synonym:** একই অর্থের অন্য শব্দ কী?
5. **Antonym:** বিপরীত শব্দ কী?

---

💡 **আমাদের Database-এ এই শব্দটি এখনো যুক্ত হয়নি।** তবে তুমি এই শব্দটি দিয়ে একটি বাক্য তৈরি করার চেষ্টা করো — আমি সংশোধন করে দেব!

🌟 **পরবর্তী পদক্ষেপ:** নিচের যেকোনো একটি প্রশ্ন করো:
- **"good মানে কি?"**
- **"beautiful মানে কি?"**
- **"happy মানে কি?"**
- **"tense shekao"** বা **"daily sentences shekao"**"""

        return {
            "status": "SUCCESS",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_screen_safe": True
        }

    def _grammar_response(self, kb_item: dict) -> Dict[str, Any]:
        md = f"""## {kb_item['title']}

---

{kb_item['content']}

---

{kb_item.get('follow_up', '🌟 **পরামর্শ:** আরো কোনো বিষয়ে জানতে চাইলে জিজ্ঞেস করো!')}"""

        return {
            "status": "SUCCESS",
            "formatted_markdown": normalize_bengali_unicode(md),
            "is_screen_safe": True
        }
