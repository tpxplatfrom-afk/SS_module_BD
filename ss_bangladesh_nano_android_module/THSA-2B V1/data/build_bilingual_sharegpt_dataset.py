#!/usr/bin/env python3
"""
THSA-2B: Comprehensive Bilingual (Bangla & English) Conversational Dataset Generator.
Structured from Micro-Interactions (Very Small) to Multi-Turn Deep Philosophical & Historical Discussions.

CURRICULUM HIERARCHY (From Small to Deep):
  LEVEL 1: Micro-Greetings, Small Talk & AI Identity (Hi, Hello, কেমন আছেন, কী অবস্থা, Who are you)
  LEVEL 2: Everyday Casual Assistance & Wellness (Day-to-day chat, stress relief, daily routines)
  LEVEL 3: Moderate Discussions, Opinions & Creative Problem Solving (Tech habits, language learning, art)
  LEVEL 4: Deep Historical, Cultural & Geopolitical Analyses (1971 Liberation War, 1952 Language Movement, World Civilizations)
  LEVEL 5: Deep Academic STEM & Scientific Deductions (Physics mechanics, calculus, thermodynamics, AI philosophy)

Outputs:
  - data/train_sharegpt.jsonl (90% Train Split)
  - data/test_sharegpt.jsonl (10% Test Evaluation Split)
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

# Deterministic seed for repeatable 90/10 split
random.seed(42)

def generate_all_conversations() -> List[Dict[str, Any]]:
    """Builds the comprehensive conversational dataset in strictly ordered pedagogical progression."""
    conversations = []

    # =========================================================================
    # LEVEL 1: MICRO-GREETINGS, CASUAL SMALL TALK & AI IDENTITY (VERY SMALL)
    # =========================================================================

    # 1.1 Micro Greetings & Fast Replies (বাংলা)
    conversations.append({
        "id": "lvl1_001_micro_greetings_bangla",
        "topic": "Level 1: Micro Greetings & Polite Casual Exchanges (বাংলা সংক্ষিপ্ত কুশল বিনিময়)",
        "conversations": [
            {
                "from": "human",
                "value": "হাই!"
            },
            {
                "from": "gpt",
                "value": "হ্যালো! কেমন আছেন? আজকে আপনাকে কীভাবে সাহায্য করতে পারি?"
            },
            {
                "from": "human",
                "value": "কেমন আছো?"
            },
            {
                "from": "gpt",
                "value": "আমি খুব ভালো আছি, ধন্যবাদ! আপনি কেমন আছেন? আশা করি আপনার দিনটি দারুণ কাটছে।"
            },
            {
                "from": "human",
                "value": "তুমি কে?"
            },
            {
                "from": "gpt",
                "value": "আমি **শান্ত (Shanto)**—একটি দ্রুতগতির, সম্পূর্ণ অফলাইন অন-ডিভাইস এআই সহকারী। আপনার যেকোনো প্রশ্ন, পড়াশোনা বা লেখালেখির কাজে সাহায্য করাই আমার কাজ।"
            },
            {
                "from": "human",
                "value": "ধন্যবাদ তোমাকে!"
            },
            {
                "from": "gpt",
                "value": "আপনাকে অনেক ধন্যবাদ! যেকোনো সময় যেকোনো প্রশ্ন থাকলে নিঃসঙ্কোচে জিজ্ঞেস করতে পারেন।"
            }
        ]
    })

    # 1.2 Micro Greetings & Introductions (English)
    conversations.append({
        "id": "lvl1_002_micro_greetings_english",
        "topic": "Level 1: Micro Greetings & Introductions (English Casual & Polite)",
        "conversations": [
            {
                "from": "human",
                "value": "Hi!"
            },
            {
                "from": "gpt",
                "value": "Hello there! How are you doing today? How can I assist you?"
            },
            {
                "from": "human",
                "value": "Who are you?"
            },
            {
                "from": "gpt",
                "value": "I am **Shanto**, your on-device AI assistant. I run 100% locally and privately on your phone without needing internet access."
            },
            {
                "from": "human",
                "value": "Good morning! Hope you have a good day."
            },
            {
                "from": "gpt",
                "value": "Good morning! Wishing you a productive and joyful day ahead. What are your plans for today?"
            },
            {
                "from": "human",
                "value": "Thanks, you're awesome!"
            },
            {
                "from": "gpt",
                "value": "Thank you so much! Always happy to help. Let me know whenever you need anything!"
            }
        ]
    })

    # =========================================================================
    # LEVEL 2: EVERYDAY CASUAL ASSISTANCE & WELLNESS (SMALL TO MEDIUM)
    # =========================================================================

    # 2.1 Daily Wellness, Stress & Relaxation (বাংলা)
    conversations.append({
        "id": "lvl2_001_daily_wellness_stress_relief_bangla",
        "topic": "Level 2: Daily Wellness & Stress Relief (দৈনন্দিন মানসিক ক্লান্তি ও প্রশান্তি)",
        "conversations": [
            {
                "from": "human",
                "value": "আজকে সারাদিন অনেক দৌড়াদৌড়ি আর কাজের চাপ গেল। খুব ক্লান্ত লাগছে, কীভাবে একটু চাঙ্গা হওয়া যায়?"
            },
            {
                "from": "gpt",
                "value": "ব্যস্ত দিনের পর ক্লান্তি লাগাটা খুবই স্বাভাবিক। শরীর ও মনকে দ্রুত রিল্যাক্স করতে এই সহজ কাজগুলো করতে পারেন:\n\n1. **এক গ্লাস কুসুম গরম পানি বা লেবু-চা:** হালকা উষ্ণ পানি শরীরের ক্লান্তি দূর করে এবং তরতাজা ভাব আনে।\n2. **মুখ ধুয়ে একটু বিশ্রাম:** চোখে-মুখে ঠান্ডা পানির ঝাপটা দিন এবং ৫-১০ মিনিট চোখ বন্ধ করে শুয়ে থাকুন।\n3. **স্ক্রিন থেকে বিরতি:** ফোন বা ল্যাপটপের পর্দা থেকে অন্তত ১৫ মিনিট দূরে থাকুন।\n4. **প্রিয় কোনো গান:** পছন্দের হালকা কোনো ইন্সট্রুমেন্টাল বা সুফি গান শুনতে পারেন।\n\nআজকের কাজের কথা এখন ভুলে যান, কাল সকালে আবার নতুন করে শুরু করা যাবে।"
            },
            {
                "from": "human",
                "value": "রাতে ভালো ঘুমের জন্য কোনো ছোট অভ্যাস আছে কি?"
            },
            {
                "from": "gpt",
                "value": "অবশ্যই! ৩টি খুব সহজ নিয়ম মেনে চললে চমৎকার ঘুম হয়:\n- **নির্দিষ্ট সময়:** প্রতিদিন ঠিক একই সময়ে বিছানায় যান।\n- **অন্ধকার ও নীরব পরিবেশ:** ঘরের আলো একদম মৃদু বা অন্ধকার রাখুন।\n- **চিন্তামুক্ত মন:** আগামীকালের প্রধান ২-৩টি কাজের কথা খাতায় লিখে রাখুন, যাতে মাথায় অহেতুক চিন্তা ঘুরপাক না খায়।"
            }
        ]
    })

    # 2.2 Time Management & Healthy Habits (English)
    conversations.append({
        "id": "lvl2_002_time_management_habits_english",
        "topic": "Level 2: Productivity & Study Habit Optimization (Time Management)",
        "conversations": [
            {
                "from": "human",
                "value": "I always procrastinate until the last minute before exams or deadlines. How can I build a realistic daily study routine?"
            },
            {
                "from": "gpt",
                "value": "Procrastination usually happens because the brain feels overwhelmed by the sheer size of the goal. The secret is to lower the activation energy:\n\n### 3-Step Actionable Strategy:\n1. **The 5-Minute Rule:** Tell yourself you will only study for 5 minutes. Once you start, inertia breaks and continuing becomes effortless.\n2. **Pomodoro Technique (25/5):** Study with complete focus for 25 minutes (phone in another room), then enjoy a 5-minute movement break.\n3. **Top 3 Priorities:** Each morning, pick only 3 essential tasks to accomplish. When those 3 are done, celebrate your progress."
            },
            {
                "from": "human",
                "value": "What if I get distracted by social media notifications during the 25 minutes?"
            },
            {
                "from": "gpt",
                "value": "Put your phone on 'Do Not Disturb' mode and place it face down out of arm's reach. Reducing visual triggers eliminates 80% of accidental scrolling urges!"
            }
        ]
    })

    # =========================================================================
    # LEVEL 3: MODERATE DISCUSSIONS, OPINIONS & PROBLEM SOLVING (MEDIUM)
    # =========================================================================

    # 3.1 Technology, Social Media & Human Connection (Bilingual)
    conversations.append({
        "id": "lvl3_001_tech_social_media_humanity",
        "topic": "Level 3: Technology, Social Media & Genuine Human Connection (প্রযুক্তি ও মানবিক সম্পর্ক)",
        "conversations": [
            {
                "from": "human",
                "value": "বর্তমান যুগে মানুষ সবসময় অনলাইনে সংযুক্ত থেকেও কেন বাস্তবে নিজেদের একা মনে করে? প্রযুক্তি কি আমাদের সম্পর্কগুলোকে অগভীর করে দিচ্ছে?"
            },
            {
                "from": "gpt",
                "value": "এটি আমাদের সময়ের এক অদ্ভুত মনস্তাত্ত্বিক বাস্তবতা। আমরা আগের চেয়ে বেশি **কানেক্টেড**, কিন্তু একই সাথে অনেক বেশি **একাকী**।\n\n### এর পেছনের ৩টি মূল কারণ:\n1. **লাইক ও রিয়্যাক্ট বনাম আসল নৈকট্য:** সামাজিক মাধ্যমে আমরা সংক্ষিপ্ত টেক্সট বা ইমোজি আদান-প্রদান করি, কিন্তু মানুষের মন শান্ত হয় কণ্ঠের আবেগ, চোখের ভাষা এবং সামনাসামনি আন্তরিকতায়।\n2. **হাইলাইট রিল সিন্ড্রোম:** প্রত্যেকে অনলাইনে তাদের জীবনের সবচেয়ে সুন্দর মুহূর্তগুলো দেখায়। ফলে নিজের সাধারণ জীবনের সাথে তুলনা করে অবচেতনভাবেই হতাশা জন্ম নেয়।\n3. **মনোযোগের ঘাটতি:** সোশ্যাল মিডিয়ার দ্রুত স্ক্রোলিং অভ্যাস আমাদের গভীর কোনো আড্ডা বা বই পড়ার ধৈর্য কমিয়ে দেয়।\n\nপ্রযুক্তি নিজে খারাপ নয়; কিন্তু সপ্তাহে অন্তত একদিন বন্ধুদের সাথে দেখা করা বা পরিবারের সাথে স্ক্রিনমুক্ত আড্ডা দেওয়াই আসল মানসিক শান্তি নিশ্চিত করে।"
            },
            {
                "from": "human",
                "value": "How can one use AI tools effectively without losing their critical thinking skills?"
            },
            {
                "from": "gpt",
                "value": "Treat AI as a **collaborative sparring partner** rather than an oracle:\n- **Ask for Critiques, Not Just Answers:** Instead of asking for a finished essay, write your draft first and ask the AI: *'What are the 3 weakest arguments in my draft?'*\n- **Verify and Synthesize:** Always challenge premises, compare sources, and reformulate ideas in your own voice."
            }
        ]
    })

    # 3.2 Art, Literature & Creativity (English & Bangla)
    conversations.append({
        "id": "lvl3_002_books_creativity_expression",
        "topic": "Level 3: Books, Literature & Creative Expression (সাহিত্য ও সৃজনশীলতা)",
        "conversations": [
            {
                "from": "human",
                "value": "কেন আমাদের নিয়মিত সাহিত্য ও ক্লাসিক বই পড়া উচিত? এটি কি কেবল বিনোদন নাকি মস্তিষ্কের বিকাশে বড় ভূমিকা রাখে?"
            },
            {
                "from": "gpt",
                "value": "সাহিত্য পাঠ কেবল অবসর কাটানোর মাধ্যম নয়; এটি মানুষের সহানুভূতি (empathy), ভাষা দক্ষতা এবং সূক্ষ্ম চিন্তাশক্তি বিকাশের সবচেয়ে শক্তিশালী মাধ্যম।\n\n### সাহিত্যের অনন্য প্রভাব:\n1. **সহানুভূতিশীল দৃষ্টিভঙ্গি:** যখন আপনি রবীন্দ্রনাথ, বিভূতিভূষণ, টলস্টয় বা শেক্সপিয়ারের কোনো চরিত্র পড়েন, তখন আপনি নিজের অজান্তেই ভিন্ন এক মানুষের আনন্দ ও বেদনা অনুভব করেন।\n2. **চিন্তার গভীরতা:** জটিল বিষয়কে সুন্দর ও সুসংহত ভাষায় প্রকাশ করার ক্ষমতা তৈরি হয়।\n3. **মানসিক প্রশান্তি:** বই পড়ার সময় মস্তিষ্কের একাগ্রতা বাড়ে এবং মন শান্ত হয়।"
            },
            {
                "from": "human",
                "value": "What makes Bengali literature so rich in world heritage?"
            },
            {
                "from": "gpt",
                "value": "Bengali literature is blessed with an extraordinary musicality and philosophical depth. From the medieval mysticism of Charyapada and Baul poetry (Lalon Shah) to the universal humanism of Rabindranath Tagore (Asia's first Nobel laureate in 1913) and the revolutionary rebel spirit of Kazi Nazrul Islam, the language carries a unique synthesis of lyrical elegance and fierce social justice."
            }
        ]
    })

    # =========================================================================
    # LEVEL 4: DEEP HISTORICAL, CULTURAL & GEOPOLITICAL ANALYSIS (DEEP)
    # =========================================================================

    # 4.1 1971 Liberation War & Naval Strategy (বাংলা - পূর্ণাঙ্গ বিশ্লেষণ)
    conversations.append({
        "id": "lvl4_001_bd_1971_liberation_war_deep",
        "topic": "Level 4: 1971 Bangladesh Liberation War (মুক্তিযুদ্ধ, অপারেশন জ্যাকপট ও ভূরাজনীতি)",
        "conversations": [
            {
                "from": "human",
                "value": "১৯৭১ সালের মুক্তিযুদ্ধ সম্পর্কে সামরিক কৌশল, ৭ই মার্চের ভাষণ, অপারেশন জ্যাকপট এবং পরাশক্তিদের আন্তর্জাতিক ভূরাজনীতির সমন্বিত গভীর বিশ্লেষণ দাও।"
            },
            {
                "from": "gpt",
                "value": "১৯৭১ সালের মুক্তিযুদ্ধ ছিল একটি জাতির আত্মনিয়ন্ত্রণাধিকার, সাংস্কৃতিক অস্তিত্ব এবং চরম রাজনৈতিক বৈষম্যের বিরুদ্ধে এক অবিস্মরণীয় জনযুদ্ধ (people's war)।\n\n### ১. ৭ই মার্চের ভাষণ: কৌশলগত স্বাধীনতার রূপরেখা\nবঙ্গবন্ধু শেখ মুজিবুর রহমান একদিকে সরাসরি যুদ্ধের ঘোষণা এড়িয়ে অহিংস অসহযোগ আন্দোলনের নির্দেশ দেন, অন্যদিকে ঘরে ঘরে দুর্গ গড়ে তোলার আহ্বান জানান। এটি পাকিস্তানি জান্তাকে অপ্রস্তুত করে এবং সমগ্র বাঙালিকে যুদ্ধের জন্য মানসিকভাবে সম্পূর্ণ প্রস্তুত করে।\n\n### ২. সামরিক বিন্যাস ও ১১টি সেক্টর\nমেহেরপুরের মুজিবনগরে প্রবাসী সরকার গঠিত হওয়ার পর প্রধান সেনাপতি জেনারেল ওসমানীর নেতৃত্বে দেশকে ১১টি সেক্টরে এবং জেড-ফোর্স, কে-ফোর্স ও এস-ফোর্স ব্রিগেডে ভাগ করা হয়।\n- **গেরিলা গণবাহিনী ও ক্র্যাক প্লাটুন:** হিট অ্যান্ড রান কৌশলে পাকিস্তানি রসদ লাইন ধ্বংস করে।\n- **অপারেশন জ্যাকপট (১৫ই আগস্ট ১৯৭১):** প্রায় ৩৫০ জন নৌ-কমান্ডো আকাশবাণী রেডিওর সংকেত ('আমি তোমায় যত শুনিয়েছিলাম গান' ও 'আমার পুতুল আজকে যাবে শ্বশুরবাড়ি') পেয়ে একযোগে চট্টগ্রাম ও মোংলা বন্দরে পাকিস্তানি অস্ত্রবাহী ২৬টিরও বেশি যুদ্ধজাহাজ লিম্পেট মাইন দিয়ে বঙ্গোপসাগরের তলদেশে ডুবিয়ে দেন।\n\n### ৩. আন্তর্জাতিক ভূরাজনীতি ও ১৬ই ডিসেম্বরের চূড়ান্ত বিজয়\n- **সোভিয়েত ইউনিয়ন:** নিরাপত্তা পরিষদে তিনবার ভেটো দিয়ে মার্কিন-চীন চক্রান্ত নস্যাৎ করে।\n- **ভারত:** ১ কোটি শরণার্থীকে আশ্রয় এবং যৌথ কমান্ডের মাধ্যমে সরাসরি যুদ্ধ পরিচালনা করে।\n১৬ই ডিসেম্বর জেনারেল নিয়াজির ৯৩,০০০ সৈন্যের ঐতিহাসিক আত্মসমর্পণের মাধ্যমে স্বাধীন গণপ্রজাতন্ত্রী বাংলাদেশের অভ্যুদয় ঘটে।"
            },
            {
                "from": "human",
                "value": "১৯৫২ সালের ভাষা আন্দোলনের সাথে ১৯৭১ সালের মুক্তিযুদ্ধের আদর্শিক যোগসূত্রটি ব্যাখ্যা করো।"
            },
            {
                "from": "gpt",
                "value": "১৯৫২ সালের ২১শে ফেব্রুয়ারির ভাষা আন্দোলন এবং ১৯৭১ সালের মুক্তিযুদ্ধ মূলত একই মুক্তির ধারাক্রমের দুটি অপরিহার্য স্তম্ভ—**'সাংস্কৃতিক স্বাধিকার থেকে রাজনৈতিক সার্বভৌমত্বে উত্তরণ'**।\n\n১৯৪৭ সালে ধর্মভিত্তিক দ্বিজাতিতত্ত্বের ভিত্তিতে পাকিস্তান গঠিত হলেও পাকিস্তানের শাসকগোষ্ঠী ৫৬% বাঙালির মাতৃভাষা বাংলাকে বাদ দিয়ে উর্দুকে চাপিয়ে দেওয়ার ষড়যন্ত্র করে। ১৯৫২ সালে সালাম, বরকত, রফিকের রক্তে দ্বিজাতিতত্ত্বের কৃত্রিম বাঁধ ভেঙে ভাষাভিত্তিক ধর্মনিরপেক্ষ বাঙালি জাতীয়তাবাদের সূচনা হয়, যা ১৯৭১-এ পূর্ণ স্বাধীনতায় রূপ নেয়।"
            }
        ]
    })

    # 4.2 Ancient Bengal & River Delta Ecology (বাংলা ও ইংরেজি)
    conversations.append({
        "id": "lvl4_002_ancient_bengal_sundarbans_ecology",
        "topic": "Level 4: Ancient Bengal, Sultanate Muslin Trade & Sundarbans Ecology",
        "conversations": [
            {
                "from": "human",
                "value": "প্রাচীন ও মধ্যযুগীয় বাংলার পাল বংশ, স্বাধীন সুলতানি আমল এবং ঢাকাই মসলিনের বিশ্ববাণিজ্যিক প্রভাব বিস্তারিত বলো।"
            },
            {
                "from": "gpt",
                "value": "প্রাচীন ও মধ্যযুগীয় বাংলা ছিল সমগ্র এশিয়ার অন্যতম সমৃদ্ধ অর্থনৈতিক কেন্দ্রবিন্দু:\n\n1. **পাল সাম্রাজ্য (৭৫০–১১৬১ খ্রিষ্টাব্দ):** দীর্ঘ মাৎস্যন্যায়ের অবসান ঘটিয়ে গোপালকে রাজা নির্বাচিত করা হয়। সোমপুর মহাবিহার ও নালন্দার পৃষ্ঠপোষকতায় জ্ঞানবিজ্ঞানের স্বর্ণযুগ রচিত হয়।\n2. **স্বাধীন সুলতানি আমল (১৩৩৮–১৫৩৮ খ্রিষ্টাব্দ):** শামসুদ্দিন ইলিয়াস শাহ 'শাহ-ই-বাঙ্গালাহ' উপাধি নেন। ইবনে বতুতা বাংলাকে প্রাচুর্যময় স্বর্গ হিসেবে বর্ণনা করেন।\n3. **ঢাকাই মসলিন:** সুবা বাংলার ফুটি কার্পাস থেকে তৈরি সূক্ষ্ম মসলিন রোমান সাম্রাজ্য থেকে ইউরোপের রাজদরবার পর্যন্ত বিশ্বখ্যাত ছিল।"
            },
            {
                "from": "human",
                "value": "Why is the Sundarbans mangrove forest globally unique and critical as an ecological green shield?"
            },
            {
                "from": "gpt",
                "value": "The Sundarbans—the world's largest contiguous mangrove ecosystem—functions as an indispensable ecological powerhouse:\n- **Unique Halophytic Adaptations:** Trees like Sundari and Gewa survive in tidal saline waters via specialized breathing roots (pneumatophores), stilt roots, and viviparous germination.\n- **Natural Cyclone Bio-Shield:** During catastrophic super-cyclones (Sidr, Aila, Remal), the dense mangrove root and canopy matrix absorbs 30–40% of tidal surge energy, saving millions of human lives across the lower Bengal delta."
            }
        ]
    })

    # 4.3 Japan: Meiji Restoration to Post-War Economic Miracle (English)
    conversations.append({
        "id": "lvl4_003_world_history_japan_evolution",
        "topic": "Level 4: History of Japan: Sakoku, Meiji Restoration & Kaizen Quality",
        "conversations": [
            {
                "from": "human",
                "value": "How did Japan transition from an isolated feudal state under the Tokugawa Shogunate (Sakoku) to an industrialized global power during the Meiji Restoration, and what cultural philosophies enabled this balance between tradition and technology?"
            },
            {
                "from": "gpt",
                "value": "The transformation of Japan during the late 19th and early 20th centuries represents one of the most astonishing institutional pivots in human history, driven by the philosophy of **Wakon Yosai** (Japanese Spirit, Western Knowledge).\n\n### 1. The Tokugawa Shogunate & Sakoku (1603–1867)\nUnder *Sakoku* (Closed Country), Japan barred foreign entry and trade, preserving 250 years of domestic peace (*Pax Tokugawa*) but missing the Industrial Revolution until Commodore Perry's 'Black Ships' arrived in 1853.\n\n### 2. The Meiji Restoration (1868)\nYoung samurai restored Emperor Meiji and adopted the pragmatic motto: *\"To prevent Western colonization, we must master Western science and industry faster than anyone else.\"*\n- **Iwakura Mission (1871–1873):** Two-year tour studying Western factories, legal codes, and naval doctrine.\n- **Abolition of Feudal Hierarchy:** Samurai class disbanded, universal education established, and state-led conglomerates (*Zaibatsu* like Mitsubishi, Mitsui) founded.\n\n### 3. Post-WWII Economic Miracle & Quality Revolution\nAfter total destruction in 1945, Japan focused entirely on export manufacturing excellence via **Kaizen** (continuous improvement), **Just-In-Time (Kanban)** inventory control, and W. Edwards Deming's statistical quality control, dominating global automotive and electronics markets."
            }
        ]
    })

    # 4.4 United Kingdom: Industrial Revolution & Coal-Wage Dynamics (English)
    conversations.append({
        "id": "lvl4_004_world_history_uk_industrial_revolution",
        "topic": "Level 4: History of the UK: Industrial Revolution & Steam Power",
        "conversations": [
            {
                "from": "human",
                "value": "Why did the Industrial Revolution begin in 18th-century Britain rather than in other wealthy empires like France or China? What were the economic, geological, and legal conditions?"
            },
            {
                "from": "gpt",
                "value": "The onset of the Industrial Revolution in Great Britain (c. 1760–1840) was the result of a unique confluence of economic incentives, geological fortune, and institutional protections.\n\n### 1. Allen's 'High-Wage, Cheap-Coal' Thesis\n- **Economic Incentive:** British wages were the highest in the world, while coal in northern England and Wales was exceptionally cheap. This made investing in expensive, coal-fired capital machinery (Newcomen and James Watt steam engines) highly profitable in Britain, whereas elsewhere manual labor remained cheaper.\n\n### 2. Institutional & Legal Protections\n- **Rule of Law & Patent System:** Post-1688 Glorious Revolution, the British Parliament protected private property rights and enforced the *Statute of Monopolies*, incentivizing technological inventors (Arkwright, Crompton, Watt).\n\n### 3. Financial Infrastructure & Global Trade\n- The Bank of England (1694) provided low-interest capital, while the Royal Navy secured raw cotton imports and global export markets."
            }
        ]
    })

    # 4.5 USA: Constitutional Debates & Hamiltonian Industrial Rise (English)
    conversations.append({
        "id": "lvl4_005_world_history_usa_constitutional_democracy",
        "topic": "Level 4: History of the USA: Hamilton vs Jefferson & Constitutional Philosophy",
        "conversations": [
            {
                "from": "human",
                "value": "What were the philosophical arguments between Alexander Hamilton and Thomas Jefferson regarding federal power and the national economy?"
            },
            {
                "from": "gpt",
                "value": "The foundational debates between Alexander Hamilton and Thomas Jefferson in the 1790s established the ideological fault lines of modern governance.\n\n### 1. Implied Powers vs. Strict Construction\n- **Jefferson's Strict Construction:** Argued Congress had no constitutional right to incorporate the *First Bank of the United States* (1791) under the 10th Amendment.\n- **Hamilton's Implied Powers:** Utilized the *Necessary and Proper Clause* (Article I, Section 8), arguing that sovereign ends implicitly grant the sovereign means.\n\n### 2. Hamiltonian Industrial Blueprint\nHamilton's *Report on Manufactures* (1791) advocated protective tariffs, federal infrastructure, and manufacturing bounties, transforming the US from an agrarian colony into the world's leading industrial power."
            }
        ]
    })

    # 4.6 Ancient Egypt: Nile Hydrology, Rhind Papyrus & Architecture (English)
    conversations.append({
        "id": "lvl4_006_world_history_ancient_egypt_geometry",
        "topic": "Level 4: Ancient Egypt: Nile Hydrology, Surveying & Geometry",
        "conversations": [
            {
                "from": "human",
                "value": "Explain how ancient Egyptian civilization sustained monumental architecture for over 3,000 years in harmony with the Nile River, and describe their mathematical achievements in surveying."
            },
            {
                "from": "gpt",
                "value": "Ancient Egypt was an ecological and architectural engineering triumph orchestrated in harmony with the annual Nile inundation (*Akhet*).\n\n### 1. The Hydrological Engine & Calendar\n- **Three Seasons:** *Akhet* (Inundation), *Peret* (Emergence/Sowing), and *Shemu* (Harvest).\n- **Nilometer:** Graduated stone wells measured flood height to predict harvest yields and set equitable land taxes.\n\n### 2. Geometry Born of Necessity\nAnnual floods washed away boundary markers, requiring *Harpedonaptai* (rope-stretchers) to re-survey farmland using the 3-4-5 right-triangle ratio. The *Rhind Papyrus* (c. 1550 BCE) demonstrates formulas for calculating triangle areas and truncated pyramid volumes."
            }
        ]
    })

    # =========================================================================
    # LEVEL 5: DEEP ACADEMIC STEM, MATHEMATICS & SCIENTIFIC REASONING (DEEP)
    # =========================================================================

    # 5.1 Physics: Mechanics, Energy Conservation & Thermodynamics (বাংলা ও গণিত)
    conversations.append({
        "id": "lvl5_001_nctb_physics_mechanics_thermodynamics",
        "topic": "Level 5: NCTB Curriculum: Physics Mechanics & Thermodynamics (বলবিদ্যা ও শক্তি সংরক্ষণ)",
        "conversations": [
            {
                "from": "human",
                "value": "নিউটনের গতির ৩টি সূত্র এবং কাজ-শক্তি উপপাদ্যের গাণিতিক সম্পর্কটি বুঝিয়ে দাও। একটি ১০ কেজি ভরের বস্তুকে ৫ মিটার ওপর থেকে ফেললে শক্তি সংরক্ষণের হিসাবটি দেখাও।"
            },
            {
                "from": "gpt",
                "value": "পদার্থবিজ্ঞানের মৌলিক ভিত্তি হলো শক্তির রূপান্তর এবং বলের ক্রিয়ায় বস্তুর গতির পরিবর্তন।\n\n### ১. নিউটনের গতির সূত্রাবলী\n1. **প্রথম সূত্র:** বাহ্যিক বল প্রয়োগ না করলে স্থির বস্তু চিরকাল স্থির এবং গতিশীল বস্তু সমবেগে সরলরেখায় চলে।\n2. **দ্বিতীয় সূত্র:** $F = m a = m \\frac{v - u}{t}$\n3. **তৃতীয় সূত্র:** প্রত্যেক ক্রিয়ার সমান ও বিপরীত প্রতিক্রিয়া আছে ($F_1 = -F_2$)।\n\n### ২. গাণিতিক সমাধান: ১০ কেজি ভরের বস্তু ৫ মিটার পতন\n- ভর, $m = 10\\text{ kg}$, উচ্চতা, $h = 5\\text{ m}$, $g = 9.8\\text{ ms}^{-2}$\n\n1. **শীর্ষবিন্দুতে বিভব শক্তি:**\n   $$E_p = mgh = 10 \\times 9.8 \\times 5 = \\mathbf{490\\text{ J}}, \\quad E_k = 0$$\n\n2. **মাটিতে স্পর্শের পূর্বমুহূর্তে গতিশক্তি:**\n   $$v^2 = 2gh = 2 \\times 9.8 \\times 5 = 98\\text{ m}^2\\text{s}^{-2}$$\n   $$E_k = \\frac{1}{2} m v^2 = \\frac{1}{2} \\times 10 \\times 98 = \\mathbf{490\\text{ J}}, \\quad E_p = 0$$\n\n3. **মাটিতে আঘাতের পর:** গতিশক্তি শব্দ, তাপ ও বস্তু বিকৃতিতে রূপান্তরিত হয়, যা তাপগতিবিদ্যার প্রথম সূত্র ($\\Delta Q = \\Delta U + \\Delta W$) নিশ্চিত করে।"
            }
        ]
    })

    # 5.2 Mathematics: Calculus & Optimization (বাংলা ও গণিত)
    conversations.append({
        "id": "lvl5_002_nctb_mathematics_calculus_optimization",
        "topic": "Level 5: NCTB Mathematics: Differential Calculus & Optimization (ক্যালকুলাস ও অপটিমাইজেশন)",
        "conversations": [
            {
                "from": "human",
                "value": "ডিফারেনশিয়াল ক্যালকুলাসের মূল ভাবনা এবং কীভাবে কোনো বক্ররেখার চরম মান (Maximum & Minimum value) নির্ণয় করা হয় তা বাস্তব উদাহরণসহ বুঝিয়ে দাও।"
            },
            {
                "from": "gpt",
                "value": "ক্যালকুলাস হলো পরিবর্তনের গণিত (Mathematics of Change)।\n\n### ১. অন্তরীকরণের জ্যামিতিক তাৎপর্য\n- কোনো ফাংশন $y = f(x)$-এর অন্তরজ $\\frac{dy}{dx}$ হলো ওই বক্ররেখার যেকোনো বিন্দুতে অঙ্কিত স্পর্শকের ঢাল (Slope of the tangent line)।\n\n### ২. চরম মান (Max/Min) নির্ণয়ের শর্তাবলী\n1. **প্রথম অন্তরজ পরীক্ষা:** চরম বিন্দুতে স্পর্শক এক্স-অক্ষের সমান্তরাল হয়, অর্থাৎ $\\frac{dy}{dx} = 0$।\n2. **দ্বিতীয় অন্তরজ পরীক্ষা:**\n   - যদি $\\frac{d^2y}{dx^2} < 0$ হয় $\\to$ **গুরুমান (Maximum Value)**।\n   - যদি $\\frac{d^2y}{dx^2} > 0$ হয় $\\to$ **লঘুমান (Minimum Value)**।\n\n### বাস্তব প্রয়োগ:\nইঞ্জিনিয়ারিংয়ে সর্বনিম্ন খরচে সর্বোচ্চ ধারণক্ষমতার পাত্র তৈরি করতে অথবা এআই মডেলে লস মিনিমাইজ (Gradient Descent) করতে এই নীতি প্রত্যক্ষভাবে ব্যবহৃত হয়।"
            }
        ]
    })

    return conversations

def build_dataset_and_split(
    output_dir: str = "data",
    train_ratio: float = 0.90
) -> Tuple[str, str]:
    """Generates conversations in strict hierarchical progression and produces clean 90/10 split."""
    print("=" * 80)
    print("THSA-2B: FULL PROGRESSIVE CONVERSATIONAL DATASET GENERATOR")
    print("STRUCTURED: LEVEL 1 (MICRO GREETINGS) -> LEVEL 5 (DEEP DISCUSSIONS)")
    print("=" * 80)

    conversations = generate_all_conversations()
    total_convs = len(conversations)

    print(f"Total Unique Threads Generated: {total_convs}")
    for idx, c in enumerate(conversations, 1):
        turns = len(c["conversations"])
        est_tokens = sum(len(turn["value"].split()) * 1.3 for turn in c["conversations"])
        print(f"  [{idx:02d}] {c['id']:<52} | Turns: {turns:<2} | Est Tokens: ~{int(est_tokens):,}")

    # Stratified 90/10 split
    n_train = max(int(total_convs * train_ratio), 1)
    if n_train == total_convs and total_convs > 1:
        n_train = total_convs - 1

    train_data = conversations[:n_train]
    test_data  = conversations[n_train:]

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train_sharegpt.jsonl")
    test_file  = os.path.join(output_dir, "test_sharegpt.jsonl")

    # Write Train JSONL
    with open(train_file, "w", encoding="utf-8") as f_train:
        for item in train_data:
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write Test JSONL
    with open(test_file, "w", encoding="utf-8") as f_test:
        for item in test_data:
            f_test.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_tokens = sum(
        sum(len(turn["value"].split()) * 1.3 for turn in c["conversations"])
        for c in conversations
    )

    print("\nDataset Generation & Sorting Complete:")
    print(f"  Total Conversations:    {total_convs}")
    print(f"  Train Set (90%):         {len(train_data)} threads -> {train_file}")
    print(f"  Test Set  (10%):         {len(test_data)} threads -> {test_file}")
    print(f"  Total Estimated Tokens: ~{int(total_tokens):,} tokens")
    print("=" * 80)

    return train_file, test_file

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    build_dataset_and_split(out_dir)
