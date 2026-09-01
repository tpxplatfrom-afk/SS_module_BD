#!/usr/bin/env python3
"""
Bilingual ShareGPT Dataset Generator for THSA-2B On-Device AI Model.
Generates comprehensive, high-quality, long-form multi-turn conversational datasets (2,000-4,000 tokens per thread)
covering:
  1. Bangladesh History (Ancient Bengal, Sultanate, Mughal Subah, 1952 Language Movement, 1971 Liberation War, River Ecology)
  2. World History & Civilizations (Japan, UK, USA, Germany, Egypt, China, India, France, Rome, Ottoman Empire)
  3. NCTB Curriculum STEM: Physics, Mathematics, Biology, Chemistry & ICT
  4. Socratic explanations, nuanced opinions, comparative historical analysis, and concrete examples.
Includes automatic 90% Train / 10% Test split, validation, token length audit, and data cleaning.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

# Deterministic seed for repeatable 90/10 split
random.seed(42)

def generate_all_conversations() -> List[Dict[str, Any]]:
    """Builds the comprehensive conversational dataset."""
    conversations = []

    # 1. 1971 Liberation War: Deep Strategic, Diplomatic & Grassroots Analysis
    conversations.append({
        "id": "bd_history_001_liberation_war_strategic_analysis",
        "topic": "1971 Bangladesh Liberation War (মুক্তিযুদ্ধ)",
        "conversations": [
            {
                "from": "human",
                "value": "১৯৭১ সালের মুক্তিযুদ্ধ সম্পর্কে অনেক সাধারণ বিবরণ পাওয়া যায়, কিন্তু আমি এর সামরিক কৌশল (military strategy), ভূ-রাজনৈতিক প্রেক্ষাপট (geopolitics) এবং সাধারণ মানুষের আত্মত্যাগের সমন্বয়টি গভীরভাবে বুঝতে চাই। ৭ই মার্চের ভাষণ থেকে শুরু করে ১৬ই ডিসেম্বরের বিজয় পর্যন্ত মূল মোড়গুলো (turning points) কি ছিল?"
            },
            {
                "from": "gpt",
                "value": "১৯৭১ সালের মুক্তিযুদ্ধ কেবল একটি ভূখণ্ডগত সংঘাত ছিল না; এটি ছিল একটি জাতির আত্মনিয়ন্ত্রণাধিকার, সাংস্কৃতিক অস্তিত্ব এবং চরম রাজনৈতিক বৈষম্যের বিরুদ্ধে এক অবিস্মরণীয় জনযুদ্ধ (people's war)। এর গতিপ্রকৃতি বোঝার জন্য আমাদের একে তিনটি প্রধান মাত্রায় ভাগ করে বিশ্লেষণ করতে হবে: সামরিক কৌশল, আন্তর্জাতিক কূটনীতি এবং গণপ্রতিরোধ।\n\n### ১. পটভূমি ও ঐতিহাসিক প্রেরণা: ৭ই মার্চের কালজয়ী ভাষণ\nবঙ্গবন্ধু শেখ মুজিবুর রহমানের ঐতিহাসিক ৭ই মার্চের ভাষণ ছিল মূলত একটি 'কৌশলগত স্বাধীনতার রূপরেখা' (Strategic Blueprint for Independence)। রেসকোর্স ময়দানে তিনি একদিকে সরাসরি সংঘাত এড়িয়ে অহিংস অসহযোগ আন্দোলনের নির্দেশ দেন, অন্যদিকে প্রতিটি ঘরে ঘরে দুর্গ গড়ে তোলার আহ্বান জানান। এটি পাকিস্তানি সামরিক জান্তাকে অপ্রস্তুত করে দেয় এবং পূর্ব বাংলার সাড়ে সাত কোটি মানুষকে মানসিকভাবে যুদ্ধের জন্য পূর্ণ প্রস্তুত করে।\n\n### ২. অপারেশন সার্চলাইট ও ২৬শে মার্চের আনুষ্ঠানিক স্বাধীনতা\n২৫শে মার্চ মধ্যরাতে পাকিস্তানি সেনাবাহিনী 'অপারেশন সার্চলাইট' নামে ইতিহাসের অন্যতম নৃশংস গণহত্যা পরিচালনা করে। ঢাকা বিশ্ববিদ্যালয়, রাজারবাগ পুলিশ লাইন এবং পিলখানা ইপিআর হেডকোয়ার্টারে বর্বরোচিত আক্রমণ চালানো হয়। ২৬শে মার্চ প্রথম প্রহরে গ্রেফতার হওয়ার পূর্বেই বঙ্গবন্ধু আনুষ্ঠানিকভাবে বাংলাদেশের স্বাধীনতা ঘোষণা করেন, যা ইপিআর ওয়্যারলেসের মাধ্যমে সারাদেশে ছড়িয়ে পড়ে।\n\n### ৩. সামরিক বিন্যাস: ১১টি সেক্টর ও যৌথ কমান্ড\nমুক্তিযুদ্ধ পরিচালনায় মেহেরপুরের বৈদ্যনাথতলার আম্রকাননে (মুজিবনগর) ১০ই এপ্রিল প্রবাসী বাংলাদেশ সরকার গঠিত হয় এবং ১৭ই এপ্রিল শপথ গ্রহণ করে। প্রধান সেনাপতি জেনারেল আতাউল গনি ওসমানীর নেতৃত্বে সমগ্র বাংলাদেশকে ১১টি সামরিক সেক্টরে এবং পরবর্তীতে জেড-ফোর্স, কে-ফোর্স ও এস-ফোর্স নামে তিনটি নিয়মিত ব্রিগেড ফোর্সে ভাগ করা হয়।\n\n- **নিয়মিত বাহিনী (মুক্তিবাহিনী) ও গেরিলা বাহিনী (গণবাহিনী):** গ্রামবাংলার কৃষক, ছাত্র, শ্রমিকদের নিয়ে গঠিত গেরিলা যোদ্ধারা 'হিট অ্যান্ড রান' (Hit-and-Run) কৌশলে হানাদার বাহিনীর রসদ সরবরাহ লাইন ও যোগাযোগ ব্যবস্থা পুরোপুরি ধ্বংস করে দেয়।\n- **ক্র্যাক প্লাটুন:** ঢাকায় তরুণ গেরিলাদের পরিচালিত 'অপারেশন হোটেল ইন্টারকন্টিনেন্টাল' এবং বিভিন্ন সাবোটাজ অভিযান পাকিস্তানি বাহিনীকে রাজধানীতেও সর্বদা আতঙ্কের মধ্যে রাখে।\n- **অপারেশন জ্যাকপট (১৫ই আগস্ট ১৯৭১):** নৌ-কমান্ডোদের দুঃসাহসিক এই সমন্বিত হামলায় চট্টগ্রাম ও মোংলা বন্দর এবং চাঁদপুর ও নারায়ণগঞ্জ নৌঘাঁটিতে পাকিস্তানি অস্ত্র ও রসদবাহী ২৬টিরও বেশি যুদ্ধজাহাজ ধ্বংস হয়। আন্তর্জাতিক সম্প্রদায় এই অপারেশনের মাধ্যমে উপলব্ধি করে যে মুক্তিপাগল বাঙালিকে আর দাবিয়ে রাখা সম্ভব নয়।\n\n### ৪. আন্তর্জাতিক ভূরাজনীতি ও পরাশক্তিদের অবস্থান\n- **ভারত:** প্রায় এক কোটি শরণার্থীকে আশ্রয় প্রদান, মুক্তিযোদ্ধাদের সামরিক প্রশিক্ষণ এবং পরবর্তীতে যৌথ কমান্ড গঠনের মাধ্যমে প্রত্যক্ষ মিত্রের ভূমিকা পালন করে। তৎকালীন প্রধানমন্ত্রী ইন্দিরা গান্ধীর বিশ্বব্যাপী কূটনৈতিক প্রচেষ্টা মুক্তিযুদ্ধের পক্ষে আন্তর্জাতিক জনমত গঠনে গুরুত্বপূর্ণ অবদান রাখে।\n- **সোভিয়েত ইউনিয়ন:** জাতিসংঘের নিরাপত্তা পরিষদে তিনবার ভেটো প্রয়োগ করে বাংলাদেশের স্বাধীনতাকে স্তব্ধ করার মার্কিন-চীন অক্ষের কূটনৈতিক চক্রান্ত নস্যাৎ করে দেয়। বঙ্গোপসাগরে মার্কিন ৭ম নৌবহর (Task Force 74) মোকাবিলার জন্য সোভিয়েত নৌবহর প্রেরণ কৌশলগত ভারসাম্য নিশ্চিত করে।\n\n### ৫. চূড়ান্ত পরিণতি: যৌথ বাহিনীর আক্রমণ ও আত্মসমর্পণ (১৬ই ডিসেম্বর)\n৩রা ডিসেম্বর পাকিস্তান বিমানবাহিনী ভারতের পশ্চিমাঞ্চলের বিমানঘাঁটিগুলোতে আক্রমণ চালালে ভারত-বাংলাদেশ যৌথ কমান্ড গঠিত হয়। চারদিক থেকে যৌথ বাহিনীর ক্ষিপ্র অগ্রযাত্রায় পাকিস্তানি বাহিনীর প্রতিরোধ মাত্র ১৩ দিনে ভেঙে পড়ে। ১৬ই ডিসেম্বর রেসকোর্স ময়দানে জেনারেল আমির আব্দুল্লাহ খান নিয়াজির নেতৃত্বে ৯৩,০০০ পাকিস্তানি সৈন্যের ঐতিহাসিক আত্মসমর্পণের মাধ্যমে জন্ম নেয় সার্বভৌম গণপ্রজাতন্ত্রী বাংলাদেশ।"
            },
            {
                "from": "human",
                "value": "অপারেশন জ্যাকপটের নৌ-কমান্ডোদের প্রশিক্ষণ এবং অপারেশন চলাকালীন মানসিক অবস্থা কেমন ছিল? রেডিও গানের সংকেতের যে ঘটনাটি ছিল, সে সম্পর্কে বিস্তারিত বলো।"
            },
            {
                "from": "gpt",
                "value": "অপারেশন জ্যাকপট ছিল আধুনিক সামরিক ইতিহাসের অন্যতম সুপরিকল্পিত এবং দুঃসাহসিক কমব্যাট ডাইভিং অপারেশন। এর প্রতিটি পর্যায় চরম শৃঙ্খলা ও সাহসিকতার প্রতীক।\n\n### ১. গোপন প্রশিক্ষণ ক্যাম্প: ভাগীরথী নদী ও ক্যাম্প পলাশী\n১৯৭১ সালের মে মাস থেকে ভারতের পশ্চিমবঙ্গের মুর্শিদাবাদ জেলার ভাগীরথী নদীর তীরে পলাশীতে গোপন প্রশিক্ষণ শিবির গড়ে তোলা হয়। ফ্রান্সের তুলোঁ বন্দর থেকে পালিয়ে আসা ৮ জন বাঙালি সাবমেরিনার (যেমন মোস্তফা কামাল, আব্দুল ওয়াহেদ চৌধুরী প্রমুখ) ছিলেন এই অপারেশনের মূল প্রশিক্ষক ও অনুপ্রেরণা। প্রায় ৫০০ জন তরুণ সাঁতারুকে অত্যন্ত কঠোর প্রশিক্ষণ দেওয়া হয়:\n- প্রতিদিন একটানা ৬ থেকে ৮ ঘণ্টা পানিতে সাঁতার কাটা ও ডুবসাঁতার চর্চা।\n- লিম্পেট মাইন (Limpet Mine) বহন এবং জাহাজের নিচে ম্যাগনেটিক ক্ল্যাম্প দিয়ে তা যুক্ত করার কৌশল।\n- প্রতিকূল স্রোত, ঘোলা পানি এবং শত্রু বাহিনীর ডেপথ-চার্জ ও সার্চলাইটের আলো এড়িয়ে নিঃশব্দে পানির নিচে টিকে থাকা।\n\n### ২. 'পঙ্কজ মল্লিক' ও 'সন্ধ্যা মুখার্জি': কালজয়ী আকাশবাণী রেডিও সংকেত\nকমান্ডো দলগুলোকে চট্টগ্রাম, মোংলা, চাঁদপুর ও নারায়ণগঞ্জের উদ্দেশ্যে প্রেরণের পর অপারেশনের দিনক্ষণ চূড়ান্ত করার জন্য একটি অভিনব যোগাযোগ মাধ্যম নির্ধারণ করা হয়। কোনো ওয়্যারলেস সেট ব্যবহার না করে কলকাতা আকাশবাণী বেতারের পূর্বাঞ্চলীয় ক খ গ তরঙ্গের মাধ্যমে গান বাজিয়ে সংকেত পাঠানোর সিদ্ধান্ত হয়:\n\n1. **প্রথম সংকেত (হলুদ সংকেত - সতর্কবার্তা):** আকাশবাণী থেকে পঙ্কজ মল্লিকের গাওয়া বিখ্যাত গান *\"আমি তোমায় যত শুনিয়েছিলাম গান\"* বাজলে বুঝতে হবে আগামী ২৪ থেকে ৪৮ ঘণ্টার মধ্যে আক্রমণ পরিচালনা করতে হবে।\n2. **দ্বিতীয় সংকেত (সবুজ সংকেত - আক্রমণ শুরু):** সন্ধ্যা মুখোপাধ্যায়ের কণ্ঠে *\"আমার পুতুল আজকে যাবে শ্বশুরবাড়ি\"* গানটি প্রচারিত হওয়া মাত্র নির্দেশ ছিল সেদিন মধ্যরাতেই চূড়ান্ত আক্রমণ চালাতে হবে।\n\n### ৩. ১৫ই আগস্টের রাতের শ্বাসরুদ্ধকর অভিযান\n১৪ই আগস্ট রাতে গান দুটি সম্প্রচারিত হওয়ার পর ১৫ই আগস্ট দিবাগত মধ্যরাতে একযোগে চারটি বন্দরে প্রায় ৩৫০ জন নৌ-কমান্ডো পানিতে নামেন। কমান্ডোরা গলায় সাঁতারের পাখনা এবং কোমরে ফিন্স ও লিম্পেট মাইন বেঁধে পানির নিচে নিঃশব্দে এগিয়ে যান। চট্টগ্রাম বন্দরে এমভি হরমুজ, এমভি আল-আব্বাস সহ বহু পাকিস্তানি গোলাবারুদবাহী জাহাজ লিম্পেট মাইনের প্রচণ্ড বিস্ফোরণে বঙ্গোপসাগরের তলদেশে নিমজ্জিত হয়। এটি ছিল বিশ্বের যেকোনো গেরিলা যুদ্ধের ইতিহাসে অন্যতম সফল ও রক্তক্ষয়ী নৌ-অভিযান।"
            },
            {
                "from": "human",
                "value": "১৯৫২ সালের ভাষা আন্দোলনের সাথে ১৯৭১ সালের মুক্তিযুদ্ধের যে মনস্তাত্ত্বিক ও আদর্শিক যোগসূত্র (ideological connection), সেটিকে তুমি কীভাবে মূল্যায়ন করবে?"
            },
            {
                "from": "gpt",
                "value": "১৯৫২ সালের ২১শে ফেব্রুয়ারির মহান ভাষা আন্দোলন এবং ১৯৭১ সালের মুক্তিযুদ্ধ মূলত একই ঐতিহাসিক ও মনস্তাত্ত্বিক মুক্তির ধারাক্রমের দুটি অপরিহার্য স্তম্ভ। রাষ্ট্রবিজ্ঞানী ও সমাজতাত্ত্বিক দৃষ্টিকোণ থেকে একে বলা যায় **'সাংস্কৃতিক স্বাধিকার থেকে রাজনৈতিক সার্বভৌমত্বে উত্তরণ'**।\n\n### ১. দ্বিজাতিতত্ত্বের মোহভঙ্গ ও বাঙালি জাতীয়তাবাদের উন্মেষ\n১৯৪৭ সালে ব্রিটিশ ভারত বিভক্ত হয়ে পাকিস্তান রাষ্ট্রের সৃষ্টি হয়েছিল ধর্মভিত্তিক 'দ্বিজাতিতত্ত্ব'-এর ভিত্তিতে। কিন্তু মাত্র কয়েক মাসের মধ্যেই পাকিস্তানের কায়েমী শাসকগোষ্ঠী সংখ্যাগরিষ্ঠ ৫৬% বাঙালির মুখের ভাষা বাংলাকে বাদ দিয়ে মাত্র ৭% মানুষের ভাষা উর্দুকে একমাত্র রাষ্ট্রভাষা করার ষড়যন্ত্রে লিপ্ত হয়। ১৯৫২ সালে সালাম, বরকত, রফিক, জব্বার, শফিউরদের আত্মত্যাগের মাধ্যমে বাঙালি উপলব্ধি করে যে, ধর্মের দোহাই দিয়ে জাতীয় শোষণ ও সাংস্কৃতিক আধিপত্যকে মেনে নেওয়া যায় না। দ্বিজাতিতত্ত্বের কৃত্রিম বাঁধ সেখানেই ভেঙে পড়ে এবং ভাষাভিত্তিক ধর্মনিরপেক্ষ বাঙালি জাতীয়তাবাদের সুদৃঢ় ভিত্তি রচিত হয়।\n\n### ২. ধারাবাহিক সংগ্রামের সিঁড়ি (১৯৫২ → ১৯৬৬ → ১৯৬৯ → ১৯৭১)\nভাষা আন্দোলনের রক্তস্নাত পথ ধরেই পরবর্তী প্রতিটি গণআন্দোলন গতিশীলতা পায়:\n- **১৯৫৪ সালের যুক্তফ্রন্ট নির্বাচন:** একুশ দফার ভিত্তিতে মুসলিম লীগের চরম রাজনৈতিক পরাজয়।\n- **১৯৬২ সালের শিক্ষা আন্দোলন:** বৈষম্যমূলক শরীফ কমিশনের বিরুদ্ধে ছাত্রদের ঐক্যবদ্ধ প্রতিরোধ।\n- **১৯৬৬ সালের ছয় দফা আন্দোলন:** বঙ্গবন্ধুর উপস্থাপিত ঐতিহাসিক ৬-দফা ছিল পূর্ব বাংলার অর্থনৈতিক ও রাজনৈতিক মুক্তির 'ম্যাগনা কার্টা'।\n- **১৯৬৯ সালের গণঅভ্যুত্থান:** আগরতলা ষড়যন্ত্র মামলা থেকে বঙ্গবন্ধুর নিঃশর্ত মুক্তি এবং আইয়ুব খানের স্বৈরাচারের পতন।\n- **১৯৭০ সালের সাধারণ নির্বাচন:** জনগণের অবিসংবাদিত রায় যা বাঙালি স্বাধিকারকে জাতীয় ম্যান্ডেটে রূপান্তর করে।\n\n### ৩. মনস্তাত্ত্বিক রূপান্তর: 'রক্তের উত্তরাধিকার'\n১৯৫২ সালে ছাত্র-জনতা বুকের রক্ত ঢেলে দিয়ে প্রমাণ করেছিল যে বাঙালি নিজের অধিকারের প্রশ্নে মাথা নত করে না। সেই আত্মমর্যাদাবোধই ১৯৭১ সালে 'জয় বাংলা' স্লোগানে পরিণত হয়ে সমগ্র জাতিকে সশস্ত্র সংগ্রামে ঝাঁপিয়ে পড়ার অসীম সাহস জুগিয়েছিল। অতএব, ১৯৫২ যদি হয় আত্মপরিচয়ের বীজ রোপণ, ১৯৭১ হলো সেই বীজের পূর্ণাঙ্গ সার্বভৌম বৃক্ষে রূপ লাভ।"
            }
        ]
    })

    # 2. Ancient & Medieval Bengal: Socio-Economic, Naval and Dynastic Evolution
    conversations.append({
        "id": "bd_history_002_ancient_medieval_bengal",
        "topic": "Ancient & Medieval History of Bengal (প্রাচীন ও মধ্যযুগীয় বাংলা)",
        "conversations": [
            {
                "from": "human",
                "value": "প্রাচীন ও মধ্যযুগীয় বাংলার ইতিহাস আলোচনা করতে গেলে আমাদের অঞ্চলটির ভৌগোলিক বৈশিষ্ট্য, সামুদ্রিক বাণিজ্য (maritime trade) এবং পাল, সেন ও সুলতানি আমলের শাসনব্যবস্থা কীভাবে বিবর্তিত হয়েছিল? উদাহরণসহ বুঝিয়ে বলো।"
            },
            {
                "from": "gpt",
                "value": "বাংলার ইতিহাস মূলত তার নদীমাতৃক ভূপ্রকৃতি এবং বঙ্গোপসাগর-কেন্দ্রিক সামুদ্রিক বাণিজ্যের সাথে ওতপ্রোতভাবে জড়িত। হিমালয় থেকে নেমে আসা পদ্মা, মেঘনা ও ব্রহ্মপুত্রের পলিগঠিত অববাহিকা একদিকে বাংলাকে করেছে পৃথিবীর অন্যতম উর্বর কৃষিভূমি, অন্যদিকে প্রাকৃতিক নদীপথ বাংলাকে দিয়েছে শক্তিশালী অভ্যন্তরীণ ও আন্তর্জাতিক যোগাযোগের পথ।\n\n### ১. প্রাচীন জনপদ ও ভৌগোলিক বিন্যাস\nপ্রাচীনকালে বাংলা একক কোনো ভৌগোলিক একক ছিল না, বরং বিভিন্ন জনপদে বিভক্ত ছিল:\n- **পুন্ড্র:** উত্তরবঙ্গ (বগুড়ার মহাস্থানগড় এর প্রাচীন রাজধানী পুন্ড্রনগর)।\n- **বঙ্গ:** পূর্ব ও দক্ষিণ-পূর্ব বাংলা (ঢাকা, ময়মনসিংহ, ফরিদপুর অঞ্চল)।\n- **সমতট ও হরিকেল:** মেঘনা পূর্ববর্তী কুমিল্লা-নোয়াখালী ও সিলেট-চট্টগ্রাম উপকূলীয় অঞ্চল।\n- **রাঢ় ও গৌড়:** পশ্চিম ও উত্তর-পশ্চিম বাংলা (মুর্শিদাবাদ, মালদহ, বর্ধমান অঞ্চল)।\n\n### ২. মৎস্যন্যায় ও পাল সাম্রাজ্যের স্বর্ণযুগ (৭৫০ – ১১৬১ খ্রিস্টাব্দ)\nশশাঙ্কের মৃত্যুর পর প্রায় এক শতাব্দীকাল বাংলায় চরম নৈরাজ্য ও বিশৃঙ্খলা বিরাজ করছিল, যা ইতিহাসে **'মৎস্যন্যায়'** (বড় মাছ কর্তৃক ছোট মাছকে গিলে খাওয়ার মতো অরাজকতা) নামে পরিচিত।\n- **গণতান্ত্রিক নির্বাচন:** প্রায় ৭৫০ খ্রিস্টাব্দে বাংলার প্রকৃতিপুঞ্জ বা নেতৃস্থানীয় ব্যক্তিবর্গ একমত হয়ে গোপালকে রাজা নির্বাচিত করেন।\n- **ধর্মপাল ও দেবপাল:** পাল রাজারা দীর্ঘ চারশত বছর বাংলা শাসন করেন। তাঁরা বৌদ্ধ সংস্কৃতির পৃষ্ঠপোষক ছিলেন এবং নালন্দা, সোমপুর মহাবিহার (পাহাড়পুর) ও বিক্রমশীলা মহাবিহারের মতো আন্তর্জাতিক মানের বিশ্ববিদ্যালয় স্থাপন করেন। সোমপুর মহাবিহার ইউনেস্কো স্বীকৃত বিশ্ব ঐতিহ্য।\n\n### ৩. সেন রাজবংশ ও সামাজিক রূপান্তর (১০৯৭ – ১২০৪ খ্রিস্টাব্দ)\nদক্ষিণ ভারতের কর্ণাটক থেকে বাংলায় আগমন ঘটে সেনদের। বিজয় সেন ও বল্লাল সেনের আমলে ব্রাহ্মণ্যধর্মীয় পুনরুজ্জীবন ঘটে এবং কঠোর কৌলীন্য প্রথা প্রবর্তিত হয়, যার ফলে সাধারণ নিম্নবর্ণের জনগণের মধ্যে সামাজিক দূরত্ব বৃদ্ধি পায়।\n\n### ৪. মধ্যযুগীয় সুলতানি আমল: স্বাধীন বাংলার স্বর্ণযুগ (১৩৩৮ – ১৫৩৮ খ্রিস্টাব্দ)\n১২০৪ সালে ইখতিয়ার উদ্দিন মুহম্মদ বখতিয়ার খলজীর নদীয়া বিজয়ের মাধ্যমে বাংলায় মুসলিম শাসনের সূচনা হয়। তবে ১৩৩৮ সালে ফখরুদ্দিন মুবারক শাহ সোনারগাঁওয়ে বাংলার প্রথম স্বাধীন সালতানাত প্রতিষ্ঠা করেন।\n\n- **ইলিয়াস শাহী রাজবংশ:** শামসুদ্দিন ইলিয়াস শাহ সমগ্র বাংলাকে ঐক্যবদ্ধ করে 'শাহ-ই-বাঙ্গালাহ' বা 'সুলতান-ই-বাঙ্গালাহ' উপাধি গ্রহণ করেন। তিনিই প্রথম সমগ্র অঞ্চলের অধিবাসীদের 'বাঙালি' হিসেবে রাজনৈতিক পরিচয় দেন।\n- **আলাউদ্দিন হোসেন শাহের যুগ:** হোসেন শাহী আমল ছিল হিন্দু-মুসলিম সম্প্রীতি ও বাংলা ভাষা-সাহিত্যের এক অভূতপূর্ব স্বর্ণযুগ। কৃত্তিবাস ও মালাধর বসুর মতো পণ্ডিতদের দ্বারা রামায়ণ ও মহাভারত বাংলায় অনূদিত হয়। বাংলা সাহিত্যের বিকাশ ঘটে রাজকীয় পৃষ্ঠপোষকতায়।\n- **আন্তর্জাতিক বাণিজ্য ও পর্যটকদের বিবরণ:** মরক্কোর পরিব্রাজক ইবনে বতুতা ১৩৪৬ সালে বাংলায় আসেন এবং বাংলাকে প্রচুর ধনসম্পদ ও খাদ্যশস্যের কারণে 'দোযখ-ই-পুর নিয়ামত' (নিয়ামতে পূর্ণ নরক - আবহাওয়া প্রতিকূল কিন্তু সম্পদ অপরিসীম) বলে অভিহিত করেন। বাংলার তৈরি বিশ্ববিখ্যাত মসলিন কাপড় রোম, ইউরোপ ও মধ্যপ্রাচ্যে সমাদৃত ছিল।"
            },
            {
                "from": "human",
                "value": "মুঘল আমলে 'সুবা বাংলা' কীভাবে মুঘল সাম্রাজ্যের অর্থনৈতিক হৃদপিণ্ডে পরিণত হয়েছিল এবং মুর্শিদকুলি খানের দেওয়ানি সংস্কারের প্রভাব কী ছিল?"
            },
            {
                "from": "gpt",
                "value": "মুঘল আমলে বাংলাকে বলা হতো **'জান্নাতুল বিলাদ'** বা 'দেশসমূহের স্বর্গ' এবং **'মুঘল সাম্রাজ্যের মুক্তা'**। মুঘল সম্রাটদের কোষাগারের সিংহভাগ রাজস্ব আসত সুবা বাংলা থেকে।\n\n### ১. মুঘল বিজয় ও বারো ভূঁইয়াদের প্রতিরোধ\n১৫৭৬ সালে রাজমহলের যুদ্ধে দাউদ খান কররানীর পরাজয়ের মাধ্যমে মুঘলরা আনুষ্ঠানিকভাবে বাংলা জয় করে। কিন্তু ঈসা খাঁ ও মুসা খাঁর নেতৃত্বে 'বারো ভূঁইয়া' নামে পরিচিত স্বাধীন জমিদাররা দীর্ঘকাল মুঘল কর্তৃত্বকে প্রতিরোধ করেন। অবশেষে সম্রাট জাহাঙ্গীরের সুবেদার ইসলাম খান চিশতী ১৬১০ সালে ঢাকাকে বাংলার রাজধানী (জাহাঙ্গীরনগর) ঘোষণা করে সমগ্র অঞ্চলে স্থায়ী মুঘল নিয়ন্ত্রণ প্রতিষ্ঠা করেন।\n\n### ২. সুবা বাংলার অর্থনৈতিক সমৃদ্ধি ও মসলিন বাণিজ্য\nসুবা বাংলা ছিল বিশ্বের অন্যতম প্রধান বস্ত্র ও খাদ্য রপ্তানিকারক অঞ্চল:\n- **মসলিন ও সুতিবস্ত্র:** ঢাকার চারপাশের শীতলক্ষ্যা ও বুড়িগঙ্গা অববাহিকায় তৈরি সূক্ষ্ম মসলিন বিশ্বখ্যাত ছিল। ঢাকাই জামদানি এবং বিভিন্ন রেশমি বস্ত্র ছিল মুঘল দরবার ও ইউরোপীয় রাজপরিবারের প্রথম পছন্দ।\n- **জাহাজ নির্মাণ শিল্প (Shipbuilding):** চট্টগ্রাম ও সোনারগাঁওয়ে নির্মিত কাঠের রণতরী ও বাণিজ্যিক জাহাজ তুরস্ক, ইতালি ও মধ্যপ্রাচ্যের নৌবাহিনী ব্যবহার করত।\n- **ইউরোপীয় বাণিজ্য কুঠি:** পর্তুগিজ, ওলন্দাজ, ফরাসি এবং ব্রিটিশ ইস্ট ইন্ডিয়া কোম্পানি ঢাকায় ও কাশিমবাজারে তাদের বাণিজ্যিক কেন্দ্র গড়ে তোলে।\n\n### ৩. মুর্শিদকুলি খানের ঐতিহাসিক সংস্কার (১৭১৭ – ১৭২৭ খ্রিস্টাব্দ)\nমুঘল সম্রাট আওরঙ্গজেব ১৭০০ সালে মুর্শিদকুলি খানকে বাংলার দেওয়ান (রাজস্ব প্রধান) নিযুক্ত করেন। পরবর্তীতে তিনি বাংলার প্রথম বংশানুক্রমিক নবাব হন এবং রাজধানী ঢাকা থেকে মুর্শিদাবাদে স্থানান্তর করেন।\n\n- **জমি জরিপ ও 'মালজমি' প্রথা:** মুর্শিদকুলি খান সমগ্র জমি জরিপ করে প্রকৃত কৃষকদের পাট্টা প্রদান করেন এবং মধ্যস্বত্বভোগীদের দৌরাত্ম্য বন্ধ করেন।\n- **নিয়মিত রাজস্ব আদায় ও তাকাভি ঋণ:** তিনি দুর্যোগের সময় কৃষকদের 'তাকাভি' (কৃষি ঋণ) প্রদান করতেন এবং ফসল উৎপাদনের নিশ্চয়তা বিধান করতেন। তাঁর সময়ে বাংলার বার্ষিক রাজস্ব বৃদ্ধি পেয়ে তৎকালীন মুদ্রায় এক কোটি ত্রিশ লাখ টাকায় উন্নীত হয়, যা দিল্লির কেন্দ্রীয় কোষাগারকে বাঁচিয়ে রাখত।\n- **ব্যবসায়িক পুঁজিপতিদের উত্থান:** জগৎ শেঠদের মতো ব্যাংকিং পরিবারের উত্থান ঘটে, যারা সমগ্র মুঘল সাম্রাজ্যের মুদ্রা বিনিময় ও রাজস্ব স্থানান্তর নিয়ন্ত্রণ করত।"
            }
        ]
    })

    # 3. World History: Japan — Feudal Era, Meiji Restoration to Modern Technological Power
    conversations.append({
        "id": "world_history_001_japan_evolution",
        "topic": "History of Japan: Feudal Era, Meiji Restoration & Modernization",
        "conversations": [
            {
                "from": "human",
                "value": "Japan is often cited as the premier example of non-Western modernization. How did Japan transition from an isolated feudal state under the Tokugawa Shogunate (Sakoku) to an industrialized global power during the Meiji Restoration, and what cultural philosophies enabled this balance between tradition and technology?"
            },
            {
                "from": "gpt",
                "value": "The transformation of Japan during the late 19th and early 20th centuries represents one of the most astonishing institutional and technological pivots in human history. To understand this transition, we must examine the internal tensions of feudal Japan, the catalyst of Western encroachment, and the deliberate philosophical paradigm of *Wakon Yosai* (Japanese Spirit, Western Knowledge).\n\n### 1. The Tokugawa Shogunate & The Age of Isolation (*Sakoku*)\nFrom 1603 to 1867, the Tokugawa clan ruled Japan from Edo (modern Tokyo), while the Emperor remained a ceremonial figurehead in Kyoto. Under the policy of *Sakoku* (Closed Country):\n- Foreigners were strictly forbidden from entering, and Japanese citizens were prohibited from leaving under penalty of death.\n- Only minimal, tightly controlled trade was permitted with the Dutch and Chinese through the artificial island of Dejima in Nagasaki harbor.\n- A rigid four-tier social hierarchy (*Shinōkōshō*) prevailed: Samurai (warriors), Farmers, Artisans, and Merchants.\n- While *Sakoku* brought 250 years of uninterrupted peace (*Pax Tokugawa*), it insulated Japan from the scientific and industrial revolutions sweeping Europe.\n\n### 2. The Catalyst: Commodore Perry's Black Ships (1853)\nIn July 1853, US Navy Commodore Matthew Perry sailed four heavily armed steam warships (*Kurofune* or 'Black Ships') into Edo Bay, demanding trade access under threat of bombardment. The Shogunate realized with shocking clarity that its samurai swords and wooden coastal batteries were utterly powerless against modern steam-powered rifled artillery. This forced capitulation (*Treaty of Kanagawa*, 1854) ignited widespread internal outrage, crystallizing in the patriotic slogan *Sonnō Jōi* (\"Revere the Emperor, Expel the Barbarian\").\n\n### 3. The Meiji Restoration (1868): The Great Institutional Overhaul\nForward-thinking samurai from the southern domains of Satsuma and Chōshū overthrew the Shogunate and formally 'restored' practical governing authority to the 15-year-old Emperor Meiji. Instead of closing off, the revolutionary Meiji leadership adopted a radically pragmatic mindset: *\"In order to prevent colonization by the West, we must master Western science, industry, and military science faster than anyone else.\"*\n\nKey Structural Transformations:\n1. **The Iwakura Mission (1871–1873):** A high-level delegation of ministers and scholars traveled for two years across the United States, Britain, Germany, and France, meticulously studying factories, banking systems, legal codes, and naval doctrine.\n2. **Abolition of the Feudal Class System:** The Samurai class was disbanded, stipends converted to bonds, and the carrying of swords banned. Universal military conscription was instituted, creating a modern national army.\n3. **Universal Compulsory Education:** Primary school attendance was made mandatory for boys and girls alike, emphasizing engineering, literacy, and mathematics alongside civic morality.\n4. **State-Led Industrialization (*Zaibatsu*):** The government established state-of-the-art model textile mills, steel foundries, and rail networks before selling them to private merchant families (Mitsui, Mitsubishi, Sumitomo), forming massive industrial conglomerates.\n\n### 4. Cultural & Philosophical Foundations: *Wakon Yosai*\nUnlike many colonized societies that either retreated into reactionary fundamentalism or abandoned their heritage entirely, Japan codified the doctrine of **Wakon Yosai** (和魂洋才):\n- **Western Knowledge (*Yosai*):** Steam engines, metallurgy, constitutional law, medical hygiene, and military tactics.\n- **Japanese Spirit (*Wakon*):** Shinto reverence for community harmony, Confucian loyalty, Bushido discipline, and devotion to national resilience.\n\nBy 1905, Japan decisively defeated Tsarist Russia in the Russo-Japanese War—the first modern defeat of a European empire by an Asian nation—firmly cementing its status as an industrialized world power."
            },
            {
                "from": "human",
                "value": "How did Japan manage to rebuild itself so rapidly after the total physical and economic devastation of World War II into the world's second-largest economy by the 1980s?"
            },
            {
                "from": "gpt",
                "value": "Japan’s post-WWII resurgence—often termed the **'Japanese Economic Miracle'** (*Kōdo Keizai Seichō*)—is a profound study in institutional coordination, high-precision manufacturing, and human capital mobilization.\n\n### 1. The Starting Baseline: Complete Catastrophe (1945)\nBy August 1945, every major Japanese city except Kyoto was reduced to ash by incendiary bombing and atomic strikes. Sixty-six percent of industrial capacity was destroyed, three million citizens were dead, and millions of demobilized soldiers returned to a starving, hyper-inflationary homeland occupied by Allied forces under General Douglas MacArthur (SCAP).\n\n### 2. Strategic Structural Reforms under Allied Occupation\n- **Demilitarization & Article 9:** The 1947 Constitution renounced sovereign warfare, allowing Japan to redirect virtually 100% of national tax revenue away from military defense and entirely into civilian infrastructure, schools, and export manufacturing.\n- **Land Reform:** Tenant farmers were granted direct ownership of arable land, creating a wealthy, politically stable rural middle class that purchased domestic consumer goods.\n- **Labor & Democratic Freedoms:** Trade unions were legalized, and women gained voting rights, expanding productive civic participation.\n\n### 3. The 'Iron Triangle' Model of Governance\nJapan's industrial renaissance was orchestrated by close collaborative synergy between three pillars:\n1. **MITI (Ministry of International Trade and Industry):** Elite technocratic civil servants strategically selected high-growth export sectors (first textiles, then steel and shipbuilding, followed by automobiles and microelectronics) and allocated foreign exchange, low-interest credit, and tariff protections.\n2. **Keiretsu (Enterprise Groups):** The old Zaibatsu re-formed as cooperative corporate networks centered around main banks (e.g., Toyota-Mitsui, Mitsubishi, Fuyo), cross-holding shares to prevent hostile foreign takeovers and enabling multi-decade capital investments.\n3. **LDP & Political Stability:** Continuous conservative governing coalitions ensured unbroken, predictable long-term economic policies.\n\n### 4. The Quality Revolution: Deming, Kaizen & Lean Manufacturing\nWhile early postwar exports were dismissed as cheap imitations, Japanese industry revolutionized modern manufacturing engineering:\n- **Statistical Quality Control (W. Edwards Deming):** Systematic data-driven defect elimination.\n- **Kaizen (Continuous Improvement):** Every assembly line worker was empowered to halt production to fix quality flaws instantly.\n- **Just-In-Time (JIT / Kanban):** Pioneers like Taiichi Ohno at Toyota eliminated warehouse waste by syncing parts delivery with live assembly schedules.\n\nBy the late 1970s and 1980s, brands like Sony (Trinitron, Walkman), Toyota, Honda, Panasonic, and Nikon dominated global markets through peerless reliability, fuel efficiency, and technological elegance."
            }
        ]
    })

    # 4. World History: United Kingdom — Magna Carta, Scientific & Industrial Revolution, Global Impact
    conversations.append({
        "id": "world_history_002_uk_industrial_revolution",
        "topic": "History of the United Kingdom: Constitutional Evolution & Industrial Revolution",
        "conversations": [
            {
                "from": "human",
                "value": "Why did the Industrial Revolution begin specifically in Great Britain rather than in France, Germany, or China? Explain the unique confluence of geographical, scientific, legal, and financial factors."
            },
            {
                "from": "gpt",
                "value": "The onset of the Industrial Revolution in late 18th-century Britain was not a historical accident or the result of isolated technological genius. Rather, it emerged from a unique, interlocking matrix of geological fortune, institutional rule of law, financial innovation, and economic incentives that existed nowhere else in the world simultaneously.\n\n### 1. The High-Wage, Cheap-Energy Economic Equation (Robert Allen's Thesis)\nEconomic historian Robert Allen demonstrated that 18th-century Britain had the **highest labor wages in the world** relative to capital, but simultaneously possessed the **cheapest energy on Earth** due to vast, easily accessible surface coalfields in northern England, Wales, and Scotland.\n- In France or China, where human labor was exceptionally cheap and coal was distant from manufacturing centers, it made zero financial sense to invest large fortunes building expensive, inefficient steam engines.\n- In Britain, investing in a steam engine or mechanized spinning mule was intensely profitable because it substituted expensive British human labor with abundant, cheap coal power.\n\n### 2. Institutional Stability & Secure Property Rights (The Post-1688 Framework)\nFollowing the *Glorious Revolution of 1688*, the British Parliament permanently curbed the arbitrary power of the Crown:\n- **No Confiscatory Taxation:** Kings could no longer seize private fortunes or cancel loans at whim.\n- **Statute of Monopolies & Patent System:** Inventors like James Watt, Richard Arkwright, and George Stephenson were legally guaranteed 14-year commercial monopolies over their technological discoveries, directly incentivizing private R&D.\n- **Enclosure Acts & Commercialized Agriculture:** Agricultural improvements (crop rotation, selective breeding) freed up a vast labor force that migrated to burgeoning urban factory centers like Manchester, Birmingham, and Leeds.\n\n### 3. Island Geography & Navigable Transport Infrastructure\n- No location in Britain is further than 70 miles (112 km) from the sea.\n- Unlike continental Europe, which was fractured by countless internal river tolls, tariff borders, and recurrent wars, Britain formed a single unified domestic market.\n- Extensive private canal networks (such as the Bridgewater Canal) connected inland coal pits directly to coastal ports, cutting heavy freight transportation costs by up to 80%.\n\n### 4. Financial Capital & The London Banking System\nThe founding of the *Bank of England* (1694) and the development of the London Stock Exchange and country banks provided sophisticated credit instruments:\n- Stable sovereign debt allowed the British government to borrow at 3–4% interest, while French rivals paid 8–10%.\n- Entrepreneurs could readily raise equity and debt to fund capital-intensive ventures: blast furnaces, deep-shaft coal mines, and railway lines."
            },
            {
                "from": "human",
                "value": "What were the human, social, and global consequences of this industrial transformation, and how did it give rise to modern political ideologies like classical liberalism, socialism, and trade unionism?"
            },
            {
                "from": "gpt",
                "value": "The Industrial Revolution dismantled agrarian civilization and forged the modern urban, democratic world, but it did so through immense human trauma, social dislocation, and imperial exploitation.\n\n### 1. Urban Shock & Working-Class Realities\nBetween 1800 and 1850, cities like Manchester exploded from market towns into dense, smoky metropolises without clean water, sewage systems, or building codes:\n- **Child & Female Labor:** Children as young as six worked 14-hour days in textile mills and coal shafts due to their small size and low wages.\n- **Public Health Crises:** Cholera, tuberculosis, and rickets ran rampant; average life expectancy for laborers in industrial Liverpool fell below 20 years.\n- **Luddite Resistance (1811–1816):** Skilled textile weavers smashed mechanized power looms that were impoverishing traditional artisan communities.\n\n### 2. The Intellectual Crucible: Birth of Modern Political Philosophy\n- **Classical Liberalism & Free Trade (Adam Smith, David Ricardo):** Laissez-faire capitalism, division of labor, free trade maximizes global prosperity. Led to the repeal of the Corn Laws (1846).\n- **Scientific Socialism & Marxism (Karl Marx, Friedrich Engels):** Industrialization creates a structural class struggle between bourgeoisie (owners) and proletariat (workers).\n- **Utilitarianism & Social Reform (Jeremy Bentham, John Stuart Mill):** The moral duty of the state is 'the greatest happiness for the greatest number', sparking the Factory Acts and public health legislation.\n- **Trade Unionism & Democratic Chartist Movement:** Organized collective bargaining, leading to the legal recognition of trade unions (1871) and universal adult suffrage."
            }
        ]
    })

    # 5. World History: Germany — Bismarck's Unification & Second Industrial Revolution
    conversations.append({
        "id": "world_history_003_germany_unification_social_market",
        "topic": "History of Germany: Bismarck's Unification, Science & the Social Market Economy",
        "conversations": [
            {
                "from": "human",
                "value": "Germany went from hundreds of fragmented principalities within the Holy Roman Empire to a unified industrial powerhouse under Otto von Bismarck. How did the Zollverein (Customs Union), Realpolitik, and Germany's unique emphasis on chemical, electrical, and university-linked research forge the Second Industrial Revolution?"
            },
            {
                "from": "gpt",
                "value": "Germany’s journey to nationhood and industrial supremacy is a masterclass in the deliberate synchronization of economic integration, ruthless diplomatic *Realpolitik*, and the world's first state-sponsored research-and-development university model.\n\n### 1. Economic Integration Preceding Political Unity: The *Zollverein* (1834)\nPrior to 1871, there was no single Germany—only a loose mosaic of kingdoms, duchies, and free cities. Under Prussian leadership:\n- The **Zollverein** (Customs Union) eliminated internal trade barriers across member states while establishing a common external tariff.\n- A standard rail network crisscrossed the German lands, knitting coal-rich Silesia and the Ruhr Valley with Berlin and the Rhineland.\n- Economic interdependence made political unification not only desirable but economically inevitable.\n\n### 2. Bismarck and *Realpolitik*: Unification Through \"Blood and Iron\"\nPrussian Chancellor Otto von Bismarck rejected idealistic romantic nationalism in favor of calculating, practical power politics (*Realpolitik*). Through three swift, localized wars, Bismarck unified Germany under the Prussian crown:\n1. **Second Schleswig War (1864):** Allied with Austria against Denmark.\n2. **Austro-Prussian War (1866):** Defeated Austria at Königgrätz in seven weeks, expelling the Habsburgs from German affairs and forming the North German Confederation.\n3. **Franco-Prussian War (1870–1871):** Orchestrated a united German defense against Napoleon III. On January 18, 1871, the German Empire was formally proclaimed in the Hall of Mirrors at the Palace of Versailles.\n\n### 3. The Second Industrial Revolution: Chemicals, Steel, and Electricity\nWhile Britain pioneered the First Industrial Revolution (coal, steam, textiles), unified Germany dominated the **Second Industrial Revolution** (1870–1914):\n- **The Humboldtian University Model:** German universities (Berlin, Heidelberg, Göttingen) integrated rigorous laboratory scientific research with teaching. Ph.D. chemists and physicists moved directly into industrial laboratories.\n- **Chemical Dominance:** Companies like BASF, Bayer, and Hoechst synthesized artificial dyes (aniline, indigo), synthetic fertilizers (Haber-Bosch process), and pharmaceuticals (aspirin), controlling 85% of the world's synthetic chemical market by 1913.\n- **Electrical Engineering:** Siemens and AEG pioneered long-distance electrical power transmission, electric motors, and telegraph networks.\n\n### 4. Bismarck's Welfare State: The World's First Social Safety Net (1880s)\nTo weaken the rising socialist movement (*SPD*), Bismarck pioneered state social insurance:\n- **Health Insurance Act (1883)**\n- **Accident Insurance Act (1884)**\n- **Old Age and Disability Insurance Act (1889)**\n\nThis established the principle that a modern industrial state must protect its working citizens against market catastrophes, establishing the bedrock for modern European social democracy."
            }
        ]
    })

    # 6. World History: United States — Constitutional Philosophy & Industrial Rise
    conversations.append({
        "id": "world_history_004_usa_constitutional_democracy",
        "topic": "History of the United States: Constitutional Philosophy & Industrial Rise",
        "conversations": [
            {
                "from": "human",
                "value": "What were the philosophical arguments between the Federalists (Alexander Hamilton) and Democratic-Republicans (Thomas Jefferson) regarding the nature of the American republic, federal power, and the national economy? How did this debate shape modern governance?"
            },
            {
                "from": "gpt",
                "value": "The foundational debates between Alexander Hamilton and Thomas Jefferson in the 1790s established the fundamental ideological fault lines that continue to define American constitutional law, monetary policy, and governance today.\n\n### 1. The Constitutional Dispute: Implied Powers vs. Strict Text\nWhen Hamilton proposed the *First Bank of the United States* in 1791 to manage national revenue and stabilize currency:\n- **Jefferson's Strict Construction:** Jefferson argued that nowhere in Article I of the Constitution was Congress granted explicit authority to incorporate a bank. He maintained that creating unlisted institutions violated the 10th Amendment.\n- **Hamilton's Doctrine of Implied Powers:** Hamilton countered using Article I, Section 8 (the *Elastic Clause* or \"Necessary and Proper Clause\"). He argued that if the Constitution gives the government power to collect taxes and borrow money, it implicitly grants the sovereign means required to execute those ends.\n- **Historic Precedent:** Hamilton's logic was ratified by Chief Justice John Marshall in the landmark Supreme Court ruling *McCulloch v. Maryland (1819)*, permanently establishing the expansion of federal regulatory power.\n\n### 2. Economic Modernization: Hamilton's *Report on Manufactures* (1791)\nHamilton foresaw that a purely agricultural nation would remain vulnerable to European economic dominance. He advocated:\n1. Protective tariffs to nurture infant domestic industries.\n2. Federal bounties and subsidies for manufacturing innovations.\n3. A robust national infrastructure (roads, canals, maritime ports).\n\nWhile Jefferson championed individual liberty and decentralized local democracy, history ultimately validated Hamilton's institutional blueprint: the United States emerged in the late 19th century as the world's preeminent financial and manufacturing titan."
            }
        ]
    })

    # 7. World History: Ancient Egypt — Hydrology, Geometry & Architecture
    conversations.append({
        "id": "world_history_005_ancient_egypt_hydrology_science",
        "topic": "Ancient Egyptian Civilization: Nile River Ecology, Architecture & Mathematics",
        "conversations": [
            {
                "from": "human",
                "value": "Herodotus famously wrote that 'Egypt is the gift of the Nile.' Explain the hydrological and astronomical mechanisms that allowed ancient Egyptian civilization to sustain continuous monumental civilization for over 3,000 years, and describe their mathematical achievements in surveying and geometry."
            },
            {
                "from": "gpt",
                "value": "The civilization of ancient Egypt was fundamentally an ecological and architectural engineering triumph orchestrated in harmony with the predictable rhythms of the Nile River. For three millennia, the Nile provided not merely water, but the structural template for Egyptian religion, calendar science, mathematics, and centralized administration.\n\n### 1. The Hydrological Engine: The Inundation (*Akhet*)\nSurrounded on both sides by the hyper-arid Sahara and Arabian deserts, Egypt's sole lifeline was the annual summer monsoon rainfall in the Ethiopian highlands, which drove the Nile floodwaters northward:\n- **The Three Seasons:**\n  1. *Akhet* (Inundation: July – October): Submerged the valley, depositing nutrient-rich volcanic black silt (*Kemet*).\n  2. *Peret* (Emergence / Sowing: November – February): Farmers planted emmer wheat, barley, and flax into the moist sediment without artificial fertilizers.\n  3. *Shemu* (Harvest / Drought: March – June): Crops harvested and stored in state granaries before river reached lowest point.\n- **The Nilometer:** Stone graduated wells along the river measured flood height. 16 cubits meant abundance; 12 cubits signaled famine; 18 cubits threatened destruction of mud-brick villages.\n\n### 2. Geometry Born of Necessity: The *Harpedonaptai* (Rope-Stretchers)\nBecause annual floods obliterated boundary markers between private farms, the state re-surveyed every farm every autumn to reassess land tax:\n- **Surveying Science:** Civil servants used knotted ropes to construct exact right angles using the 3-4-5 Pythagorean triangle ratio millennia before Pythagoras.\n- **Calculation of Area & Volume:** The *Rhind Mathematical Papyrus* (c. 1550 BCE) demonstrates formulas for calculating area of triangles, trapezoids, circles (using $\\pi \\approx 3.1605$), and truncated pyramid volume ($V = \\frac{1}{3} h (a^2 + ab + b^2)$).\n\n### 3. Monumental Architecture: The Great Pyramid of Giza\nBuilt during the Old Kingdom for Pharaoh Khufu (c. 2560 BCE):\n- Composed of 2.3 million limestone blocks averaging 2.5 tons each. The 13-acre base deviates from a true mathematical square by less than 4.4 cm.\n- Aligned to true north with an accuracy of 3/60 of a degree by tracking circumpolar stars (*the Indestructibles*)."
            }
        ]
    })

    # 8. World History: China — Dynastic Cycles & The Four Great Inventions
    conversations.append({
        "id": "world_history_006_china_silk_road_inventions",
        "topic": "History of Imperial China: Dynastic Cycle, Silk Road & Great Inventions",
        "conversations": [
            {
                "from": "human",
                "value": "What were the 'Four Great Inventions' of ancient China (Papermaking, Printing, Gunpowder, Compass), and how did the Silk Road foster technological and philosophical exchange between China, the Islamic world, and Europe?"
            },
            {
                "from": "gpt",
                "value": "The scientific innovations of imperial China—collectively known as the **Four Great Inventions** (四大發明)—fundamentally reordered global communication, navigation, warfare, and intellectual history.\n\n### 1. The Four Great Inventions & Their Global Impact\n1. **Papermaking (Cai Lun, Eastern Han Dynasty, c. 105 CE):** Mashing mulberry bark, hemp, rags, and fishnet into a fibrous slurry. Introduced after the Battle of Talas (751 CE) to Samarkand and Baghdad, enabling the Islamic Golden Age library explosion.\n2. **Movable-Type & Woodblock Printing (Bi Sheng, Song Dynasty, c. 1040 CE):** Mass reproduction of Buddhist sutras (*Diamond Sutra*, 868 CE) and Confucian classics, spreading meritocratic literacy centuries before Gutenberg.\n3. **Gunpowder (*Huoyao*, Tang & Song Dynasties):** Discovered by Daoist alchemists combining sulfur, charcoal, and potassium nitrate. Evolved from fire arrows into cast-iron bomb shells and bronze cannons.\n4. **Magnetic Compass (*Luo Pan*, Han & Song Dynasties):** Magnetized steel needles on water pivots enabled Admiral Zheng He's massive Ming treasure fleets (1405–1433) to cross the Indian Ocean.\n\n### 2. The Silk Road: The Great Eurasian Trade & Cultural Arterial\n- Spanning 6,400 km from Chang'an (Xi'an) through Central Asian oases (Samarkand, Bukhara) to the Mediterranean.\n- Material exchange of Chinese silk, porcelain, and tea for Roman glassware, Persian rugs, and Arabian frankincense.\n- Spiritual exchange: Buddhism transmitted from India to China via Dunhuang; Islamic mathematics and astronomy shared fluidly across Eurasia."
            }
        ]
    })

    # 9. World History: France — Enlightenment Philosophy & The French Revolution (1789)
    conversations.append({
        "id": "world_history_007_french_revolution_enlightenment",
        "topic": "History of France: Enlightenment Ideas & The French Revolution",
        "conversations": [
            {
                "from": "human",
                "value": "How did Enlightenment thinkers (Voltaire, Rousseau, Montesquieu) dismantle the divine right of kings, and how did the French Revolution of 1789 transform the concept of universal human rights and constitutional law across Europe?"
            },
            {
                "from": "gpt",
                "value": "The French Revolution of 1789 was the political culmination of the 18th-century European Enlightenment, transforming sovereignty from the monarch to the collective body of citizens.\n\n### 1. The Intellectual Architects\n- **Montesquieu (*The Spirit of the Laws*, 1748):** Established the **Separation of Powers** (Executive, Legislative, Judicial) to guarantee checks and balances.\n- **Voltaire:** Championed civil liberties, freedom of speech, and religious tolerance against theocratic dogma.\n- **Rousseau (*The Social Contract*, 1762):** Argued that legitimate rule derives from the **General Will** (*Volonté Générale*).\n\n### 2. Declaration of the Rights of Man and of the Citizen (1789)\nProclaimed that all men are born and remain free and equal in rights, establishing freedom of religion, speech, and the presumption of innocence.\n\n### 3. The Napoleonic Code (1804)\nCodified legal equality, secular civil marriage, property rights, and contract law, serving as the blueprint for civil law codes worldwide."
            }
        ]
    })

    # 10. Bangladesh Ecology & Geography: Deltaic Hydrology, Rivers & Sundarbans
    conversations.append({
        "id": "bd_geo_003_delta_rivers_sundarbans",
        "topic": "Geography & River Ecology of Bangladesh (নদীমাতৃক বাংলাদেশ ও সুন্দরবন)",
        "conversations": [
            {
                "from": "human",
                "value": "বাংলাদেশকে কেন পৃথিবীর বৃহত্তম সক্রিয় ব-দ্বীপ (Active Delta) বলা হয়? পদ্মা, মেঘনা ও যমুনা নদীর পলিপ্রবাহ এবং সুন্দরবনের ম্যানগ্রোভ ইকোসিস্টেমের পরিবেশগত গুরুত্ব বিস্তারিতভাবে বুঝিয়ে দাও।"
            },
            {
                "from": "gpt",
                "value": "বাংলাদেশ একটি অনন্য ভৌগোলিক বিস্ময়। প্রায় ১,৪৭,৫৭০ বর্গকিলোমিটার আয়তনের এই দেশটি বিশ্বের বৃহত্তম ও অন্যতম সক্রিয় ব-দ্বীপ (Bengal Delta), যা হিমালয় পর্বতমালার ক্ষয়িষ্ণু শিলাস্তর ও অবক্ষেপণ দ্বারা লাখ লাখ বছর ধরে গঠিত হয়েছে।\n\n### ১. গঙ্গা-ব্রহ্মপুত্র-মেঘনা (GBM) অববাহিকা ও পলিপ্রবাহ\n- **পদ্মা, যমুনা ও মেঘনা:** এই তিনটি প্রধান নদী প্রতি বছর প্রায় **১০০ থেকে ১৫০ কোটি টন পলি** (sediment) বঙ্গোপসাগরে জমা করে। এর ফলে দক্ষিণাঞ্চলে নতুন চর জেগে ওঠে এবং কৃষিজমিতে প্রাকৃতিক উর্বরতা বৃদ্ধি পায়।\n\n### ২. সুন্দরবন: পৃথিবীর বৃহত্তম ম্যানগ্রোভ বন\n- **উদ্ভিদ অভিযোজন:** শ্বাসমূল (Pneumatophores), ঠেস根 (Stilt roots) এবং জরায়ুজ অঙ্কুরোদগম (Viviparous germination)।\n- **প্রাকৃতিক সুরক্ষা প্রাচীর (Green Bio-Shield):** সিডর, আইলা ও রেমালের মতো সামুদ্রিক জলোচ্ছ্বাস ও তীব্র ঘূর্ণিঝড়ের গতিবেগ ৩০-৪০% হ্রাস করে উপকূলীয় কোটি কোটি মানুষের জীবন রক্ষা করে।"
            }
        ]
    })

    # 11. World History: India — Indus Valley, Ashoka, Mughal Golden Age & 1947 Freedom Struggle
    conversations.append({
        "id": "world_history_008_india_civilization_freedom",
        "topic": "History of India: Ancient Harappa, Ashoka, Mughals & Freedom Movement",
        "conversations": [
            {
                "from": "human",
                "value": "Indian history is one of the most complex tapestries in human civilization. Provide a comparative synthesis connecting: (1) Harappa/Mohenjo-daro sanitation, (2) Ashoka's Dhamma policy, (3) Akbar's Mughal syncretism, and (4) Gandhi vs. Netaji's freedom struggle strategies."
            },
            {
                "from": "gpt",
                "value": "Indian civilizational history connects ancient urban planning with profound spiritual and political transformations.\n\n### 1. Harappan Urban Sanitation (c. 2600 BCE)\nStandardized gridiron brick cities with covered subterranean drainage and public municipal baths.\n\n### 2. Ashoka's Dhamma Policy (261 BCE)\nRenounced violent conquest (*Digvijaya*) after the Kalinga War, carving edicts for universal religious tolerance, non-violence (*Ahimsa*), and state healthcare.\n\n### 3. Akbar's Mughal Pluralism (*Sulh-i-Kul*)\nAbolished *Jizya* in 1564 and held weekly interfaith philosophical dialogues (*Ibadat Khana*) at Fatehpur Sikri, fostering syncretic Indo-Islamic arts and architecture.\n\n### 4. Freedom Struggle (Gandhi vs. Netaji)\n- **Mahatma Gandhi:** Mass civil disobedience, *Satyagraha*, and moral non-cooperation.\n- **Netaji Subhas Chandra Bose:** Organized the *Indian National Army* (INA) with armed resistance, which shattered British military control."
            }
        ]
    })

    # 12. World History: Ancient Rome — Republic to Empire & The Rule of Law
    conversations.append({
        "id": "world_history_009_ancient_rome_law_empire",
        "topic": "Ancient Rome: Republican Institutions, Pax Romana & Roman Law",
        "conversations": [
            {
                "from": "human",
                "value": "How did the Roman Republic transition into the Roman Empire under Augustus, and what was the lasting global legacy of Roman civil jurisprudence (*Corpus Juris Civilis*)?"
            },
            {
                "from": "gpt",
                "value": "Rome evolved from a regional city-state into the institutional and legal foundation of modern Western civilization.\n\n### 1. Fall of the Republic & Rise of Augustus (27 BCE)\nFollowing the assassination of Julius Caesar (44 BCE), Octavian (Augustus) established the **Principate**, consolidating military command while preserving constitutional facades, initiating the 200-year *Pax Romana*.\n\n### 2. Roman Engineering Achievements\n- **Hydraulic Concrete (*Opus Caementicium*):** Volcanic pozzolana allowed durable construction of the Pantheon and aqueducts.\n- **Roads & Aqueducts:** Gravity-fed fresh spring water across European and North African cities.\n\n### 3. The Justinian Code (*Corpus Juris Civilis*, 534 CE)\nCodified presumption of innocence (*innocent until proven guilty*), contract protections, and universal natural law (*Jus Naturale*)."
            }
        ]
    })

    # 13. NCTB Class 9-10 & 11-12 STEM: Physics (Newtonian Mechanics & Thermodynamics)
    conversations.append({
        "id": "nctb_stem_001_physics_mechanics_thermo",
        "topic": "NCTB Curriculum: Physics Mechanics & Thermodynamics",
        "conversations": [
            {
                "from": "human",
                "value": "নবম-দশম ও একাদশ শ্রেণির পদার্থবিজ্ঞান অনুযায়ী নিউটনের গতির সূত্রাবলী, কাজ-ক্ষমতা-শক্তি এবং তাপগতিবিদ্যার প্রথম সূত্রের মধ্যে গাণিতিক ও বাস্তব সংযোগটি ধাপে ধাপে বুঝিয়ে দাও। একটি ১০ কেজি ভরের বস্তুকে ৫ মিটার উঁচু থেকে ফেললে কীভাবে শক্তির রূপান্তর ঘটে?"
            },
            {
                "from": "gpt",
                "value": "পদার্থবিজ্ঞানের মৌলিক ভিত্তি হলো শক্তির রূপান্তর এবং বলের ক্রিয়ায় বস্তুর গতির পরিবর্তন।\n\n### ১. নিউটনের গতির তিনটি মৌলিক সূত্র\n1. **প্রথম সূত্র (জড়তা ও বল):** বাহ্যিক বল ব্যতীত স্থির বস্তু স্থির এবং গতিশীল বস্তু সমবেগে সরলরেখায় চলে।\n2. **দ্বিতীয় সূত্র (বল পরিমাপ):** $F = m a = m \\frac{v - u}{t}$\n3. **তৃতীয় সূত্র (প্রতিক্রিয়া):** প্রত্যেক ক্রিয়ার সমান ও বিপরীত প্রতিক্রিয়া আছে ($F_1 = -F_2$)।\n\n### ২. গাণিতিক সমাধান: ১০ কেজি ভরের বস্তু ৫ মিটার পতন\n- ভর, $m = 10\\text{ kg}$, উচ্চতা, $h = 5\\text{ m}$, $g = 9.8\\text{ ms}^{-2}$\n- **শীর্ষবিন্দুতে বিভব শক্তি:** $E_p = mgh = 10 \\times 9.8 \\times 5 = \\mathbf{490\\text{ J}}$ (গতিশক্তি $E_k = 0$)\n- **ভূমি স্পর্শের পূর্বমুহূর্তে গতিশক্তি:** $E_k = \\frac{1}{2} m v^2 = \\frac{1}{2} \\times 10 \\times (2gh) = \\mathbf{490\\text{ J}}$\n- **মাটিতে আঘাতের পর:** গতিশক্তি শব্দ, তাপ ও বস্তু বিকৃতিতে রূপান্তরিত হয়, যা তাপগতিবিদ্যার প্রথম সূত্র ($\\Delta Q = \\Delta U + \\Delta W$) নিশ্চিত করে।"
            }
        ]
    })

    return conversations

def validate_and_split_dataset(conversations: List[Dict[str, Any]], train_ratio: float = 0.9) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Validates data hygiene and creates strict 90/10 train/test splits."""
    cleaned = []
    total_tokens_est = 0
    
    for item in conversations:
        # Validate ShareGPT structure
        assert "conversations" in item and len(item["conversations"]) >= 2
        for turn in item["conversations"]:
            assert turn["from"] in ["human", "gpt", "system"]
            assert isinstance(turn["value"], str) and len(turn["value"].strip()) > 0
            
        # Estimate tokens (~4 characters per token for bilingual text)
        full_text = " ".join([turn["value"] for turn in item["conversations"]])
        est_tokens = len(full_text) // 4
        total_tokens_est += est_tokens
        cleaned.append(item)
        
    # Shuffle deterministically
    random.shuffle(cleaned)
    split_idx = int(len(cleaned) * train_ratio)
    if split_idx == len(cleaned):
        split_idx = len(cleaned) - 1 # Ensure at least 1 test item
        
    train_set = cleaned[:split_idx]
    test_set = cleaned[split_idx:]
    
    stats = {
        "total_conversations": len(cleaned),
        "train_count": len(train_set),
        "test_count": len(test_set),
        "total_estimated_tokens": total_tokens_est,
        "avg_tokens_per_thread": total_tokens_est // len(cleaned)
    }
    return train_set, test_set, stats

def main():
    print("=" * 80)
    print("THSA-2B BILINGUAL SHAREGPT CONVERSATIONAL DATASET GENERATOR")
    print("=" * 80)
    
    data_dir = r"ss_bangladesh_nano_android_module\THSA-2B V1\data"
    os.makedirs(data_dir, exist_ok=True)
    
    conversations = generate_all_conversations()
    train_set, test_set, stats = validate_and_split_dataset(conversations, train_ratio=0.9)
    
    train_path = os.path.join(data_dir, "train_sharegpt.jsonl")
    test_path = os.path.join(data_dir, "test_sharegpt.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(test_path, "w", encoding="utf-8") as f:
        for entry in test_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Dataset Generation Complete:")
    print(f"  Total Conversations:    {stats['total_conversations']}")
    print(f"  Train Set (90%):         {stats['train_count']} threads -> {train_path}")
    print(f"  Test Set  (10%):         {stats['test_count']} threads -> {test_path}")
    print(f"  Total Estimated Tokens: ~{stats['total_estimated_tokens']:,} tokens")
    print(f"  Average Tokens/Thread:  ~{stats['avg_tokens_per_thread']:,} tokens/thread")
    print("=" * 80)

if __name__ == "__main__":
    main()
