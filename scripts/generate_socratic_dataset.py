"""
SS Tutor BD - Synthetic Socratic Hint Dataset Generator (Phase 4)
Generates structured JSONL training examples for Socratic tutoring and direct answer withholding.
"""

import sys
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "phase4" / "socratic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HINT_PATTERNS = [
    {
        "topic": "fraction_addition",
        "questions": ["৩/৪ + ৫/৬ কীভাবে করব? উত্তর বলো না।", "ভগ্নাংশ যোগ করার প্রথম ধাপ কী?", "২/৩ + ৪/৫ যোগের ইঙ্গিত দাও।"],
        "hints": [
            "প্রথমে দুটি ভগ্নাংশের হরের ল.সা.গু নির্ণয় করো। এরপর হরগুলোকে সমান করে লবগুলো যোগ করার চেষ্টা করো।",
            "প্রথমেই হরের দিকে লক্ষ্য করো। হর দুটি অসমান হলে ল.সা.গু বের করে সমহর বিশিষ্ট ভগ্নাংশ তৈরি করতে হবে।"
        ]
    },
    {
        "topic": "simple_interest",
        "questions": ["৫০০০ টাকায় ১০% হারে ৩ বছরের মুনাফার সূত্র কী?", "মুনাফা কীভাবে হিসাব করব? শুধু ইঙ্গিত দাও।"],
        "hints": [
            "সরল মুনাফা নির্ণয়ের জন্য I = Prn সূত্রটি মনে করো, যেখানে P হলো আসল, r হলো সুদের হার এবং n হলো সময়।",
            "এখানে আসল P এবং সময় n দেওয়া আছে। সূত্র I = Prn প্রয়োগ করে I-এর মান বের করার চেষ্টা করো।"
        ]
    },
    {
        "topic": "quadratic_equation",
        "questions": ["x² + 7x + 12 = 0 এর মধ্যপদ কীভাবে ভাঙব? উত্তর দিও না।", "দ্বিঘাত সমীকরণ সমাধানের ইঙ্গিত দাও।"],
        "hints": [
            "এমন দুটি সংখ্যা খুঁজে বের করো যাদের গুণফল ১২ এবং যোগফল ৭। তারপর মধ্যপদ ৭x কে সেই দুটি অংশে বিভক্ত করো।",
            "ধ্রুবক পদ ১২-এর গুণনীয়কগুলো দেখো। কোন জোড়া গুণনীয়ক যোগ করলে মধ্যপদের সহগ পাওয়া যায়?"
        ]
    },
    {
        "topic": "pythagoras",
        "questions": ["সমকোণী ত্রিভুজের অতিভুজ বের করতে কোন সূত্র লাগবে?", "ভূমি ৬ ও লম্ব ৮ হলে অতিভুজ বের করার নিয়ম কী?"],
        "hints": [
            "সমকোণী ত্রিভুজের ক্ষেত্রে পিথাগোরাসের উপপাদ্য প্রযোজ্য: অতিভুজ² = ভূমি² + লম্ব²। মানগুলো বসিয়ে বর্গমূল করো।",
            "ভূমি ও লম্বের বর্গ করে যোগ করো, তারপর যোগফলের বর্গমূল নির্ণয় করলে অতিভুজ পাবে।"
        ]
    },
    {
        "topic": "circle_area",
        "questions": ["৭ সেমি ব্যাসার্ধের বৃত্তের ক্ষেত্রফল কীভাবে বের করব? শুধু সূত্র বলো।", "বৃত্তের পরিধি নির্ণয়ের ধাপ কী?"],
        "hints": [
            "বৃত্তের ক্ষেত্রফলের সূত্র হলো A = πr² এবং পরিধির সূত্র হলো P = 2πr। এখানে r = ৭ মানটি সূত্রে বসাও।",
            "ব্যাসার্ধ r দেওয়া আছে। ক্ষেত্রফল পেতে π × r² হিসাব করো।"
        ]
    }
]


def generate_socratic_examples(count: int = 3000) -> list:
    examples = []
    for _ in range(count):
        pat = random.choice(HINT_PATTERNS)
        q = random.choice(pat["questions"])
        hint = random.choice(pat["hints"])
        examples.append({
            "mode": "hint",
            "category": pat["topic"],
            "instruction": q,
            "context": "[HINT] উত্তর সরাসরি বলা নিষেধ। শিক্ষার্থীকে চিন্তা করতে সাহায্য করো।",
            "response": hint
        })
    return examples


def main():
    random.seed(42)
    data = generate_socratic_examples(3000)
    out_file = DATA_DIR / "socratic_hints.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Generated {len(data)} synthetic Socratic hint examples: {out_file}")


if __name__ == "__main__":
    main()
