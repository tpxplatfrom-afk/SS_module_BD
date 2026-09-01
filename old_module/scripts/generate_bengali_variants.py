"""
SS Tutor BD - Bengali Linguistic Variants Generator (Phase 4)
Generates formal Bengali, colloquial Bengali, student shorthand, and Banglish input variations.
"""

import sys
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "phase4" / "bengali"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS_MAP = [
    {
        "intent": "fraction_addition",
        "variants": [
            "ভগ্নাংশ যোগ করার নিয়ম কী?",
            "ভগ্নাংশ কেমনে যোগ করে?",
            "fraction jog korar niyom ki?",
            "vognangsho jog korbo kivabe?",
            "ভগ্নাংশ যোগের সূত্রটা বুঝিয়ে বলো তো"
        ],
        "response": "ভগ্নাংশ যোগ করতে প্রথমে হরগুলোর ল.সা.গু নির্ণয় করে সমহর বিশিষ্ট ভগ্নাংশে রূপান্তর করতে হয়, এরপর লবগুলো যোগ করতে হয়।"
    },
    {
        "intent": "simple_interest",
        "variants": [
            "সরল মুনাফার সূত্র কী?",
            "shorol munafa formula ki?",
            "munafa kivabe ber kore?",
            "মুনাফা বের করার সহজ নিয়ম কী?",
            "I = Prn সূত্রে r এর মানে কী?"
        ],
        "response": "সরল মুনাফার সূত্র হলো I = Prn, যেখানে I = মোট মুনাফা, P = আসল, r = মুনাফার শতকরা হার এবং n = সময়।"
    },
    {
        "intent": "pythagoras",
        "variants": [
            "পিথাগোরাস উপপাদ্য কী?",
            "pythagoras theorem ki?",
            "অতিভুজ বের করার নিয়ম বলো",
            "shomokoni trivujer otivuj formula ki?",
            "পিথাগোরাসের সূত্রটা সহজ করে বুঝিয়ে দাও"
        ],
        "response": "সমকোণী ত্রিভুজের ক্ষেত্রে পিথাগোরাসের উপপাদ্য হলো: অতিভুজ² = ভূমি² + লম্ব² (c² = a² + b²)।"
    }
]


def generate_variant_examples(count: int = 2000) -> list:
    examples = []
    for _ in range(count):
        item = random.choice(VARIANTS_MAP)
        q = random.choice(item["variants"])
        examples.append({
            "mode": "explanation",
            "category": item["intent"],
            "instruction": q,
            "context": "[C] সহজ বাংলায় উত্তর দাও।",
            "response": item["response"]
        })
    return examples


def main():
    random.seed(42)
    data = generate_variant_examples(2000)
    out_file = DATA_DIR / "bengali_variants.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Generated {len(data)} synthetic Bengali variant examples: {out_file}")


if __name__ == "__main__":
    main()
