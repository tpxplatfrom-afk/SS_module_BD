"""
SS Tutor BD - Synthetic Grounding & Anti-Hallucination Dataset Generator (Phase 4)
Generates structured JSONL training examples for strict factual grounding and unsupported fact refusal.
"""

import sys
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "phase4" / "grounding"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GROUNDED_FACTS = [
    {
        "context": "সরল মুনাফার ক্ষেত্রে মুনাফা = আসল × মুনাফার হার × সময় অর্থাৎ I = Prn।",
        "supported_q": "সরল মুনাফা নির্ণয়ের সূত্রটি কী?",
        "supported_ans": "প্রদত্ত পাঠ্যপুস্তকের তথ্য অনুযায়ী, সরল মুনাফার সূত্র হলো I = Prn, যেখানে I = মুনাফা, P = আসল, r = মুনাফার হার এবং n = সময়।",
        "unsupported_q": "ব্যাংক কর্মচারীর বেতন কত ছিল?",
        "refusal_ans": "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না।"
    },
    {
        "context": "সমকোণী ত্রিভুজের অতিভুজের উপর অঙ্কিত বর্গক্ষেত্রের ক্ষেত্রফল অপর দুই বাহুর উপর অঙ্কিত বর্গক্ষেত্রদ্বয়ের ক্ষেত্রফলের সমষ্টির সমান।",
        "supported_q": "পিথাগোরাসের উপপাদ্য কী?",
        "supported_ans": "প্রদত্ত তথ্য অনুসারে, সমকোণী ত্রিভুজের অতিভুজের বর্গ অপর দুই বাহুর বর্গের সমষ্টির সমান (c² = a² + b²)।",
        "unsupported_q": "পিথাগোরাস কোন সালে জন্মগ্রহণ করেন?",
        "refusal_ans": "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না।"
    },
    {
        "context": "বৃত্তের পরিধি এবং ব্যাসের অনুপাত সর্বদা একটি ধ্রুবক সংখ্যা, যাকে গ্রিক বর্ণ π দ্বারা প্রকাশ করা হয়। π এর আসন্ন মান ২২/৭ বা ৩.১৪১৬।",
        "supported_q": "বৃত্তের পাই (π) এর আসন্ন মান কত?",
        "supported_ans": "প্রদত্ত পাঠ্যবইয়ের তথ্য অনুযায়ী, π এর আসন্ন মান হলো ২২/৭ বা প্রায় ৩.১৪১৬।",
        "unsupported_q": "বৃত্তের আবিষ্কারক কে ছিলেন?",
        "refusal_ans": "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না।"
    },
    {
        "context": "দুটি রাশির একটি অপরটির কত গুণ বা কত অংশ তা একটি ভগ্নাংশ দ্বারা প্রকাশ করা যায়। এই ভগ্নাংশটিকে রাশি দুটির অনুপাত বলে।",
        "supported_q": "অনুপাত কাকে বলে?",
        "supported_ans": "প্রদত্ত তথ্য অনুযায়ী, দুটি রাশির একটি অপরটির কত গুণ বা কত অংশ তা প্রকাশকারী ভগ্নাংশটিকে অনুপাত বলে।",
        "unsupported_q": "সাকিব আল হাসানের গড় রান কত?",
        "refusal_ans": "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না।"
    }
]


def generate_grounding_examples(count: int = 3000) -> list:
    examples = []
    for _ in range(count):
        item = random.choice(GROUNDED_FACTS)
        is_refusal = random.choice([True, False])
        if not is_refusal:
            examples.append({
                "mode": "grounded",
                "category": "factual_adherence",
                "instruction": item["supported_q"],
                "context": f"[FACT] {item['context']}",
                "response": item["supported_ans"]
            })
        else:
            examples.append({
                "mode": "grounded",
                "category": "anti_hallucination_refusal",
                "instruction": item["unsupported_q"],
                "context": f"[FACT] {item['context']}",
                "response": item["refusal_ans"]
            })
    return examples


def main():
    random.seed(42)
    data = generate_grounding_examples(3000)
    out_file = DATA_DIR / "grounding_dataset.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Generated {len(data)} synthetic grounding examples: {out_file}")


if __name__ == "__main__":
    main()
