"""
SS Tutor BD - Bengali Tokenizer Efficiency Benchmark (Phase 3C)
Evaluates tokenizer suitability for NCTB Bengali math content.
Measures: chars/token, words/token, sentence/token, math+Bengali efficiency.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Representative NCTB Class 8 Mathematics content samples
BENGALI_SAMPLES = {
    "pure_bengali_definition": [
        "সরল মুনাফা হলো মূল অর্থের উপর নির্দিষ্ট হারে নির্দিষ্ট সময়ের জন্য অর্জিত মুনাফা।",
        "ভগ্নাংশের লব ও হরের মধ্যে সর্বোচ্চ সাধারণ গুণনীয়ক দিয়ে ভাগ করলে লঘিষ্ঠ আকার পাওয়া যায়।",
        "পিথাগোরাস উপপাদ্য অনুযায়ী সমকোণী ত্রিভুজের অতিভুজের বর্গ অপর দুই বাহুর বর্গের সমষ্টির সমান।",
        "বৃত্তের পরিধি = ২πr যেখানে r হলো বৃত্তের ব্যাসার্ধ।",
    ],
    "math_expressions": [
        "৩/৪ + ৫/৬ এর যোগফল নির্ণয় করো।",
        "৫০০০ টাকায় ১০% হারে ৩ বছরের সরল মুনাফা কত?",
        "x² + 7x + 12 = 0 সমীকরণটি সমাধান করো।",
        "একটি ত্রিভুজের ভূমি ১২ সেমি এবং উচ্চতা ৮ সেমি হলে ক্ষেত্রফল কত?",
    ],
    "mixed_bengali_english": [
        "Simple Interest = P × R × T / 100 সূত্র ব্যবহার করে সমাধান করো।",
        "Profit = Selling Price - Cost Price অর্থাৎ লাভ = বিক্রয়মূল্য - ক্রয়মূল্য।",
        "LCM বা ল.সা.গু হলো দুটি বা তার বেশি সংখ্যার ক্ষুদ্রতম সাধারণ গুণিতক।",
    ],
    "compound_math_bengali": [
        "চক্রবৃদ্ধি মুনাফার সূত্র: C = P(1 + r)^n যেখানে P = আসল, r = সুদের হার এবং n = সময়।",
        "১ থেকে ১০০ পর্যন্ত ক্রমিক স্বাভাবিক সংখ্যার যোগফল Sₙ = n(n+1)/2 সূত্র দ্বারা নির্ণয় করা যায়।",
        "সমকোণী ত্রিভুজে যদি ভূমি ৩ সেমি এবং লম্ব ৪ সেমি হয় তাহলে অতিভুজ = √(3² + 4²) = ৫ সেমি।",
    ]
}

ALL_SAMPLES_FLAT = []
for cat, samples in BENGALI_SAMPLES.items():
    ALL_SAMPLES_FLAT.extend(samples)


def count_bengali_chars(text: str) -> int:
    return sum(1 for c in text if '\u0980' <= c <= '\u09FF')


def count_words(text: str) -> int:
    return len(text.split())


def evaluate_tokenizer(tokenizer_repo: str, candidate_id: str) -> Dict[str, Any]:
    """Load tokenizer from HF and benchmark against NCTB samples."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    except Exception as e:
        return {"error": str(e), "tokenizer_repo": tokenizer_repo}

    category_results = {}
    total_chars = 0
    total_words = 0
    total_tokens = 0
    total_samples = 0

    for category, samples in BENGALI_SAMPLES.items():
        cat_tokens_list = []
        for sample in samples:
            tokens = tokenizer.encode(sample)
            n_chars = len(sample)
            n_words = count_words(sample)
            n_bn_chars = count_bengali_chars(sample)
            cat_tokens_list.append({
                "text_preview": sample[:50] + "...",
                "total_chars": n_chars,
                "bengali_chars": n_bn_chars,
                "words": n_words,
                "tokens": len(tokens),
                "chars_per_token": round(n_chars / max(1, len(tokens)), 2),
                "bengali_chars_per_token": round(n_bn_chars / max(1, len(tokens)), 2),
                "words_per_token": round(n_words / max(1, len(tokens)), 3),
            })
            total_chars += n_chars
            total_words += n_words
            total_tokens += len(tokens)
            total_samples += 1
        category_results[category] = cat_tokens_list

    avg_chars_per_token = round(total_chars / max(1, total_tokens), 2)
    avg_words_per_token = round(total_words / max(1, total_tokens), 3)
    avg_words_per_token_inverse = round(1.0 / avg_words_per_token, 2) if avg_words_per_token > 0 else 0
    vocab_size = getattr(tokenizer, "vocab_size", "unknown")

    return {
        "candidate_id": candidate_id,
        "tokenizer_repo": tokenizer_repo,
        "vocab_size": vocab_size,
        "total_samples": total_samples,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "avg_chars_per_token": avg_chars_per_token,
        "avg_words_per_token": avg_words_per_token,
        "avg_bengali_tokens_per_word": avg_words_per_token_inverse,
        "category_results": category_results
    }


def print_tokenizer_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"  TOKENIZER BENCHMARK: {result.get('candidate_id', '?')}")
    print("=" * 60)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  Tokenizer Repo:        {result['tokenizer_repo']}")
    print(f"  Vocabulary Size:       {result['vocab_size']:,}")
    print(f"  Samples Tested:        {result['total_samples']}")
    print(f"  Total Tokens Produced: {result['total_tokens']}")
    print(f"  Total Characters:      {result['total_chars']}")
    print(f"  Avg Chars / Token:     {result['avg_chars_per_token']}")
    print(f"  Avg Words / Token:     {result['avg_words_per_token']}")
    print(f"  Avg Bengali Tokens/Word: {result['avg_bengali_tokens_per_word']}")
    print("-" * 60)
    print(f"  EFFICIENCY VERDICT: ", end="")
    tpw = result['avg_bengali_tokens_per_word']
    if tpw <= 2.0:
        print("EXCELLENT (≤ 2 tok/word)")
    elif tpw <= 4.0:
        print("ACCEPTABLE (2–4 tok/word)")
    elif tpw <= 6.0:
        print("POOR (4–6 tok/word)")
    else:
        print("DISQUALIFYING (> 6 tok/word — byte expansion)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    RESULTS_DIR = PROJECT_ROOT / "results" / "phase3c"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Candidates for tokenizer evaluation (offline feasibility screening)
    candidates_to_evaluate = [
        {"id": "CAND-03",  "repo": "HuggingFaceTB/SmolLM2-135M-Instruct"},
        {"id": "CAND-01",  "repo": "Qwen/Qwen2.5-0.5B-Instruct"},
    ]

    if len(sys.argv) >= 3:
        candidates_to_evaluate = [{"id": sys.argv[1], "repo": sys.argv[2]}]

    all_results = []
    for cand in candidates_to_evaluate:
        print(f"[Tokenizer Benchmark] Evaluating {cand['id']} ({cand['repo']})...")
        res = evaluate_tokenizer(cand["repo"], cand["id"])
        print_tokenizer_report(res)
        all_results.append(res)

    out_path = RESULTS_DIR / "tokenizer_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")
