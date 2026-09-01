"""
SS Tutor BD - Bengali Tokenizer Benchmark
Evaluates subword fragmentation, token expansion ratios, and unicode efficiency
for candidate model tokenizers against official NCTB Bengali samples.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "benchmarks" / "tokenizer" / "dataset.json"


def load_dataset() -> List[Dict[str, str]]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])


def evaluate_tokenizer(tokenizer_name_or_path: str) -> Dict[str, Any]:
    """Evaluates a tokenizer against the Bengali benchmark dataset using the tokenizers library."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(tokenizer_name_or_path)
    vocab_size = tok.get_vocab_size()
    samples = load_dataset()
    sample_results = []
    
    total_chars = 0
    total_words = 0
    total_tokens = 0

    for sample in samples:
        text = sample["text"]
        chars = len(text)
        words = len(text.split())
        encoded = tok.encode(text)
        num_tokens = len(encoded.ids)

        tok_per_word = round(num_tokens / words, 2) if words > 0 else 0.0
        tok_per_char = round(num_tokens / chars, 3) if chars > 0 else 0.0

        total_chars += chars
        total_words += words
        total_tokens += num_tokens

        sample_results.append({
            "id": sample["id"],
            "category": sample["category"],
            "chars": chars,
            "words": words,
            "tokens": num_tokens,
            "tok_per_word": tok_per_word,
            "tok_per_char": tok_per_char
        })

    overall_tok_per_word = round(total_tokens / total_words, 2) if total_words > 0 else 0.0
    overall_tok_per_char = round(total_tokens / total_chars, 3) if total_chars > 0 else 0.0

    return {
        "tokenizer": tokenizer_name_or_path,
        "vocab_size": vocab_size,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "avg_tokens_per_word": overall_tok_per_word,
        "avg_tokens_per_char": overall_tok_per_char,
        "samples": sample_results
    }


def print_comparison_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 90)
    print("SS TUTOR BD - BENGALI TOKENIZER EFFICIENCY BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Tokenizer Identifier':<38} {'Vocab Size':<12} {'Words':<8} {'Tokens':<8} {'Tok/Word':<10} {'Tok/Char'}")
    print("-" * 90)
    for r in results:
        print(f"{r['tokenizer']:<38} {r['vocab_size']:<12} {r['total_words']:<8} {r['total_tokens']:<8} {r['avg_tokens_per_word']:<10} {r['avg_tokens_per_char']}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    candidates_to_test = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]
    
    if len(sys.argv) > 1:
        candidates_to_test = sys.argv[1:]

    all_results = []
    for cand in candidates_to_test:
        try:
            res = evaluate_tokenizer(cand)
            all_results.append(res)
        except Exception as e:
            print(f"Failed to evaluate {cand}: {str(e)}")

    print_comparison_table(all_results)
