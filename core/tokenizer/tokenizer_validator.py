"""
SS Tutor BD - Bengali Tokenizer Validator (Phase 4)
Validates that mathematical symbols, Bengali conjuncts, English technical terms,
and NCTB formulas are perfectly encoded and decoded without character loss.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import PreTrainedTokenizerFast

TEST_CASES = [
    # Math formulas and symbols
    "I = Prn",
    "C = P(1 + r)^n",
    "c² = a² + b²",
    "A = πr²",
    "P = 2πr",
    "Sₙ = n(n + 1) / 2",
    "x² + 7x + 12 = 0",
    "৩/৪ + ৫/৬ = ১৯/১২",
    "√16 = 4 এবং √25 = 5",
    "১০% মুনাফা হারে ৩ বছর",
    "∠ABC = ৯০°",
    "x ≤ 10 এবং y ≥ 5",
    "a ≠ b এবং x ≈ 3.14",
    # Complex Bengali sentences with conjuncts
    "সমকোণী ত্রিভুজের অতিভুজ² = ভূমি² + লম্ব²।",
    "চক্রবৃদ্ধি মুনাফা এবং সরল মুনাফার পার্থক্য নির্ণয় করো।",
    "ভগ্নাংশের লব ও হরের লঘিষ্ঠ সাধারণ গুণিতক (ল.সা.গু)।",
    "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না।",
    # Mixed English/Bengali
    "Simple Interest = P × R × T / 100 বাংলায় সরল মুনাফা।",
    "Profit = SP - CP অর্থাৎ লাভ = বিক্রয়মূল্য - ক্রয়মূল্য।"
]


def validate_tokenizer_integrity(tokenizer_dir: Path = None) -> Dict[str, Any]:
    """Tests encode -> decode roundtrip for all test cases."""
    tokenizer_dir = tokenizer_dir or (PROJECT_ROOT / "models" / "tokenizer_bengali_16k")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

    passed = 0
    failures = []

    for text in TEST_CASES:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        # Normalize whitespace
        if decoded.strip() == text.strip():
            passed += 1
        else:
            failures.append({
                "original": text,
                "decoded": decoded,
                "encoded_tokens": encoded
            })

    all_passed = (len(failures) == 0)
    print(f"\n--- Bengali Tokenizer Symbol & Script Validation ---")
    print(f"  Result: {passed} / {len(TEST_CASES)} PASSED")
    if failures:
        print(f"  Failures ({len(failures)}):")
        for f in failures:
            print(f"    Original: {f['original']}")
            print(f"    Decoded:  {f['decoded']}")
    else:
        print("  100% Roundtrip Integrity Verified ✅")
    print("----------------------------------------------------\n")

    return {
        "total_cases": len(TEST_CASES),
        "passed": passed,
        "failed": len(failures),
        "all_passed": all_passed,
        "failures": failures
    }


if __name__ == "__main__":
    validate_tokenizer_integrity()
