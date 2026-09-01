"""
SS Tutor BD - Dedicated Bengali Educational Tokenizer Trainer (Phase 4)
Trains a 16,000-vocabulary BPE tokenizer optimized for Bengali script,
English technical terms, and mathematical/scientific symbols.
"""

import sys
import os
import json
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Special tokens
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>",
    "[TASK]",
    "[FACT]",
    "[RESULT]",
    "[GOAL]",
    "[HINT]",
    "[CONSTRAINT]",
    "[T]",
    "[F]",
    "[R]",
    "[G]",
    "[H]",
    "[C]"
]

MATH_SYMBOLS = [
    "x²", "y²", "a²", "b²", "c²", "r²", "n²", "x³", "a³", "b³",
    "√", "π", "≤", "≥", "≠", "≈", "±", "×", "÷", "−", "=", "+", "%",
    "Sₙ", "I=Prn", "C=P(1+r)^n", "a²+b²=c²", "πr²", "2πr", "1/2", "1/3", "1/4",
    "3/4", "5/6", "7/12", "19/12", "°", "₁", "₂", "ₙ"
]


def build_training_corpus(output_path: Path) -> int:
    """Creates a comprehensive Bengali math and educational corpus for training the tokenizer."""
    corpus_lines = []

    # 1. Math formulas and rule statements in Bengali
    core_rules = [
        "সরল মুনাফার সূত্র হলো I = Prn যেখানে P = আসল, r = মুনাফার হার, n = সময় এবং I = মোট মুনাফা।",
        "চক্রবৃদ্ধি মূলধন C = P(1 + r)^n এবং চক্রবৃদ্ধি মুনাফা = C - P।",
        "সমকোণী ত্রিভুজের ক্ষেত্রে পিথাগোরাসের উপপাদ্য: অতিভুজ² = ভূমি² + লম্ব² অর্থাৎ c² = a² + b²।",
        "বৃত্তের পরিধি = 2πr এবং বৃত্তের ক্ষেত্রফল = πr² যেখানে r হলো বৃত্তের ব্যাসার্ধ এবং π ≈ 22/7।",
        "ভগ্নাংশের যোগ করার নিয়ম: প্রথমে হরগুলোর ল.সা.গু নির্ণয় করে সমহর বিশিষ্ট ভগ্নাংশে রূপান্তর করতে হয়।",
        "৩/৪ + ৫/৬ এর ল.সা.গু হলো ১২। ৯/১২ + ১০/১২ = ১৯/১২ = ১ সমস্ত ৭/১২।",
        "বীজগাণিতিক সূত্রাবলী: (a + b)² = a² + 2ab + b² এবং (a - b)² = a² - 2ab + b²।",
        "a² - b² = (a + b)(a - b) এবং (a + b)³ = a³ + 3a²b + 3ab² + b³।",
        "দ্বিঘাত সমীকরণ x² + 7x + 12 = 0 এর উৎপাদকে বিশ্লেষণ হলো (x + 3)(x + 4) = 0।",
        "১ থেকে n পর্যন্ত স্বাভাবিক ক্রমিক সংখ্যার সমষ্টির সূত্র: Sₙ = n(n + 1) / 2।",
        "লাভ = বিক্রয়মূল্য - ক্রয়মূল্য এবং ক্ষতি = ক্রয়মূল্য - বিক্রয়মূল্য।",
        "শতকরা লাভ = (মোট লাভ / ক্রয়মূল্য) × ১০০%।",
        "অনুপাত হলো একই জাতীয় দুটি রাশির মধ্যকার তুলনামূলক সম্পর্ক।"
    ]
    corpus_lines.extend(core_rules * 50)

    # 2. Add numbers and mathematical expressions
    for i in range(1, 101):
        corpus_lines.append(f"সংখ্যা {i} এর বর্গ হলো {i**2} এবং ঘন হলো {i**3}।")
        bn_num = "".join(["০১২৩৪৫৬৭৮৯"[int(d)] for d in str(i)])
        corpus_lines.append(f"বাংলা সংখ্যা: {bn_num} এর সমাধান ধাপ {bn_num}।")

    # 3. Add textbook chapters from SQLite database if available
    db_path = PROJECT_ROOT / "packs" / "class8_math" / "index.db"
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT content_text FROM chunks")
            for row in cursor.fetchall():
                corpus_lines.append(row[0])
        except Exception:
            pass
        finally:
            conn.close()

    # 4. Ingest all Phase 4 synthetic datasets from data/phase4/
    data_phase4_dir = PROJECT_ROOT / "data" / "phase4"
    if data_phase4_dir.exists():
        for jfile in data_phase4_dir.rglob("*.jsonl"):
            try:
                with open(jfile, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        for field in ["instruction", "response", "context"]:
                            if field in data and data[field]:
                                corpus_lines.append(data[field])
            except Exception:
                pass

    # 5. Add common tutoring dialogues and prompts
    dialogues = [
        "প্রশ্ন: এই সমস্যার সমাধান কীভাবে করব? শিক্ষক: প্রথমে সূত্রের চলকগুলো চিহ্নিত করো।",
        "ইঙ্গিত: সরাসরি উত্তর না দিয়ে ভেবে দেখো সূত্রের সাথে কোন তথ্যটি মিলছে।",
        "পাঠ্যপুস্তকের তথ্য অনুযায়ী এটি সঠিক এবং প্রমাণিত।",
        "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না। অনুগ্রহ করে আরও তথ্য দিন।",
        "শিক্ষার্থীর উত্তর যাচাই: তোমার হিসাব সঠিক হয়েছে, চমৎকার!",
        "ভুল সংশোধনের নির্দেশনা: এখানে হরের লসাগু করার সময় একটু ভুল হয়েছে, পুনরায় চেষ্টা করো।"
    ]
    corpus_lines.extend(dialogues * 30)

    # 6. Add all special math symbols
    corpus_lines.append(" ".join(MATH_SYMBOLS) * 20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    return len(corpus_lines)


def train_bengali_tokenizer(vocab_size: int = 16000, output_dir: Path = None) -> Path:
    """Trains a ByteLevel BPE tokenizer using Hugging Face tokenizers library."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    output_dir = output_dir or (PROJECT_ROOT / "models" / "tokenizer_bengali_16k")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = output_dir / "corpus.txt"

    print(f"[Tokenizer Trainer] Building training corpus at: {corpus_file}")
    num_lines = build_training_corpus(corpus_file)
    print(f"[Tokenizer Trainer] Corpus built with {num_lines} lines.")

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False
    )

    print(f"[Tokenizer Trainer] Training BPE tokenizer (Vocab: {vocab_size})...")
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)

    # Save tokenizer.json
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # Also wrap into Hugging Face PreTrainedTokenizerFast for full transformers compatibility
    from transformers import PreTrainedTokenizerFast
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>"
    )
    wrapped_tokenizer.save_pretrained(str(output_dir))

    print(f"[Tokenizer Trainer] Successfully trained & saved to: {output_dir}")
    print(f"[Tokenizer Trainer] Actual vocabulary size: {tokenizer.get_vocab_size()}")

    # Clean corpus file to save disk
    if corpus_file.exists():
        corpus_file.unlink()

    return output_dir


if __name__ == "__main__":
    train_bengali_tokenizer()
