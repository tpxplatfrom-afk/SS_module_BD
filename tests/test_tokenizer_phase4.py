"""SS Tutor BD — Unit Tests: Phase 4 Dedicated Bengali Tokenizer"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import PreTrainedTokenizerFast
from core.tokenizer.tokenizer_validator import validate_tokenizer_integrity


def test_tokenizer_loads_properly():
    tok_dir = PROJECT_ROOT / "models" / "tokenizer_bengali_16k"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    assert tokenizer.vocab_size > 500, f"Vocab too small: {tokenizer.vocab_size}"
    print(f"test_tokenizer_loads_properly: PASSED (Vocab: {tokenizer.vocab_size})")


def test_special_tokens_present():
    tok_dir = PROJECT_ROOT / "models" / "tokenizer_bengali_16k"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    tokens = tokenizer.encode("<|im_start|>[T] [F] [R] [G] [H] [C]<|im_end|>")
    assert len(tokens) > 0
    print("test_special_tokens_present: PASSED")


def test_math_formula_roundtrip():
    res = validate_tokenizer_integrity()
    assert res["all_passed"] == True, f"Failed roundtrip test cases: {res['failed']}"
    print("test_math_formula_roundtrip: PASSED (100% roundtrip)")


def test_bengali_token_efficiency_gate():
    from core.tokenizer.tokenizer_benchmark import benchmark_single_tokenizer
    tok_dir = PROJECT_ROOT / "models" / "tokenizer_bengali_16k"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    res = benchmark_single_tokenizer(tokenizer, "Custom Bengali-16K", tokenizer.vocab_size)
    tpw = res["tokens_per_bengali_word"]
    assert tpw <= 4.0, f"Failed token efficiency gate: {tpw:.2f} tok/word"
    print(f"test_bengali_token_efficiency_gate: PASSED ({tpw:.2f} tok/word <= 4.0)")


def run_all():
    print("\n--- Phase 4 Bengali Tokenizer Tests ---")
    test_tokenizer_loads_properly()
    test_special_tokens_present()
    test_math_formula_roundtrip()
    test_bengali_token_efficiency_gate()
    print("--- All Phase 4 Tokenizer Tests PASSED (4 / 4) ---\n")


if __name__ == "__main__":
    run_all()
