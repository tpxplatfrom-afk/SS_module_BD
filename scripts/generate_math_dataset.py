"""
SS Tutor BD - Synthetic Math Verbalization Dataset Generator (Phase 4)
Generates structured JSONL training examples for verbalizing deterministic tool results.
"""

import sys
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "phase4" / "math"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from core.math.fraction import FractionHelper
from core.math.calculator import MathCalculator
from core.math.equation_solver import EquationSolver


def generate_fraction_examples(count: int = 1500) -> list:
    examples = []
    pairs = [
        ((1, 2), (1, 3)), ((1, 4), (1, 2)), ((3, 4), (5, 6)), ((2, 3), (4, 5)),
        ((1, 6), (1, 3)), ((3, 8), (1, 4)), ((5, 12), (7, 18)), ((2, 5), (3, 10)),
        ((1, 3), (2, 9)), ((3, 5), (4, 15)), ((5, 8), (3, 16)), ((7, 10), (9, 20))
    ]
    for _ in range(count):
        f1, f2 = random.choice(pairs)
        # Random scale
        scale = random.choice([1, 2, 3])
        n1, d1 = f1[0] * scale, f1[1] * scale
        n2, d2 = f2[0], f2[1]
        res = FractionHelper.add((n1, d1), (n2, d2))
        f1_str = f"{FractionHelper.to_bengali_number(n1)}/{FractionHelper.to_bengali_number(d1)}"
        f2_str = f"{FractionHelper.to_bengali_number(n2)}/{FractionHelper.to_bengali_number(d2)}"
        q_bengali = f"{f1_str} এবং {f2_str} এর যোগফল কত?"
        tool_result = f"যোগফল = {res['final_answer_bengali']}"
        steps_text = " ".join(res["steps_bengali"])
        response = f"প্রথমে হরদ্বয়ের ল.সা.গু নির্ণয় করি। {steps_text} অতএব নির্ণেয় যোগফল = {res['final_answer_bengali']}।"
        examples.append({
            "mode": "tool_result",
            "category": "fraction_addition",
            "instruction": q_bengali,
            "context": f"[RESULT] {tool_result}",
            "response": response
        })
    return examples


def generate_interest_examples(count: int = 1500) -> list:
    examples = []
    principals = [1000, 2000, 3000, 5000, 8000, 10000, 15000]
    rates = [5, 6, 8, 10, 12, 15]
    times = [1, 2, 3, 4, 5]

    for _ in range(count):
        P = random.choice(principals)
        r = random.choice(rates)
        n = random.choice(times)
        is_compound = random.choice([True, False])

        if not is_compound:
            res = MathCalculator.simple_interest(P, r, n)
            q = f"{P} টাকায় বার্ষিক {r}% মুনাফায় {n} বছরের সরল মুনাফা কত?"
            tool_result = f"I = Prn, মুনাফা = {res['interest']} টাকা"
            response = f"সরল মুনাফার সূত্র I = Prn ব্যবহার করে পাই: I = {P} × ({r}/100) × {n} = {res['interest']} টাকা। মোট সবৃদ্ধিমূল = {res['total_amount']} টাকা।"
            mode = "simple_interest"
        else:
            res = MathCalculator.compound_interest(P, r, n)
            q = f"{P} টাকায় বার্ষিক {r}% চক্রবৃদ্ধি মুনাফায় {n} বছর পর সবৃদ্ধিমূল কত?"
            tool_result = f"C = P(1+r)^n, সবৃদ্ধিমূল = {res['compound_amount']} টাকা"
            response = f"চক্রবৃদ্ধি মূলধনের সূত্র C = P(1 + r)^n অনুযায়ী: C = {P} × (1 + {r}/100)^{n} = {res['compound_amount']} টাকা। চক্রবৃদ্ধি মুনাফা = {res['compound_interest']} টাকা।"
            mode = "compound_interest"

        examples.append({
            "mode": "tool_result",
            "category": mode,
            "instruction": q,
            "context": f"[RESULT] {tool_result}",
            "response": response
        })
    return examples


def generate_geometry_and_series_examples(count: int = 2000) -> list:
    examples = []
    # Pythagoras
    pyth_triples = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15)]
    for _ in range(count // 2):
        a, b, c = random.choice(pyth_triples)
        res = MathCalculator.pythagoras(a=float(a), b=float(b))
        q = f"একটি সমকোণী ত্রিভুজের ভূমি {a} সেমি এবং লম্ব {b} সেমি হলে অতিভুজ কত?"
        tool_result = f"c² = a² + b² = {a**2} + {b**2} = {c**2}, অতিভুজ = {c} সেমি"
        response = f"পিথাগোরাসের উপপাদ্য অনুসারে, অতিভুজ c = √(a² + b²) = √({a}² + {b}²) = √({a**2 + b**2}) = {c} সেমি।"
        examples.append({
            "mode": "tool_result",
            "category": "pythagoras",
            "instruction": q,
            "context": f"[RESULT] {tool_result}",
            "response": response
        })

    # Series sum
    series_limits = [10, 20, 50, 100, 200]
    for _ in range(count // 2):
        n = random.choice(series_limits)
        res = MathCalculator.series_sum(1, n)
        q = f"১ থেকে {n} পর্যন্ত স্বাভাবিক সংখ্যার যোগফল কত?"
        tool_result = f"Sₙ = n(n+1)/2, যোগফল = {res['sum']}"
        response = f"১ থেকে {n} পর্যন্ত ক্রমিক সংখ্যার সমষ্টির সূত্র Sₙ = n(n + 1) / 2 ব্যবহার করে পাই: Sₙ = {n} × ({n} + 1) / 2 = {res['sum']}।"
        examples.append({
            "mode": "tool_result",
            "category": "series_sum",
            "instruction": q,
            "context": f"[RESULT] {tool_result}",
            "response": response
        })
    return examples


def main():
    random.seed(42)
    fractions = generate_fraction_examples(1500)
    interest = generate_interest_examples(1500)
    geom_series = generate_geometry_and_series_examples(2000)
    all_data = fractions + interest + geom_series
    random.shuffle(all_data)

    out_file = DATA_DIR / "math_verbalization.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in all_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_data)} synthetic math training examples: {out_file}")


if __name__ == "__main__":
    main()
