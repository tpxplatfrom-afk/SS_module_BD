"""
Runs user request for Class 9 Math Chapter 3 Ex 3.1 Q 2 (খ, গ, ঘ)
and saves output into a clean Markdown / Text document for verification against the guide book.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.engine.universal_tutor_engine import UniversalTutorEngine

engine = UniversalTutorEngine()
user_prompt = "Write down the numbers 'খ' 'গ' 'ঘ' in Exercise 3.1 of Chapter 3 of Class 9."
res = engine.ask(user_prompt)

# Save to file
out_file = MODULE_ROOT / "class9_math_ex3_1_solutions.md"
out_file.write_text(res["markdown"], encoding="utf-8")

print(f"[SUCCESS] Solutions saved to: {out_file}")
print("=" * 80)
print(res["markdown"])
