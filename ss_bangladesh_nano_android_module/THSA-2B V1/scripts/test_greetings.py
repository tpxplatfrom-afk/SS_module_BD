import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(r'ss_bangladesh_nano_android_module\THSA-2B V1')))
from src.engine.universal_tutor_engine import UniversalTutorEngine

engine = UniversalTutorEngine()
tests = ['hi', 'hello', 'হ্যালো', 'কেমন আছ', 'ধন্যবাদ', 'bye', 'ok']
print("=" * 60)
print("GREETING HANDLER TEST")
print("=" * 60)
for t in tests:
    r = engine.ask(t)
    txt = r["text"][:100]
    print(f"[PROMPT]: {t}")
    print(f"[REPLY] : {txt}")
    print()
