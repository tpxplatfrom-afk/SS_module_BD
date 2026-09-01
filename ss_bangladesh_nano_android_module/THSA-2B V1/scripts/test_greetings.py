import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(r'ss_bangladesh_nano_android_module\THSA-2B V1')))
from src.engine.universal_tutor_engine import UniversalTutorEngine

engine = UniversalTutorEngine()

# All the "mature chat" variations that the old system would have failed on
hard_tests = [
    "hi there",
    "hey!",
    "হাই ভাই",
    "হ্যালো ভাই",
    "কেমন আছেন?",
    "তুমি কে?",
    "তুমি কি পারো?",
    "অনেক ধন্যবাদ",
    "আচ্ছা বাই",
    "see you",
    "পড়তে ভালো লাগছে না",
    "অংক বুঝি না",
    "কাল পরীক্ষা",
    "মাথা ব্যথা করছে",
    "তুমি কি আমার বন্ধু?",
    "সাহায্য লাগবে",
    "ইন্টারনেট ছাড়া কাজ করো?",
    "এখন পড়তে বসব",
]

print("=" * 65)
print("MATURE CONVERSATIONAL HANDLER - EXTENDED TEST")
print("=" * 65)
passed = 0
for t in hard_tests:
    r = engine.ask(t)
    txt = r["text"]
    hit = txt != r["markdown"] or len(txt) < 200  # Not a math/curriculum response
    # Check it's NOT a math fallback
    is_conv = "গণিত সমাধান" not in txt and "অনুশীলনী" not in txt and "NCTB" not in txt
    status = "✅ PASS" if is_conv else "❌ FAIL (went to curriculum engine)"
    if is_conv:
        passed += 1
    print(f"{status} [{t}]")
    print(f"         → {txt[:90]}")
    print()

print(f"\n{'='*65}")
print(f"RESULT: {passed}/{len(hard_tests)} PASSED")
print(f"{'='*65}")
