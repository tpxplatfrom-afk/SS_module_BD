import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(r'ss_bangladesh_nano_android_module\THSA-2B V1')))
from src.engine.universal_tutor_engine import UniversalTutorEngine

engine = UniversalTutorEngine()

noisy_prompts = [
    ('1. Misspelled CQ Rules', 'সৃজনশিল লেখার নিয়ম টা কি?'),
    ('2. Slang / Colloquial Airplane', 'বিমান ক্যামনে উড়ে ভাই?'),
    ('3. Super Short Math', '৩.১ ২ এর গ'),
    ('4. Spelling Error Pass Hack', 'পরিক্ষায় সহজে পাশ করবার উপায় কি?'),
    ('5. Super Short Science', 'আকাশ নিল কেন'),
    ('6. Banglish / Typo English CV', 'cv english teacher er jonno'),
    ('7. Exam Panic Slang', 'কালকে পরিক্ষা কি আসবে একটু সাজেশন দেন'),
    ('8. Frustrated Attitude', 'ধুর কিছুই পারি না অংক ভুয়া লাগে'),
    ('9. Super Short Grammar', 'right form verbs ৫টা নিয়ম'),
    ('10. Super Short Newton', 'নিউটন ২য় সূত্র')
]

print('=' * 85)
print('AUDIT REPORT: NOISY, MISSPELLED & SHORT REAL-STUDENT PROMPTS')
print('=' * 85)

for label, p in noisy_prompts:
    res = engine.ask(p)
    txt = res['text'][:140].replace('\n', ' ')
    print(f'[{label}]')
    print(f'User Input: \"{p}\"')
    print(f'Output:    {txt}...')
    print('-' * 85)
