import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(r'ss_bangladesh_nano_android_module\THSA-2B V1')))
from src.engine.universal_tutor_engine import UniversalTutorEngine

engine = UniversalTutorEngine()

prompts = [
    ("1. Vocab: Single word meaning (user wants to learn 'good')", "What does 'good' mean?"),
    ("2. Vocab: Bengali asks English word meaning", "beautiful মানে কি?"),
    ("3. Tense: Explain present tense step by step", "Present tense ki? step by step shekao"),
    ("4. Grammar: Explain the difference between 'is are am was were'", "is are am was were er difference koi?"),
    ("5. Sentence: How to make a simple English sentence", "simple english sentence kivabe banabo?"),
    ("6. Spoken: User wants to speak English with module", "tumi ki amar shathe english e kotha bolbe?"),
    ("7. Vocabulary: How do I improve my English vocabulary?", "how can i improve my english vocabulary?"),
    ("8. Writing: User wants to write a paragraph about trees", "trees er upore ekta paragraph likhe dao"),
    ("9. Grammar: What is noun, pronoun, verb?", "noun pronoun verb ki?"),
    ("10. Tense: Past tense rules and examples", "past tense er rules and examples dao"),
    ("11. Daily use: Teach me 5 daily use English sentences", "daily life e use hoy emon 5 ta english sentence shekao"),
    ("12. Proficiency Check: How much English do I know?", "I want to talk to you in english. how much english do i know?"),
]

print('=' * 88)
print('ENGLISH LEARNING AUDIT: 12-PROMPT CONVERSATIONAL EXCELLENCE TEST')
print('=' * 88)

for label, q in prompts:
    res = engine.ask(q)
    txt = res['text']
    snippet = txt[:200].replace('\n', ' ')
    print(f'\n[{label}]')
    print(f'Prompt : "{q}"')
    print(f'Output : {snippet}...')
    print('-' * 88)
