import sys
from pathlib import Path

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(r"c:\Users\User\Desktop\SS_module_BD")
sys.path.insert(0, str(ROOT_DIR / "ss_bangladesh_nano_android_module" / "THSA-2B V1"))

from src.engine.universal_tutor_engine import UniversalTutorEngine

user_prompt = """আমি এখন তোমাকে একটি নতুন, unseen reasoning test দিচ্ছি। কোনো পাঠ্যবই, predefined template, static answer, keyword rule বা hardcoded response ব্যবহার না করে প্রশ্নগুলোর উত্তর দাও।

১) 487 × 36 কত? ধাপে ধাপে হিসাব দেখাও।

২) যদি a + b = 7 এবং ab = 12 হয়, তাহলে a² + b² কত? সূত্র ব্যবহার করে দেখাও।

৩) পানির অণুর H₂O-তে H–O–H bond angle প্রায় 104.5° কেন? সহজ বাংলায় ব্যাখ্যা করো, তবে “electron pair”, “lone pair” এবং “VSEPR” শব্দগুলো English-এ রাখো।

৪) একজন শিক্ষার্থী বলল: “শব্দ vacuum-এর মধ্যেও বাতাসের মতো ছড়িয়ে পড়ে।” তার বক্তব্যটি সঠিক না ভুল? কেন?

৫) এখন আগের চারটি উত্তরের তথ্য ব্যবহার করে ৩ লাইনে একটি সংক্ষিপ্ত বাংলা summary দাও।

গুরুত্বপূর্ণ:

কোনো প্রশ্ন এড়িয়ে যেও না।
যেখানে গণনা আছে সেখানে প্রকৃত হিসাব করো।
কোনো তথ্য নিশ্চিত না হলে তা স্পষ্টভাবে বলো।
প্রশ্নের সঙ্গে সম্পর্কহীন NCTB/template explanation দিও না।
প্রতিটি অংশকে ১–৫ নম্বর দিয়ে আলাদা করো।"""

engine = UniversalTutorEngine()
res = engine.ask(user_prompt)

print("=== RAW ENGINE OUTPUT START ===")
print(res.get("text", ""))
print("=== RAW ENGINE OUTPUT END ===")
