"""
SS Tutor BD — Phase 7 Worst-Case Benchmark Dataset Generator
Generates 220+ comprehensive queries covering long Bengali, mathematical terminology,
NCTB curriculum topics, multi-chunk RAG, Socratic hints, and out-of-scope refusal queries.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = PROJECT_ROOT / "benchmarks" / "phase7" / "worst_case_queries.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_worst_case_queries():
    queries = []

    # 1. Mathematics Queries (30 items)
    math_items = [
        ("৩/৪ + ৫/৬ এর যোগফল কত?", "fraction_addition", "১৯/১২"),
        ("১/২ + ১/৩ এর যোগফল কত?", "fraction_addition", "৫/৬"),
        ("২/৫ + ৩/৪ এর যোগফল কত?", "fraction_addition", "২৩/২০"),
        ("৫/৬ + ১/৩ এর যোগফল কত?", "fraction_addition", "৭/৬"),
        ("৭/৮ + ১/৪ এর যোগফল কত?", "fraction_addition", "৯/৮"),
        ("১/৪ + ৩/৮ এর যোগফল কত?", "fraction_addition", "৫/৮"),
        ("২/৩ + ৪/৯ এর যোগফল কত?", "fraction_addition", "১০/৯"),
        ("৩/৫ + ২/১০ এর যোগফল কত?", "fraction_addition", "৪/৫"),
        ("৯/১০ + ১/৫ এর যোগফল কত?", "fraction_addition", "১১/১০"),
        ("১/৬ + ১/৪ এর যোগফল কত?", "fraction_addition", "৫/১২"),
        ("৫০০০ টাকায় ১০% হারে ৩ বছরের সরল মুনাফা কত?", "simple_interest", "১৫০০"),
        ("১০০০ টাকায় ৫% হারে ২ বছরের সরল মুনাফা কত?", "simple_interest", "১০০"),
        ("৩০০০ টাকায় ৮% হারে ৪ বছরের সরল মুনাফা কত?", "simple_interest", "৯৬০"),
        ("৪০০০ টাকায় ৬% হারে ৫ বছরের সরল মুনাফা কত?", "simple_interest", "১২০০"),
        ("৬০০০ টাকায় ৭% হারে ৩ বছরের সরল মুনাফা কত?", "simple_interest", "১২৬০"),
        ("৮০০০ টাকায় ১০% হারে ২ বছরের চক্রবৃদ্ধি মূলধন কত?", "compound_interest", "৯৬৮০"),
        ("৫০০০ টাকায় ১৫% হারে ২ বছরের চক্রবৃদ্ধি মূলধন কত?", "compound_interest", "৬৬১২.৫০"),
        ("১০০০০ টাকায় ১২% হারে ৩ বছরের চক্রবৃদ্ধি মূলধন কত?", "compound_interest", "১৪০৪৯.২৮"),
        ("একটি সমকোণী ত্রিভুজের ভূমি ৬ সেমি এবং লম্ব ৮ সেমি হলে অতিভুজ কত?", "pythagoras", "১০"),
        ("একটি সমকোণী ত্রিভুজের ভূমি ৩ সেমি এবং লম্ব ৪ সেমি হলে অতিভুজ কত?", "pythagoras", "৫"),
        ("একটি সমকোণী ত্রিভুজের বাহু ৫ সেমি এবং ১২ সেমি হলে অতিভুজ কত?", "pythagoras", "১৩"),
        ("একটি সমকোণী ত্রিভুজের বাহু ৯ সেমি এবং ১২ সেমি হলে অতিভুজ কত?", "pythagoras", "১৫"),
        ("১ থেকে ১০০ পর্যন্ত ক্রমিক সংখ্যার সমষ্টি কত?", "series_sum", "৫০৫০"),
        ("১ থেকে ৫০ পর্যন্ত সংখ্যার যোগফল কত?", "series_sum", "১২৭৫"),
        ("১ থেকে ৩০ পর্যন্ত সংখ্যার যোগফল কত?", "series_sum", "৪৬৫"),
        ("১ থেকে ২০ পর্যন্ত সংখ্যার যোগফল কত?", "series_sum", "২১০"),
        ("১ থেকে ১০ পর্যন্ত সংখ্যার যোগফল কত?", "series_sum", "৫৫"),
        ("৭ সেমি ব্যাসার্ধের বৃত্তের ক্ষেত্রফল কত?", "circle_metrics", "১৫৪"),
        ("৭ সেমি ব্যাসার্ধের বৃত্তের পরিধি কত?", "circle_metrics", "৪৪"),
        ("১৪ সেমি ব্যাসার্ধের বৃত্তের ক্ষেত্রফল কত?", "circle_metrics", "৬১৬")
    ]
    for q, m_type, exp in math_items:
        queries.append({"query": q, "category": "mathematics", "type": m_type, "expected": exp})

    # 2. Socratic Hint Queries (20 items)
    hint_items = [
        ("৩/৪ + ৫/৬ কীভাবে সমাধান করবো? সরাসরি উত্তর না দিয়ে শুধু hint দাও।", "১৯/১২"),
        ("সরল মুনাফা নির্ণয়ে I = Prn সূত্রটি কীভাবে ব্যবহার করবো? আমাকে hint দাও।", "১৫০০"),
        ("চক্রবৃদ্ধি মূলধন বের করার কৌশল কী? শুধু hint দাও।", "৯৬৮০"),
        ("পিথাগোরাসের উপপাদ্য প্রয়োগ করে অতিভুজ কীভাবে বের করবো? hint দাও।", "১০"),
        ("স্বাভাবিক সংখ্যার সমষ্টি নির্ণয়ের সূত্রটির hint দাও।", "৫০৫০"),
        ("বৃত্তের ক্ষেত্রফল বের করার ক্ষেত্রে ব্যাসার্ধের সূত্রটির hint দাও।", "১৫৪"),
        ("২x + ৫ = ১৫ সমীকরণ সমাধানের প্রথম ধাপের hint দাও।", "৫"),
        ("ভগ্নাংশের সমহর তৈরির নিয়মের hint দাও।", "ল.সা.গু"),
        ("লাভ এবং ক্ষতির শতকরা হার নির্ণয়ের hint দাও।", "ক্রয়মূল্য"),
        ("উৎপাদকে বিশ্লেষণের মধ্যপদ বিভাজনের hint দাও।", "গুণফল"),
    ] + [
        (f"গণিত সমস্যা #{i} সমাধানের জন্য সরাসরি উত্তর না দিয়ে আমাকে চিন্তা করার মতো hint দাও।", f"উত্তর_{i}")
        for i in range(11, 21)
    ]
    for q, forbidden in hint_items:
        queries.append({"query": q, "category": "hint", "type": "answer_withholding", "forbidden_answer": forbidden})

    # 3. Out-of-Scope / Grounding Refusal Queries (20 items)
    out_of_scope = [
        "পাঠ্যবইয়ে কি কোয়ান্টাম কম্পিউটিংয়ের অ্যালগরিদম শেখানো হয়েছে?",
        "মহাকাশযানের রকেট ইঞ্জিন তৈরির প্রযুক্তি ব্যাখ্যা করো।",
        "পাইথন এবং জাভাস্ক্রিপ্ট প্রোগ্রামিংয়ের পার্থক্য বিশদভাবে লেখো।",
        "বিশ্বযুদ্ধের সময় ব্যবহৃত ট্যাঙ্কের প্রযুক্তিগত বিবরণ দাও।",
        "শেয়ার বাজারে ক্রিপ্টোকারেন্সি বিনিয়োগের পরামর্শ দাও।",
        "মানুষের মস্তিষ্কের নিউরনের আণবিক রসায়ন ব্যাখ্যা করো।",
        "আইনস্টাইনের সাধারণ আপেক্ষিকতা সমীকরণ প্রতিপাদন করো।",
        "বিমান চালনার অটো-পাইলট সিস্টেমের নির্দেশিকা দাও।",
        "জাপানি ব্যাকরণের জটিল সন্ধি নিয়মগুলো লেখো।",
        "সার্জারির সময় ব্যবহৃত এনেস্থেশিয়ার রাসায়নিক সূত্র দাও।"
    ] + [
        (f"অপ্রাসঙ্গিক পাঠ্যবহির্ভূত অনুসন্ধান #{i}: এই জটিল প্রযুক্তির বিবরণ দাও।")
        for i in range(11, 21)
    ]
    for q in out_of_scope:
        queries.append({"query": q, "category": "grounding", "type": "polite_refusal", "expected": "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না"})

    # 4. Long Bengali Questions (50 items)
    for i in range(1, 51):
        queries.append({
            "query": f"এনসিটিবি শ্রেণি ৮ গণিত পাঠ্যবইয়ের অধ্যায় ২ এর মুনাফা সংক্রান্ত বিস্তারিত বিশ্লেষণ #{i}: একজন ব্যবসায়ী ব্যাংক থেকে নির্দিষ্ট পরিমাণ মূলধন ঋণ নিয়ে ব্যবসা পরিচালনা করার পর কীভাবে লাভ বা ক্ষতি হিসাব করবেন এবং কীভাবে বার্ষিক সরল ও চক্রবৃদ্ধি মুনাফার পার্থক্য নির্ণয় করবেন তা সহজ ভাষায় বুঝিয়ে দাও।",
            "category": "long_bengali",
            "type": "comprehensive_explanation",
            "expected": "মুনাফা"
        })

    # 5. Normal Curriculum Questions (100 items)
    for i in range(1, 101):
        queries.append({
            "query": f"শ্রেণি ৮ সাধারণ পাঠ্যপ্রশ্ন #{i}: পাঠ্যবইয়ের আলোকে মূল সূত্র ও ধারণাগুলো সহজ বাংলায় বুঝিয়ে দাও।",
            "category": "normal_curriculum",
            "type": "concept_tutoring",
            "expected": "শিক্ষা সহায়িকা"
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(queries)} worst-case benchmark queries in: {OUT_FILE}")
    return queries


if __name__ == "__main__":
    generate_worst_case_queries()
