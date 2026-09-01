import sys
import re
import sqlite3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYNONYMS = {
    "প্রাইম": ["মৌলিক", "prime"],
    "সুদ": ["মুনাফা", "সুদের", "মুনাফার"],
    "সুদের": ["মুনাফা", "মুনাফার"],
    "কম্পাউন্ড": ["চক্রবৃদ্ধি", "compound"],
    "ইন্টারেস্ট": ["মুনাফা", "interest"],
    "স্কয়ার": ["বর্গ", "square"],
    "হোল": ["বর্গ"],
    "ফ্যাক্টরাইজেশন": ["উৎপাদক", "উৎপাদকে", "বিশ্লেষণ"],
    "মিডল": ["মধ্যপদ", "middle"],
    "টার্ম": ["বিভাজন", "term"],
    "এলিমিনেশন": ["অপনয়ন", "elimination"],
    "সাবস্টিটিউশন": ["প্রতিস্থাপন", "substitution"],
    "সমষ্টি": ["যোগফল", "সমষ্টি"],
    "পেরিমিটার": ["পরিসীমা", "perimeter"],
    "এরিয়া": ["ক্ষেত্রফল", "area"],
    "দৈঘ্য": ["দৈর্ঘ্য"],
    "প্রস্থের": ["প্রস্থ"],
    "সঙ্খ্যা": ["সংখ্যা"],
    "বরগ": ["বর্গ"],
    "সুত্র": ["সূত্র"],
    "প্রার্থক্য": ["পার্থক্য"],
    "এলজেব্রিক": ["বীজগণিতীয়", "algebraic"],
    "অ্যালজেব্রিক": ["বীজগণিতীয়", "algebraic"]
}

STOPWORDS = {
    "জন্য", "কী", "কি", "কীভাবে", "কেমন", "করে", "হলে", "কত",
    "পদ্ধতি", "হিসাব", "উপায়", "নির্ণয়", "করব", "করা", "থেকে",
    "এবং", "অথবা", "কিন্তু", "একটি", "দুটি", "তিনটি", "বলুন", "বলো", "দাও",
    "তার", "তাদের", "মধ্যে", "কোন", "কোনো", "কীসে", "পান", "সাল", "সালে",
    "হয়", "হলে", "হলো", "হবে", "আছে", "ছিল", "করুন", "করো"
}

def extract_bengali_tokens(query: str):
    # Match intact Bengali words, English words, math expressions
    raw_tokens = re.findall(r"[\u0980-\u09FFa-zA-Z0-9_\+\-\*\/\^\=]+", query)
    expanded = []
    
    for tok in raw_tokens:
        tok_low = tok.lower()
        if tok_low in SYNONYMS:
            expanded.extend(SYNONYMS[tok_low])
            expanded.append(tok)
        elif tok not in STOPWORDS and len(tok) > 1:
            expanded.append(tok)
            
    if not expanded:
        expanded = [t for t in raw_tokens if len(t) > 1]
        
    return list(dict.fromkeys(expanded))

def build_fts_query(tokens):
    clauses = []
    for t in tokens:
        # Exact token match
        clauses.append(f'"{t}"')
    return " OR ".join(clauses)

conn = sqlite3.connect("packs/class8_math/index.db")
conn.row_factory = sqlite3.Row

test_queries = [
    "প্রাইম নাম্বার খোঁজার জন্য গ্রিক বিজ্ঞানীর ছাঁকনি পদ্ধতি",
    "১ থেকে ১০০ পর্যন্ত ক্রমিক সংখ্যার সমষ্টি কেমন করে বের করব",
    "ব্যাংক থেকে ঋণ নিলে সুদের হিসাব কীভাবে করে",
    "বছরের পর বছর মূলধনের সাথে লাভ যোগ হয়ে বৃদ্ধি পাওয়া সুদ",
    "জমির চারপাশের সীমানা বা ঘেরের দৈর্ঘ্য মাপার সূত্র",
    "বর্গ রাশির সূত্র ভেঙে গুণফল আকারে রূপান্তর করার কৌশল",
    "বীজগণিতের রাশিকে ভেঙে খণ্ড খণ্ড রাশিতে গুণ আকারে প্রকাশ",
    "দুইটা অ্যালজেব্রিক ভগ্নাংশ এক সাথে যুক্ত করার উপায়",
    "একটি সমীকরণ থেকে মান নিয়ে অন্যটায় বসিয়ে এক্স ওয়াই বের করা",
    "সমকোণী ত্রিভুজের তিন বাহুর মধ্যকার সম্পর্কের প্রাচীন সূত্র"
]

for q in test_queries:
    tokens = extract_bengali_tokens(q)
    fts_q = build_fts_query(tokens)
    sql = """
        SELECT 
            k.chunk_id, k.chapter_id, k.chapter_title,
            bm25(fts_knowledge) as rank_score
        FROM fts_knowledge
        JOIN knowledge_chunks k ON fts_knowledge.chunk_id = k.chunk_id
        WHERE fts_knowledge MATCH ?
        ORDER BY rank_score ASC LIMIT 3
    """
    cursor = conn.execute(sql, [fts_q])
    rows = cursor.fetchall()
    print(f"\nQuery: {q}")
    print(f"Tokens: {tokens}")
    print(f"FTS Query: {fts_q}")
    print(f"Results ({len(rows)}):")
    for r in rows:
        print(f"  [{r['chapter_id']}] {r['chapter_title']} (score: {round(r['rank_score'], 3)})")

conn.close()
