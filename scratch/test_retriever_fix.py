import sys
import re
import sqlite3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYNONYMS = {
    "প্রাইম": "মৌলিক",
    "সুদ": "মুনাফা",
    "কম্পাউন্ড": "চক্রবৃদ্ধি",
    "স্কয়ার": "বর্গ",
    "ফ্যাক্টরাইজেশন": "উৎপাদক",
    "এলিমিনেশন": "অপনয়ন",
    "সাবস্টিটিউশন": "প্রতিস্থাপন",
    "সমষ্টি": "যোগফল",
    "পেরিমিটার": "পরিসীমা",
    "এরিয়া": "ক্ষেত্রফল"
}

STOPWORDS = {
    "জন্য", "কী", "কি", "কীভাবে", "কেমন", "করে", "হলে", "কত",
    "পদ্ধতি", "হিসাব", "উপায়", "নির্ণয়", "করব", "করা", "থেকে",
    "এবং", "অথবা", "কিন্তু", "একটি", "দুটি", "তিনটি", "বলুন", "বলো", "দাও"
}

def normalize_query(query: str) -> str:
    cleaned = re.sub(r"[^\w\s\+\-\*\/\^\=]", " ", query)
    words = cleaned.split()
    expanded_terms = []
    
    for w in words:
        w_lower = w.lower()
        if w_lower in SYNONYMS:
            expanded_terms.append(SYNONYMS[w_lower])
            expanded_terms.append(w)
        elif w not in STOPWORDS and len(w) > 1:
            expanded_terms.append(w)

    if not expanded_terms:
        expanded_terms = [w for w in words if len(w) > 1]

    # Create FTS5 query with OR
    fts_terms = []
    for t in expanded_terms:
        # If Bengali word, use exact and prefix
        fts_terms.append(f'"{t}"')
        if len(t) >= 3:
            fts_terms.append(f'"{t}"*')

    return " OR ".join(fts_terms)

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
    fts_q = normalize_query(q)
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
    print(f"FTS Query: {fts_q}")
    print(f"Results ({len(rows)}):")
    for r in rows:
        print(f"  [{r['chapter_id']}] {r['chapter_title']} (score: {round(r['rank_score'], 3)})")

conn.close()
