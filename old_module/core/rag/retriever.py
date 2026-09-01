"""
SS Tutor BD - Offline Knowledge Retriever
Performs ranked full-text search (BM25) over indexed NCTB curriculum packs with Bengali query normalization.
"""

import re
import sqlite3
import time
from typing import List, Dict, Any, Optional
from core.rag.schema import KnowledgeChunk
from core.rag.indexer import KnowledgeIndexer

# Common Bengali-English synonyms and mathematical transliterations
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
    "অ্যালজেব্রিক": ["বীজগণিতীয়", "algebraic"],
    "পাই": ["৩.১৪১৬", "২২/৭", "π"],
    "বৃত্ত": ["পরিধি", "বৃত্তের", "ব্যাসার্ধ"],
    "ক্ষেত্রফল": ["ক্ষেত্রফল", "পরিমাপ"]
}

STOPWORDS = {
    "জন্য", "কী", "কি", "কীভাবে", "কেমন", "করে", "হলে", "কত",
    "পদ্ধতি", "হিসাব", "উপায়", "নির্ণয়", "করব", "করা", "থেকে",
    "এবং", "অথবা", "কিন্তু", "একটি", "দুটি", "তিনটি", "বলুন", "বলো", "দাও",
    "তার", "তাদের", "মধ্যে", "কোন", "কোনো", "কীসে", "পান", "সাল", "সালে",
    "হয়", "হলো", "হবে", "আছে", "ছিল", "করুন", "করো"
}


class KnowledgeRetriever:
    def __init__(self, indexer: KnowledgeIndexer):
        self.indexer = indexer

    def _normalize_query(self, query: str) -> str:
        """Extracts intact Bengali words, mathematical expressions, and expands synonyms for FTS5."""
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

        unique_tokens = list(dict.fromkeys(expanded))
        if not unique_tokens:
            return '""'

        clauses = []
        for t in unique_tokens:
            clauses.append(f'"{t}"')
            if len(t) >= 4:
                clauses.append(f'"{t}"*')
            if t.endswith("ফল") and len(t) > 3:
                clauses.append(f'"{t[:-2]}"')
                clauses.append(f'"{t[:-2]}"*')

        return " OR ".join(clauses)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        class_filter: Optional[str] = None,
        subject_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant knowledge chunks for the given query using FTS5 BM25 ranking.
        Returns list of dicts with 'chunk', 'score', and 'retrieval_time_ms'.
        """
        t0 = time.perf_counter()
        fts_query = self._normalize_query(query)

        sql = """
            SELECT 
                k.*,
                bm25(fts_knowledge) as rank_score,
                snippet(fts_knowledge, 3, '<b>', '</b>', '...', 20) as matched_snippet
            FROM fts_knowledge
            JOIN knowledge_chunks k ON fts_knowledge.chunk_id = k.chunk_id
            WHERE fts_knowledge MATCH ?
        """
        params = [fts_query]

        if class_filter:
            sql += " AND k.class_level = ?"
            params.append(class_filter)
        if subject_filter:
            sql += " AND k.subject = ?"
            params.append(subject_filter)

        sql += " ORDER BY rank_score ASC LIMIT ?"
        params.append(top_k)

        try:
            cursor = self.indexer.conn.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            rows = []

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        results = []
        for r in rows:
            chunk = KnowledgeChunk(
                chunk_id=r["chunk_id"],
                pack_id=r["pack_id"],
                class_level=r["class_level"],
                subject=r["subject"],
                book_title=r["book_title"],
                chapter_id=r["chapter_id"],
                chapter_title=r["chapter_title"],
                section_id=r["section_id"],
                section_title=r["section_title"],
                content_text=r["content_text"],
                content_type=r["content_type"],
                keywords=[],
                metadata={}
            )
            results.append({
                "chunk": chunk,
                "score": round(abs(r["rank_score"]), 3),
                "matched_snippet": r["matched_snippet"] if "matched_snippet" in r.keys() else "",
                "retrieval_time_ms": duration_ms
            })

        return results
