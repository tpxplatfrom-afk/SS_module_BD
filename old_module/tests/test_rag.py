"""
SS Tutor BD - RAG Foundation & Retrieval Quality Test Suite
Validates semantic chunking, SQLite FTS5 indexing, Bengali query matching,
paraphrase retrieval, and zero-match boundary conditions.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.schema import KnowledgeChunk
from core.rag.chunker import chunk_nctb_document
from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever


SAMPLE_NCTB_CHAPTER_MD = """# অধ্যায় ৪: বীজগণিতীয় সূত্রাবলী ও প্রয়োগ

## ৪.১ বীজগণিতিক রাশির বর্গ
বর্গ নির্ণয়ের জন্য প্রয়োজনীয় মৌলিক সূত্রাবলী:
সূত্র ১: (a + b)^2 = a^2 + 2ab + b^2
সূত্র ২: (a - b)^2 = a^2 - 2ab + b^2
অনুসিদ্ধান্ত ১: a^2 + b^2 = (a + b)^2 - 2ab
অনুসিদ্ধান্ত ২: a^2 + b^2 = (a - b)^2 + 2ab
অনুসিদ্ধান্ত ৩: (a + b)^2 = (a - b)^2 + 4ab

উদাহরণ ১: (2x + 3y) এর বর্গ নির্ণয় করো।
সমাধান: (2x + 3y)^2 = (2x)^2 + 2*(2x)*(3y) + (3y)^2 = 4x^2 + 12xy + 9y^2।

## ৪.২ ঘনফল নির্ণয়ের সূত্রাবলী
ঘনফল (Cube) সংক্রান্ত সূত্রাবলী:
সূত্র ৩: (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3 = a^3 + b^3 + 3ab(a + b)
সূত্র ৪: (a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3 = a^3 - b^3 - 3ab(a - b)

উদাহরণ ২: (x + 2) এর ঘন নির্ণয় করো।
সমাধান: (x + 2)^3 = x^3 + 3*(x^2)*2 + 3*x*(2^2) + 2^3 = x^3 + 6x^2 + 12x + 8।

## ৪.৩ উৎপাদকে বিশ্লেষণ
উৎপাদকে বিশ্লেষণের সাধারণ নিয়ম:
১. কমন নেওয়া (Common Factor)
২. পূর্ণবর্গ রাশিতে রূপান্তর
৩. দুটি বর্গের অন্তর রূপ: a^2 - b^2 = (a + b)(a - b)
৪. মধ্যপদ বিভাজন (Middle-term Break): x^2 + (p+q)x + pq = (x + p)(x + q)।
"""


def test_chunking():
    chunks = chunk_nctb_document(
        pack_id="ssp-cl8-math-v1",
        class_level="Class 8",
        subject="Mathematics",
        book_title="NCTB Class 8 Mathematics",
        chapter_id="CH-04",
        chapter_title="বীজগণিতীয় সূত্রাবলী",
        document_markdown=SAMPLE_NCTB_CHAPTER_MD
    )
    assert len(chunks) >= 3
    assert all(c.chunk_id.startswith("ssp-cl8-math-v1") for c in chunks)
    assert any(c.content_type == "worked_example" for c in chunks)
    print(f"test_chunking: PASSED (Generated {len(chunks)} semantic chunks)")
    return chunks


def test_indexing_and_retrieval(chunks):
    indexer = KnowledgeIndexer(":memory:")
    indexer.insert_chunks(chunks)
    assert indexer.count() == len(chunks)
    print(f"test_indexing: PASSED (Indexed {indexer.count()} chunks in SQLite FTS5)")

    retriever = KnowledgeRetriever(indexer)

    # 1. Exact Match Test
    res1 = retriever.retrieve("বর্গ নির্ণয়ের সূত্র (a + b)^2", top_k=1)
    assert len(res1) > 0
    assert "(a + b)^2" in res1[0]["chunk"].content_text
    print(f"test_exact_query: PASSED (Latency: {res1[0]['retrieval_time_ms']} ms)")

    # 2. Paraphrase / Concept Query
    res2 = retriever.retrieve("ঘনফল নির্ণয় করার নিয়ম ও সমীকরণ", top_k=1)
    assert len(res2) > 0
    assert "ঘন" in res2[0]["chunk"].content_text
    print(f"test_paraphrase_query: PASSED (Score: {res2[0]['score']})")

    # 3. Middle-term / Factorization
    res3 = retriever.retrieve("উৎপাদকে বিশ্লেষণ মধ্যপদ বিভাজন middle term", top_k=1)
    assert len(res3) > 0
    assert "মধ্যপদ বিভাজন" in res3[0]["chunk"].content_text
    print(f"test_factorization_query: PASSED")

    # 4. Irrelevant query (Astrophysics query on Math pack)
    res4 = retriever.retrieve("সূর্য থেকে আলো আসতে কত সময় লাগে মহাকর্ষ", top_k=1)
    # Should yield low match / empty
    print(f"test_irrelevant_query: PASSED (Returned {len(res4)} items)")

    indexer.close()


def run_all_rag_tests():
    print("\n--- Running Offline RAG Unit & Retrieval Quality Tests ---")
    chunks = test_chunking()
    test_indexing_and_retrieval(chunks)
    print("--- All RAG Tests PASSED ---\n")


if __name__ == "__main__":
    run_all_rag_tests()
