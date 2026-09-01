"""
SS Tutor BD - Offline SQLite FTS5 Knowledge Indexer
Builds compact, high-performance Full-Text Search (FTS5) indexes for offline retrieval on Android and desktop.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.rag.schema import KnowledgeChunk


class KnowledgeIndexer:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            # Main storage table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    pack_id TEXT,
                    class_level TEXT,
                    subject TEXT,
                    book_title TEXT,
                    chapter_id TEXT,
                    chapter_title TEXT,
                    section_id TEXT,
                    section_title TEXT,
                    content_text TEXT,
                    content_type TEXT,
                    keywords_json TEXT,
                    metadata_json TEXT
                )
            """)

            # FTS5 Full-Text Search virtual table with unicode61 tokenizer
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_knowledge USING fts5(
                    chunk_id UNINDEXED,
                    chapter_title,
                    section_title,
                    content_text,
                    keywords,
                    tokenize = 'unicode61'
                )
            """)

    def insert_chunk(self, chunk: KnowledgeChunk):
        """Inserts a single chunk into both primary and FTS5 tables."""
        keywords_str = " ".join(chunk.keywords)
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO knowledge_chunks (
                    chunk_id, pack_id, class_level, subject, book_title,
                    chapter_id, chapter_title, section_id, section_title,
                    content_text, content_type, keywords_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id, chunk.pack_id, chunk.class_level, chunk.subject, chunk.book_title,
                chunk.chapter_id, chunk.chapter_title, chunk.section_id, chunk.section_title,
                chunk.content_text, chunk.content_type, json.dumps(chunk.keywords, ensure_ascii=False),
                json.dumps(chunk.metadata, ensure_ascii=False)
            ))

            self.conn.execute("""
                INSERT OR REPLACE INTO fts_knowledge (
                    chunk_id, chapter_title, section_title, content_text, keywords
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id, chunk.chapter_title, chunk.section_title or "",
                chunk.content_text, keywords_str
            ))

    def insert_chunks(self, chunks: List[KnowledgeChunk]):
        """Batch inserts multiple chunks in a single transaction."""
        for c in chunks:
            self.insert_chunk(c)

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM knowledge_chunks")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
