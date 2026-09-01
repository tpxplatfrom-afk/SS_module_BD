"""
SS Tutor BD - Class 8 Mathematics Content Ingestion Script
Parses manifest.json and Markdown chapters, chunks them semantically, and builds SQLite FTS5 index.
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.chunker import chunk_nctb_document
from core.rag.indexer import KnowledgeIndexer

PACK_DIR = PROJECT_ROOT / "packs" / "class8_math"
MANIFEST_PATH = PACK_DIR / "manifest.json"
DB_PATH = PACK_DIR / "index.db"


def ingest_class8_math():
    print(f"[Content Ingestion] Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Remove old index if exists
    if DB_PATH.exists():
        DB_PATH.unlink()

    indexer = KnowledgeIndexer(str(DB_PATH))
    total_chunks = 0

    for ch in manifest.get("chapters", []):
        ch_file = PACK_DIR / ch["file"]
        if not ch_file.exists():
            print(f"Warning: File {ch_file} not found. Skipping.")
            continue

        with open(ch_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        chunks = chunk_nctb_document(
            pack_id=manifest["pack_id"],
            class_level=manifest["class_level"],
            subject=manifest["subject"],
            book_title="NCTB Class 8 Mathematics",
            chapter_id=ch["id"],
            chapter_title=ch["title"],
            document_markdown=md_content,
            base_metadata={"chapter_id": ch["id"], "file": ch["file"]}
        )

        indexer.insert_chunks(chunks)
        total_chunks += len(chunks)
        print(f"  Ingested {ch['id']} ({ch['title']}): {len(chunks)} chunks")

    print(f"\n[Ingestion Complete] Total Chunks Indexed: {total_chunks}")
    print(f"SQLite FTS5 Database: {DB_PATH} ({round(DB_PATH.stat().st_size / 1024, 2)} KB)")
    indexer.close()
    return total_chunks


if __name__ == "__main__":
    ingest_class8_math()
