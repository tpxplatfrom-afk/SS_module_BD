"""
SS Tutor BD - Semantic Content Chunker
Performs deterministic, semantic-boundary chunking on NCTB Bengali curriculum materials.
Preserves formula integrity, worked examples, and section hierarchies.
"""

import re
from typing import List, Dict, Any, Optional
from core.rag.schema import KnowledgeChunk


def extract_keywords_from_bengali(text: str) -> List[str]:
    """Extracts key indexing words (Bengali nouns, math terms, formulas)."""
    # Find mathematical formulas like (a+b)^2, a^2+2ab+b^2, F=ma, V=IR
    math_terms = re.findall(r"[a-zA-Z0-9_\^\+\-\*\/\=\(\)\.]{2,}", text)
    # Find Bengali key terms (words with 3+ characters)
    bn_words = re.findall(r"[\u0980-\u09FF]{3,}", text)
    # Common stop words to exclude
    stopwords = {"এবং", "অথবা", "কিন্তু", "হলে", "হলো", "হবে", "একটি", "দুটি", "তিনটি", "জন্য", "থেকে", "দিয়ে", "করা"}
    filtered_bn = [w for w in bn_words if w not in stopwords]
    
    combined = list(dict.fromkeys(filtered_bn + math_terms))
    return combined[:15]


def chunk_nctb_document(
    pack_id: str,
    class_level: str,
    subject: str,
    book_title: str,
    chapter_id: str,
    chapter_title: str,
    document_markdown: str,
    base_metadata: Optional[Dict[str, Any]] = None
) -> List[KnowledgeChunk]:
    """
    Chunks a structured Markdown textbook chapter into deterministic KnowledgeChunks.
    Splits by Sections (## / ###), Worked Examples (উদাহরণ), and Exercises (অনুশীলনী).
    """
    metadata = base_metadata or {}
    chunks: List[KnowledgeChunk] = []

    # Split document by Markdown headers or section markers
    section_blocks = re.split(r"(?m)^(?=##\s+)", document_markdown)
    chunk_counter = 1

    for block in section_blocks:
        block_clean = block.strip()
        if not block_clean:
            continue

        # Extract section header if present
        header_match = re.match(r"^##\s+(.*?)$", block_clean, re.MULTILINE)
        section_title = header_match.group(1).strip() if header_match else "সাধারণ আলোচনা"
        section_id = f"SEC-{chunk_counter:02d}"

        # Subdivide block by Worked Examples (উদাহরণ) or Exercises if large
        sub_blocks = re.split(r"(?m)(?=^(?:উদাহরণ|অনুশীলনী|সূত্রাবলী)\s*[\d০-৯\.]*:?)", block_clean)

        for sub in sub_blocks:
            sub_clean = sub.strip()
            if not sub_clean or len(sub_clean) < 15:
                continue

            # Determine content type
            content_type = "general"
            if re.search(r"^(?:উদাহরণ|Example)", sub_clean, re.MULTILINE):
                content_type = "worked_example"
            elif re.search(r"^(?:অনুশীলনী|Exercise)", sub_clean, re.MULTILINE):
                content_type = "exercise"
            elif re.search(r"(?:সূত্র|সমীকরণ|Formula)", sub_clean):
                content_type = "formula"
            elif re.search(r"(?:সংজ্ঞা|কাকে বলে|Definition)", sub_clean):
                content_type = "definition"

            chunk_id = f"{pack_id}-{chapter_id.lower()}-{section_id.lower()}-c{chunk_counter:03d}"
            keywords = extract_keywords_from_bengali(sub_clean)

            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                pack_id=pack_id,
                class_level=class_level,
                subject=subject,
                book_title=book_title,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                section_id=section_id,
                section_title=section_title,
                content_text=sub_clean,
                content_type=content_type,
                keywords=keywords,
                metadata=metadata
            )
            chunks.append(chunk)
            chunk_counter += 1

    return chunks
