"""
SS Tutor BD - RAG Content Schema & Data Models
Defines structured knowledge unit specifications compatible with the .ssp pack architecture.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import json


@dataclass
class KnowledgeChunk:
    chunk_id: str             # Stable unique ID, e.g. "nctb-cl8-math-ch04-sec01-c001"
    pack_id: str              # e.g. "ssp-cl8-math-v1"
    class_level: str          # e.g. "Class 8"
    subject: str              # e.g. "Mathematics"
    book_title: str           # e.g. "NCTB Class 8 Mathematics"
    chapter_id: str           # e.g. "CH-04"
    chapter_title: str        # e.g. "বীজগাণিতীয় সূত্রাবলী ও প্রয়োগ (Algebraic Formulae)"
    section_id: Optional[str] # e.g. "SEC-4.1"
    section_title: Optional[str] # e.g. "বীজগাণিতিক রাশির বর্গ"
    content_text: str         # Clean normalized textbook Bengali text
    content_type: str         # "definition", "formula", "worked_example", "exercise", "summary"
    keywords: List[str]       # Key indexing terms in Bengali & English
    metadata: Dict[str, Any]  # Page number, curriculum year, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeChunk":
        return cls(
            chunk_id=data["chunk_id"],
            pack_id=data.get("pack_id", "ssp-core"),
            class_level=data.get("class_level", "Class 8"),
            subject=data.get("subject", "General"),
            book_title=data.get("book_title", "NCTB"),
            chapter_id=data.get("chapter_id", "CH-01"),
            chapter_title=data.get("chapter_title", ""),
            section_id=data.get("section_id"),
            section_title=data.get("section_title"),
            content_text=data["content_text"],
            content_type=data.get("content_type", "general"),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {})
        )
