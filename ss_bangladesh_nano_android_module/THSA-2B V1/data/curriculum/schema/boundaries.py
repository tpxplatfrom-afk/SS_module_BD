"""
SS Tutor BD — Package-Ready Knowledge Boundaries (Phase 8)
Defines architectural boundaries (CurriculumScope, KnowledgeUnit, KnowledgePackMetadata)
enabling future modular extraction without implementing package distribution now.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class CurriculumScope:
    scope_id: str
    grades: List[int] = field(default_factory=lambda: [6, 7, 8, 9, 10])
    subjects: List[str] = field(default_factory=lambda: ["mathematics", "science", "bengali", "english"])
    chapters: Optional[List[int]] = None
    is_full_core: bool = True

    def matches(self, grade: int, subject: str, chapter: int) -> bool:
        if grade not in self.grades:
            return False
        if subject.lower() not in [s.lower() for s in self.subjects]:
            return False
        if self.chapters is not None and chapter not in self.chapters:
            return False
        return True


@dataclass
class KnowledgeUnit:
    unit_id: str
    concept_id: str
    scope: CurriculumScope
    content_bengali: str
    content_type: str  # "definition", "formula", "example", "socratic_hint", "misconception"
    token_count: int
    fts_indexed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scope"] = asdict(self.scope)
        return d


@dataclass
class KnowledgePackMetadata:
    pack_id: str
    pack_name: str
    pack_name_bengali: str
    version: str
    scope: CurriculumScope
    total_units: int
    total_concepts: int
    db_size_bytes: int
    format_version: str = "ssp_v1"
    license: str = "CC0-1.0 / NCTB Open Framework"
    build_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scope"] = asdict(self.scope)
        return d
