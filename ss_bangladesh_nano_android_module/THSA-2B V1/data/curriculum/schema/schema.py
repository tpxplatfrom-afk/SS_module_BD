"""
SS Tutor BD — Curriculum Knowledge Schema (Phase 8)
Defines the strict hierarchical ontology and deterministic ID structure for Bangladesh NCTB Class 6-10:
Grade -> Subject -> Book -> Chapter -> Topic -> Concept.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class CurriculumConcept:
    concept_id: str  # e.g., "g08.math.ch02.topic01.concept01"
    title_bengali: str
    title_english: str
    grade: int  # 6, 7, 8, 9, 10
    subject: str  # "mathematics", "science", "bengali", "english"
    chapter_number: int
    chapter_title: str
    topic_number: int
    topic_title: str
    definition: str
    formula: Optional[str] = None
    explanation: str = ""
    example: str = ""
    exercise_problem: str = ""
    common_mistake: str = ""
    socratic_hint: str = ""
    expected_answer: str = ""
    difficulty: str = "medium"  # "basic", "medium", "advanced"
    source: str = "NCTB"
    content_type: str = "concept"  # "definition", "formula", "procedure", "theorem"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def generate_id(grade: int, subject: str, ch_num: int, top_num: int, con_num: int) -> str:
        """Deterministically generates standardized concept IDs."""
        subj_abbr = subject.lower()[:4]
        return f"g{grade:02d}.{subj_abbr}.ch{ch_num:02d}.t{top_num:02d}.c{con_num:02d}"


@dataclass
class CurriculumTopic:
    topic_id: str
    title_bengali: str
    title_english: str
    topic_number: int
    concepts: List[CurriculumConcept] = field(default_factory=list)


@dataclass
class CurriculumChapter:
    chapter_id: str
    chapter_number: int
    title_bengali: str
    title_english: str
    topics: List[CurriculumTopic] = field(default_factory=list)


@dataclass
class CurriculumSubject:
    subject_id: str
    subject_name: str
    subject_name_bengali: str
    chapters: List[CurriculumChapter] = field(default_factory=list)


@dataclass
class CurriculumGrade:
    grade_level: int
    subjects: List[CurriculumSubject] = field(default_factory=list)
