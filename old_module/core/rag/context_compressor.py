"""
SS Tutor BD - RAG Context Compressor (Phase 3C)
Compresses retrieved textbook passages into high-density factual statements,
preserving mathematical formulas, definitions, and NCTB terminology while eliminating bloated prose.
"""

import re
from typing import List, Dict, Any


class ContextCompressor:
    @staticmethod
    def extract_key_formulas_and_facts(raw_text: str, max_words: int = 50) -> str:
        """
        Extracts mathematical formulas, definitions, and key rules from text.
        Removes conversational filler, duplicate lines, and excess punctuation.
        """
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        selected_facts = []
        seen_core = set()

        for line in lines:
            # Clean heading markers
            clean_line = re.sub(r"^#+\s*", "", line).strip()
            if not clean_line or len(clean_line) < 4:
                continue

            # Prioritize formulas, definitions, and rules
            is_formula = any(char in clean_line for char in ["=", "+", "-", "*", "/", "^", "(", ")", "%", "π"])
            is_definition = any(w in clean_line for w in ["কাকে বলে", "সংজ্ঞা", "সূত্র", "নিয়ম", "বলা হয়", "সমান", "পদ্ধতি"])

            # Deduplicate by key terms
            core_words = tuple(clean_line.split()[:4])
            if core_words in seen_core:
                continue
            seen_core.add(core_words)

            if is_formula or is_definition or len(selected_facts) < 2:
                selected_facts.append(clean_line)

        if not selected_facts:
            words = raw_text.split()
            return " ".join(words[:max_words])

        combined = " | ".join(selected_facts)
        words = combined.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return combined

    @classmethod
    def compress_retrieved_chunks(cls, retrieved_items: List[Dict[str, Any]], max_total_words: int = 60) -> str:
        """
        Takes raw retrieved chunks and returns a compact, deduplicated context string.
        """
        if not retrieved_items:
            return ""

        compressed_parts = []
        words_allocated = 0
        per_chunk_limit = max_total_words // max(1, min(len(retrieved_items), 2))

        for item in retrieved_items[:2]:
            chunk = item.get("chunk")
            if chunk is None:
                continue
            fact = cls.extract_key_formulas_and_facts(chunk.content_text, max_words=per_chunk_limit)
            fact_words = len(fact.split())
            if words_allocated + fact_words <= max_total_words + 10:
                compressed_parts.append(f"[{chunk.chapter_id}] {fact}")
                words_allocated += fact_words

        return "\n".join(compressed_parts)
