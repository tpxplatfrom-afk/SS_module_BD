"""
SS Tutor BD - Format & Tag Validator (Phase 4)
Sanitizes raw model output, strips delimiter tags and tool markers.
"""

import re
from typing import Dict, Any


class FormatValidator:
    PROMPT_TAGS = ["[T]", "[F]", "[R]", "[G]", "[H]", "[C]", "[TASK]", "[FACT]", "[RESULT]", "[GOAL]", "[HINT]", "[CONSTRAINT]"]

    @classmethod
    def clean_output_format(cls, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        had_tags = False

        for tag in cls.PROMPT_TAGS:
            if tag in cleaned:
                cleaned = cleaned.replace(tag, "").strip()
                had_tags = True

        # Remove control tokens
        for ctok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|pad|>", "<|unk|>"]:
            if ctok in cleaned:
                cleaned = cleaned.replace(ctok, "").strip()
                had_tags = True

        # Normalize excess whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        return {
            "cleaned_text": cleaned,
            "had_tags": had_tags,
            "length": len(cleaned)
        }
