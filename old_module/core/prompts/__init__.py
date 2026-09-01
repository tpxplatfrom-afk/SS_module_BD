"""
SS Tutor BD - Prompt Scaffolding Package
"""
from core.prompts.tutor_templates import (
    get_base_system_prompt,
    build_socratic_hint_prompt,
    build_step_by_step_math_prompt,
    build_grounded_rag_prompt,
    build_adaptive_simplification_prompt
)

__all__ = [
    "get_base_system_prompt",
    "build_socratic_hint_prompt",
    "build_step_by_step_math_prompt",
    "build_grounded_rag_prompt",
    "build_adaptive_simplification_prompt"
]
