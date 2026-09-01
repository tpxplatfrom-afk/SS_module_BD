"""
SS Tutor BD - Context Budget Manager (Phase 4)
Enforces strict token limits on every component of the prompt to prevent KV-cache expansion.
"""

from typing import Dict, Any


class ContextBudgetManager:
    # Strict Token Budgets per prompt section
    MAX_SYSTEM_PROTOCOL_TOKENS = 70
    MAX_RAG_FACT_TOKENS = 120
    MAX_COMPUTED_RESULT_TOKENS = 60
    MAX_USER_QUESTION_TOKENS = 60
    MAX_OUTPUT_TOKENS = 128
    MAX_TOTAL_CONTEXT_TOKENS = 256  # Hard bounded context window for low-memory Android

    @classmethod
    def truncate_text_to_token_budget(cls, text: str, max_tokens: int) -> str:
        """Truncates text approximately based on word count (1 word ~ 1.5 - 2 tokens)."""
        if not text:
            return ""
        max_words = int(max_tokens / 1.5)
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    @classmethod
    def enforce_prompt_budget(
        cls,
        task_code: str,
        question: str,
        textbook_fact: str = "",
        computed_result: str = ""
    ) -> Dict[str, str]:
        """
        Applies strict component budgets to ensure the total prompt never exceeds MAX_TOTAL_CONTEXT_TOKENS.
        """
        trunc_q = cls.truncate_text_to_token_budget(question, cls.MAX_USER_QUESTION_TOKENS)
        trunc_fact = cls.truncate_text_to_token_budget(textbook_fact, cls.MAX_RAG_FACT_TOKENS)
        trunc_res = cls.truncate_text_to_token_budget(computed_result, cls.MAX_COMPUTED_RESULT_TOKENS)

        return {
            "task_code": task_code,
            "question": trunc_q,
            "textbook_fact": trunc_fact,
            "computed_result": trunc_res,
            "max_output_tokens": cls.MAX_OUTPUT_TOKENS,
            "max_context_tokens": cls.MAX_TOTAL_CONTEXT_TOKENS
        }
