"""
SS Tutor BD - Hybrid Grounded Tutoring Engine
Orchestrates Intent Detection, Deterministic Mathematics Calculations, SQLite FTS5 RAG,
Compact Prompt Protocol, Micro-LLM Generation, Output Sanitization, and Cross-Validation.
"""

import sys
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.schema import KnowledgeChunk
from core.rag.indexer import KnowledgeIndexer
from core.rag.retriever import KnowledgeRetriever
from core.prompts.compact_tutor_templates import (
    get_compact_system_prompt,
    build_compact_prompt
)
from core.sanitization.cleaner import sanitize_tutor_output
from core.math.fraction import FractionHelper
from core.math.calculator import MathCalculator
from core.math.equation_solver import EquationSolver
from core.math.unit_converter import UnitConverter
from core.math.expression_parser import ExpressionParser
from core.math.validator import MathValidator
from runtimes.base import ModelRuntime, GenerationResult


@dataclass
class TutorResponse:
    query: str
    mode: str
    pipeline_type: str  # "llm_only", "llm_rag", "hybrid_rag_tools"
    final_text: str
    raw_text: str
    retrieved_chunks: List[Dict[str, Any]]
    deterministic_result: Optional[Dict[str, Any]]
    grounding_status: str
    was_math_task: bool
    was_corrected_by_validator: bool
    prompt_tokens: int
    generated_tokens: int
    retrieval_latency_ms: float
    math_engine_latency_ms: float
    inference_latency_ms: float
    sanitization_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float
    peak_rss_mb: float
    sanitization_flags: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroundedTutorEngine:
    def __init__(
        self,
        retriever: Optional[KnowledgeRetriever] = None,
        runtime: Optional[ModelRuntime] = None,
        default_top_k: int = 2,
        temperature: float = 0.1,
        repeat_penalty: float = 1.15,
        max_tokens: int = 256
    ):
        self.retriever = retriever
        self.runtime = runtime
        self.default_top_k = default_top_k
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.max_tokens = max_tokens

    def process_query(
        self,
        query: str,
        mode: str = "auto",
        pipeline_type: str = "hybrid_rag_tools",  # "llm_only", "llm_rag", "hybrid_rag_tools"
        student_level: str = "Class 8",
        temperature: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> TutorResponse:
        t_start = time.perf_counter()

        # 1. Math Intent Detection & Deterministic Calculation
        t_math_0 = time.perf_counter()
        math_intent = ExpressionParser.detect_math_intent(query)
        deterministic_res = None
        verified_steps_str = ""

        if pipeline_type == "hybrid_rag_tools" and math_intent.get("intent") != "general_or_concept":
            intent_name = math_intent.get("intent")
            if intent_name == "fraction_addition":
                deterministic_res = FractionHelper.add(math_intent["fraction1"], math_intent["fraction2"])
                deterministic_res["intent"] = intent_name
                deterministic_res["final_answer"] = deterministic_res["final_answer_bengali"]
                deterministic_res["steps"] = deterministic_res["steps_bengali"]
            elif intent_name == "simple_interest":
                deterministic_res = MathCalculator.simple_interest(
                    math_intent["principal"], math_intent["rate_pct"], math_intent["time_years"]
                )
                deterministic_res["intent"] = intent_name
            elif intent_name == "compound_interest":
                deterministic_res = MathCalculator.compound_interest(
                    math_intent["principal"], math_intent["rate_pct"], math_intent["time_years"]
                )
                deterministic_res["intent"] = intent_name
            elif intent_name == "series_sum":
                deterministic_res = MathCalculator.series_sum(
                    math_intent["first_term"], math_intent["last_term"]
                )
                deterministic_res["intent"] = intent_name
            elif intent_name == "pythagoras_leg":
                deterministic_res = MathCalculator.pythagoras(
                    c=math_intent["hypotenuse"], a=math_intent["leg"]
                )
                deterministic_res["intent"] = intent_name
            elif intent_name == "pythagoras_hypotenuse":
                deterministic_res = MathCalculator.pythagoras(
                    a=math_intent["leg1"], b=math_intent["leg2"]
                )
                deterministic_res["intent"] = intent_name
            elif intent_name == "circle_metrics":
                deterministic_res = MathCalculator.circle_metrics(math_intent["radius"])
                deterministic_res["intent"] = intent_name
            elif intent_name == "factorization":
                deterministic_res = EquationSolver.factorize_quadratic(math_intent["b"], math_intent["c"])
                deterministic_res["intent"] = intent_name
                deterministic_res["final_answer"] = deterministic_res.get("bengali_expression", "")

            if deterministic_res and "steps" in deterministic_res:
                verified_steps_str = "\n".join(deterministic_res["steps"])

        math_latency = (time.perf_counter() - t_math_0) * 1000

        # 2. Retrieval Stage (Skipped in llm_only mode)
        retrieval_results = []
        ret_latency = 0.0
        combined_context = ""

        if pipeline_type in ["llm_rag", "hybrid_rag_tools"] and self.retriever is not None:
            t_ret_0 = time.perf_counter()
            k = top_k or self.default_top_k
            retrieval_results = self.retriever.retrieve(query, top_k=k)
            ret_latency = (time.perf_counter() - t_ret_0) * 1000

            context_parts = []
            for r in retrieval_results[:2]:
                c = r["chunk"]
                words = c.content_text.split()
                trunc_text = " ".join(words[:60]) if len(words) > 60 else c.content_text
                context_parts.append(f"[{c.chapter_id}: {c.section_title}]\n{trunc_text}")
            combined_context = "\n\n".join(context_parts) if context_parts else ""

        # 3. Mode Determination & Prompt Construction
        if mode == "auto":
            if any(w in query for w in ["ইঙ্গিত", "hint", "বলবে না", "উত্তর দেবে না", "সরাসরি বলবে না"]):
                chosen_mode = "HINT"
            elif any(w in query for w in ["সহজ", "বুঝিনি", "আবার"]):
                chosen_mode = "EXPLAIN"
            elif any(w in query for w in ["সমাধান", "কত", "মান", "নির্ণয়", "হিসাব", "যোগফল"]):
                chosen_mode = "SOLVE"
            else:
                chosen_mode = "EXPLAIN"
        else:
            chosen_mode = mode.upper()

        final_prompt = build_compact_prompt(
            task_mode=chosen_mode,
            user_query=query,
            textbook_context=combined_context if pipeline_type != "llm_only" else None,
            verified_result=verified_steps_str if (pipeline_type == "hybrid_rag_tools" and chosen_mode != "HINT") else None,
            student_level=student_level
        )
        system_prompt = get_compact_system_prompt()

        # 4. Model Generation
        temp = temperature if temperature is not None else self.temperature
        rep_pen = repeat_penalty if repeat_penalty is not None else self.repeat_penalty
        m_tokens = max_tokens if max_tokens is not None else self.max_tokens

        t_gen_0 = time.perf_counter()
        raw_text = ""
        prompt_tokens = len(final_prompt.split())
        gen_tokens = 0

        if self.runtime is not None:
            if hasattr(self.runtime, "_llm") and self.runtime._llm is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ]
                response = self.runtime._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=m_tokens,
                    temperature=temp,
                    repeat_penalty=rep_pen,
                    stop=["<|im_end|>", "<|endoftext|>", "[TASK:", "User:", "Question:"]
                )
                raw_text = response["choices"][0]["message"].get("content", "").strip()
                usage = response.get("usage", {})
                gen_tokens = usage.get("completion_tokens", len(raw_text.split()))
                prompt_tokens = usage.get("prompt_tokens", len(final_prompt.split()))
            else:
                gen_res = self.runtime.generate(
                    prompt=final_prompt,
                    system_prompt=system_prompt,
                    max_tokens=m_tokens,
                    temperature=temp
                )
                raw_text = gen_res.text
                gen_tokens = gen_res.generated_tokens
                prompt_tokens = gen_res.prompt_tokens
        else:
            # Fallback if no LLM runtime is attached (Pure Deterministic RAG Mode)
            if deterministic_res and "steps" in deterministic_res:
                raw_text = "সমাধান:\n" + "\n".join(deterministic_res["steps"]) + f"\n\nঅতএব, নির্ণেয় উত্তর: {deterministic_res.get('final_answer', '')}।"
            else:
                raw_text = combined_context if combined_context else "সংশ্লিষ্ট পাঠ্যপুস্তক তথ্য পাওয়া যায়নি।"

        inf_latency = (time.perf_counter() - t_gen_0) * 1000
        tok_per_sec = (gen_tokens / (inf_latency / 1000)) if inf_latency > 0 else 0.0

        # 5. Output Sanitization Stage
        t_san_0 = time.perf_counter()
        san_res = sanitize_tutor_output(raw_text, user_prompt=query)
        san_latency = (time.perf_counter() - t_san_0) * 1000
        cleaned_text = san_res["cleaned_text"]

        # 6. Math Validator Cross-Check
        was_corrected = False
        if pipeline_type == "hybrid_rag_tools" and deterministic_res is not None and chosen_mode != "HINT":
            val_res = MathValidator.validate_and_correct(query, cleaned_text, deterministic_res)
            if val_res.get("corrected", False):
                cleaned_text = val_res["verified_text"]
                was_corrected = True

        # 7. Grounding Status
        grounding_status = "GROUNDED"
        if pipeline_type == "llm_only" or not retrieval_results:
            grounding_status = "UNGROUNDED"
        else:
            grounding_status = "GROUNDED"

        total_latency = (time.perf_counter() - t_start) * 1000
        peak_rss = self.runtime.get_current_rss_mb() if self.runtime else 0.0

        return TutorResponse(
            query=query,
            mode=chosen_mode,
            pipeline_type=pipeline_type,
            final_text=cleaned_text,
            raw_text=raw_text,
            retrieved_chunks=[
                {
                    "chunk_id": r["chunk"].chunk_id,
                    "chapter_id": r["chunk"].chapter_id,
                    "chapter_title": r["chunk"].chapter_title,
                    "score": r["score"]
                }
                for r in retrieval_results
            ],
            deterministic_result=deterministic_res,
            grounding_status=grounding_status,
            was_math_task=deterministic_res is not None,
            was_corrected_by_validator=was_corrected,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            retrieval_latency_ms=round(ret_latency, 2),
            math_engine_latency_ms=round(math_latency, 2),
            inference_latency_ms=round(inf_latency, 2),
            sanitization_latency_ms=round(san_latency, 2),
            total_latency_ms=round(total_latency, 2),
            tokens_per_second=round(tok_per_sec, 2),
            peak_rss_mb=round(peak_rss, 2),
            sanitization_flags={
                "had_control_tokens": san_res["had_control_tokens"],
                "had_prompt_echo": san_res["had_prompt_echo"],
                "had_repetition_loop": san_res["had_repetition_loop"]
            }
        )
