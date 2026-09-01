"""
SS Tutor BD - Micro-Runtime Adapter (Phase 3C)
Abstract adapter interface over multiple inference backends with explicit memory bounds.
Supports: llama_cpp (bounded), ONNX Runtime, and Deterministic-Fallback (no neural).

All implementations enforce the Phase 3C KV cache contract:
  MAX_CONTEXT_TOKENS (policy-driven)
  MAX_OUTPUT_TOKENS (policy-driven)
"""

import os
import time
import psutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from core.runtime.memory_budget import MemoryBudgetManager
from core.runtime.device_profile import DeviceProfiler


@dataclass
class MicroGenResult:
    text: str
    tokens_per_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    peak_rss_mb: float = 0.0
    backend: str = "unknown"
    truncated: bool = False


class MicroRuntimeBase(ABC):
    """Abstract interface every runtime backend must implement."""

    @abstractmethod
    def load(self, model_path: str, max_context: int = 256, max_output: int = 96) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 96,
        temperature: float = 0.15,
        repeat_penalty: float = 1.2,
        stop_sequences: Optional[List[str]] = None,
    ) -> MicroGenResult: ...

    @abstractmethod
    def memory_usage_mb(self) -> float: ...

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_context_limit(self) -> bool:
        return True


class LlamaCppMicroRuntime(MicroRuntimeBase):
    """
    llama.cpp backend with explicitly bounded KV cache.
    Uses the smallest viable context (policy-driven) to respect the 200 MB ceiling.
    """

    def __init__(self, model_id: str, threads: int = 4, tokenizer_repo: Optional[str] = None):
        self.model_id = model_id
        self.threads = threads
        self.tokenizer_repo = tokenizer_repo
        self._llm = None
        self._max_context = 256

    def load(self, model_path: str, max_context: int = 256, max_output: int = 96) -> None:
        from llama_cpp import Llama
        self._max_context = max_context
        self._llm = Llama(
            model_path=model_path,
            n_ctx=max_context,
            n_threads=self.threads,
            verbose=False
        )

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 96,
        temperature: float = 0.15,
        repeat_penalty: float = 1.2,
        stop_sequences: Optional[List[str]] = None,
    ) -> MicroGenResult:
        if self._llm is None:
            return MicroGenResult(text="[Runtime not loaded]", backend="llama_cpp")

        rss_before = MemoryBudgetManager.get_current_rss_mb()
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        stops = stop_sequences or ["[T]", "[F]", "[R]", "User:", "প্রশ্ন:", "<|im_end|>", "<|endoftext|>"]

        try:
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                stop=stops
            )
            text = result["choices"][0]["message"]["content"].strip()
            usage = result.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
        except ValueError as e:
            text = f"[Context overflow: {str(e)[:60]}]"
            in_tok, out_tok = 0, 0

        latency_ms = (time.perf_counter() - t0) * 1000
        peak_rss = MemoryBudgetManager.get_current_rss_mb()
        tok_s = (out_tok / (latency_ms / 1000)) if latency_ms > 0 and out_tok > 0 else 0.0

        return MicroGenResult(
            text=text,
            tokens_per_sec=round(tok_s, 2),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            peak_rss_mb=peak_rss,
            backend="llama_cpp"
        )

    def memory_usage_mb(self) -> float:
        return MemoryBudgetManager.get_current_rss_mb()


class OnnxMicroRuntime(MicroRuntimeBase):
    """
    ONNX Runtime backend (future Phase 3C candidate).
    Uses fixed-allocation approach for deterministic memory footprint.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._session = None
        self._tokenizer = None

    def load(self, model_path: str, max_context: int = 256, max_output: int = 96) -> None:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options=opts)
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")

    def unload(self) -> None:
        self._session = None

    def generate(self, system_prompt, user_prompt, max_tokens=96, temperature=0.15,
                 repeat_penalty=1.2, stop_sequences=None) -> MicroGenResult:
        # ONNX generation requires model-specific decoding loop.
        # Placeholder until a Bengali-capable ONNX model is selected.
        return MicroGenResult(
            text="[ONNX Runtime: model-specific decoder not yet implemented]",
            backend="onnx_runtime"
        )

    def memory_usage_mb(self) -> float:
        return MemoryBudgetManager.get_current_rss_mb()


class DeterministicFallbackRuntime(MicroRuntimeBase):
    """
    No-neural-model fallback runtime.
    Produces structured Bengali responses from deterministic templates only.
    Used when neural runtime is unavailable, OOM, or in TIER_ULTRA_LOW mode.
    """

    def load(self, model_path: str = "", max_context: int = 256, max_output: int = 96) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, system_prompt, user_prompt, max_tokens=96, temperature=0.15,
                 repeat_penalty=1.2, stop_sequences=None) -> MicroGenResult:
        response = self._template_response(user_prompt)
        return MicroGenResult(text=response, tokens_per_sec=999.0, backend="deterministic_template")

    def _template_response(self, prompt: str) -> str:
        if "[R]" in prompt:
            import re
            m = re.search(r"\[R\]\s*(.+)", prompt)
            if m:
                return f"গণনার ফলাফল: {m.group(1).strip()}\n(সম্পূর্ণ ব্যাখ্যার জন্য মডেল লোড করুন।)"
        return "এই প্রশ্নের উত্তর পাঠ্যপুস্তক থেকে পাওয়া যাবে। অনুগ্রহ করে পাঠ্যবই দেখুন।"

    def memory_usage_mb(self) -> float:
        return MemoryBudgetManager.get_current_rss_mb()


class MicroRuntimeFactory:
    @staticmethod
    def create(backend: str, model_id: str, threads: int = 4, tokenizer_repo: Optional[str] = None) -> MicroRuntimeBase:
        b = backend.lower()
        if b == "llama_cpp":
            return LlamaCppMicroRuntime(model_id=model_id, threads=threads, tokenizer_repo=tokenizer_repo)
        elif b == "onnx":
            return OnnxMicroRuntime(model_id=model_id)
        elif b in ("deterministic", "template", "fallback"):
            return DeterministicFallbackRuntime()
        else:
            raise ValueError(f"Unknown micro-runtime backend: {backend}")
