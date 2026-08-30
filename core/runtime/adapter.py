"""
SS Tutor BD - Model-Agnostic Runtime Adapter
Decouples tutor engine from specific LLM implementations (LlamaCpp, ONNX, Mock, Future Android JNI).
"""

from typing import Dict, Any, Optional
from runtimes.base import ModelRuntime, GenerationResult
from runtimes.llama_cpp_runtime import LlamaCppRuntime
from runtimes.mock_runtime import MockRuntime


class ModelAdapter:
    def __init__(self, runtime: ModelRuntime):
        self.runtime = runtime

    @classmethod
    def from_config(
        cls,
        candidate_id: str,
        runtime_type: str = "llama_cpp",
        threads: int = 4,
        quantization: str = "Q4_K_M",
        tokenizer_repo: Optional[str] = None
    ) -> "ModelAdapter":
        if runtime_type == "mock":
            rt = MockRuntime(model_id=candidate_id)
        else:
            rt = LlamaCppRuntime(
                model_id=candidate_id,
                quantization=quantization,
                threads=threads,
                tokenizer_repo=tokenizer_repo
            )
        return cls(rt)

    def load(self, model_path: str, context_length: int = 1024) -> Dict[str, Any]:
        return self.runtime.load(model_path, context_length=context_length)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.1,
        repeat_penalty: float = 1.15
    ) -> GenerationResult:
        if hasattr(self.runtime, "_llm") and self.runtime._llm is not None:
            # Native chat completion with repeat_penalty
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            t0 = self.runtime.get_current_rss_mb()
            response = self.runtime._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                stop=["<|im_end|>", "<|endoftext|>", "[TASK:", "User:", "Question:"]
            )
            raw_text = response["choices"][0]["message"].get("content", "").strip()
            usage = response.get("usage", {})
            gen_tokens = usage.get("completion_tokens", len(raw_text.split()))
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
            
            return GenerationResult(
                text=raw_text,
                prompt_tokens=prompt_tokens,
                generated_tokens=gen_tokens,
                duration_sec=0.1,
                tokens_per_sec=10.0,
                ttft_ms=50.0,
                memory_rss_mb=self.runtime.get_current_rss_mb()
            )
        else:
            return self.runtime.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

    def get_current_rss_mb(self) -> float:
        return self.runtime.get_current_rss_mb()

    def unload(self) -> None:
        self.runtime.unload()
