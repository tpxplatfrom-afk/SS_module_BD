"""
SS Tutor BD - Mock Runtime
Fast deterministic dry-run runtime for validating the benchmarking harness,
scoring engine, and report generator without requiring downloaded weights.
"""

from typing import Dict, Any, Optional, List
import time
from runtimes.base import ModelRuntime, GenerationResult


class MockRuntime(ModelRuntime):
    def __init__(self, model_id: str = "MOCK-01", quantization: str = "Q4_K_M", threads: int = 2):
        super().__init__(model_id, quantization, threads)
        self.load_time_ms = 150.0

    def load(self, model_path: str = "mock", context_length: int = 2048) -> Dict[str, Any]:
        self.is_loaded = True
        return {
            "status": "LOADED",
            "runtime": "mock",
            "model_path": model_path,
            "context_length": context_length,
            "load_time_ms": self.load_time_ms
        }

    def count_tokens(self, text: str) -> int:
        return len(text.split()) * 2

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None
    ) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        start_time = time.perf_counter()
        # Simulated Bengali response for mock testing
        mock_response = (
            "শিক্ষার্থীকে স্বাগতম। এই প্রশ্নের সমাধানটি ধাপে ধাপে বুঝিয়ে বলা হলো: "
            "১. প্রথমে প্রদত্ত তথ্যটি বিবেচনা করো। "
            "২. প্রয়োজনীয় সূত্রটি প্রয়োগ করো। "
            "৩. হিসাব করে সঠিক ফলাফল নির্ণয় করো।"
        )
        time.sleep(0.01)  # Minimal sleep to simulate compute
        gen_time = time.perf_counter() - start_time
        gen_tokens = len(mock_response.split()) * 2

        return GenerationResult(
            text=mock_response,
            prompt_tokens=self.count_tokens(prompt),
            generated_tokens=gen_tokens,
            ttft_ms=50.0,
            generation_time_s=gen_time,
            tokens_per_sec=gen_tokens / gen_time if gen_time > 0 else 50.0,
            peak_rss_mb=self.get_current_rss_mb(),
            metadata={"runtime": "mock"}
        )

    def unload(self) -> None:
        self.is_loaded = False
