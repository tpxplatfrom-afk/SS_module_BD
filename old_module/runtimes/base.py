"""
SS Tutor BD - Model Runtime Abstract Base Class
Defines the standard inference contract across all candidate model engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import psutil
import os


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    generation_time_s: float
    tokens_per_sec: float
    peak_rss_mb: float
    metadata: Dict[str, Any]


class ModelRuntime(ABC):
    """Abstract interface for local offline model execution."""

    def __init__(self, model_id: str, quantization: str = "Q4_K_M", threads: int = 2):
        self.model_id = model_id
        self.quantization = quantization
        self.threads = threads
        self.is_loaded = False

    @abstractmethod
    def load(self, model_path: str, context_length: int = 2048) -> Dict[str, Any]:
        """Loads model weights into memory/mmap."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None
    ) -> GenerationResult:
        """Executes autoregressive text generation."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Returns exact token count for the given string."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Releases memory and closes runtime handles."""
        pass

    def get_current_rss_mb(self) -> float:
        """Returns resident memory of the current process in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
