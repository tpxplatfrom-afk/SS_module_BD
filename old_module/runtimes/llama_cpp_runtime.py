"""
SS Tutor BD - llama.cpp GGUF Runtime Adapter
Provides local CPU-driven GGUF model execution using:
  1. llama-cpp-python  (primary — native GGUF via Python bindings)
  2. standalone llama-cli subprocess (secondary — if binary present in runtimes/bin/)
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from runtimes.base import ModelRuntime, GenerationResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_ROOT / "runtimes" / "bin"


class LlamaCppRuntime(ModelRuntime):
    """Runtime implementation for GGUF models via llama.cpp."""

    def __init__(
        self,
        model_id: str,
        quantization: str = "Q4_K_M",
        threads: int = 2,
        tokenizer_repo: Optional[str] = None
    ):
        super().__init__(model_id, quantization, threads)
        self.model_path: Optional[Path] = None
        self.context_length = 2048
        self.tokenizer_repo = tokenizer_repo
        self.tokenizer = None
        self._llm = None  # llama_cpp.Llama instance

        # Determine which backend to use
        self.cli_path: Optional[Path] = self._find_llama_cli()
        self.has_llama_cpp_python = self._check_llama_cpp_python()

        if self.has_llama_cpp_python:
            self.active_backend = "llama_cpp_python"
        elif self.cli_path:
            self.active_backend = "llama_cli"
        else:
            self.active_backend = "none"

        self._init_tokenizer()

    def _find_llama_cli(self) -> Optional[Path]:
        """Looks for llama-cli or main executable on PATH or in runtimes/bin."""
        candidates = [
            BIN_DIR / "llama-cli.exe",
            BIN_DIR / "llama-cli",
            BIN_DIR / "main.exe",
            BIN_DIR / "main"
        ]
        for c in candidates:
            if c.exists():
                return c
        # Also check system PATH
        cli_name = "llama-cli.exe" if os.name == "nt" else "llama-cli"
        for p in os.environ.get("PATH", "").split(os.pathsep):
            target = Path(p) / cli_name
            if target.exists():
                return target
        return None

    def _check_llama_cpp_python(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def _init_tokenizer(self):
        """Initializes HF tokenizer for fast token counting without loading the full model."""
        if self.tokenizer_repo:
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_pretrained(self.tokenizer_repo)
            except Exception:
                self.tokenizer = None

    def load(self, model_path: str, context_length: int = 4096) -> Dict[str, Any]:
        """Loads the GGUF model file into memory."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"GGUF model file not found at: {model_path}")

        self.model_path = path
        self.context_length = context_length
        t0 = time.perf_counter()

        if self.active_backend == "llama_cpp_python":
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=context_length,
                n_threads=self.threads,
                n_gpu_layers=0,   # CPU-only
                verbose=False
            )

        load_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.is_loaded = True

        return {
            "status": "LOADED",
            "runtime": "llama.cpp",
            "backend": self.active_backend,
            "model_path": str(self.model_path),
            "file_size_mb": round(self.model_path.stat().st_size / (1024 * 1024), 2),
            "context_length": context_length,
            "threads": self.threads,
            "load_time_ms": load_ms,
            "initial_rss_mb": round(self.get_current_rss_mb(), 2)
        }

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text).ids)
            except Exception:
                pass
        return max(1, int(len(text.split()) * 1.5))

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None
    ) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if self.active_backend == "llama_cpp_python":
            return self._generate_via_llama_cpp_python(
                prompt, system_prompt, max_tokens, temperature, stop_sequences
            )
        elif self.active_backend == "llama_cli":
            return self._generate_via_cli(
                prompt, system_prompt, max_tokens, temperature
            )
        else:
            raise RuntimeError(
                "No usable inference backend found.\n"
                "Install llama-cpp-python:  pip install llama-cpp-python --prefer-binary\n"
                "Or place llama-cli.exe in: runtimes/bin/"
            )

    def _build_chatml_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)

    def _generate_via_llama_cpp_python(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]]
    ) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stop = stop_sequences or ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]

        t0 = time.perf_counter()
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        gen_time = time.perf_counter() - t0

        raw_text = response["choices"][0]["message"].get("content", "").strip()
        usage = response.get("usage", {})
        gen_tokens = usage.get("completion_tokens", self.count_tokens(raw_text))
        prompt_tokens = usage.get("prompt_tokens", self.count_tokens(prompt))
        ttft_ms = round((gen_time / max(gen_tokens, 1)) * 1000, 2)
        tok_per_sec = round(gen_tokens / gen_time, 2) if gen_time > 0 else 0.0
        peak_rss = self.get_current_rss_mb()

        return GenerationResult(
            text=raw_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            ttft_ms=ttft_ms,
            generation_time_s=round(gen_time, 3),
            tokens_per_sec=tok_per_sec,
            peak_rss_mb=round(peak_rss, 2),
            metadata={
                "runtime": "llama.cpp",
                "backend": "llama_cpp_python",
                "model_id": self.model_id,
                "quantization": self.quantization,
                "threads": self.threads
            }
        )

    def _generate_via_cli(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> GenerationResult:
        full_prompt = self._build_chatml_prompt(prompt, system_prompt)
        prompt_tokens = self.count_tokens(full_prompt)

        cmd = [
            str(self.cli_path),
            "-m", str(self.model_path),
            "-p", full_prompt,
            "-n", str(max_tokens),
            "-c", str(self.context_length),
            "-t", str(self.threads),
            "--temp", str(temperature),
            "--no-display-prompt"
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        gen_time = time.perf_counter() - t0

        raw_text = proc.stdout
        for stop in ["<|im_end|>", "<|endoftext|>", "</s>"]:
            if stop in raw_text:
                raw_text = raw_text.split(stop)[0]

        gen_tokens = self.count_tokens(raw_text)
        tok_per_sec = round(gen_tokens / gen_time, 2) if gen_time > 0 else 0.0

        return GenerationResult(
            text=raw_text.strip(),
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            ttft_ms=round(gen_time * 500, 2),
            generation_time_s=round(gen_time, 3),
            tokens_per_sec=tok_per_sec,
            peak_rss_mb=round(self.get_current_rss_mb(), 2),
            metadata={"runtime": "llama.cpp", "backend": "llama_cli",
                      "model_id": self.model_id, "quantization": self.quantization}
        )

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self.model_path = None
        self.is_loaded = False
