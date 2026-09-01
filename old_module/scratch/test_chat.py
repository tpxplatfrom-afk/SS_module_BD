import sys
import time
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llama_cpp import Llama

def test_chat():
    llm = Llama(
        model_path="models/active/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        n_ctx=2048,
        n_threads=2,
        n_gpu_layers=0,
        verbose=False
    )

    messages = [
        {"role": "system", "content": "You are SS Tutor BD, an expert AI tutor for Bangladesh High School (Class 6-10). Explain clearly in natural Bengali."},
        {"role": "user", "content": "বাংলাদেশের রাজধানীর নাম কী? এক বাক্যে বলো।"}
    ]

    t0 = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=64,
        temperature=0.0
    )
    gen_time = time.perf_counter() - t0

    choice = response["choices"][0]["message"]["content"]
    usage = response["usage"]

    print("--- Output ---")
    print(choice)
    print("--------------")
    print(f"Completion tokens: {usage['completion_tokens']}")
    print(f"Gen time: {gen_time:.3f} s")
    print(f"Tokens/s: {usage['completion_tokens']/gen_time:.2f}")

if __name__ == "__main__":
    test_chat()
