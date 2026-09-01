import sys
import time
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llama_cpp import Llama

def test_questions():
    llm = Llama(
        model_path="models/active/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        n_ctx=2048,
        n_threads=2,
        n_gpu_layers=0,
        verbose=False
    )

    prompts = [
        "প্রশ্ন: ৫ + ৭ = কত?",
        "প্রশ্ন: সালোকসংশ্লেষণ কী? সংক্ষেপে বলো।",
        "প্রশ্ন: (a + b)^2 এর সূত্রটি কী?"
    ]

    for p in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant who answers in Bengali."},
            {"role": "user", "content": p}
        ]
        res = llm.create_chat_completion(messages=messages, max_tokens=128, temperature=0.0)
        content = res["choices"][0]["message"]["content"]
        print(f"\nUser: {p}")
        print(f"Assistant: {content}")

if __name__ == "__main__":
    test_questions()
