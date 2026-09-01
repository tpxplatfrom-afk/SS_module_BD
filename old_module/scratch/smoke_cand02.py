import sys
import time
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runtimes.llama_cpp_runtime import LlamaCppRuntime

def smoke_test_cand02():
    print("Testing CAND-02 (Qwen2.5-1.5B-Instruct Q4_K_M) Smoke Inference...")
    rt = LlamaCppRuntime(
        model_id="CAND-02",
        quantization="Q4_K_M",
        threads=2,
        tokenizer_repo="Qwen/Qwen2.5-1.5B-Instruct"
    )
    print("Active backend:", rt.active_backend)

    t0 = time.perf_counter()
    info = rt.load("models/active/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    load_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"Model loaded in {load_ms} ms. Initial RSS: {info['initial_rss_mb']} MB")

    prompt = "বাংলাদেশের জাতীয় কবির নাম কী এবং তাঁর দুটি বিখ্যাত কবিতার নাম বলো।"
    system_prompt = "You are SS Tutor BD, an expert AI tutor for Bangladesh High School (Class 6-10). Answer in natural Bengali."

    print(f"\nPrompt: {prompt}\nGenerating response...")
    result = rt.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=128,
        temperature=0.0
    )

    print("\n" + "=" * 40)
    print("Generated Text:")
    print(result.text)
    print("=" * 40)
    print(f"Generated Tokens:  {result.generated_tokens}")
    print(f"Generation Time:   {result.generation_time_s} s")
    print(f"Tokens Per Second: {result.tokens_per_sec} tok/s")
    print(f"Peak RSS:          {result.peak_rss_mb} MB")

    rt.unload()
    print("\nSmoke test complete!")

if __name__ == "__main__":
    smoke_test_cand02()
