#!/usr/bin/env python3
"""
THSA-2B: Test & Export Fine-Tuned Foundation Model.
Verifies interactive text generation directly from fine-tuned weights
and converts to on-device formats.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Test & Export Fine-Tuned Foundation Model")
    parser.add_argument("--model_dir", type=str, default="checkpoints/thsa_foundation_merged",
                        help="Directory containing merged fine-tuned model")
    parser.add_argument("--prompt", type=str, default="Hi Shanto, who are you and how can you help me?",
                        help="Test prompt to verify text generation")
    return parser.parse_args()

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Slice output tokens
    gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

def main():
    args = parse_args()
    print("=" * 80)
    print(f"LOADING FINE-TUNED MODEL FOR GENERATION TEST: {args.model_dir}")
    print("=" * 80)

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    test_prompts = [
        "Hi Shanto! Kemon acho? Tumi ke?",
        "Tell me about Operation Jackpot during 1971 Liberation War of Bangladesh.",
        "If a 10 kg object falls from 5 meters, calculate its energy."
    ]

    print("\n--- RUNNING LIVE NEURAL GENERATION TESTS ---")
    for idx, p in enumerate(test_prompts, 1):
        print(f"\n[Prompt {idx}]: {p}")
        resp = generate_response(model, tokenizer, p)
        print(f"[Generated Response]:\n{resp}")
        print("-" * 60)

if __name__ == "__main__":
    main()
