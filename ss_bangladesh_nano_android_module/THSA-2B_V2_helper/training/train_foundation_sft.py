#!/usr/bin/env python3
"""
THSA-2B: Real Foundation Model SFT + LoRA Training Pipeline.
Fine-tunes a pretrained multilingual foundation LLM (Qwen2.5-0.5B-Instruct or Qwen2.5-1.5B-Instruct)
on the 5-Tier Bilingual (Bangla & English) ShareGPT Dataset.

Resilient to all PEFT/torchao/transformers versions on Python 3.10-3.13.
"""

import os
import sys
import json
import argparse
import torch
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="THSA-2B Foundation SFT + LoRA Training")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace pretrained foundation model ID")
    parser.add_argument("--train_data", type=str, default="data/train_sharegpt.jsonl",
                        help="Path to training ShareGPT JSONL file")
    parser.add_argument("--test_data", type=str, default="data/test_sharegpt.jsonl",
                        help="Path to test ShareGPT JSONL file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/thsa_foundation_merged",
                        help="Directory to save merged fine-tuned model and tokenizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    return parser.parse_args()

def load_sharegpt_as_chatml(jsonl_path: str) -> List[List[Dict[str, str]]]:
    """Reads ShareGPT JSONL and converts to list of messages [{'role': ..., 'content': ...}]."""
    dialogues = []
    if not os.path.exists(jsonl_path):
        print(f"Warning: File not found {jsonl_path}")
        return dialogues
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = []
            for turn in item.get("conversations", []):
                role = "user" if turn["from"] == "human" else "assistant"
                messages.append({"role": role, "content": turn["value"]})
            if messages:
                dialogues.append(messages)
    return dialogues

def main():
    args = parse_args()
    print("=" * 80)
    print("THSA-2B: REAL FOUNDATION MODEL SFT + LoRA TRAINING PIPELINE")
    print(f"Base Model:       {args.base_model}")
    print(f"Train Dataset:    {args.train_data}")
    print(f"Test Dataset:     {args.test_data}")
    print(f"Output Directory: {args.output_dir}")
    print(f"LoRA Rank:        {args.lora_r} (alpha={args.lora_alpha})")
    print(f"Epochs:           {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # 1. Load Tokenizer & Pretrained Foundation Model
    print(f"\n[1/5] Loading pretrained tokenizer and model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in native precision without device_map="auto" to avoid PEFT torchao dispatch hook errors
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    # 2. Configure LoRA (Compatible with all PEFT versions)
    print(f"[2/5] Injecting LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        print(f"  Note: Retrying with standard attention projections due to: {e}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)

    # Move to GPU after adapter injection
    if device == "cuda":
        model = model.to(device)

    model.print_trainable_parameters()

    # 3. Process ShareGPT Datasets
    print(f"\n[3/5] Ingesting and formatting multi-turn ShareGPT dialogues...")
    train_dialogues = load_sharegpt_as_chatml(args.train_data)
    test_dialogues  = load_sharegpt_as_chatml(args.test_data)

    if not train_dialogues:
        print("Error: No training dialogues found!")
        sys.exit(1)

    def tokenize_dialogue(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, max_length=args.max_seq_len, truncation=True, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_encoded = [tokenize_dialogue(d) for d in train_dialogues]
    test_encoded  = [tokenize_dialogue(d) for d in test_dialogues] if test_dialogues else train_encoded[:2]

    train_dataset = Dataset.from_list(train_encoded)
    test_dataset  = Dataset.from_list(test_encoded)

    print(f"  Train Set: {len(train_dataset)} multi-turn threads")
    print(f"  Test Set:  {len(test_dataset)} multi-turn threads")

    # 4. Training Arguments & Execution
    print(f"\n[4/5] Starting SFT + LoRA Training...")
    training_args = TrainingArguments(
        output_dir="checkpoints/lora_temp",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        eval_strategy="epoch" if test_dialogues else "no",
        save_strategy="no",
        fp16=(device == "cuda"),
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    trainer.train()

    # Evaluate Validation Perplexity
    if test_dialogues:
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", 0.0)
        perplexity = torch.exp(torch.tensor(eval_loss)).item()
        print(f"\n✅ Training Complete!")
        print(f"   Validation Loss:       {eval_loss:.4f}")
        print(f"   Validation Perplexity: {perplexity:.2f}")

    # 5. Merge LoRA Adapters & Export Final Standalone Model
    print(f"\n[5/5] Merging LoRA adapters into base weights and saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("✅ SUCCESS: STANDALONE FINE-TUNED MODEL & TOKENIZER EXPORTED!")
    print(f"Location: {os.path.abspath(args.output_dir)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
