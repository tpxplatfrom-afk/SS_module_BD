"""
SS Tutor BD - Bengali Educational Micro-Model Trainer (Phase 4)
Builds and trains a ~70M parameter Transformer using the dedicated 16K Bengali tokenizer
on synthetic NCTB educational datasets (CPU-compatible, $0 cost).
"""

import sys
import os
import json
import time
import math
import torch
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import (
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset

TOKENIZER_DIR = PROJECT_ROOT / "models" / "tokenizer_bengali_16k"
OUTPUT_DIR = PROJECT_ROOT / "models" / "sstutor_bengali_70m_edu"
CONFIG_FILE = PROJECT_ROOT / "configs" / "phase4_training.json"


class BengaliEduDataset(Dataset):
    def __init__(self, data_files: List[Path], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for fpath in data_files:
            if not fpath.exists():
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.examples.append(json.loads(line))
                        except Exception:
                            pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        mode = item.get("mode", "explain")
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        response = item.get("response", "")

        # Format with micro prompt protocol
        prompt = f"[T] {mode}\n{context}\nপ্রশ্ন: {instruction}\n[G] সহজ বাংলায় উত্তর দাও।\nউত্তর: "
        full_text = prompt + response + "<|eos|>"

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask prompt tokens in labels so loss is only computed on response
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:prompt_len] = -100
        # Also mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def build_70m_micro_model(vocab_size: int = 16000) -> LlamaForCausalLM:
    """Instantiates the ~70M parameter Transformer architecture."""
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=576,
        intermediate_size=2304,
        num_hidden_layers=10,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=256,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        tie_word_embeddings=False
    )
    model = LlamaForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Builder] Instantiated 70M Architecture:")
    print(f"  - Hidden Size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Heads: {config.num_attention_heads}")
    print(f"  - Total Parameters: {total_params:,} ({round(total_params/1e6, 2)}M)")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    return model


def train_micro_model(num_steps: int = 100, batch_size: int = 4):
    print("\n" + "=" * 70)
    print("      SS TUTOR BD — PHASE 4 MICRO-MODEL TRAINING HARNESS")
    print("=" * 70)

    # 1. Load Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    vocab_size = tokenizer.vocab_size
    print(f"[Trainer] Loaded tokenizer with vocab size: {vocab_size}")

    # 2. Build Dataset
    data_files = [
        PROJECT_ROOT / "data" / "phase4" / "math" / "math_verbalization.jsonl",
        PROJECT_ROOT / "data" / "phase4" / "socratic" / "socratic_hints.jsonl",
        PROJECT_ROOT / "data" / "phase4" / "grounding" / "grounding_dataset.jsonl",
        PROJECT_ROOT / "data" / "phase4" / "bengali" / "bengali_variants.jsonl"
    ]
    dataset = BengaliEduDataset(data_files, tokenizer, max_length=256)
    print(f"[Trainer] Dataset loaded with {len(dataset)} examples.")

    # 3. Instantiate ~70M Model
    model = build_70m_micro_model(vocab_size=vocab_size)

    # 4. Training Arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=20,
        save_steps=num_steps,
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        fp16=False,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)
    )

    print(f"[Trainer] Starting training on CPU ({num_steps} steps)...")
    t0 = time.perf_counter()
    train_result = trainer.train()
    duration = time.perf_counter() - t0
    print(f"[Trainer] Training completed in {duration:.2f} seconds!")

    # 5. Save Final Exportable Model & Tokenizer
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"[Trainer] Model and Tokenizer saved to: {OUTPUT_DIR}")

    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "duration_seconds": round(duration, 2),
        "train_loss": round(train_result.training_loss, 4) if hasattr(train_result, "training_loss") else 0.0,
        "output_dir": str(OUTPUT_DIR)
    }


if __name__ == "__main__":
    train_micro_model(num_steps=100)
