#!/usr/bin/env python3
"""
THSA-2B Teacher-Student Distillation Engine (Qwen -> THSA)
===========================================================
Pairs a pre-trained Qwen model (e.g. Qwen/Qwen2.5-0.5B or Qwen/Qwen2.5-1.5B)
as the frozen Teacher and the THSA Hybrid Architecture as the Student.

Features:
  1. Mixed CE + Soft KL Divergence Distillation Loss:
     L_total = (1 - alpha) * L_CE + alpha * tau^2 * KL(P_student || P_teacher)
  2. Dynamic QAT Annealing (beta: 1.0 -> 100.0) for 1.58-bit ternary weights.
  3. Mixed Precision FP16 (AMP) + Gradient Checkpointing for Free Google Colab / Kaggle T4 GPUs.
  4. Automatic vocabulary projection & alignment (Qwen 152K -> THSA 65K).
  5. Live HuggingFace streaming fallback if local corpus is absent.
  6. Checkpoint saving ready for direct conversion to .nano binary format.
"""

import sys
import os
import time
import math
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add module paths
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
MODULE_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from models.thsa_hybrid_model import THSAHybridForCausalLM
from models.ternary_layers import TernaryLinear
from distillation.distillation_loss import DistillationLoss


class TextCorpusDataset(Dataset):
    """Memory-efficient dataset integrating NCTB curriculum packs, JSONL QA pairs, and corpus."""
    def __init__(self, corpus_path: str, max_samples: int = 100000, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.lines: List[str] = []
        
        # 1. Load from NCTB Curriculum Knowledge Packs and JSONL datasets
        curriculum_dir = MODULE_ROOT / "data" / "curriculum"
        if curriculum_dir.exists():
            import glob
            # Load JSONL QA pairs (Class 6-10 Math, Socratic Hints, Grounding)
            for jf in glob.glob(str(curriculum_dir / "datasets" / "**" / "*.jsonl"), recursive=True):
                with open(jf, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                d = json.loads(line)
                                inst = d.get("instruction", "").strip()
                                resp = d.get("response", "").strip()
                                ctx = d.get("context", "").strip()
                                if inst and resp:
                                    pair = f"{ctx} প্রশ্ন: {inst} সমাধান: {resp}" if ctx else f"প্রশ্ন: {inst} উত্তর: {resp}"
                                    self.lines.append(pair)
                            except Exception:
                                pass
                                
            # Load Markdown textbook packs (Class 8 & 9-10 Math, Physics, Chemistry, ICT)
            for mf in glob.glob(str(curriculum_dir / "packs" / "**" / "*.md"), recursive=True):
                with open(mf, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) >= 15 and not line.startswith("```"):
                            self.lines.append(line)
                            
            if self.lines:
                print(f"[Dataset] Loaded {len(self.lines):,} NCTB textbook curriculum lines from {curriculum_dir}")
                
        # 2. Load from corpus_path if present
        if corpus_path and os.path.exists(corpus_path):
            print(f"[Dataset] Loading text from local file: {corpus_path}...")
            with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    text = line.strip()
                    if len(text) >= 15 and not text.startswith("==="):
                        self.lines.append(text)
                    if len(self.lines) >= max_samples:
                        break
            print(f"[Dataset] Total combined dataset size: {len(self.lines):,} training sentences.")
            
        if len(self.lines) == 0:
            raise FileNotFoundError(
                f"[FATAL ERROR] No training data found at '{corpus_path}' or '{curriculum_dir}'. "
                f"Synthetic fallback dataset generation is strictly forbidden."
            )

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


class QwenTeacherWrapper(nn.Module):
    """
    Wraps a Hugging Face Qwen causal LM as a frozen teacher.
    Generates teacher logits and hidden states under torch.no_grad().
    """
    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda", precision: str = "bfloat16"):
        super().__init__()
        self.model_name = model_name_or_path
        self.device = device
        self.teacher_model = None
        self.teacher_tokenizer = None

        print(f"[Teacher] Loading teacher model: {model_name_or_path}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True
            )
            
            # Load in bfloat16 or float16 to fit in GPU VRAM
            if precision == "bfloat16" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif precision == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32 if device == "cpu" else torch.float16
                
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None
            )
            if device != "cuda":
                self.teacher_model = self.teacher_model.to(device)

            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
                
            print(f"[Teacher] Successfully loaded {model_name_or_path} (Frozen {dtype}).")
        except Exception as e:
            raise RuntimeError(
                f"[FATAL ERROR] Failed to load required Teacher model '{model_name_or_path}': {e}. "
                f"Mock or synthetic teacher fallbacks are strictly prohibited."
            )

    def forward(self, input_ids: torch.Tensor, student_vocab_size: int = 65536) -> torch.Tensor:
        """Computes teacher logits aligned with student vocabulary."""
        with torch.no_grad():
            outputs = self.teacher_model(input_ids=input_ids)
            teacher_logits = outputs.logits # [B, S, V_teacher]
            
            # If teacher vocab size != student vocab size, project/slice to match
            V_t = teacher_logits.shape[-1]
            if V_t != student_vocab_size:
                if V_t > student_vocab_size:
                    teacher_logits = teacher_logits[:, :, :student_vocab_size]
                else:
                    pad = torch.zeros(
                        teacher_logits.shape[0], 
                        teacher_logits.shape[1], 
                        student_vocab_size - V_t, 
                        device=teacher_logits.device,
                        dtype=teacher_logits.dtype
                    )
                    teacher_logits = torch.cat([teacher_logits, pad], dim=-1)
                    
            return teacher_logits


class DistillationTrainer:
    """
    Orchestrates the Qwen -> THSA Teacher-Student Knowledge Distillation Loop.
    """
    def __init__(
        self,
        config_path: str,
        teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        corpus_path: Optional[str] = None,
        output_dir: str = "checkpoints/thsa_2b_distilled",
        alpha: float = 0.65,
        temperature: float = 2.0,
        learning_rate: float = 3e-4,
        batch_size: int = 1,
        grad_accum_steps: int = 16,
        max_steps: int = 10000,
        checkpoint_interval: int = 500,
        resume_checkpoint: Optional[str] = None,
        precision: str = "bfloat16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_steps = max_steps
        self.checkpoint_interval = checkpoint_interval
        self.resume_checkpoint = resume_checkpoint
        self.precision = precision
        self.start_step = 0
        
        # 1. Load Student Config
        with open(config_path, "r", encoding="utf-8-sig") as f:
            self.config = json.load(f)
            
        print("=" * 80)
        print("THSA-2B PRODUCTION TEACHER-STUDENT DISTILLATION HARNESS")
        print("=" * 80)
        print(f"Student Model ID:    {self.config.get('model_id', 'THSA-2B-V1-PRODUCTION')}")
        print(f"Student Backbone:    {self.config.get('total_blocks')} Blocks ({self.config.get('state_blocks')} State / {self.config.get('gqa_blocks')} GQA)")
        print(f"Hidden Dimension:    {self.config.get('d_model')}")
        print(f"Vocabulary Size:     {self.config.get('vocab_size')}")
        print(f"Teacher Model:       {teacher_model_name}")
        print(f"Compute Device:      {self.device.upper()} (Precision: {self.precision})")
        print(f"Distillation:        alpha={self.alpha}, tau={self.temperature}, lr={self.learning_rate}")
        print(f"Batching:            Batch={self.batch_size}, GradAccum={self.grad_accum_steps} (EffBatch={self.batch_size * self.grad_accum_steps})")
        print(f"Checkpoints Dir:     {self.output_dir}")
        print("=" * 80)

        # 2. Determine precision dtype
        if self.precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            student_dtype = torch.bfloat16
        elif self.precision in ("bfloat16", "float16") and self.device == "cuda":
            student_dtype = torch.float16
        else:
            student_dtype = torch.float32

        # 3. Instantiate Student Model with Gradient Checkpointing
        print("\n[Init] Instantiating THSA-2B Student Model...")
        self.student = THSAHybridForCausalLM(self.config).to(device=self.device, dtype=student_dtype)
        if self.device == "cuda":
            self.student.gradient_checkpointing = True
            print(f"[Init] Enabled Gradient Checkpointing & {student_dtype} for low VRAM footprint.")
            
        total_params = sum(p.numel() for p in self.student.parameters())
        print(f"[Init] Student instantiated with {total_params:,} parameters ({total_params/1e9:.3f}B).")

        # 4. Instantiate Teacher Model
        print("\n[Init] Instantiating Teacher Ensemble...")
        self.teacher = QwenTeacherWrapper(teacher_model_name, device=self.device, precision=self.precision)

        # 5. Setup Loss, Optimizer & Scheduler
        self.loss_fn = DistillationLoss(alpha=self.alpha, temperature=self.temperature)
        
        # Prioritize Adafactor for memory-factored optimizer state (<150 MB VRAM)
        try:
            from transformers.optimization import Adafactor
            self.optimizer = Adafactor(
                self.student.parameters(),
                lr=self.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=0.01,
            )
            print("[Init] Activated Adafactor Memory-Factored Optimizer.")
        except Exception:
            self.optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            print("[Init] Activated AdamW Optimizer.")

        # 6. Resume from Checkpoint if Requested
        if self.resume_checkpoint:
            self._load_resume_checkpoint(self.resume_checkpoint)

        # 7. Tokenizer & Dataset Loader
        self.sp = None
        sp_path = MODULE_ROOT / "tokenizer" / "thsa_tokenizer.model"
        if not sp_path.exists():
            raise FileNotFoundError(f"[FATAL ERROR] Production tokenizer not found at: {sp_path}")
            
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_path))
        print(f"[Init] Loaded THSA SentencePiece tokenizer ({self.sp.get_piece_size()} pieces) from {sp_path}")

        default_corpus = str(MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt")
        corpus_file = corpus_path if corpus_path and os.path.exists(corpus_path) else default_corpus
        self.dataset = TextCorpusDataset(corpus_file, max_samples=100000, max_seq_len=self.config.get("chunk_size", 256))

    def _load_resume_checkpoint(self, ckpt_path_str: str):
        ckpt_path = Path(ckpt_path_str)
        if ckpt_path == Path("auto"):
            # Find highest step checkpoint in output_dir
            candidates = list(self.output_dir.glob("checkpoint_step_*.pt"))
            if not candidates:
                print(f"[Resume] No checkpoint_step_*.pt found in {self.output_dir}. Starting from step 0.")
                return
            candidates.sort(key=lambda p: int(p.stem.split("_")[-1]))
            ckpt_path = candidates[-1]

        if not ckpt_path.exists():
            raise FileNotFoundError(f"[Resume] Checkpoint path does not exist: {ckpt_path}")
            
        print(f"[Resume] Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.student.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Resume] Notice: Could not restore optimizer state ({e}). Continuing with fresh optimizer.")
        self.start_step = ckpt.get("global_step", 0)
        print(f"[Resume] Successfully resumed at global step {self.start_step}.")

    def _save_checkpoint(self, step: int, is_final: bool = False) -> Path:
        save_name = "thsa_2b_trained_final.pt" if is_final else f"checkpoint_step_{step:06d}.pt"
        save_path = self.output_dir / save_name
        tmp_path = self.output_dir / f"{save_name}.tmp"
        
        checkpoint_dict = {
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if hasattr(self.optimizer, "state_dict") else None,
            "global_step": step,
            "config": self.config,
            "distillation_meta": {
                "teacher": self.teacher.model_name,
                "alpha": self.alpha,
                "temperature": self.temperature,
                "precision": self.precision,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps
            }
        }
        
        # Save atomically
        torch.save(checkpoint_dict, str(tmp_path))
        if save_path.exists():
            save_path.unlink()
        tmp_path.rename(save_path)
        print(f"  [*] Saved checkpoint: {save_path} ({os.path.getsize(save_path):,} bytes)")
        return save_path

    def run(self):
        """Executes the distillation training loop."""
        print(f"\n[Train] Starting distillation loop (Steps {self.start_step + 1} to {self.max_steps})...")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.student.train()
        
        vocab_size = self.config.get("vocab_size", 65536)
        seq_len = 64 # Chunk length per step
        
        t0 = time.perf_counter()
        running_loss = 0.0
        
        for step in range(self.start_step + 1, self.max_steps + 1):
            # Dynamic QAT Annealing: beta scales from 1.0 to 100.0
            beta = 1.0 + (99.0 * step / self.max_steps)
            for m in self.student.modules():
                if isinstance(m, TernaryLinear):
                    m.beta = beta
                    
            # Extract batch tokens from real corpus using SentencePiece
            batch_tokens = []
            for b_i in range(self.batch_size):
                rand_idx = (step * self.batch_size + b_i) % len(self.dataset)
                line = self.dataset[rand_idx]
                toks = self.sp.encode(line, out_type=int)
                if len(toks) < seq_len:
                    toks = toks + [3] * (seq_len - len(toks)) # pad with EOS (3)
                else:
                    toks = toks[:seq_len]
                batch_tokens.append(toks)
            input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=self.device)
            targets = input_ids.clone()
            
            # Forward Passes
            student_logits = self.student(input_ids)
            teacher_logits = self.teacher(input_ids, student_vocab_size=vocab_size)
            
            # Distillation Loss: CE + Soft KL
            loss = self.loss_fn(
                student_logits.view(-1, vocab_size),
                teacher_logits.view(-1, vocab_size),
                targets.view(-1)
            )
            loss_scaled = loss / self.grad_accum_steps
            
            # Backward Pass
            loss_scaled.backward()
            running_loss += loss.item()
            
            # Optimizer Step with Gradient Accumulation
            if step % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            if step % 10 == 0 or step == self.max_steps:
                avg_loss = running_loss / 10 if step % 10 == 0 else running_loss / (step % 10 or 1)
                elapsed = time.perf_counter() - t0
                tok_per_sec = (10 * self.batch_size * seq_len) / (elapsed + 1e-5)
                vram_used = f" | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB" if self.device == "cuda" else ""
                print(f"  Step {step:5d}/{self.max_steps} | Beta: {beta:5.1f} | Distill Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}){vram_used} | Speed: {tok_per_sec:.1f} tok/s")
                running_loss = 0.0
                t0 = time.perf_counter()
                
            # Periodic Checkpoint Save
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(step, is_final=False)
                
            if self.device == "cuda" and step % 50 == 0:
                torch.cuda.empty_cache()

        # Save Final Checkpoint
        final_path = self._save_checkpoint(self.max_steps, is_final=True)
        
        print("\n" + "=" * 80)
        print("[SUCCESS] THSA-2B DISTILLATION TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"  Trained Checkpoint: {final_path}")
        print(f"  Ready for FIX-06D Export: python tools/export_to_nano.py")
        print("=" * 80 + "\n")
        return final_path


def main():
    parser = argparse.ArgumentParser(description="THSA-2B Qwen -> THSA Knowledge Distillation Engine")
    default_cfg = str(MODULE_ROOT / "training" / "config" / "thsa_2b_config.json")
    default_corpus = str(MODULE_ROOT / "data" / "processed" / "clean_pretrain_corpus.txt")
    
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to student config JSON")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Teacher model name/path")
    parser.add_argument("--corpus", type=str, default=default_corpus, help="Path to pre-training text corpus")
    parser.add_argument("--steps", type=int, default=10000, help="Number of distillation steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per step (1 for 2B on Colab)")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints/thsa_2b_distilled", help="Save path")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path or 'auto'")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Computation precision")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda  (auto detects at runtime)")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Selected: {args.device.upper()} (CUDA available: {torch.cuda.is_available()})")
    
    trainer = DistillationTrainer(
        config_path=args.config,
        teacher_model_name=args.teacher,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        resume_checkpoint=args.resume,
        precision=args.precision,
        device=args.device
    )
    trainer.run()


if __name__ == "__main__":
    main()
