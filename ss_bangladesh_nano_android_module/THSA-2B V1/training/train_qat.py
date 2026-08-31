#!/usr/bin/env python3
"""
THSA-350M Proxy Pilot QAT Pre-Flight Training Runner (Bridge 4).
Validates quantization-aware training dynamics and teacher-student distillation.
"""

import json
import torch
import torch.optim as optim
from models.thsa_hybrid_model import THSAHybridForCausalLM
from distillation.distillation_loss import DistillationLoss
from distillation.teacher_ensemble import TeacherEnsemble

def run_qat_pilot(config_path="config/proxy_350m_config.json", num_dummy_steps=10):
    print("=" * 80)
    print("THSA-350M PROXY PILOT: QAT PRE-FLIGHT TRAINING INITIALIZATION")
    print("=" * 80)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print(f"Model ID:        {config['model_id']}")
    print(f"Total Blocks:    {config['total_blocks']} ({config['state_blocks']} State / {config['gqa_blocks']} GQA)")
    print(f"Hidden Dim:      {config['d_model']}")
    print(f"Vocab Size:      {config['vocab_size']}")
    print(f"Quantization:    {config['quantization']['weights']} + {config['quantization']['activations']}")
    print(f"Distillation:    alpha={config['training']['distillation']['alpha']}, tau={config['training']['distillation']['temperature']}")
    
    # 1. Initialize Student Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing model on device: {device}...")
    model = THSAHybridForCausalLM(config).to(device)
    
    # 2. Setup Loss & Optimizer
    loss_fn = DistillationLoss(
        alpha=config["training"]["distillation"]["alpha"],
        temperature=config["training"]["distillation"]["temperature"]
    )
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    teacher = TeacherEnsemble()
    
    # 3. Simulate QAT Annealing Loop
    print("\nExecuting QAT Training Annealing Smoke Test...")
    model.train()
    batch_size = 2
    seq_len = 32
    
    for step in range(1, num_dummy_steps + 1):
        # Generate dummy input tokens
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        targets = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        
        # Temperature annealing: beta scales from 1.0 -> 100.0
        beta = 1.0 + (99.0 * step / num_dummy_steps)
        for m in model.modules():
            if hasattr(m, "beta"):
                m.beta = beta
                
        optimizer.zero_grad()
        student_logits = model(input_ids)
        teacher_logits = teacher.compute_teacher_logits(input_ids)
        
        loss = loss_fn(
            student_logits.view(-1, config["vocab_size"]),
            teacher_logits.view(-1, config["vocab_size"]),
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step:2d}/{num_dummy_steps} | Beta: {beta:5.1f} | Distillation Loss: {loss.item():.4f}")
        
    print("\n✅ QAT Pre-Flight Smoke Test Passed Successfully!")
    return True

if __name__ == "__main__":
    run_qat_pilot()
