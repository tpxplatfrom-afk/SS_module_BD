"""
BitNet-Style Ternary Linear Layer with Straight-Through Estimator (STE),
Temperature-Annealed Quantization-Aware Training (QAT), and native LoRA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightQuantizerSTE(torch.autograd.Function):
    """Straight-Through Estimator for 1.58-bit Ternary Weights {-1, 0, +1}."""
    @staticmethod
    def forward(ctx, weight, beta=100.0):
        gamma = weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        if beta < 50.0:
            w_quant = torch.clamp(torch.round(torch.tanh(beta * (weight / gamma))), -1.0, 1.0)
        else:
            w_quant = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        ctx.save_for_backward(weight)
        return w_quant * gamma

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        # STE: Pass gradient straight through within [-1.5, +1.5] clipping window
        return grad_output * (weight.abs() <= 1.5), None

class ActivationQuantizerSTE(torch.autograd.Function):
    """Dynamic INT8 Activation Quantizer with STE."""
    @staticmethod
    def forward(ctx, x):
        scale = (x.abs().max(dim=-1, keepdim=True)[0] / 127.0).clamp(min=1e-5)
        x_quant = torch.clamp(torch.round(x / scale), -128.0, 127.0)
        return x_quant * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class TernaryLinear(nn.Module):
    """Ternary Linear Projection Layer (W in {-1, 0, +1}, INT8 activations)."""
    def __init__(self, in_features, out_features, bias=False, is_sensitive=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_sensitive = is_sensitive # If true, stays in INT8/FP16 (Bridge 1 Sensitive Shield)
        self.beta = 100.0 # QAT temperature annealing factor
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.is_sensitive:
            # Sensitive Layer Shield: Standard linear projection
            return F.linear(x, self.weight, self.bias)
        
        # Quantize activations to INT8 & weights to Ternary
        x_q = ActivationQuantizerSTE.apply(x)
        w_q = WeightQuantizerSTE.apply(self.weight, self.beta)
        
        return F.linear(x_q, w_q, self.bias)

class TernaryLoRALinear(TernaryLinear):
    """
    Ternary Linear Projection Layer with integrated Low-Rank Adaptation (LoRA).
    Enables ultra-fast, memory-efficient QAT fine-tuning on Google Colab GPUs (T4/V100/A100).
    """
    def __init__(self, in_features, out_features, bias=False, is_sensitive=False,
                 lora_r=16, lora_alpha=32.0, lora_dropout=0.05):
        super().__init__(in_features, out_features, bias=bias, is_sensitive=is_sensitive)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.scaling = lora_alpha / lora_r if lora_r > 0 else 1.0
        self.merged = False
        
        if lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # Freeze base weights when LoRA is active for fast training
            self.weight.requires_grad = False
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def merge_lora_weights(self):
        """Merges LoRA adapter weights into base weights for zero-overhead .nano export."""
        if self.lora_r > 0 and not self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.weight.data += delta_w
                self.merged = True

    def forward(self, x):
        if self.merged or self.lora_r == 0:
            return super().forward(x)
            
        base_out = super().forward(x)
        # LoRA bypass branch: (x @ A^T) @ B^T * scaling
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out
