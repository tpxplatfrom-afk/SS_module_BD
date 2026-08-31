"""
BitNet-Style Ternary Linear Layer with Straight-Through Estimator (STE)
and Temperature-Annealed Quantization-Aware Training (QAT).
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
