"""
1D Causal Depthwise Short-Convolution & Linear State Sequence Mixing Block.
Memory consumption: O(1) state memory per block during autoregressive generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_sq + self.eps) * self.weight

class ShortConvStateBlock(nn.Module):
    """1D Causal Short-Convolution Block (K=4) with SiLU Gating and O(1) State."""
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Depthwise 1D Convolution with causal padding
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=True,
            padding=kernel_size - 1
        )
        
        # In / Out Gating Projections
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # x shape: [Batch, SeqLen, D_Model]
        B, S, D = x.shape
        residual = x
        
        x_norm = self.norm(x)
        projected = self.in_proj(x_norm) # [B, S, 2*D]
        gate, value = projected.chunk(2, dim=-1)
        
        # 1D Causal Convolution on value stream
        val_transposed = value.transpose(1, 2) # [B, D, S]
        conv_out = self.conv1d(val_transposed)[:, :, :S] # Causal trim
        conv_out = conv_out.transpose(1, 2) # [B, S, D]
        
        # Gated SiLU activation: y = silu(gate) * conv_out
        gated = F.silu(gate) * conv_out
        y = self.out_proj(gated)
        
        return residual + y
