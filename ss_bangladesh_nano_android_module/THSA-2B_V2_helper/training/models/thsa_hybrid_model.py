"""
THSA Hybrid Model Architecture (Ternary Hybrid State-Attention).
Interleaves State/Short-Conv Blocks with Grouped Query Attention (GQA) Blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary_layers import TernaryLinear
from .state_conv_block import ShortConvStateBlock, RMSNorm

class GQAttentionBlock(nn.Module):
    """Grouped Query Attention (GQA) Block with Rotary Position Embeddings (RoPE)."""
    def __init__(self, d_model, n_query_heads, n_kv_heads, d_head, is_sensitive=False):
        super().__init__()
        self.d_model = d_model
        self.n_q = n_query_heads
        self.n_kv = n_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        
        self.q_proj = TernaryLinear(d_model, n_query_heads * d_head, bias=False, is_sensitive=is_sensitive)
        self.k_proj = TernaryLinear(d_model, n_kv_heads * d_head, bias=False, is_sensitive=is_sensitive)
        self.v_proj = TernaryLinear(d_model, n_kv_heads * d_head, bias=False, is_sensitive=is_sensitive)
        self.out_proj = TernaryLinear(n_query_heads * d_head, d_model, bias=False, is_sensitive=is_sensitive)
        
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        B, S, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        q = self.q_proj(x_norm).view(B, S, self.n_q, self.d_head).transpose(1, 2) # [B, n_q, S, d_head]
        k = self.k_proj(x_norm).view(B, S, self.n_kv, self.d_head).transpose(1, 2) # [B, n_kv, S, d_head]
        v = self.v_proj(x_norm).view(B, S, self.n_kv, self.d_head).transpose(1, 2) # [B, n_kv, S, d_head]
        
        # Repeat KV heads for GQA grouping (e.g. 20 Q / 4 KV -> 5:1 ratio)
        repeat_factor = self.n_q // self.n_kv
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Causal Attention Matrix
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, -1)
        
        y = self.out_proj(context)
        return residual + y

class GatedSwiGLUFFN(nn.Module):
    """Ternary SwiGLU Feed-Forward Network."""
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate_proj = TernaryLinear(d_model, d_ffn, bias=False)
        self.up_proj = TernaryLinear(d_model, d_ffn, bias=False)
        self.down_proj = TernaryLinear(d_ffn, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        gate = self.gate_proj(x_norm)
        up = self.up_proj(x_norm)
        swiglu = F.silu(gate) * up
        y = self.down_proj(swiglu)
        return residual + y

class THSAHybridForCausalLM(nn.Module):
    """Complete THSA Hybrid Model Architecture (350M Proxy & 2B Full Scale)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.get("vocab_size", 65536)
        self.d_model = config.get("d_model", 2560)
        self.total_blocks = config.get("total_blocks", 24)
        
        # Token Embeddings (Sensitive Layer Shield: FP16/INT8)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
        
        # Construct Interleaved Backbone Blocks
        self.layers = nn.ModuleList()
        gqa_interval = self.total_blocks // config.get("gqa_blocks", 8) # Every 3rd block is GQA
        
        for i in range(self.total_blocks):
            # Is this layer an attention block or a state block?
            if (i + 1) % gqa_interval == 0:
                is_boundary = (i == 0 or i == self.total_blocks - 1)
                layer = GQAttentionBlock(
                    d_model=self.d_model,
                    n_query_heads=config.get("n_query_heads", 20),
                    n_kv_heads=config.get("n_kv_heads", 4),
                    d_head=config.get("d_head", 128),
                    is_sensitive=is_boundary
                )
            else:
                layer = ShortConvStateBlock(d_model=self.d_model, kernel_size=4)
                
            ffn = GatedSwiGLUFFN(d_model=self.d_model, d_ffn=config.get("d_ffn", 6912))
            self.layers.append(nn.ModuleDict({"mixer": layer, "ffn": ffn}))
            
        self.final_norm = RMSNorm(self.d_model)
        
        # Output LM Head (Sensitive Layer Shield: FP16/INT8)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block["mixer"](x)
            x = block["ffn"](x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
