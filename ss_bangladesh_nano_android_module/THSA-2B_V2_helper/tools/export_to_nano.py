#!/usr/bin/env python3
"""
THSA-2B / THSA-350M Model Serializer & Binary Exporter Tool.
Compiles PyTorch weights (trained .pt checkpoints or configs) into 64-byte SIMD-aligned .nano binary distribution packages.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import zlib
import struct
import argparse
from typing import List, Dict, Any, Tuple

# Quantization Types
NANO_QUANT_FP32 = 0
NANO_QUANT_FP16 = 1
NANO_QUANT_INT8 = 2
NANO_QUANT_INT4 = 3
NANO_QUANT_TERNARY_2BIT = 4

MAGIC_NANO = b"NANO" # 0x4E414E4F

def pack_ternary_tensor(weight_tensor) -> Tuple[bytes, float]:
    """
    Quantizes float weights to ternary {-1, 0, +1} and packs into 2-bit format (4 weights/byte).
    Returns (packed_bytes, scale_gamma).
    """
    import torch
    if isinstance(weight_tensor, torch.Tensor):
        w = weight_tensor.detach().cpu().float()
    else:
        w = torch.tensor(weight_tensor, dtype=torch.float32)
        
    gamma = w.abs().mean().clamp(min=1e-5).item()
    w_ternary = torch.clamp(torch.round(w / gamma), -1.0, 1.0).to(torch.int8)
    
    flat = w_ternary.view(-1).tolist()
    packed = bytearray((len(flat) + 3) // 4)
    for i, val in enumerate(flat):
        byte_idx = i // 4
        shift = (i % 4) * 2
        code = 1 if val > 0 else (2 if val < 0 else 0)
        packed[byte_idx] |= (code << shift)
    return bytes(packed), float(gamma)

def quantize_int8_tensor(weight_tensor) -> Tuple[bytes, float]:
    """Quantizes float weights to symmetric INT8 [-127, +127]."""
    import torch
    if isinstance(weight_tensor, torch.Tensor):
        w = weight_tensor.detach().cpu().float()
    else:
        w = torch.tensor(weight_tensor, dtype=torch.float32)
        
    scale = (w.abs().max() / 127.0).clamp(min=1e-5).item()
    w_int8 = torch.clamp(torch.round(w / scale), -127.0, 127.0).to(torch.int8)
    flat = w_int8.view(-1).tolist()
    # Convert signed int8 to unsigned bytes
    data_bytes = bytes([b if b >= 0 else b + 256 for b in flat])
    return data_bytes, float(scale)

def align_to(offset: int, alignment: int = 64) -> int:
    """Aligns an integer offset to the specified byte boundary."""
    return (offset + (alignment - 1)) & ~(alignment - 1)

def export_model_to_nano(
    config_path: str,
    output_nano_path: str,
    checkpoint_path: str = None,
    dry_run: bool = False
) -> str:
    print("=" * 80)
    print("THSA-2B: MODEL EXPORTER & 64-BYTE ALIGNED .NANO SERIALIZER")
    print("=" * 80)
    
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)
        
    state_dict = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        import torch
        print(f"Loading trained weights from checkpoint: {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        print(f"  Loaded state_dict with {len(state_dict)} tensor keys.")
    else:
        print("No trained checkpoint provided; generating structured format scaffold.")
        
    print(f"Source Model Config: {config['model_id']}")
    print(f"Target Binary Path:  {output_nano_path}")
    
    total_blocks = config["total_blocks"]
    state_blocks = config["state_blocks"]
    gqa_blocks   = config["gqa_blocks"]
    d_model      = config["d_model"]
    d_ffn        = config["d_ffn"]
    n_q          = config["n_query_heads"]
    n_kv         = config["n_kv_heads"]
    d_head       = config["d_head"]
    vocab_size   = config["vocab_size"]
    max_context  = config["max_context_tokens"]
    
    # 1. Build Tensor Manifest
    # Tensors list of tuples: (name, quant_type, shape, scale, data_bytes)
    tensors = []
    
    # (A) Token Embeddings (INT8 Sensitive Shield)
    if state_dict and "embed_tokens.weight" in state_dict:
        embed_bytes, embed_scale = quantize_int8_tensor(state_dict["embed_tokens.weight"])
    else:
        embed_size = vocab_size * d_model
        embed_bytes = bytes([i % 127 for i in range(embed_size if not dry_run else 1024)])
        embed_scale = 0.02
    tensors.append(("embed_tokens", NANO_QUANT_INT8, (vocab_size, d_model), embed_scale, embed_bytes))
    
    # (B) Backbone Layers (Ternary FFN + State/GQA)
    for l_idx in range(total_blocks):
        is_gqa = ((l_idx + 1) % (total_blocks // gqa_blocks) == 0)
        
        if is_gqa:
            # GQA Attention Projections (Q, K, V, Out)
            for proj_name, out_dim, in_dim in [
                ("q", n_q * d_head, d_model),
                ("k", n_kv * d_head, d_model),
                ("v", n_kv * d_head, d_model),
                ("out", d_model, n_q * d_head)
            ]:
                key = f"layers.{l_idx}.mixer.{proj_name}_proj.weight"
                if state_dict and key in state_dict:
                    data_bytes, scale = pack_ternary_tensor(state_dict[key])
                else:
                    sz = (out_dim * in_dim) // 4
                    data_bytes = bytes(sz if not dry_run else 64)
                    scale = 0.04
                tensors.append((f"layer_{l_idx}_attn_{proj_name}", NANO_QUANT_TERNARY_2BIT, (out_dim, in_dim), scale, data_bytes))
        else:
            # 1D Short-Conv State weights (FP32)
            conv_key = f"layers.{l_idx}.mixer.conv1d.weight"
            if state_dict and conv_key in state_dict:
                import torch
                conv_t = state_dict[conv_key].detach().cpu().float().view(-1)
                conv_bytes = struct.pack(f"<{len(conv_t)}f", *conv_t.tolist())
            else:
                conv_w_size = 4 * d_model
                conv_bytes = struct.pack(f"<{conv_w_size if not dry_run else 16}f", *([1.0] * (conv_w_size if not dry_run else 16)))
            tensors.append((f"layer_{l_idx}_state_conv_w", NANO_QUANT_FP32, (4, d_model), 1.0, conv_bytes))
            
        # FFN Weights (Gate, Up, Down)
        for ffn_name, out_dim, in_dim in [
            ("gate", d_ffn, d_model),
            ("up", d_ffn, d_model),
            ("down", d_model, d_ffn)
        ]:
            key = f"layers.{l_idx}.ffn.{ffn_name}_proj.weight"
            if state_dict and key in state_dict:
                data_bytes, scale = pack_ternary_tensor(state_dict[key])
            else:
                sz = (out_dim * in_dim) // 4
                data_bytes = bytes(sz if not dry_run else 64)
                scale = 0.035
            tensors.append((f"layer_{l_idx}_ffn_{ffn_name}", NANO_QUANT_TERNARY_2BIT, (out_dim, in_dim), scale, data_bytes))
            
    # (C) Final RMSNorm Gamma (FP32)
    norm_key = "final_norm.weight"
    if state_dict and norm_key in state_dict:
        norm_t = state_dict[norm_key].detach().cpu().float().view(-1)
        norm_bytes = struct.pack(f"<{len(norm_t)}f", *norm_t.tolist())
    else:
        norm_bytes = struct.pack(f"<{d_model if not dry_run else 16}f", *([1.0] * (d_model if not dry_run else 16)))
    tensors.append(("final_norm", NANO_QUANT_FP32, (d_model,), 1.0, norm_bytes))
    
    # (D) LM Head (INT8 Sensitive Shield)
    lm_head_key = "lm_head.weight"
    if state_dict and lm_head_key in state_dict:
        lm_head_bytes, lm_head_scale = quantize_int8_tensor(state_dict[lm_head_key])
    else:
        lm_head_size = d_model * vocab_size
        lm_head_bytes = bytes([i % 127 for i in range(lm_head_size if not dry_run else 1024)])
        lm_head_scale = 0.025
    tensors.append(("lm_head", NANO_QUANT_INT8, (vocab_size, d_model), lm_head_scale, lm_head_bytes))
    
    tensor_count = len(tensors)
    print(f"Generated Tensor Manifest: {tensor_count} tensors.")
    
    # 2. Serialize to Binary File with 64-byte Alignment
    header_size = 64
    descriptor_table_size = tensor_count * 32
    raw_payload_start = header_size + descriptor_table_size
    payload_start = align_to(raw_payload_start, 64)
    
    descriptors = []
    current_offset = payload_start
    payload_bytes_list = []
    
    for t_id, (name, q_type, shape, scale, data) in enumerate(tensors):
        aligned_offset = align_to(current_offset, 64)
        pad_needed = aligned_offset - current_offset
        if pad_needed > 0:
            payload_bytes_list.append(bytes(pad_needed))
            
        data_len = len(data)
        desc = struct.pack("<IIQQfI", t_id, q_type, aligned_offset, data_len, scale, 0)
        descriptors.append(desc)
        
        payload_bytes_list.append(data)
        current_offset = aligned_offset + data_len
        
    desc_block = b"".join(descriptors)
    pad_to_payload = bytes(payload_start - (header_size + len(desc_block)))
    payload_block = b"".join(payload_bytes_list)
    
    crc_value = zlib.crc32(desc_block + pad_to_payload + payload_block)
    
    header = struct.pack(
        "<4sHHHHIIHHHHI I I I 20s",
        MAGIC_NANO,
        config.get("format_version", 1),
        total_blocks,
        state_blocks,
        gqa_blocks,
        d_model,
        d_ffn,
        n_q,
        n_kv,
        d_head,
        0, # padding
        vocab_size,
        max_context,
        crc_value,
        tensor_count,
        bytes(20) # reserved
    )
    assert len(header) == 64, f"Header size is {len(header)}, expected 64"
    
    os.makedirs(os.path.dirname(output_nano_path) or ".", exist_ok=True)
    with open(output_nano_path, "wb") as f_out:
        f_out.write(header)
        f_out.write(desc_block)
        f_out.write(pad_to_payload)
        f_out.write(payload_block)
        
    total_file_size = os.path.getsize(output_nano_path)
    print(f"\n[SUCCESS] Serialized {output_nano_path}")
    print(f"   Total Size:     {total_file_size / (1024*1024):.2f} MB")
    print(f"   Header Size:    {len(header)} bytes")
    print(f"   CRC32 Checksum: 0x{crc_value:08X}")
    print(f"   Payload Offset: {payload_start} bytes (64-byte aligned: {payload_start % 64 == 0})")
    return output_nano_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to .nano binary distribution format")
    parser.add_argument("--config", type=str, default="training/config/proxy_350m_config.json", help="Path to architecture config JSON")
    parser.add_argument("--output", type=str, default="models/model_trained.nano", help="Output .nano binary path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to trained PyTorch .pt checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Generate small dry-run test artifact")
    args = parser.parse_args()
    
    export_model_to_nano(
        config_path=args.config,
        output_nano_path=args.output,
        checkpoint_path=args.checkpoint,
        dry_run=args.dry_run
    )
