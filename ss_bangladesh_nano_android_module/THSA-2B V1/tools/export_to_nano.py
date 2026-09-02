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
    checkpoint_path: str
) -> str:
    print("=" * 80)
    print("THSA-2B: STRICT MODEL EXPORTER & 64-BYTE ALIGNED .NANO SERIALIZER")
    print("=" * 80)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[FATAL ERROR] Checkpoint path is mandatory and must exist: {checkpoint_path}. "
            f"Synthetic or dummy exports are strictly prohibited."
        )
        
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)
        
    import torch
    print(f"Loading trained weights from checkpoint: {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"[FATAL ERROR] Checkpoint is not a valid dict, got {type(ckpt)}")
        
    state_dict = ckpt.get("model_state_dict", ckpt)
    print(f"  Successfully loaded state_dict with {len(state_dict)} tensor keys.")
    print(f"Source Model Config: {config.get('model_id', 'UNKNOWN')}")
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
    
    # Strict Format V2 Pre-Serialization Contract Audit
    format_version = config.get("format_version", 2)
    if format_version != 2:
        raise ValueError(f"[FATAL ERROR] Format version must be 0x0002 for THSA-2B production export, got {format_version}")

    expected_keys = {"embed_tokens.weight", "final_norm.weight", "lm_head.weight"}
    for l_idx in range(total_blocks):
        is_gqa = ((l_idx + 1) % (total_blocks // gqa_blocks) == 0)
        if is_gqa:
            expected_keys.update({
                f"layers.{l_idx}.mixer.q_proj.weight",
                f"layers.{l_idx}.mixer.k_proj.weight",
                f"layers.{l_idx}.mixer.v_proj.weight",
                f"layers.{l_idx}.mixer.out_proj.weight",
                f"layers.{l_idx}.mixer.norm.weight",
            })
        else:
            expected_keys.update({
                f"layers.{l_idx}.mixer.conv1d.weight",
                f"layers.{l_idx}.mixer.conv1d.bias",
                f"layers.{l_idx}.mixer.in_proj.weight",
                f"layers.{l_idx}.mixer.out_proj.weight",
                f"layers.{l_idx}.mixer.norm.weight",
            })
        expected_keys.update({
            f"layers.{l_idx}.ffn.gate_proj.weight",
            f"layers.{l_idx}.ffn.up_proj.weight",
            f"layers.{l_idx}.ffn.down_proj.weight",
            f"layers.{l_idx}.ffn.norm.weight",
        })

    if len(state_dict) != 219:
        raise ValueError(f"[FATAL ERROR] Expected exactly 219 tensor keys in checkpoint, got {len(state_dict)}")

    missing_keys = expected_keys - set(state_dict.keys())
    extra_keys = set(state_dict.keys()) - expected_keys
    if missing_keys:
        raise KeyError(f"[FATAL ERROR] Checkpoint missing required keys: {sorted(list(missing_keys))[:5]} (total {len(missing_keys)})")
    if extra_keys:
        raise ValueError(f"[FATAL ERROR] Checkpoint contains unrecognized extra keys: {sorted(list(extra_keys))[:5]} (total {len(extra_keys)})")

    total_params = sum(t.numel() for t in state_dict.values())
    if total_params != 2050296320:
        raise ValueError(f"[FATAL ERROR] Parameter count mismatch: expected 2,050,296,320, got {total_params}")

    print(f"  --> Pre-serialization Contract Audit: 219 keys verified, {total_params:,} parameters exact.")

    # 1. Build Tensor Manifest from REAL Checkpoint
    tensors = []
    
    # (A) Token Embeddings (INT8 Sensitive Shield)
    if "embed_tokens.weight" not in state_dict:
        raise KeyError("[FATAL ERROR] Missing required tensor 'embed_tokens.weight' in checkpoint.")
    embed_t = state_dict["embed_tokens.weight"]
    if list(embed_t.shape) != [vocab_size, d_model]:
        raise ValueError(
            f"[FATAL ERROR] Architecture shape mismatch in 'embed_tokens.weight': "
            f"expected [{vocab_size}, {d_model}], got {list(embed_t.shape)}"
        )
    embed_bytes, embed_scale = quantize_int8_tensor(embed_t)
    tensors.append(("embed_tokens", NANO_QUANT_INT8, (vocab_size, d_model), embed_scale, embed_bytes))
    
    # (B) Backbone Layers (9 tensors per layer * 24 layers = 216 tensors)
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
                if key not in state_dict:
                    raise KeyError(f"[FATAL ERROR] Missing required GQA tensor '{key}' in checkpoint.")
                proj_t = state_dict[key]
                if list(proj_t.shape) != [out_dim, in_dim]:
                    raise ValueError(f"[FATAL ERROR] Shape mismatch for '{key}': expected [{out_dim}, {in_dim}], got {list(proj_t.shape)}")
                data_bytes, scale = pack_ternary_tensor(proj_t)
                tensors.append((f"layer_{l_idx}_attn_{proj_name}", NANO_QUANT_TERNARY_2BIT, (out_dim, in_dim), scale, data_bytes))
            
            # GQA Mixer RMSNorm (FP32)
            mixer_norm_key = f"layers.{l_idx}.mixer.norm.weight"
            if mixer_norm_key not in state_dict:
                raise KeyError(f"[FATAL ERROR] Missing required mixer norm tensor '{mixer_norm_key}' in checkpoint.")
            norm_t = state_dict[mixer_norm_key].detach().cpu().float().view(-1)
            if norm_t.numel() != d_model:
                raise ValueError(f"[FATAL ERROR] Shape mismatch for '{mixer_norm_key}': expected {d_model}, got {norm_t.numel()}")
            norm_bytes = struct.pack(f"<{len(norm_t)}f", *norm_t.tolist())
            tensors.append((f"layer_{l_idx}_mixer_norm", NANO_QUANT_FP32, (d_model,), 1.0, norm_bytes))
        else:
            # State Conv1D Weights (FP32)
            conv_key = f"layers.{l_idx}.mixer.conv1d.weight"
            if conv_key not in state_dict:
                raise KeyError(f"[FATAL ERROR] Missing required state conv tensor '{conv_key}' in checkpoint.")
            conv_t = state_dict[conv_key].detach().cpu().float().view(-1)
            expected_numel = 4 * d_model
            if conv_t.numel() != expected_numel:
                raise ValueError(f"[FATAL ERROR] Shape mismatch for '{conv_key}': expected {expected_numel} elements, got {conv_t.numel()}")
            conv_bytes = struct.pack(f"<{len(conv_t)}f", *conv_t.tolist())
            tensors.append((f"layer_{l_idx}_state_conv_w", NANO_QUANT_FP32, (4, d_model), 1.0, conv_bytes))
            
            # State Conv1D Bias (FP32)
            bias_key = f"layers.{l_idx}.mixer.conv1d.bias"
            if bias_key not in state_dict:
                raise KeyError(f"[FATAL ERROR] Missing required state conv bias tensor '{bias_key}' in checkpoint.")
            bias_t = state_dict[bias_key].detach().cpu().float().view(-1)
            if bias_t.numel() != d_model:
                raise ValueError(f"[FATAL ERROR] Shape mismatch for '{bias_key}': expected {d_model}, got {bias_t.numel()}")
            bias_bytes = struct.pack(f"<{len(bias_t)}f", *bias_t.tolist())
            tensors.append((f"layer_{l_idx}_state_conv_b", NANO_QUANT_FP32, (d_model,), 1.0, bias_bytes))
            
            # State In-Projection (Ternary 2-bit, [5120, 2560])
            in_key = f"layers.{l_idx}.mixer.in_proj.weight"
            if in_key not in state_dict:
                raise KeyError(f"[FATAL ERROR] Missing required state in_proj tensor '{in_key}' in checkpoint.")
            in_t = state_dict[in_key]
            if list(in_t.shape) != [2 * d_model, d_model]:
                raise ValueError(f"[FATAL ERROR] Shape mismatch for '{in_key}': expected [{2 * d_model}, {d_model}], got {list(in_t.shape)}")
            in_bytes, in_scale = pack_ternary_tensor(in_t)
            tensors.append((f"layer_{l_idx}_state_in_proj", NANO_QUANT_TERNARY_2BIT, (2 * d_model, d_model), in_scale, in_bytes))
            
            # State Out-Projection (Ternary 2-bit, [2560, 2560])
            out_key = f"layers.{l_idx}.mixer.out_proj.weight"
            if out_key not in state_dict:
                raise KeyError(f"[FATAL ERROR] Missing required state out_proj tensor '{out_key}' in checkpoint.")
            out_t = state_dict[out_key]
            if list(out_t.shape) != [d_model, d_model]:
                raise ValueError(f"[FATAL ERROR] Shape mismatch for '{out_key}': expected [{d_model}, {d_model}], got {list(out_t.shape)}")
            out_bytes, out_scale = pack_ternary_tensor(out_t)
            tensors.append((f"layer_{l_idx}_state_out_proj", NANO_QUANT_TERNARY_2BIT, (d_model, d_model), out_scale, out_bytes))
            
            # State Mixer RMSNorm (FP32)
            mixer_norm_key = f"layers.{l_idx}.mixer.norm.weight"
            if mixer_norm_key not in state_dict:
                raise KeyError(f"[FATAL ERROR] Missing required mixer norm tensor '{mixer_norm_key}' in checkpoint.")
            norm_t = state_dict[mixer_norm_key].detach().cpu().float().view(-1)
            if norm_t.numel() != d_model:
                raise ValueError(f"[FATAL ERROR] Shape mismatch for '{mixer_norm_key}': expected {d_model}, got {norm_t.numel()}")
            norm_bytes = struct.pack(f"<{len(norm_t)}f", *norm_t.tolist())
            tensors.append((f"layer_{l_idx}_mixer_norm", NANO_QUANT_FP32, (d_model,), 1.0, norm_bytes))
            
        # FFN Weights (Gate, Up, Down)
        for ffn_name, out_dim, in_dim in [
            ("gate", d_ffn, d_model),
            ("up", d_ffn, d_model),
            ("down", d_model, d_ffn)
        ]:
            key = f"layers.{l_idx}.ffn.{ffn_name}_proj.weight"
            if key not in state_dict:
                raise KeyError(f"[FATAL ERROR] Missing required FFN tensor '{key}' in checkpoint.")
            ffn_t = state_dict[key]
            if list(ffn_t.shape) != [out_dim, in_dim]:
                raise ValueError(f"[FATAL ERROR] Shape mismatch for '{key}': expected [{out_dim}, {in_dim}], got {list(ffn_t.shape)}")
            data_bytes, scale = pack_ternary_tensor(ffn_t)
            tensors.append((f"layer_{l_idx}_ffn_{ffn_name}", NANO_QUANT_TERNARY_2BIT, (out_dim, in_dim), scale, data_bytes))
            
        # FFN RMSNorm (FP32)
        ffn_norm_key = f"layers.{l_idx}.ffn.norm.weight"
        if ffn_norm_key not in state_dict:
            raise KeyError(f"[FATAL ERROR] Missing required FFN norm tensor '{ffn_norm_key}' in checkpoint.")
        fn_t = state_dict[ffn_norm_key].detach().cpu().float().view(-1)
        if fn_t.numel() != d_model:
            raise ValueError(f"[FATAL ERROR] Shape mismatch for '{ffn_norm_key}': expected {d_model}, got {fn_t.numel()}")
        fn_bytes = struct.pack(f"<{len(fn_t)}f", *fn_t.tolist())
        tensors.append((f"layer_{l_idx}_ffn_norm", NANO_QUANT_FP32, (d_model,), 1.0, fn_bytes))
            
    # (C) Final RMSNorm Gamma (FP32)
    norm_key = "final_norm.weight"
    if norm_key not in state_dict:
        raise KeyError(f"[FATAL ERROR] Missing required final norm tensor '{norm_key}' in checkpoint.")
    norm_t = state_dict[norm_key].detach().cpu().float().view(-1)
    if norm_t.numel() != d_model:
        raise ValueError(f"[FATAL ERROR] Shape mismatch for '{norm_key}': expected {d_model}, got {norm_t.numel()}")
    norm_bytes = struct.pack(f"<{len(norm_t)}f", *norm_t.tolist())
    tensors.append(("final_norm", NANO_QUANT_FP32, (d_model,), 1.0, norm_bytes))
    
    # (D) LM Head (INT8 Sensitive Shield)
    lm_head_key = "lm_head.weight"
    if lm_head_key not in state_dict:
        raise KeyError(f"[FATAL ERROR] Missing required LM head tensor '{lm_head_key}' in checkpoint.")
    lm_head_t = state_dict[lm_head_key]
    if list(lm_head_t.shape) != [vocab_size, d_model]:
        raise ValueError(f"[FATAL ERROR] Shape mismatch for '{lm_head_key}': expected [{vocab_size}, {d_model}], got {list(lm_head_t.shape)}")
    lm_head_bytes, lm_head_scale = quantize_int8_tensor(lm_head_t)
    tensors.append(("lm_head", NANO_QUANT_INT8, (vocab_size, d_model), lm_head_scale, lm_head_bytes))
    
    tensor_count = len(tensors)
    if tensor_count != 219:
        raise ValueError(f"[FATAL ERROR] Exported tensor count must be 219, got {tensor_count}")
    print(f"Extracted and quantized {tensor_count} verified tensors from checkpoint.")
    
    # 2. Serialize to Binary File with 64-byte Alignment
    header_size = 64
    descriptor_table_size = tensor_count * 32
    raw_payload_start = header_size + descriptor_table_size
    payload_start = align_to(raw_payload_start, 64)
    if payload_start != 7104:
        raise ValueError(f"[FATAL ERROR] V2 first payload offset must be 7104, got {payload_start}")
    if struct.calcsize("<IIQQfI") != 32:
        raise ValueError(f"[FATAL ERROR] Descriptor ABI size must be 32 bytes, got {struct.calcsize('<IIQQfI')}")
    
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
        config.get("format_version", 2),
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
    parser = argparse.ArgumentParser(description="Strict PyTorch model to .nano binary exporter")
    parser.add_argument("--config", type=str, required=True, help="Path to architecture config JSON")
    parser.add_argument("--output", type=str, required=True, help="Output .nano binary path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained PyTorch .pt checkpoint")
    args = parser.parse_args()
    
    export_model_to_nano(
        config_path=args.config,
        output_nano_path=args.output,
        checkpoint_path=args.checkpoint
    )
