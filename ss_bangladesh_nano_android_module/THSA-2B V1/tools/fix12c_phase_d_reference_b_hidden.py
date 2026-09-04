#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX-12C Phase D — Reference-B Layer-by-Layer Hidden State Capture
================================================================
Dumps all 25 intermediate checkpoints for each canonical prompt as raw float32
little-endian binary files into:
  tools/fix12c/reference_b/prompt_{pi}/ckpt*.bin
"""

import os
import sys
import json
import struct
import hashlib
import time
import math
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SCRIPT_DIR.parent
NANO_PATH = MODULE_ROOT / "android" / "src" / "main" / "assets" / "model.nano"
BASE_OUT_DIR = SCRIPT_DIR / "fix12c" / "reference_b"
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA = "0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64"
EXPECTED_SIZE = 765_477_824

PROMPTS = [
    ("TEST-A", "2+2=?"),
    ("TEST-B", "বাংলাদেশের রাজধানী কী?"),
    ("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?"),
    ("TEST-D", "১২ × ৮ = ?"),
    ("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।"),
]

TOKEN_IDS = {
    "TEST-A": [360, 43226, 64782, 64792],
    "TEST-B": [1620, 3715, 3101, 64792],
    "TEST-C": [4874, 6494, 4186, 4289, 1357, 263, 5821, 19591, 64792],
    "TEST-D": [2232, 15325, 1656, 1718, 2667],
    "TEST-E": [2829, 1620, 3715, 64705],
}

DETAILED_BLOCKS = {0, 1, 2, 3, 4, 5, 12, 23}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

HEADER_FMT = "<4sHHHHIIHHHHIIII20s"   # 64 bytes
DESC_FMT   = "<IIQQfI"                  # 32 bytes

def parse_header(f):
    raw = f.read(64)
    magic, version, total_blocks, state_blocks, gqa_blocks, \
    d_model, d_ffn, n_q, n_kv, d_head, pad, vocab_size, max_context, \
    crc32, tensor_count, _res = struct.unpack(HEADER_FMT, raw)
    assert magic == b"NANO", f"Bad magic: {magic}"
    return dict(version=version, total_blocks=total_blocks,
                state_blocks=state_blocks, gqa_blocks=gqa_blocks,
                d_model=d_model, d_ffn=d_ffn, n_q=n_q, n_kv=n_kv,
                d_head=d_head, vocab_size=vocab_size, max_context=max_context,
                crc32=hex(crc32), tensor_count=tensor_count)

def parse_descriptors(f, n):
    descs = []
    for _ in range(n):
        raw = f.read(32)
        tid, qt, off, sz, scale, _pad = struct.unpack(DESC_FMT, raw)
        descs.append(dict(id=tid, qt=qt, off=off, sz=sz, scale=scale))
    return descs

QT_FP32    = 0
QT_INT8    = 2
QT_TERNARY = 4

def load_tensor_data(nano_path, desc):
    with open(nano_path, "rb") as f:
        f.seek(desc["off"])
        return f.read(desc["sz"])

def dequant_fp32(data):
    return np.frombuffer(data, dtype=np.float32).copy()

def dequant_int8(data, scale):
    return np.frombuffer(data, dtype=np.int8).astype(np.float32) * scale

def ternary_row_matmul(packed, n_rows, n_cols, scale, x):
    vals_per_byte = 4
    bytes_per_row = (n_cols + vals_per_byte - 1) // vals_per_byte
    out = np.zeros(n_rows, dtype=np.float32)
    x32 = x.astype(np.float32)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    for r in range(n_rows):
        row_bytes = packed_arr[r * bytes_per_row : (r+1) * bytes_per_row]
        b0 = (row_bytes & 0x03).astype(np.int8)
        b1 = ((row_bytes >> 2) & 0x03).astype(np.int8)
        b2 = ((row_bytes >> 4) & 0x03).astype(np.int8)
        b3 = ((row_bytes >> 6) & 0x03).astype(np.int8)
        codes = np.stack([b0, b1, b2, b3], axis=1).ravel()[:n_cols]
        row = np.where(codes == 0, 0.0,
              np.where(codes == 1, scale, -scale)).astype(np.float32)
        out[r] = np.dot(row, x32)
    return out

def apply_weight(nano_path, desc, shape, x):
    data = load_tensor_data(nano_path, desc)
    qt = desc["qt"]
    if qt == QT_TERNARY:
        n_rows, n_cols = shape[0], shape[1]
        return ternary_row_matmul(data, n_rows, n_cols, desc["scale"], x)
    elif qt == QT_INT8:
        w = dequant_int8(data, desc["scale"]).reshape(shape)
        return (w @ x.astype(np.float32)).astype(np.float32)
    else:
        w = np.frombuffer(data, dtype=np.float32).reshape(shape)
        return (w @ x.astype(np.float32)).astype(np.float32)

def load_fp32_weight(nano_path, desc, shape):
    data = load_tensor_data(nano_path, desc)
    return dequant_fp32(data).reshape(shape)

def is_gqa(li):
    return (li + 1) % 3 == 0

def tensor_base(li):
    return 1 + li * 9

def rms_norm(x, w, eps=1e-6):
    x = x.astype(np.float32)
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * w.astype(np.float32)

def silu(x):
    return (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)

def dump_vector(p_dir: Path, name: str, vec: np.ndarray) -> dict:
    f32 = vec.astype(np.float32)
    raw = f32.tobytes()
    out_file = p_dir / f"{name}.bin"
    with open(out_file, "wb") as f:
        f.write(raw)
    h = sha256_bytes(raw)
    v_dbl = f32.astype(np.float64).ravel()
    l2 = float(np.sqrt(np.dot(v_dbl, v_dbl)))
    return {
        "name": name,
        "file": str(out_file.name),
        "shape": list(vec.shape),
        "dim": len(v_dbl),
        "min": float(v_dbl.min()),
        "max": float(v_dbl.max()),
        "mean": float(v_dbl.mean()),
        "l2_norm": l2,
        "sha256": h,
    }

def main():
    print("=" * 80)
    print("FIX-12C PHASE D — REFERENCE-B LAYERWISE HIDDEN STATE CAPTURE")
    print("=" * 80)

    sz = NANO_PATH.stat().st_size
    print(f"model.nano: {sz:,} bytes")
    assert sz == EXPECTED_SIZE, "model.nano size mismatch!"
    nano_sha = sha256_file(NANO_PATH)
    assert nano_sha == EXPECTED_SHA, "model.nano SHA mismatch!"
    print(f"model.nano SHA: {nano_sha} [PASS]")

    with open(NANO_PATH, "rb") as f:
        hdr = parse_header(f)
        descs = parse_descriptors(f, hdr["tensor_count"])

    D = hdr["d_model"]   # 2560
    F = hdr["d_ffn"]     # 6912
    V = hdr["vocab_size"] # 65536
    NQ = hdr["n_q"]      # 20
    NKV = hdr["n_kv"]    # 4
    DH = hdr["d_head"]   # 128
    L = hdr["total_blocks"] # 24

    # Preload embedding & lm_head
    desc_emb = descs[0]
    emb_data = load_tensor_data(NANO_PATH, desc_emb)
    emb_weights = dequant_int8(emb_data, desc_emb["scale"]).reshape(V, D)
    del emb_data
    print("Embedding weights preloaded.")

    desc_fn = descs[217]
    fn_gamma = load_fp32_weight(NANO_PATH, desc_fn, (D,))

    desc_lm = descs[218]
    lm_data = load_tensor_data(NANO_PATH, desc_lm)
    lm_weights = dequant_int8(lm_data, desc_lm["scale"]).reshape(V, D)
    del lm_data
    print("LM head weights preloaded.")

    all_prompt_records = {}

    for pi, (label, prompt_text) in enumerate(PROMPTS):
        token_ids = TOKEN_IDS[label]
        num_toks = len(token_ids)
        print(f"\n[{label}] '{prompt_text}' ({num_toks} tokens: {token_ids})")

        p_dir = BASE_OUT_DIR / f"prompt_{pi}"
        p_dir.mkdir(parents=True, exist_ok=True)

        prompt_ckpts = {}

        # Reset session state across blocks
        # Conv state: K-1 = 3 history vectors per state block
        conv_states = {l: np.zeros((3, D), dtype=np.float32) for l in range(L) if not is_gqa(l)}
        # GQA KV cache: list of [NKV, DH] for each GQA block
        kv_k = {l: [] for l in range(L) if is_gqa(l)}
        kv_v = {l: [] for l in range(L) if is_gqa(l)}

        # Process prompt tokens sequentially
        for t_idx, tok_id in enumerate(token_ids):
            is_last = (t_idx == num_toks - 1)

            # 1. Embedding lookup
            h = emb_weights[tok_id].copy().astype(np.float32)
            if is_last:
                prompt_ckpts["ckpt01_embed"] = dump_vector(p_dir, "ckpt01_embed", h)

            # 2. Backbone 24 blocks
            for li in range(L):
                base = tensor_base(li)
                is_g = is_gqa(li)
                is_detailed = (li in DETAILED_BLOCKS and is_last)

                # Capture Block Input
                if is_last:
                    prompt_ckpts[f"ckpt02_block_{li:02d}_input"] = dump_vector(p_dir, f"ckpt02_block_{li:02d}_input", h)

                if not is_g:
                    # ── State Block ──
                    # slot 4: mixer.norm
                    norm_w = load_fp32_weight(NANO_PATH, descs[base + 4], (D,))
                    h_normed = rms_norm(h, norm_w)
                    if is_detailed:
                        prompt_ckpts[f"ckpt03_block_{li:02d}_state_norm"] = dump_vector(p_dir, f"ckpt03_block_{li:02d}_state_norm", h_normed)

                    # slot 2: in_proj [5120, D]
                    ip = apply_weight(NANO_PATH, descs[base + 2], (5120, D), h_normed)
                    if is_detailed:
                        prompt_ckpts[f"ckpt04_block_{li:02d}_state_in_proj"] = dump_vector(p_dir, f"ckpt04_block_{li:02d}_state_in_proj", ip)

                    gate_s, value_s = ip[:D], ip[D:2*D]
                    if is_detailed:
                        prompt_ckpts[f"ckpt05a_block_{li:02d}_state_gate"] = dump_vector(p_dir, f"ckpt05a_block_{li:02d}_state_gate", gate_s)
                        prompt_ckpts[f"ckpt05b_block_{li:02d}_state_value"] = dump_vector(p_dir, f"ckpt05b_block_{li:02d}_state_value", value_s)

                    # slot 0: conv1d.weight [D, 1, 4]
                    # slot 1: conv1d.bias [D]
                    conv_w = load_fp32_weight(NANO_PATH, descs[base + 0], (D, 1, 4))
                    conv_b = load_fp32_weight(NANO_PATH, descs[base + 1], (D,))

                    # Causal Conv1D using conv_state history (PyTorch Conv1d exact equivalence)
                    s0 = conv_states[li][0] # t-3
                    s1 = conv_states[li][1] # t-2
                    s2 = conv_states[li][2] # t-1
                    # In PyTorch F.conv1d(padding=3):
                    # conv_w[:, 0, 0] = W_0 (t-3), conv_w[:, 0, 1] = W_1 (t-2), conv_w[:, 0, 2] = W_2 (t-1), conv_w[:, 0, 3] = W_3 (t)
                    conv_out = (s0 * conv_w[:, 0, 0] +
                                s1 * conv_w[:, 0, 1] +
                                s2 * conv_w[:, 0, 2] +
                                value_s * conv_w[:, 0, 3] + conv_b)
                    # Update state FIFO
                    conv_states[li][0] = s1
                    conv_states[li][1] = s2
                    conv_states[li][2] = value_s

                    if is_detailed:
                        prompt_ckpts[f"ckpt06_block_{li:02d}_state_conv"] = dump_vector(p_dir, f"ckpt06_block_{li:02d}_state_conv", conv_out)

                    silu_gate = silu(gate_s)
                    if is_detailed:
                        prompt_ckpts[f"ckpt07_block_{li:02d}_state_silu"] = dump_vector(p_dir, f"ckpt07_block_{li:02d}_state_silu", silu_gate)

                    gated = silu_gate * conv_out
                    if is_detailed:
                        prompt_ckpts[f"ckpt08_block_{li:02d}_state_gated"] = dump_vector(p_dir, f"ckpt08_block_{li:02d}_state_gated", gated)

                    # slot 3: out_proj [D, D]
                    state_out = apply_weight(NANO_PATH, descs[base + 3], (D, D), gated)
                    if is_detailed:
                        prompt_ckpts[f"ckpt09_block_{li:02d}_state_out_proj"] = dump_vector(p_dir, f"ckpt09_block_{li:02d}_state_out_proj", state_out)

                    h = h + state_out
                    if is_detailed:
                        prompt_ckpts[f"ckpt10_block_{li:02d}_state_residual"] = dump_vector(p_dir, f"ckpt10_block_{li:02d}_state_residual", h)

                else:
                    # ── GQA Block ──
                    # slot 4: mixer.norm
                    norm_w = load_fp32_weight(NANO_PATH, descs[base + 4], (D,))
                    h_normed = rms_norm(h, norm_w)
                    if is_detailed:
                        prompt_ckpts[f"ckpt11_block_{li:02d}_gqa_norm"] = dump_vector(p_dir, f"ckpt11_block_{li:02d}_gqa_norm", h_normed)

                    # slot 0: q_proj [NQ*DH, D]
                    # slot 1: k_proj [NKV*DH, D]
                    # slot 2: v_proj [NKV*DH, D]
                    q = apply_weight(NANO_PATH, descs[base + 0], (NQ * DH, D), h_normed)
                    k = apply_weight(NANO_PATH, descs[base + 1], (NKV * DH, D), h_normed)
                    v = apply_weight(NANO_PATH, descs[base + 2], (NKV * DH, D), h_normed)

                    if is_detailed:
                        prompt_ckpts[f"ckpt12a_block_{li:02d}_gqa_q"] = dump_vector(p_dir, f"ckpt12a_block_{li:02d}_gqa_q", q)
                        prompt_ckpts[f"ckpt12b_block_{li:02d}_gqa_k"] = dump_vector(p_dir, f"ckpt12b_block_{li:02d}_gqa_k", k)
                        prompt_ckpts[f"ckpt12c_block_{li:02d}_gqa_v"] = dump_vector(p_dir, f"ckpt12c_block_{li:02d}_gqa_v", v)

                    # Update GQA KV cache
                    kv_k[li].append(k.reshape(NKV, DH))
                    kv_v[li].append(v.reshape(NKV, DH))
                    seq_len_active = len(kv_k[li])

                    # Multi-token causal GQA attention
                    context = np.zeros(NQ * DH, dtype=np.float32)
                    scale_attn = 1.0 / math.sqrt(DH)
                    gqa_group_size = NQ // NKV

                    for q_head in range(NQ):
                        kv_head = q_head // gqa_group_size
                        q_h = q[q_head * DH : (q_head + 1) * DH]
                        scores = np.zeros(seq_len_active, dtype=np.float32)
                        max_score = -1e9
                        for past_t in range(seq_len_active):
                            k_h = kv_k[li][past_t][kv_head]
                            score = float(np.dot(q_h, k_h)) * scale_attn
                            scores[past_t] = score
                            if score > max_score:
                                max_score = score
                        exp_scores = np.exp(scores - max_score)
                        attn_weights = exp_scores / (np.sum(exp_scores) + 1e-9)
                        out_h = np.zeros(DH, dtype=np.float32)
                        for past_t in range(seq_len_active):
                            out_h += attn_weights[past_t] * kv_v[li][past_t][kv_head]
                        context[q_head * DH : (q_head + 1) * DH] = out_h

                    if is_detailed:
                        prompt_ckpts[f"ckpt13_block_{li:02d}_gqa_attention"] = dump_vector(p_dir, f"ckpt13_block_{li:02d}_gqa_attention", context)

                    # slot 3: out_proj [D, D]
                    gqa_out = apply_weight(NANO_PATH, descs[base + 3], (D, D), context)
                    if is_detailed:
                        prompt_ckpts[f"ckpt14_block_{li:02d}_gqa_out_proj"] = dump_vector(p_dir, f"ckpt14_block_{li:02d}_gqa_out_proj", gqa_out)

                    h = h + gqa_out
                    if is_detailed:
                        prompt_ckpts[f"ckpt15_block_{li:02d}_gqa_residual"] = dump_vector(p_dir, f"ckpt15_block_{li:02d}_gqa_residual", h)

                # ── FFN Block (All 24 blocks) ──
                # slot 8: ffn.norm
                ffn_norm_w = load_fp32_weight(NANO_PATH, descs[base + 8], (D,))
                h_ffn_norm = rms_norm(h, ffn_norm_w)
                if is_detailed:
                    prompt_ckpts[f"ckpt16_block_{li:02d}_ffn_norm"] = dump_vector(p_dir, f"ckpt16_block_{li:02d}_ffn_norm", h_ffn_norm)

                # slot 5: gate_proj [FFN, D]
                # slot 6: up_proj [FFN, D]
                gate_f = apply_weight(NANO_PATH, descs[base + 5], (F, D), h_ffn_norm)
                up_f   = apply_weight(NANO_PATH, descs[base + 6], (F, D), h_ffn_norm)
                if is_detailed:
                    prompt_ckpts[f"ckpt17_block_{li:02d}_ffn_gate"] = dump_vector(p_dir, f"ckpt17_block_{li:02d}_ffn_gate", gate_f)
                    prompt_ckpts[f"ckpt18_block_{li:02d}_ffn_up"] = dump_vector(p_dir, f"ckpt18_block_{li:02d}_ffn_up", up_f)

                swiglu = silu(gate_f) * up_f
                if is_detailed:
                    prompt_ckpts[f"ckpt19_block_{li:02d}_ffn_activation"] = dump_vector(p_dir, f"ckpt19_block_{li:02d}_ffn_activation", swiglu)

                # slot 7: down_proj [D, F]
                ffn_out = apply_weight(NANO_PATH, descs[base + 7], (D, F), swiglu)
                if is_detailed:
                    prompt_ckpts[f"ckpt20_block_{li:02d}_ffn_down"] = dump_vector(p_dir, f"ckpt20_block_{li:02d}_ffn_down", ffn_out)

                h = h + ffn_out
                if is_last:
                    prompt_ckpts[f"ckpt21_block_{li:02d}_ffn_residual"] = dump_vector(p_dir, f"ckpt21_block_{li:02d}_ffn_residual", h)

            # End of 24 blocks
            if is_last:
                h_final = rms_norm(h, fn_gamma)
                prompt_ckpts["ckpt22_final_norm"] = dump_vector(p_dir, "ckpt22_final_norm", h_final)
                prompt_ckpts["ckpt23_lm_head_input"] = dump_vector(p_dir, "ckpt23_lm_head_input", h_final)

                logits = (lm_weights @ h_final.astype(np.float32)).astype(np.float32)
                prompt_ckpts["ckpt24_logits"] = dump_vector(p_dir, "ckpt24_logits", logits)

                am = int(np.argmax(logits))
                top5 = np.argsort(logits)[-5:][::-1].tolist()
                print(f"  Final Token Logits: Argmax={am}, Top5={top5}")
                print(f"  Checkpoints captured: {len(prompt_ckpts)} files in {p_dir}")

        all_prompt_records[label] = {
            "label": label,
            "prompt": prompt_text,
            "token_ids": token_ids,
            "checkpoints": prompt_ckpts,
        }

    meta_path = BASE_OUT_DIR / "reference_b_checkpoints_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_prompt_records, f, indent=2)
    print(f"\nAll Reference-B checkpoints captured and saved to {meta_path}")

if __name__ == "__main__":
    main()
