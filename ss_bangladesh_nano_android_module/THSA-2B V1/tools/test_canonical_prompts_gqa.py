import struct
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

NANO_PATH = Path("android/src/main/assets/model.nano")
HEADER_FMT = "<4sHHHHIIHHHHIIII20s"
DESC_FMT   = "<IIQQfI"

def parse_nano():
    with open(NANO_PATH, "rb") as f:
        raw = f.read(64)
        magic, version, total_blocks, state_blocks, gqa_blocks, \
        d_model, d_ffn, n_q, n_kv, d_head, pad, vocab_size, max_context, \
        crc32, tensor_count, _res = struct.unpack(HEADER_FMT, raw)
        
        descs_raw = f.read(tensor_count * 32)
        descs = []
        for i in range(tensor_count):
            tid, qt, off, sz, scale, _pad = struct.unpack(DESC_FMT, descs_raw[i*32:(i+1)*32])
            descs.append(dict(id=tid, qt=qt, off=off, sz=sz, scale=scale))
            
    return dict(d_model=d_model, n_q=n_q, n_kv=n_kv, d_head=d_head, vocab_size=vocab_size), descs

def load_data(desc):
    with open(NANO_PATH, "rb") as f:
        f.seek(desc["off"])
        return f.read(desc["sz"])

def unpack_ternary(packed, rows, cols, scale):
    k_bytes = cols // 4
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    W = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        row_b = packed_arr[r * k_bytes : (r+1) * k_bytes]
        b0 = (row_b & 1) - ((row_b >> 1) & 1)
        b1 = ((row_b >> 2) & 1) - ((row_b >> 3) & 1)
        b2 = ((row_b >> 4) & 1) - ((row_b >> 5) & 1)
        b3 = ((row_b >> 6) & 1) - ((row_b >> 7) & 1)
        codes = np.stack([b0, b1, b2, b3], axis=1).ravel()[:cols]
        W[r] = codes.astype(np.float32) * scale
    return W

def rms_norm(x, w, eps=1e-5):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w

def quantize_int8(x):
    max_v = max(1e-6, np.max(np.abs(x)))
    scale = max_v / 127.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale

def quantize_int4_head(v):
    max_v = max(1e-6, np.max(np.abs(v)))
    scale = max_v / 7.0
    q = np.clip(np.round(v / scale) + 7, 0, 15).astype(np.uint8)
    deq = (q.astype(np.float32) - 7.0) * scale
    return q, scale, deq

def calc_metrics(ref, act):
    diff = np.abs(ref - act)
    max_d = float(np.max(diff))
    mean_d = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    norm_ref = float(np.linalg.norm(ref))
    norm_act = float(np.linalg.norm(act))
    l2_rel = float(rmse * np.sqrt(ref.size) / (norm_ref + 1e-12))
    cos = float(np.dot(ref.ravel(), act.ravel()) / (norm_ref * norm_act + 1e-12))
    return {
        "max_abs_diff": max_d,
        "mean_abs_diff": mean_d,
        "rmse": rmse,
        "l2_rel_err": l2_rel,
        "cosine": cos,
        "norm_ref": norm_ref,
        "norm_act": norm_act,
        "min_act": float(np.min(act)),
        "max_act": float(np.max(act))
    }

cfg, descs = parse_nano()
D = cfg["d_model"]
NQ = cfg["n_q"]
NKV = cfg["n_kv"]
DH = cfg["d_head"]
scale_attn = 1.0 / math.sqrt(DH)

# Canonical token sequences:
# TEST-A: "2+2=?" -> tokens [610, 503, 610, 574, 1506] (T=5)
# TEST-D: "১২ × ৮ = ?" -> tokens [13245, 12890, 893, 14002, 574, 1506] (T=6)
prompts = {
    "TEST-A (2+2=?)": [610, 503, 610, 574, 1506],
    "TEST-D (১২ × ৮ = ?)": [13245, 12890, 893, 14002, 574, 1506]
}

# Embedding table is INT8 (qt=2)
raw_emb = np.frombuffer(load_data(descs[0]), dtype=np.int8).astype(np.float32) * descs[0]["scale"]
w_emb = raw_emb.reshape(cfg["vocab_size"], D)

print("================================================================================")
print("CANONICAL PROMPTS TEST-A AND TEST-D: GQA BRANCH CAPTURE (LAYERS 2 & 23)")
print("================================================================================")

for p_name, token_ids in prompts.items():
    T = len(token_ids)
    safe_name = p_name.encode("ascii", "backslashreplace").decode("ascii")
    print(f"\nPrompt: {safe_name} | Sequence Length T={T} | Token IDs: {token_ids}")
    
    # Embedding sequence
    x_seq = np.array([w_emb[tok] for tok in token_ids]) # [T, D]
    
    for layer in [2, 23]:
        base = 1 + layer * 9
        W_q = unpack_ternary(load_data(descs[base + 0]), NQ * DH, D, descs[base + 0]["scale"])
        W_k = unpack_ternary(load_data(descs[base + 1]), NKV * DH, D, descs[base + 1]["scale"])
        W_v = unpack_ternary(load_data(descs[base + 2]), NKV * DH, D, descs[base + 2]["scale"])
        W_out = unpack_ternary(load_data(descs[base + 3]), D, NQ * DH, descs[base + 3]["scale"])
        gamma = np.frombuffer(load_data(descs[base + 4]), dtype=np.float32)
        
        # We track KV cache across tokens
        k_cache_fp32 = []
        v_cache_fp32 = []
        k_cache_int4 = []
        v_cache_int4 = []
        
        for t in range(T):
            x_t = x_seq[t]
            h_norm = rms_norm(x_t, gamma)
            h_i8, x_scale = quantize_int8(h_norm)
            h_q = h_i8.astype(np.float32) * x_scale
            
            q_t = (W_q @ h_q).reshape(NQ, DH)
            k_t = (W_k @ h_q).reshape(NKV, DH)
            v_t = (W_v @ h_q).reshape(NKV, DH)
            
            # INT4
            k_i4 = np.zeros_like(k_t)
            v_i4 = np.zeros_like(v_t)
            for h in range(NKV):
                _, _, k_i4[h] = quantize_int4_head(k_t[h])
                _, _, v_i4[h] = quantize_int4_head(v_t[h])
                
            k_cache_fp32.append(k_t)
            v_cache_fp32.append(v_t)
            k_cache_int4.append(k_i4)
            v_cache_int4.append(v_i4)
            
        # Evaluate attention for last token (t = T - 1)
        cur_len = T
        q_last = q_t
        ctx_fp32 = np.zeros((NQ, DH), dtype=np.float32)
        ctx_int4 = np.zeros((NQ, DH), dtype=np.float32)
        
        for qh in range(NQ):
            kvh = qh // 5
            q_vec = q_last[qh]
            
            # FP32
            scores_fp = np.zeros(cur_len, dtype=np.float32)
            for tau in range(cur_len):
                scores_fp[tau] = np.dot(q_vec, k_cache_fp32[tau][kvh]) * scale_attn
            w_fp = np.exp(scores_fp - np.max(scores_fp))
            w_fp /= np.sum(w_fp)
            for tau in range(cur_len):
                ctx_fp32[qh] += w_fp[tau] * v_cache_fp32[tau][kvh]
                
            # INT4
            scores_i4 = np.zeros(cur_len, dtype=np.float32)
            for tau in range(cur_len):
                scores_i4[tau] = np.dot(q_vec, k_cache_int4[tau][kvh]) * scale_attn
            w_i4 = np.exp(scores_i4 - np.max(scores_i4))
            w_i4 /= np.sum(w_i4)
            for tau in range(cur_len):
                ctx_int4[qh] += w_i4[tau] * v_cache_int4[tau][kvh]
                
        # Out-proj
        ctx_flat = ctx_int4.ravel()
        ctx_i8, c_scale = quantize_int8(ctx_flat)
        gqa_out = W_out @ (ctx_i8.astype(np.float32) * c_scale)
        
        m_attn = calc_metrics(ctx_fp32, ctx_int4)
        l2_out = float(np.linalg.norm(gqa_out))
        rms_out = float(np.sqrt(np.mean(gqa_out ** 2)))
        nz_out = int(np.count_nonzero(gqa_out))
        min_out = float(np.min(gqa_out))
        max_out = float(np.max(gqa_out))
        
        print(f"  [Layer {layer:02d} GQA Branch Telemetry]")
        print(f"    Q Norm: {np.linalg.norm(q_last):.4e} | K Norm: {np.linalg.norm(k_t):.4e} | V Norm: {np.linalg.norm(v_t):.4e}")
        print(f"    Attn Context Cosine (FP32 vs INT4): {m_attn['cosine']:.8f}")
        print(f"    Attn Context MaxAbsDiff:            {m_attn['max_abs_diff']:.4f}")
        print(f"    Attn Context RMSE:                  {m_attn['rmse']:.4f}")
        print(f"    GQA Out L2 Norm:                    {l2_out:.6e}")
        print(f"    GQA Out RMS:                        {rms_out:.6f}")
        print(f"    GQA Out Range [Min, Max]:           [{min_out:.4f}, {max_out:.4f}]")
        print(f"    GQA Out Non-Zero Count:             {nz_out} / {D} ({(nz_out/D)*100:.1f}%)")
        print(f"    GQA Out First 8: {[round(x, 4) for x in gqa_out[:8].tolist()]}")
