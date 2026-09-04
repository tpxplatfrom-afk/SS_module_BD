import struct
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

NANO_PATH = Path("android/src/main/assets/model.nano")
HEADER_FMT = "<4sHHHHIIHHHHIIII20s"   # 64 bytes
DESC_FMT   = "<IIQQfI"                  # 32 bytes

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

print("================================================================================")
print("FIX-C.1 FORENSIC NUMERICAL SUITE (HOST PYTHON VERIFIER)")
print("================================================================================")

cfg, descs = parse_nano()
D = cfg["d_model"]
NQ = cfg["n_q"]
NKV = cfg["n_kv"]
DH = cfg["d_head"]
scale_attn = 1.0 / math.sqrt(DH)

# ------------------------------------------------------------------------------
# 1. T=1 Hard Invariant
# ------------------------------------------------------------------------------
print("\n--- TEST 1: T=1 Hard Invariant Check ---")
np.random.seed(12345)
q_t1 = np.random.randn(NQ * DH).astype(np.float32)
k_t1 = np.random.randn(NKV * DH).astype(np.float32)
v_t1 = np.random.randn(NKV * DH).astype(np.float32)

context_t1_expected = np.zeros(NQ * DH, dtype=np.float32)
for qh in range(NQ):
    kvh = qh // 5
    context_t1_expected[qh * DH : (qh + 1) * DH] = v_t1[kvh * DH : (kvh + 1) * DH]

context_t1_scalar = np.zeros(NQ * DH, dtype=np.float32)
for qh in range(NQ):
    kvh = qh // 5
    qh_v = q_t1[qh * DH : (qh + 1) * DH]
    kh_v = k_t1[kvh * DH : (kvh + 1) * DH]
    vh_v = v_t1[kvh * DH : (kvh + 1) * DH]
    score = np.dot(qh_v, kh_v) * scale_attn
    w = 1.0
    context_t1_scalar[qh * DH : (qh + 1) * DH] = w * vh_v

t1_diff = np.max(np.abs(context_t1_expected - context_t1_scalar))
t1_cos = np.dot(context_t1_expected, context_t1_scalar) / (np.linalg.norm(context_t1_expected) * np.linalg.norm(context_t1_scalar))
print(f"  T=1 MaxAbsDiff: {t1_diff:.2e} | Cosine: {t1_cos:.10f}")

# ------------------------------------------------------------------------------
# 2. Multi-Token Traces: T=1, 2, 4, 8 & INT4 Isolation
# ------------------------------------------------------------------------------
print("\n--- TEST 2: Multi-Token Traces & INT4 Quantization Isolation (T=1, 2, 4, 8) ---")
for T in [1, 2, 4, 8]:
    np.random.seed(42 + T)
    Q = np.random.randn(NQ, DH).astype(np.float32) * 0.5
    K_fp = np.random.randn(T, NKV, DH).astype(np.float32) * 0.8
    V_fp = np.random.randn(T, NKV, DH).astype(np.float32) * 1.2
    
    K_deq = np.zeros_like(K_fp)
    V_deq = np.zeros_like(V_fp)
    for t in range(T):
        for h in range(NKV):
            _, _, K_deq[t, h] = quantize_int4_head(K_fp[t, h])
            _, _, V_deq[t, h] = quantize_int4_head(V_fp[t, h])
            
    ctx_fp32 = np.zeros((NQ, DH), dtype=np.float32)
    ctx_int4 = np.zeros((NQ, DH), dtype=np.float32)
    
    for qh in range(NQ):
        kvh = qh // 5
        q_vec = Q[qh]
        
        scores_fp = np.zeros(T, dtype=np.float32)
        for t in range(T):
            scores_fp[t] = np.dot(q_vec, K_fp[t, kvh]) * scale_attn
        w_fp = np.exp(scores_fp - np.max(scores_fp))
        w_fp /= np.sum(w_fp)
        for t in range(T):
            ctx_fp32[qh] += w_fp[t] * V_fp[t, kvh]
            
        scores_i4 = np.zeros(T, dtype=np.float32)
        for t in range(T):
            scores_i4[t] = np.dot(q_vec, K_deq[t, kvh]) * scale_attn
        w_i4 = np.exp(scores_i4 - np.max(scores_i4))
        w_i4 /= np.sum(w_i4)
        for t in range(T):
            ctx_int4[qh] += w_i4[t] * V_deq[t, kvh]
            
    m_k = calc_metrics(K_fp, K_deq)
    m_v = calc_metrics(V_fp, V_deq)
    m_attn = calc_metrics(ctx_fp32, ctx_int4)
    
    print(f"  T={T}:")
    print(f"    K-Cache  : Cosine={m_k['cosine']:.8f} | MaxAbs={m_k['max_abs_diff']:.4f} | RMSE={m_k['rmse']:.4f}")
    print(f"    V-Cache  : Cosine={m_v['cosine']:.8f} | MaxAbs={m_v['max_abs_diff']:.4f} | RMSE={m_v['rmse']:.4f}")
    print(f"    Attention: Cosine={m_attn['cosine']:.8f} | MaxAbs={m_attn['max_abs_diff']:.4f} | RMSE={m_attn['rmse']:.4f}")

# ------------------------------------------------------------------------------
# 3. Real Model Weights (Layers 2 and 23) + Out Norm Forensic Inspection
# ------------------------------------------------------------------------------
print("\n--- TEST 3: Real Model Layers 2 & 23 + Out Norm Forensic Audit ---")
for layer in [2, 23]:
    base = 1 + layer * 9
    W_q = unpack_ternary(load_data(descs[base + 0]), NQ * DH, D, descs[base + 0]["scale"])
    W_k = unpack_ternary(load_data(descs[base + 1]), NKV * DH, D, descs[base + 1]["scale"])
    W_v = unpack_ternary(load_data(descs[base + 2]), NKV * DH, D, descs[base + 2]["scale"])
    W_out = unpack_ternary(load_data(descs[base + 3]), D, NQ * DH, descs[base + 3]["scale"])
    gamma = np.frombuffer(load_data(descs[base + 4]), dtype=np.float32)
    
    np.random.seed(100 + layer)
    h_in = np.cos(np.arange(D, dtype=np.float32) * 0.05) * 0.75
    h_norm = rms_norm(h_in, gamma)
    
    h_norm_i8, x_scale = quantize_int8(h_norm)
    h_norm_q = h_norm_i8.astype(np.float32) * x_scale
    
    q = (W_q @ h_norm_q).reshape(NQ, DH)
    k = (W_k @ h_norm_q).reshape(NKV, DH)
    v = (W_v @ h_norm_q).reshape(NKV, DH)
    
    k_deq = np.zeros_like(k)
    v_deq = np.zeros_like(v)
    for h in range(NKV):
        _, _, k_deq[h] = quantize_int4_head(k[h])
        _, _, v_deq[h] = quantize_int4_head(v[h])
        
    ctx_fp32 = np.zeros((NQ, DH), dtype=np.float32)
    ctx_int4 = np.zeros((NQ, DH), dtype=np.float32)
    for qh in range(NQ):
        kvh = qh // 5
        ctx_fp32[qh] = v[kvh]
        ctx_int4[qh] = v_deq[kvh]
        
    ctx_i4_flat = ctx_int4.ravel()
    ctx_i8, ctx_scale = quantize_int8(ctx_i4_flat)
    gqa_out = W_out @ (ctx_i8.astype(np.float32) * ctx_scale)
    
    l2_norm = float(np.linalg.norm(gqa_out))
    rms_val = float(np.sqrt(np.mean(gqa_out ** 2)))
    min_val = float(np.min(gqa_out))
    max_val = float(np.max(gqa_out))
    nz_count = int(np.count_nonzero(gqa_out))
    first_8 = gqa_out[:8].tolist()
    last_8 = gqa_out[-8:].tolist()
    
    cos_attn = float(np.dot(ctx_fp32.ravel(), ctx_int4.ravel()) / (np.linalg.norm(ctx_fp32) * np.linalg.norm(ctx_int4)))
    
    print(f"\n  [Layer {layer} Audit]")
    print(f"    Attention Cosine (FP32 vs INT4): {cos_attn:.8f}")
    print(f"    GQA Out L2 Norm:                 {l2_norm:.6e}")
    print(f"    GQA Out RMS:                     {rms_val:.6f}")
    print(f"    GQA Out Min / Max:               [{min_val:.6f}, {max_val:.6f}]")
    print(f"    Non-Zero Element Count:          {nz_count} / {D} ({(nz_count/D)*100:.1f}%)")
    print(f"    First 8 Elements: {[round(x, 5) for x in first_8]}")
    print(f"    Last 8 Elements:  {[round(x, 5) for x in last_8]}")
    
    buggy_rmse = float(np.sqrt(np.mean((gqa_out - gqa_out) ** 2)))
    print(f"    Buggy calc_rmse(gqa_out, gqa_out) printed: {buggy_rmse:.4f}")
