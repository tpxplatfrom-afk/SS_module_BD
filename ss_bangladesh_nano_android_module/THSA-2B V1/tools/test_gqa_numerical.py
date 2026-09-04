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
            
    return dict(d_model=d_model, n_q=n_q, n_kv=n_kv, d_head=d_head), descs

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

cfg, descs = parse_nano()
D = cfg["d_model"]
NQ = cfg["n_q"]
NKV = cfg["n_kv"]
DH = cfg["d_head"]
scale_attn = 1.0 / math.sqrt(DH)

base = 19
W_q = unpack_ternary(load_data(descs[base + 0]), NQ * DH, D, descs[base + 0]["scale"])
W_k = unpack_ternary(load_data(descs[base + 1]), NKV * DH, D, descs[base + 1]["scale"])
W_v = unpack_ternary(load_data(descs[base + 2]), NKV * DH, D, descs[base + 2]["scale"])
W_out = unpack_ternary(load_data(descs[base + 3]), D, NQ * DH, descs[base + 3]["scale"])
gamma = np.frombuffer(load_data(descs[base + 4]), dtype=np.float32)

def rms_norm(x, w, eps=1e-5):
    # PyTorch formula: x / sqrt(mean(x^2) + eps) * w
    var = torch.mean(x ** 2, dim=-1, keepdim=True)
    return x / torch.sqrt(var + eps) * w

def pytorch_gqa_forward(x_seq):
    # x_seq: [1, S, D] torch.float32
    S = x_seq.shape[1]
    t_gamma = torch.from_numpy(gamma)
    t_wq = torch.from_numpy(W_q)
    t_wk = torch.from_numpy(W_k)
    t_wv = torch.from_numpy(W_v)
    t_wout = torch.from_numpy(W_out)
    
    x_n = rms_norm(x_seq, t_gamma)
    
    q = F.linear(x_n, t_wq).view(1, S, NQ, DH).transpose(1, 2)   # [1, NQ, S, DH]
    k = F.linear(x_n, t_wk).view(1, S, NKV, DH).transpose(1, 2)  # [1, NKV, S, DH]
    v = F.linear(x_n, t_wv).view(1, S, NKV, DH).transpose(1, 2)  # [1, NKV, S, DH]
    
    repeat_factor = NQ // NKV
    k_exp = k.repeat_interleave(repeat_factor, dim=1) # [1, NQ, S, DH]
    v_exp = v.repeat_interleave(repeat_factor, dim=1) # [1, NQ, S, DH]
    
    scores = torch.matmul(q, k_exp.transpose(-1, -2)) * scale_attn # [1, NQ, S, S]
    causal_mask = torch.triu(torch.full((S, S), float('-inf'), dtype=scores.dtype), diagonal=1)
    scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
    attn_w = F.softmax(scores, dim=-1, dtype=torch.float32)
    context = torch.matmul(attn_w, v_exp).transpose(1, 2).contiguous().view(1, S, D)
    gqa_out = F.linear(context, t_wout)
    residual = x_seq + gqa_out
    
    return {
        "x_n": x_n,
        "q": q,
        "k": k,
        "v": v,
        "scores": scores,
        "attn_w": attn_w,
        "context": context,
        "gqa_out": gqa_out,
        "residual": residual
    }

# Native C++ GQA simulation (with FP32 KV and INT4 KV)
def native_gqa_step(q_vec, k_cache, v_cache, seq_len, use_int4=False):
    # q_vec: [2560] float32
    # k_cache: list of seq_len vectors, each [4, 128]
    # v_cache: list of seq_len vectors, each [4, 128]
    out_attn = np.zeros(NQ * DH, dtype=np.float32)
    gqa_group_size = NQ // NKV # 5
    
    scores = np.zeros(seq_len, dtype=np.float32)
    
    for q_head in range(NQ):
        kv_head = q_head // gqa_group_size
        q_h = q_vec[q_head * DH : (q_head + 1) * DH]
        out_h = np.zeros(DH, dtype=np.float32)
        
        # 1. Dot Products
        max_score = -1e9
        for t in range(seq_len):
            k_h = k_cache[t][kv_head]
            if use_int4:
                # INT4 quantize/dequantize
                max_v = max(1e-6, np.max(np.abs(k_h)))
                scale = max_v / 7.0
                q_int = np.clip(np.round(k_h / scale) + 7, 0, 15).astype(int)
                k_h = (q_int - 7) * scale
                
            dot = np.dot(q_h, k_h)
            score = dot * scale_attn
            scores[t] = score
            if score > max_score:
                max_score = score
                
        # 2. Softmax
        exp_scores = np.exp(scores[:seq_len] - max_score)
        exp_sum = np.sum(exp_scores)
        attn_w = exp_scores / (exp_sum + 1e-9)
        
        # 3. Weighted Accumulation
        for t in range(seq_len):
            w = attn_w[t]
            if w < 1e-7:
                continue
            v_h = v_cache[t][kv_head]
            if use_int4:
                max_v = max(1e-6, np.max(np.abs(v_h)))
                scale = max_v / 7.0
                q_int = np.clip(np.round(v_h / scale) + 7, 0, 15).astype(int)
                v_h = (q_int - 7) * scale
            out_h += w * v_h
            
        out_attn[q_head * DH : (q_head + 1) * DH] = out_h
        
    return out_attn

print("\nRunning T=1, 2, 4, 8 Numerical Validation...")
for T in [1, 2, 4, 8]:
    torch.manual_seed(42)
    x_input = torch.randn(1, T, D)
    py_res = pytorch_gqa_forward(x_input)
    
    # Run native step-by-step
    k_cache_fp32 = []
    v_cache_fp32 = []
    
    for t in range(T):
        x_t = x_input[:, t:t+1, :]
        x_n_t = rms_norm(x_t, torch.from_numpy(gamma)).numpy()[0, 0, :]
        q_t = (W_q @ x_n_t)
        k_t = (W_k @ x_n_t).reshape(NKV, DH)
        v_t = (W_v @ x_n_t).reshape(NKV, DH)
        
        k_cache_fp32.append(k_t)
        v_cache_fp32.append(v_t)
        
        # At step t, seq_len = t + 1
        native_ctx_fp32 = native_gqa_step(q_t, k_cache_fp32, v_cache_fp32, t + 1, use_int4=False)
        native_ctx_int4 = native_gqa_step(q_t, k_cache_fp32, v_cache_fp32, t + 1, use_int4=True)
        
    # Compare last token (t = T - 1)
    py_ctx_last = py_res["context"][0, -1, :].numpy()
    
    # Cosine & max abs diff: PyTorch vs Native FP32
    cos_fp32 = np.dot(py_ctx_last, native_ctx_fp32) / (np.linalg.norm(py_ctx_last) * np.linalg.norm(native_ctx_fp32))
    max_d_fp32 = np.max(np.abs(py_ctx_last - native_ctx_fp32))
    
    # Cosine & max abs diff: PyTorch vs Native INT4
    cos_int4 = np.dot(py_ctx_last, native_ctx_int4) / (np.linalg.norm(py_ctx_last) * np.linalg.norm(native_ctx_int4))
    max_d_int4 = np.max(np.abs(py_ctx_last - native_ctx_int4))
    
    print(f"  T={T}:")
    print(f"    PyTorch vs Native FP32: Cosine={cos_fp32:.10f}, MaxAbsDiff={max_d_fp32:.2e}")
    print(f"    PyTorch vs Native INT4: Cosine={cos_int4:.10f}, MaxAbsDiff={max_d_int4:.2e}")
