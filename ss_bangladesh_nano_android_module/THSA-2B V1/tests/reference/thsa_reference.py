"""
THSA-2B Independent Python / NumPy Reference Forward Implementation.
Memory-efficient implementation: unpacks ternary weights per-layer on-the-fly,
allocates KV cache only for the 8 GQA layers, and computes exact reference forward passes.
"""

import struct
import numpy as np
from pathlib import Path

def unpack_ternary_2bit(w_packed, M, K):
    """
    Vectorized NumPy unpack 2-bit packed ternary weights:
    00 -> 0.0
    01 -> +1.0
    10 -> -1.0
    11 -> 0.0 (reserved)
    Layout: M rows, K columns, packed into M * (K // 4) bytes.
    """
    k_bytes = K // 4
    b = w_packed.reshape(M, k_bytes)
    c0 = (b >> 0) & 0x03
    c1 = (b >> 2) & 0x03
    c2 = (b >> 4) & 0x03
    c3 = (b >> 6) & 0x03
    
    lut = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    w_mat = np.stack([lut[c0], lut[c1], lut[c2], lut[c3]], axis=-1).reshape(M, K)
    return w_mat

def fast_silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def rms_norm(x, gamma=None, eps=1e-5):
    mean_sq = np.mean(x ** 2)
    rsqrt = 1.0 / np.sqrt(mean_sq + eps)
    normed = x * rsqrt
    if gamma is not None and not np.all(gamma == 0):
        normed = normed * gamma
    return normed

def quantize_int8(x):
    max_abs = np.max(np.abs(x))
    if max_abs < 1e-6:
        max_abs = 1e-6
    scale = max_abs / 127.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale

class THSAReferenceModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        with open(self.model_path, "rb") as f:
            hdr = f.read(64)
            (self.magic, self.version, self.total_blocks, self.state_blocks, self.gqa_blocks,
             self.d_model, self.d_ffn, self.n_q, self.n_kv, self.d_head, self.pad,
             self.vocab_size, self.max_context, self.stored_crc, self.tensor_count, _) = struct.unpack("<4sHHHHIIHHHHI I I I 20s", hdr)
            
            desc_bytes = f.read(self.tensor_count * 32)
            self.descriptors = []
            for i in range(self.tensor_count):
                t_id, q_type, offset, size_bytes, scale, pad = struct.unpack("<IIQQfI", desc_bytes[i*32:(i+1)*32])
                self.descriptors.append({
                    "id": t_id, "type": q_type, "offset": offset, "size": size_bytes, "scale": scale
                })
                
            # Load Tensor 0: embed_tokens (INT8)
            d0 = self.descriptors[0]
            f.seek(d0["offset"])
            raw_emb = f.read(self.vocab_size * self.d_model)
            self.embed_tokens = np.frombuffer(raw_emb, dtype=np.int8).reshape(self.vocab_size, self.d_model)
            self.embed_scale = d0["scale"]
            
            # Load 24 Layers metadata & packed buffers
            self.layers = []
            curr_idx = 1
            for l in range(self.total_blocks):
                is_gqa = ((l + 1) % 3 == 0)
                layer_info = {"is_gqa": is_gqa}
                
                if is_gqa:
                    # Q, K, V, Out
                    for name, M, K in [("q", self.n_q * self.d_head, self.d_model),
                                       ("k", self.n_kv * self.d_head, self.d_model),
                                       ("v", self.n_kv * self.d_head, self.d_model),
                                       ("out", self.d_model, self.n_q * self.d_head)]:
                        d = self.descriptors[curr_idx]
                        f.seek(d["offset"])
                        packed = f.read(d["size"])
                        layer_info[f"w_{name}_packed"] = np.frombuffer(packed, dtype=np.uint8)
                        layer_info[f"scale_{name}"] = d["scale"]
                        layer_info[f"shape_{name}"] = (M, K)
                        curr_idx += 1
                else:
                    # State conv weights (FP32)
                    d = self.descriptors[curr_idx]
                    f.seek(d["offset"])
                    raw_conv = f.read(d["size"])
                    layer_info["conv_weights"] = np.frombuffer(raw_conv, dtype=np.float32).reshape(4, self.d_model)
                    curr_idx += 1
                    
                # FFN: Gate, Up, Down
                for name, M, K in [("gate", self.d_ffn, self.d_model),
                                   ("up", self.d_ffn, self.d_model),
                                   ("down", self.d_model, self.d_ffn)]:
                    d = self.descriptors[curr_idx]
                    f.seek(d["offset"])
                    packed = f.read(d["size"])
                    layer_info[f"w_{name}_packed"] = np.frombuffer(packed, dtype=np.uint8)
                    layer_info[f"scale_{name}"] = d["scale"]
                    layer_info[f"shape_{name}"] = (M, K)
                    curr_idx += 1
                    
                self.layers.append(layer_info)
                
            # Tensor 121: final_norm
            d_fn = self.descriptors[curr_idx]
            f.seek(d_fn["offset"])
            self.final_norm_gamma = np.frombuffer(f.read(d_fn["size"]), dtype=np.float32)
            curr_idx += 1
            
            # Tensor 122: lm_head
            d_lm = self.descriptors[curr_idx]
            f.seek(d_lm["offset"])
            self.lm_head = np.frombuffer(f.read(self.vocab_size * self.d_model), dtype=np.int8).reshape(self.vocab_size, self.d_model)
            self.lm_head_scale = d_lm["scale"]

    def reset_session(self):
        self.conv_states = [np.zeros((3, self.d_model), dtype=np.float32) for _ in range(self.total_blocks)]
        # Only 8 GQA layers allocate KV cache (1000 max test tokens)
        self.kv_k = {}
        self.kv_v = {}
        for l in range(self.total_blocks):
            if self.layers[l]["is_gqa"]:
                self.kv_k[l] = np.zeros((self.n_kv, 1024, self.d_head), dtype=np.float32)
                self.kv_v[l] = np.zeros((self.n_kv, 1024, self.d_head), dtype=np.float32)
        self.active_seq_len = 0

    def forward_single_token(self, token_id, checkpoints=None):
        # 1. Embedding lookup
        h = self.embed_tokens[token_id].astype(np.float32) * self.embed_scale
        if checkpoints is not None:
            checkpoints["embedding_out"] = h.copy()
            
        t_idx = min(self.active_seq_len, 1023)
        
        # 2. 24 Backbone layers
        for l_idx, layer in enumerate(self.layers):
            if layer["is_gqa"]:
                # GQA Block
                x_q, x_scale = quantize_int8(h)
                
                # Unpack Q, K, V on the fly
                w_q = unpack_ternary_2bit(layer["w_q_packed"], *layer["shape_q"])
                w_k = unpack_ternary_2bit(layer["w_k_packed"], *layer["shape_k"])
                w_v = unpack_ternary_2bit(layer["w_v_packed"], *layer["shape_v"])
                
                q_act = (w_q @ x_q) * (layer["scale_q"] * x_scale)
                k_act = (w_k @ x_q) * (layer["scale_k"] * x_scale)
                v_act = (w_v @ x_q) * (layer["scale_v"] * x_scale)
                
                # Append to KV cache
                for h_i in range(self.n_kv):
                    self.kv_k[l_idx][h_i, t_idx] = k_act[h_i * self.d_head : (h_i + 1) * self.d_head]
                    self.kv_v[l_idx][h_i, t_idx] = v_act[h_i * self.d_head : (h_i + 1) * self.d_head]
                    
                # GQA Attention
                attn_out = np.zeros(self.n_q * self.d_head, dtype=np.float32)
                gqa_group = self.n_q // self.n_kv # 5
                inv_sqrt_d = 1.0 / np.sqrt(self.d_head)
                
                for q_h in range(self.n_q):
                    kv_h = q_h // gqa_group
                    q_vec = q_act[q_h * self.d_head : (q_h + 1) * self.d_head]
                    
                    scores = np.zeros(t_idx + 1, dtype=np.float32)
                    for t in range(t_idx + 1):
                        k_vec = self.kv_k[l_idx][kv_h, t]
                        scores[t] = np.dot(q_vec, k_vec) * inv_sqrt_d
                        
                    # Softmax
                    max_s = np.max(scores)
                    exp_s = np.exp(scores - max_s)
                    weights = exp_s / (np.sum(exp_s) + 1e-9)
                    
                    out_head = np.zeros(self.d_head, dtype=np.float32)
                    for t in range(t_idx + 1):
                        out_head += weights[t] * self.kv_v[l_idx][kv_h, t]
                    attn_out[q_h * self.d_head : (q_h + 1) * self.d_head] = out_head
                    
                # Out projection
                attn_q, attn_scale = quantize_int8(attn_out)
                w_out = unpack_ternary_2bit(layer["w_out_packed"], *layer["shape_out"])
                attn_proj = (w_out @ attn_q) * (layer["scale_out"] * attn_scale)
                
                h = h + attn_proj
            else:
                # State Block: 1D causal short-conv (K=4)
                w = layer["conv_weights"]
                s = self.conv_states[l_idx]
                conv_val = s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + h * w[3]
                
                # Update FIFO
                s[0] = s[1].copy()
                s[1] = s[2].copy()
                s[2] = h.copy()
                
                h = h + conv_val
                
            # SwiGLU FFN
            ffn_q, ffn_scale = quantize_int8(h)
            w_gate = unpack_ternary_2bit(layer["w_gate_packed"], *layer["shape_gate"])
            w_up   = unpack_ternary_2bit(layer["w_up_packed"], *layer["shape_up"])
            
            gate = (w_gate @ ffn_q) * (layer["scale_gate"] * ffn_scale)
            up   = (w_up @ ffn_q) * (layer["scale_up"] * ffn_scale)
            
            swiglu_act = fast_silu(gate) * up
            
            act_q, act_scale = quantize_int8(swiglu_act)
            w_down = unpack_ternary_2bit(layer["w_down_packed"], *layer["shape_down"])
            ffn_out = (w_down @ act_q) * (layer["scale_down"] * act_scale)
            
            h = h + ffn_out
            if checkpoints is not None and l_idx == 0:
                checkpoints["layer_0_out"] = h.copy()
                
        # 3. Final RMSNorm
        h_norm = rms_norm(h, self.final_norm_gamma)
        if checkpoints is not None:
            checkpoints["final_norm_out"] = h_norm.copy()
            
        # 4. LM Head Projection (INT8)
        norm_q, norm_scale = quantize_int8(h_norm)
        # Vectorized dot product against lm_head matrix
        logits = (self.lm_head.astype(np.float32) @ norm_q.astype(np.float32)) * (self.lm_head_scale * norm_scale)
        if checkpoints is not None:
            checkpoints["logits"] = logits.copy()
            
        self.active_seq_len += 1
        return logits, int(np.argmax(logits))
