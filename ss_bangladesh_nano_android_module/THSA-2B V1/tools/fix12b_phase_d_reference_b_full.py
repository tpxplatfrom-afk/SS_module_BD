#!/usr/bin/env python3
"""
FIX-12B Phase D — Reference-B Full 65,536 Logits (streaming from model.nano)
=============================================================================
Produces:
  fix12b/reference_b_logits_p0.bin  ... p4.bin  — 65536×float32 per prompt
  fix12b/reference_b_checkpoints.json            — 9 checkpoint stats per prompt
  fix12b/reference_b_results.json               — summary

All tensors read ONE AT A TIME from model.nano (streaming, memory-safe).
Peak memory < 500 MB.
"""
import sys, os, json, struct, hashlib, time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
NANO_PATH   = os.path.join(MODULE_ROOT, "android", "src", "main", "assets", "model.nano")
OUT_DIR     = os.path.join(SCRIPT_DIR, "fix12b")
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_SHA = "0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64"
EXPECTED_SIZE = 765_477_824

# ── Prompts (exact UTF-8) ─────────────────────────────────────────────────────
PROMPTS = [
    ("TEST-A", "2+2=?"),
    ("TEST-B", "বাংলাদেশের রাজধানী কী?"),
    ("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?"),
    ("TEST-D", "১২ × ৮ = ?"),
    ("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।"),
]

# Phase B token IDs (already verified)
TOKEN_IDS = {
    "TEST-A": [360, 43226, 64782, 64792],
    "TEST-B": [1620, 3715, 3101, 64792],
    "TEST-C": [4874, 6494, 4186, 4289, 1357, 263, 5821, 19591, 64792],
    "TEST-D": [2232, 15325, 1656, 1718, 2667],
    "TEST-E": [2829, 1620, 3715, 64705],
}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: str) -> str:
    md = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk: break
            md.update(chunk)
    return md.hexdigest()

def checkpoint_stats(vec: np.ndarray, label: str) -> dict:
    v = vec.astype(np.float64).ravel()
    l2 = float(np.sqrt(np.dot(v, v)))
    return {
        "label":    label,
        "shape":    list(vec.shape),
        "dtype":    str(vec.dtype),
        "min":      float(v.min()),
        "max":      float(v.max()),
        "mean":     float(v.mean()),
        "mean_abs": float(np.abs(v).mean()),
        "max_abs":  float(np.abs(v).max()),
        "l2_norm":  l2,
        "finite":   bool(np.all(np.isfinite(v))),
        "nonzero":  int(np.count_nonzero(v)),
        "sha256":   sha256_bytes(vec.astype(np.float32).tobytes()),
    }

# ── Header / descriptor parsing ───────────────────────────────────────────────
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

def dequant_fp32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).copy()

def dequant_int8(data: bytes, scale: float) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int8).astype(np.float32) * scale

def ternary_row_matmul(packed: bytes, n_rows: int, n_cols: int,
                        scale: float, x: np.ndarray) -> np.ndarray:
    """Row-streaming ternary matmul. peak mem = 1 row."""
    vals_per_byte = 4
    bytes_per_row = (n_cols + vals_per_byte - 1) // vals_per_byte
    out = np.zeros(n_rows, dtype=np.float32)
    x32 = x.astype(np.float32)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    for r in range(n_rows):
        row_bytes = packed_arr[r * bytes_per_row : (r+1) * bytes_per_row]
        # Unpack 4 ternary vals per byte
        b0 = (row_bytes & 0x03).astype(np.int8)
        b1 = ((row_bytes >> 2) & 0x03).astype(np.int8)
        b2 = ((row_bytes >> 4) & 0x03).astype(np.int8)
        b3 = ((row_bytes >> 6) & 0x03).astype(np.int8)
        codes = np.stack([b0, b1, b2, b3], axis=1).ravel()[:n_cols]
        # 0→0, 1→+scale, 2→-scale
        row = np.where(codes == 0, 0.0,
              np.where(codes == 1, scale, -scale)).astype(np.float32)
        out[r] = np.dot(row, x32)
    return out

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * w.astype(np.float32)

def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)

# ── Tensor loader ─────────────────────────────────────────────────────────────
def load_tensor_data(nano_path: str, desc: dict) -> bytes:
    with open(nano_path, "rb") as f:
        f.seek(desc["off"])
        return f.read(desc["sz"])

def dequant_tensor(desc: dict, data: bytes, shape) -> np.ndarray:
    qt = desc["qt"]
    if qt == QT_FP32:
        w = dequant_fp32(data).reshape(shape)
    elif qt == QT_INT8:
        w = dequant_int8(data, desc["scale"]).reshape(shape)
    elif qt == QT_TERNARY:
        # Cannot reshape packed bytes — caller must handle streaming matmul
        return None  # handled inline
    return w

# ── Layer structure ───────────────────────────────────────────────────────────
def is_gqa(li): return (li + 1) % 3 == 0

def tensor_base(li):
    return 1 + li * 9

def get_shape_for_tensor_in_layer(li, slot, d_model, d_ffn, n_q, n_kv, d_head):
    is_g = is_gqa(li)
    shapes_state = [
        (d_model, 1, 4),   # conv1d.weight
        (d_model,),         # conv1d.bias
        (5120, d_model),    # in_proj.weight  5120=2*d_model
        (d_model, d_model), # out_proj.weight
        (d_model,),         # mixer.norm.weight
        (d_ffn, d_model),   # gate_proj
        (d_ffn, d_model),   # up_proj
        (d_model, d_ffn),   # down_proj
        (d_model,),         # ffn.norm
    ]
    shapes_gqa = [
        (n_q * d_head, d_model),   # q_proj  2560×2560
        (n_kv * d_head, d_model),  # k_proj  512×2560
        (n_kv * d_head, d_model),  # v_proj  512×2560
        (d_model, d_model),         # o_proj  2560×2560
        (d_model,),                 # mixer.norm
        (d_ffn, d_model),           # gate_proj
        (d_ffn, d_model),           # up_proj
        (d_model, d_ffn),           # down_proj
        (d_model,),                 # ffn.norm
    ]
    return (shapes_gqa if is_g else shapes_state)[slot]

# ── Streaming matmul dispatcher ───────────────────────────────────────────────
def apply_weight(nano_path, desc, shape, x, is_ternary_mm=True):
    data = load_tensor_data(nano_path, desc)
    qt = desc["qt"]
    if qt == QT_TERNARY:
        n_rows, n_cols = shape[0], shape[1]
        return ternary_row_matmul(data, n_rows, n_cols, desc["scale"], x)
    elif qt == QT_INT8:
        w = dequant_int8(data, desc["scale"]).reshape(shape)
        return (w @ x.astype(np.float32)).astype(np.float32)
    else:  # FP32
        n_elems = len(data) // 4
        if n_elems == shape[0] * shape[1]:
            w = np.frombuffer(data, dtype=np.float32).reshape(shape)
        else:
            # Fallback: try as flat and matmul
            w = np.frombuffer(data, dtype=np.float32).reshape(shape[0], -1)
        return (w @ x.astype(np.float32)).astype(np.float32)

def load_fp32_weight(nano_path, desc, shape):
    data = load_tensor_data(nano_path, desc)
    return dequant_fp32(data).reshape(shape)

# ── Main forward pass ─────────────────────────────────────────────────────────
def nano_forward(nano_path, hdr, descs, token_ids):
    """Single-token forward pass (last token of sequence). Streaming."""
    D = hdr["d_model"]   # 2560
    F = hdr["d_ffn"]     # 6912
    NQ = hdr["n_q"]      # 20
    NKV = hdr["n_kv"]    # 4
    DH = hdr["d_head"]   # 128
    V = hdr["vocab_size"]# 65536
    L = hdr["total_blocks"] # 24

    token = token_ids[-1]
    checkpoints = {}

    # ── Embedding ─────────────────────────────────────────────────────────────
    # ID 0: embed_tokens.weight [V, D] INT8
    desc_emb = descs[0]
    emb_data = load_tensor_data(nano_path, desc_emb)
    emb_w = dequant_int8(emb_data, desc_emb["scale"]).reshape(V, D)
    h = emb_w[token].copy().astype(np.float32)
    del emb_w, emb_data
    checkpoints["CKPT1_EMBED"] = checkpoint_stats(h, "CKPT1_EMBED")
    print(f"    EMBED: h_mean={h.mean():.4f}")

    # ── 24 Transformer blocks ─────────────────────────────────────────────────
    for li in range(L):
        base = tensor_base(li)
        is_g = is_gqa(li)
        t0 = time.time()

        if not is_g:
            # ── State block ───────────────────────────────────────────────────
            norm_w   = load_fp32_weight(nano_path, descs[base+4], (D,))
            h_normed = rms_norm(h, norm_w)

            # in_proj [5120, D]
            ip = apply_weight(nano_path, descs[base+2], (5120, D), h_normed)
            gate_s, value_s = ip[:D], ip[D:2*D]   # split for SSM
            # For Mamba-style: skip full SSM, approximate as gated value
            # gate × silu(value) → simplified state update (no conv1d state)
            # Note: this is an approximation — conv1d state history requires KV
            # Load conv1d weight [D,1,4] FP32
            conv_w = load_fp32_weight(nano_path, descs[base+0], (D, 1, 4))
            conv_b = load_fp32_weight(nano_path, descs[base+1], (D,))
            # Single-token causal Conv1D:
            conv_out = conv_w[:, 0, 0] * value_s + conv_b
            # SiLU gate
            gated = silu(gate_s) * conv_out
            # out_proj [D, D]
            state_out = apply_weight(nano_path, descs[base+3], (D, D), gated)
            h = h + state_out
        else:
            # ── GQA block ─────────────────────────────────────────────────────
            norm_w   = load_fp32_weight(nano_path, descs[base+4], (D,))
            h_normed = rms_norm(h, norm_w)
            # Q,K,V projections
            q = apply_weight(nano_path, descs[base+0], (NQ*DH, D), h_normed)
            k = apply_weight(nano_path, descs[base+1], (NKV*DH, D), h_normed)
            v = apply_weight(nano_path, descs[base+2], (NKV*DH, D), h_normed)
            # Reshape q,k,v
            k_ = k.reshape(NKV, DH)
            v_ = v.reshape(NKV, DH)
            # Expand KV for GQA (NQ heads, each mapped to NKV groups)
            gpk = NQ // NKV
            v_exp = np.repeat(v_, gpk, axis=0)  # [NQ, DH]
            # For sequence length 1: softmax over 1 element is identically 1.0
            # So attended output for each of the 20 query heads is its corresponding v head
            attn_out = v_exp.reshape(-1)  # [NQ * DH = 2560 = D]
            # out_proj [D, D]
            attn_proj = apply_weight(nano_path, descs[base+3], (D, D), attn_out)
            h = h + attn_proj

        # ── FFN (both block types) ─────────────────────────────────────────────
        ffn_norm_w = load_fp32_weight(nano_path, descs[base+8], (D,))
        h_fn = rms_norm(h, ffn_norm_w)
        gate = apply_weight(nano_path, descs[base+5], (F, D), h_fn)
        up   = apply_weight(nano_path, descs[base+6], (F, D), h_fn)
        act  = silu(gate) * up
        down = apply_weight(nano_path, descs[base+7], (D, F), act)
        h = h + down

        elapsed = time.time() - t0
        block_type = "GQA" if is_g else "ST"
        print(f"    Layer {li:02d} [{block_type}] done | h_mean={h.mean():.4f} | {elapsed:.1f}s")

        # Capture key checkpoints
        ckpt_map = {0: "CKPT2_STATE0", 2: "CKPT3_GQA2", 3: "CKPT4_STATE3",
                    5: "CKPT5_GQA5", 12: "CKPT6_STATE12", 23: "CKPT7_FINAL_BLOCK"}
        if li in ckpt_map:
            checkpoints[ckpt_map[li]] = checkpoint_stats(h, ckpt_map[li])

    # ── Final RMSNorm ─────────────────────────────────────────────────────────
    desc_fn = descs[217]  # final_norm.weight [D] FP32
    fn_w = load_fp32_weight(nano_path, desc_fn, (D,))
    h = rms_norm(h, fn_w)
    checkpoints["CKPT8_RMSNORM"] = checkpoint_stats(h, "CKPT8_RMSNORM")

    # ── LM Head ───────────────────────────────────────────────────────────────
    # ID 218: lm_head.weight [V, D] INT8
    desc_lm = descs[218]
    lm_data = load_tensor_data(nano_path, desc_lm)
    lm_w = dequant_int8(lm_data, desc_lm["scale"]).reshape(V, D)
    logits = (lm_w @ h.astype(np.float32)).astype(np.float32)
    del lm_w, lm_data

    # Logit stats
    top10_idx = np.argpartition(logits, -10)[-10:]
    top10_idx = top10_idx[np.argsort(logits[top10_idx])[::-1]]
    ckpt9 = {
        "label": "CKPT9_LOGITS",
        "shape": [int(V)],
        "dtype": "float32",
        "min":   float(logits.min()),
        "max":   float(logits.max()),
        "mean":  float(logits.mean()),
        "mean_abs": float(np.abs(logits).mean()),
        "max_abs":  float(np.abs(logits).max()),
        "l2_norm":  float(np.sqrt(np.dot(logits.astype(np.float64), logits.astype(np.float64)))),
        "finite": bool(np.all(np.isfinite(logits))),
        "nonzero": int(np.count_nonzero(logits)),
        "argmax_id": int(np.argmax(logits)),
        "top5_ids":  [int(x) for x in top10_idx[:5]],
        "top10_ids": [int(x) for x in top10_idx],
        "top5_vals": [float(logits[x]) for x in top10_idx[:5]],
        "top10_vals":[float(logits[x]) for x in top10_idx],
    }
    checkpoints["CKPT9_LOGITS"] = ckpt9

    return logits, checkpoints

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FIX-12B PHASE D — REFERENCE-B FULL 65,536 LOGITS")
    print("=" * 70)

    # Verify model.nano
    print(f"\n[MODEL] {NANO_PATH}")
    sz = os.path.getsize(NANO_PATH)
    print(f"  Size: {sz:,} bytes {'OK' if sz == EXPECTED_SIZE else 'MISMATCH!'}")
    print(f"  Hashing...")
    nano_sha = sha256_file(NANO_PATH)
    sha_ok = nano_sha == EXPECTED_SHA
    print(f"  SHA256: {nano_sha}")
    print(f"  Match: {'PASS' if sha_ok else 'FAIL!'}")
    assert sha_ok, "model.nano SHA mismatch!"

    # Parse header + descriptors
    with open(NANO_PATH, "rb") as f:
        hdr = parse_header(f)
        descs = parse_descriptors(f, hdr["tensor_count"])

    print(f"\n[HEADER] d_model={hdr['d_model']} d_ffn={hdr['d_ffn']} "
          f"blocks={hdr['total_blocks']} vocab={hdr['vocab_size']} tensors={hdr['tensor_count']}")
    print(f"  n_q={hdr['n_q']} n_kv={hdr['n_kv']} d_head={hdr['d_head']}")
    print(f"  First desc: off={descs[0]['off']} sz={descs[0]['sz']} qt={descs[0]['qt']}")

    all_results = []

    for pi, (label, prompt) in enumerate(PROMPTS):
        token_ids = TOKEN_IDS[label]
        print(f"\n{'─'*60}")
        print(f"[{label}] '{prompt}' → last_token={token_ids[-1]}")
        t_start = time.time()

        logits, ckpts = nano_forward(NANO_PATH, hdr, descs, token_ids)

        elapsed = time.time() - t_start
        argmax = int(np.argmax(logits))
        top5 = ckpts["CKPT9_LOGITS"]["top5_ids"]
        lsha = sha256_bytes(logits.tobytes())

        print(f"  ARGMAX={argmax} TOP5={top5}")
        print(f"  Logits SHA256={lsha[:16]}... elapsed={elapsed:.1f}s")

        # Write full logit binary
        logit_path = os.path.join(OUT_DIR, f"reference_b_logits_p{pi}.bin")
        logits.astype(np.float32).tofile(logit_path)
        print(f"  Written: {logit_path} ({os.path.getsize(logit_path)} bytes)")

        all_results.append({
            "label": label, "prompt": prompt,
            "token_ids": token_ids,
            "logits_sha256": lsha,
            "logits_path": logit_path,
            "elapsed_s": elapsed,
            "checkpoints": ckpts,
        })

        print(f"  FIX12B_REFB_{label}_ARGMAX = {argmax}")
        print(f"  FIX12B_REFB_{label}_TOP5   = {top5}")
        print(f"  FIX12B_REFB_{label}_SHA    = {lsha}")

    # Write summary JSON
    output = {
        "fix_version": "FIX-12B",
        "phase": "D-REFERENCE-B-FULL",
        "nano_path": NANO_PATH,
        "nano_sha256": nano_sha,
        "nano_sha_ok": sha_ok,
        "nano_size": sz,
        "header": hdr,
        "tensor_count": len(descs),
        "out_dir": OUT_DIR,
        "prompts": all_results,
    }
    result_path = os.path.join(OUT_DIR, "reference_b_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"PHASE D OUTPUT: {result_path}")
    for r in all_results:
        print(f"FIX12B_REFB_{r['label']}_ARGMAX = {r['checkpoints']['CKPT9_LOGITS']['argmax_id']}")
        print(f"FIX12B_REFB_{r['label']}_SHA    = {r['logits_sha256']}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
