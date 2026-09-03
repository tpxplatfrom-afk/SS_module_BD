#!/usr/bin/env python3
"""
FIX-12 Phase C/D — Nano-from-file Reference Forward (REFERENCE-B)
==================================================================
Streaming forward pass — tensors are loaded ONE AT A TIME from model.nano,
applied to the activation, then immediately discarded. Peak RAM ≈ 1 tensor
at a time (max ~268 MB for [65536,2560] INT8) + activation buffers.

Output: tools/fix12_phase_cd_reference_results.json
"""

import sys, os, json, struct, hashlib, gc, math
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
NANO_PATH   = os.path.join(MODULE_ROOT, "android", "src", "main", "assets", "model.nano")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "fix12_phase_cd_reference_results.json")

D = 2560; F = 6912; V = 65536
N_LAYERS = 24; GQA_INTERVAL = 3
NQ, NKV, DH = 20, 4, 128

HEADER_SIZE     = 64
DESCRIPTOR_SIZE = 32
NANO_QUANT_FP32 = 0; NANO_QUANT_INT8 = 2; NANO_QUANT_TERNARY_2BIT = 4

PROMPTS = [
    ("TEST-A", "2+2=?",                             [360, 43226, 64782, 64792]),
    ("TEST-B", "বাংলাদেশের রাজধানী কী?",           [1620, 3715, 3101, 64792]),
    ("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?", [4874,6494,4186,4289,1357,263,5821,19591,64792]),
    ("TEST-D", "১২ × ৮ = ?",                        [2232,15325,1656,1718,2667]),
    ("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।",         [2829,1620,3715,64705]),
]

def sha256_b(b): return hashlib.sha256(b).hexdigest()

def tensor_stats(arr):
    a = arr.ravel().astype(np.float64)
    l2 = float(np.sqrt(np.dot(a, a)))
    return {"shape": list(arr.shape), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "mean_abs": float(np.abs(a).mean()),
            "max_abs": float(np.abs(a).max()), "l2_norm": l2,
            "sha256": sha256_b(arr.astype(np.float32).tobytes())}

def logit_stats(logits):
    s = tensor_stats(logits)
    f = logits.ravel()
    top10 = np.argpartition(f, -10)[-10:]
    top10 = top10[np.argsort(f[top10])[::-1]]
    s["argmax_id"]  = int(np.argmax(f))
    s["argmax_val"] = float(f[s["argmax_id"]])
    s["top5_ids"]   = [int(x) for x in top10[:5]]
    s["top5_vals"]  = [float(f[x]) for x in top10[:5]]
    s["top10_ids"]  = [int(x) for x in top10]
    s["top10_vals"] = [float(f[x]) for x in top10]
    return s

def dequant_int8_matvec(f_bin, d_desc, shape, vec):
    """Read INT8 weight, compute matmul with vec [N] → [M], then discard."""
    f_bin.seek(d_desc["offset"])
    raw = f_bin.read(d_desc["byte_size"])
    sc  = d_desc["scale"]
    w   = np.frombuffer(raw, dtype=np.int8).reshape(shape).astype(np.float32) * sc
    out = w @ vec.astype(np.float32)
    del w, raw
    return out

def dequant_ternary_matvec(f_bin, d_desc, shape, vec):
    """Read ternary packed weight [M,N], matmul → [M], discard. Memory-safe."""
    f_bin.seek(d_desc["offset"])
    raw  = f_bin.read(d_desc["byte_size"])
    sc   = np.float32(d_desc["scale"])
    ne   = shape[0] * shape[1]
    pkd  = np.frombuffer(raw, dtype=np.uint8)
    # Compute M×N matmul via packed ternary: w·v = Σ w_i*v_i
    # Unpack row by row to keep RAM = 1 row at a time for large matrices
    M, N = shape
    out  = np.zeros(M, dtype=np.float32)
    v    = vec.astype(np.float32)
    bytes_per_row = (N + 3) // 4
    for row in range(M):
        row_packed = pkd[row * bytes_per_row: (row+1) * bytes_per_row]
        # Decode ternary for this row
        w_row = np.zeros(bytes_per_row * 4, dtype=np.float32)
        for shift in range(4):
            codes = (row_packed >> np.uint8(shift * 2)) & np.uint8(0x03)
            seg = w_row[shift::4]
            seg[codes == np.uint8(1)] = sc
            seg[codes == np.uint8(2)] = -sc
        # dot product of this row with v
        out[row] = np.dot(w_row[:N], v)
    del pkd, raw
    gc.collect()
    return out

def dequant_fp32_read(f_bin, d_desc, shape):
    """Read FP32 tensor — small (norm/conv), always safe."""
    f_bin.seek(d_desc["offset"])
    raw = f_bin.read(d_desc["byte_size"])
    return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)

def rmsnorm(x, gamma, eps=1e-5):
    x64 = x.astype(np.float64)
    rms = np.sqrt(np.mean(x64**2) + eps)
    return ((x64 / rms) * gamma.astype(np.float64)).astype(np.float32)

def silu(x):
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-x64))).astype(np.float32)

def parse_nano(nano_path):
    with open(nano_path, "rb") as f:
        hdr = f.read(HEADER_SIZE)
    magic    = hdr[0:4]
    version  = struct.unpack_from("<H", hdr, 4)[0]
    crc32_h  = struct.unpack_from("<I", hdr, 36)[0]
    n_tensors= struct.unpack_from("<I", hdr, 40)[0]
    assert magic == b"NANO" and version == 0x0002 and n_tensors == 219
    print(f"  Header OK: magic={magic} version=0x{version:04X} tensors={n_tensors} CRC=0x{crc32_h:08X}")

    with open(nano_path, "rb") as f:
        f.seek(HEADER_SIZE)
        desc_raw = f.read(n_tensors * DESCRIPTOR_SIZE)
    descs = []
    for i in range(n_tensors):
        o = i * DESCRIPTOR_SIZE
        descs.append({
            "tid": struct.unpack_from("<I", desc_raw, o)[0],
            "qt":  struct.unpack_from("<I", desc_raw, o+4)[0],
            "offset": struct.unpack_from("<Q", desc_raw, o+8)[0],
            "byte_size": struct.unpack_from("<Q", desc_raw, o+16)[0],
            "scale": struct.unpack_from("<f", desc_raw, o+24)[0],
        })
    print(f"  First: off={descs[0]['offset']} sz={descs[0]['byte_size']} qt={descs[0]['qt']}")
    return descs

def run_forward(nano_path, descs, token_id, label):
    """Streaming single-token forward pass. Returns checkpoint stats dict."""
    ckpts = {}
    D_ = D; F_ = F; V_ = V
    idx = 0
    def d(): nonlocal idx; r = descs[idx]; idx += 1; return r

    with open(nano_path, "rb") as fb:

        # ── Embed (INT8 [V,D]) ────────────────────────────────────────────────
        d0 = d()  # embed_tokens
        fb.seek(d0["offset"] + token_id * D_)
        row = np.frombuffer(fb.read(D_), dtype=np.int8).astype(np.float32) * d0["scale"]
        h = row.copy()
        ckpts["CKPT1_EMBED"] = tensor_stats(h)

        # ── 24 Backbone blocks ────────────────────────────────────────────────
        for li in range(N_LAYERS):
            is_gqa = ((li + 1) % GQA_INTERVAL == 0)

            if not is_gqa:
                # State block: conv_w FP32, conv_b FP32, in_proj TERNARY, out_proj TERNARY, norm FP32
                d_cw  = d();  d_cb  = d();  d_ip  = d();  d_op  = d();  d_nm  = d()
                norm_w  = dequant_fp32_read(fb, d_nm, (D_,))
                conv_w  = dequant_fp32_read(fb, d_cw, (D_,1,4))  # [D,1,4]
                conv_b  = dequant_fp32_read(fb, d_cb, (D_,))

                x_norm = rmsnorm(h, norm_w)

                # In-proj [5120, D] ternary → [5120]
                proj = dequant_ternary_matvec(fb, d_ip, (2*D_, D_), x_norm)
                gate_s = proj[:D_]; val_s = proj[D_:]
                del proj

                # Conv1D (kernel=4, single token → tap 0 only)
                conv_out = conv_w[:, 0, 0] * val_s + conv_b
                del conv_w, conv_b

                # Gated SiLU × conv_out
                gated = silu(gate_s) * conv_out
                del gate_s, conv_out

                # Out-proj [D, D] ternary → [D]
                y = dequant_ternary_matvec(fb, d_op, (D_, D_), gated)
                del gated
                h = h + y; del y

            else:
                # GQA block: q,k,v,out TERNARY, norm FP32
                d_q = d(); d_k = d(); d_v = d(); d_o = d(); d_nm = d()
                norm_w = dequant_fp32_read(fb, d_nm, (D_,))
                x_norm = rmsnorm(h, norm_w)

                q = dequant_ternary_matvec(fb, d_q, (NQ*DH, D_), x_norm)  # [2560]
                k = dequant_ternary_matvec(fb, d_k, (NKV*DH, D_), x_norm) # [512]
                v = dequant_ternary_matvec(fb, d_v, (NKV*DH, D_), x_norm) # [512]

                # Seq=1 GQA: trivial attention
                q_h = q.reshape(NQ, DH); k_h = k.reshape(NKV, DH); v_h = v.reshape(NKV, DH)
                del q, k, v
                k_rep = np.repeat(k_h, NQ//NKV, axis=0)
                v_rep = np.repeat(v_h, NQ//NKV, axis=0)
                context = v_rep.reshape(-1)  # seq=1: attn_weights=1 trivially
                del k_h, v_h, q_h, k_rep, v_rep

                y = dequant_ternary_matvec(fb, d_o, (D_, NQ*DH), context)
                del context
                h = h + y; del y

            # FFN: gate TERNARY [F,D], up TERNARY [F,D], down TERNARY [D,F], ffn_norm FP32 [D]
            d_gate = d(); d_up = d(); d_dn = d(); d_fn = d()
            ffn_norm_w = dequant_fp32_read(fb, d_fn, (D_,))
            x_ffn = rmsnorm(h, ffn_norm_w)
            del ffn_norm_w

            gate = dequant_ternary_matvec(fb, d_gate, (F_, D_), x_ffn)  # [F]
            up   = dequant_ternary_matvec(fb, d_up,   (F_, D_), x_ffn)  # [F]
            del x_ffn
            swiglu = silu(gate) * up
            del gate, up
            y = dequant_ternary_matvec(fb, d_dn, (D_, F_), swiglu)
            del swiglu
            h = h + y; del y
            gc.collect()

            # Checkpoint captures
            if li == 0:  ckpts["CKPT2_STATE0"]     = tensor_stats(h)
            elif li == 2:  ckpts["CKPT3_GQA2"]     = tensor_stats(h)
            elif li == 3:  ckpts["CKPT4_STATE3"]   = tensor_stats(h)
            elif li == 5:  ckpts["CKPT5_GQA5"]     = tensor_stats(h)
            elif li == 12: ckpts["CKPT6_STATE12"]  = tensor_stats(h)

            if li % 4 == 0:
                print(f"    Layer {li:02d} {'GQA' if is_gqa else 'State'} done | h_mean={h.mean():.4f}")

        ckpts["CKPT7_FINAL_BLOCK"] = tensor_stats(h)

        # Final RMSNorm
        d_fn  = d()  # final_norm FP32 [D]
        fn_w  = dequant_fp32_read(fb, d_fn, (D_,))
        h_norm = rmsnorm(h, fn_w); del h, fn_w
        ckpts["CKPT8_RMSNORM"] = tensor_stats(h_norm)

        # LM Head INT8 [V, D]
        d_lm = d()  # lm_head
        logits = dequant_int8_matvec(fb, d_lm, (V_, D_), h_norm)
        del h_norm

    ckpts["CKPT9_LOGITS"] = logit_stats(logits)
    logits_sha = sha256_b(logits.astype(np.float32).tobytes())
    del logits
    gc.collect()
    print(f"    ARGMAX={ckpts['CKPT9_LOGITS']['argmax_id']} TOP5={ckpts['CKPT9_LOGITS']['top5_ids']}")
    return ckpts, logits_sha

def main():
    print("=" * 70)
    print("FIX-12 PHASE C/D — NANO REFERENCE FORWARD (REFERENCE-B, STREAMING)")
    print("=" * 70)

    assert os.path.isfile(NANO_PATH), f"Not found: {NANO_PATH}"
    nano_size = os.path.getsize(NANO_PATH)
    print(f"\nHashing model.nano ({nano_size:,} bytes)...")
    nano_sha = sha256_b(open(NANO_PATH,"rb").read())
    EXP = "0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64"
    sha_ok = (nano_sha == EXP)
    print(f"SHA256: {nano_sha}")
    print(f"MATCH:  {'PASS' if sha_ok else 'FAIL — ABORT'}"); assert sha_ok

    descs = parse_nano(NANO_PATH)
    all_results = []

    for label, prompt, token_ids in PROMPTS:
        last_tok = token_ids[-1]
        print(f"\n[{label}] {repr(prompt)} → token {last_tok}")
        try:
            ckpts, logits_sha = run_forward(NANO_PATH, descs, last_tok, label)
            all_results.append({
                "label": label, "prompt": prompt,
                "token_ids": token_ids, "forward_token_id": last_tok,
                "checkpoints": ckpts, "logits_sha256": logits_sha, "status": "OK",
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append({"label": label, "prompt": prompt, "status": f"ERROR:{e}"})
        gc.collect()

    output = {
        "fix_version": "FIX-12", "phase": "C-D",
        "nano_path": NANO_PATH, "nano_size": nano_size,
        "nano_sha256": nano_sha, "nano_sha_ok": sha_ok,
        "tensor_count": 219, "prompts": all_results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\nPHASE C/D OUTPUT: {OUTPUT_JSON}")
    for r in all_results:
        lbl = r["label"]
        if r.get("status") == "OK":
            ck = r["checkpoints"]
            print(f"FIX12_REFB_{lbl}_ARGMAX = {ck['CKPT9_LOGITS']['argmax_id']}")
            print(f"FIX12_REFB_{lbl}_TOP5   = {ck['CKPT9_LOGITS']['top5_ids']}")
            print(f"FIX12_REFB_{lbl}_SHA    = {r['logits_sha256']}")
        else:
            print(f"FIX12_REFB_{lbl}_STATUS = {r['status']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
