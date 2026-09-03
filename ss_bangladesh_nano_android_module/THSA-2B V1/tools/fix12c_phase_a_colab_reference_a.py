#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX-12C Phase A — Reference-A: Step-30 PyTorch Checkpoint Layerwise Hidden-State Capture
=======================================================================================
Google Colab execution script for the authoritative Step-30 PyTorch model.
Loads checkpoint_step_000030.pt (READ-ONLY, SHA256 verified before & after).
Captures all 25 layerwise intermediate checkpoints for the final prompt token:
  fix12c/reference_a/prompt_{pi}/ckpt*.bin (float32 little-endian)

CHECKPOINT INTEGRITY:
  Expected size:   4,106,953,961 bytes
  Expected SHA256: 0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667
"""

# Mount Drive if on Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("Drive mounted.")
except Exception:
    print("Local or non-Colab execution.")

import sys, os, json, hashlib, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Common checkpoint paths on Colab
CANDIDATE_PATHS = [
    "/content/drive/MyDrive/checkpoint_step_000030.pt",
    "/content/drive/MyDrive/checkpoints/checkpoint_step_000030.pt",
    "/content/drive/MyDrive/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B V1/training/checkpoints/checkpoint_step_000030.pt",
    "/content/checkpoint_step_000030.pt",
    "checkpoint_step_000030.pt",
]

CHECKPOINT_PATH = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        CHECKPOINT_PATH = p
        break

if CHECKPOINT_PATH is None:
    print("Candidate paths searched:", CANDIDATE_PATHS)
    CHECKPOINT_PATH = input("Enter exact path to checkpoint_step_000030.pt: ").strip()

OUT_DIR = Path("/content/fix12c/reference_a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA  = "0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667"
EXPECTED_SIZE = 4_106_953_961

TOKEN_IDS = {
    "TEST-A": [360, 43226, 64782, 64792],
    "TEST-B": [1620, 3715, 3101, 64792],
    "TEST-C": [4874, 6494, 4186, 4289, 1357, 263, 5821, 19591, 64792],
    "TEST-D": [2232, 15325, 1656, 1718, 2667],
    "TEST-E": [2829, 1620, 3715, 64705],
}

PROMPTS = [
    ("TEST-A", "2+2=?"),
    ("TEST-B", "বাংলাদেশের রাজধানী কী?"),
    ("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?"),
    ("TEST-D", "১২ × ৮ = ?"),
    ("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।"),
]

DETAILED_BLOCKS = {0, 1, 2, 3, 4, 5, 12, 23}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def dump_vector(p_dir: Path, name: str, tensor: torch.Tensor) -> dict:
    f32_np = tensor.detach().float().cpu().numpy().astype(np.float32)
    raw = f32_np.tobytes()
    out_file = p_dir / f"{name}.bin"
    with open(out_file, "wb") as f:
        f.write(raw)
    h = hashlib.sha256(raw).hexdigest()
    v_dbl = f32_np.astype(np.float64).ravel()
    l2 = float(np.sqrt(np.dot(v_dbl, v_dbl)))
    return {
        "name": name,
        "file": str(out_file.name),
        "shape": list(f32_np.shape),
        "dim": len(v_dbl),
        "min": float(v_dbl.min()),
        "max": float(v_dbl.max()),
        "mean": float(v_dbl.mean()),
        "l2_norm": l2,
        "sha256": h,
    }

# ─── Model Architecture (Exact Step-30 THSA-2B) ──────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight.float()).to(x.dtype)

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, is_sensitive=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
    def forward(self, x):
        out = F.linear(x, self.weight)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out

def main():
    print("=" * 80)
    print("FIX-12C PHASE A — STEP-30 PYTORCH LAYERWISE HIDDEN STATE CAPTURE")
    print("=" * 80)

    # 1. Checkpoint Verification BEFORE loading
    print("\n[CHECKPOINT INTEGRITY AUDIT — BEFORE LOAD]")
    sz = os.path.getsize(CHECKPOINT_PATH)
    print(f"  Path: {CHECKPOINT_PATH}")
    print(f"  Size: {sz:,} bytes (Expected: {EXPECTED_SIZE:,})")
    assert sz == EXPECTED_SIZE, f"Size mismatch! Got {sz}, expected {EXPECTED_SIZE}"
    sha_before = sha256_file(CHECKPOINT_PATH)
    print(f"  SHA256: {sha_before}")
    assert sha_before == EXPECTED_SHA, f"SHA256 mismatch! Got {sha_before}, expected {EXPECTED_SHA}"
    print("  Checkpoint SHA: PASS (verified against authoritative hash)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 2. Load Checkpoint
    print("\n[LOADING CHECKPOINT] ...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    print(f"  Tensors in state_dict: {len(state_dict)}")
    total_params = sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))
    print(f"  Total parameters: {total_params:,} (Expected: 2,050,296,320)")
    assert len(state_dict) == 219, f"Expected 219 tensors, got {len(state_dict)}"

    # 3. Extract parameter weights directly for clean, isolated layer-by-layer forward pass
    # Token Embedding [65536, 2560]
    embed_w = state_dict["embed_tokens.weight"].to(device)
    final_norm_w = state_dict["final_norm.weight"].to(device)
    lm_head_w = state_dict["lm_head.weight"].to(device)

    D = 2560
    F_DIM = 6912
    NQ = 20
    NKV = 4
    DH = 128
    scale_attn = 1.0 / math.sqrt(DH)

    all_prompt_records = {}

    with torch.inference_mode():
        for pi, (label, prompt_text) in enumerate(PROMPTS):
            token_ids = TOKEN_IDS[label]
            S = len(token_ids)
            print(f"\n[{label}] '{prompt_text}' ({S} tokens: {token_ids})")

            p_dir = OUT_DIR / f"prompt_{pi}"
            p_dir.mkdir(parents=True, exist_ok=True)
            prompt_ckpts = {}

            # Input token tensor [1, S]
            inp_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

            # 1. Embedding [1, S, D]
            h = F.embedding(inp_ids, embed_w) # [1, S, D]
            prompt_ckpts["ckpt01_embed"] = dump_vector(p_dir, "ckpt01_embed", h[0, -1, :])

            # 2. Backbone 24 blocks
            for li in range(24):
                is_g = ((li + 1) % 3 == 0)
                is_detailed = (li in DETAILED_BLOCKS)
                prefix = f"layers.{li}."

                # Residual stream entering block
                prompt_ckpts[f"ckpt02_block_{li:02d}_input"] = dump_vector(p_dir, f"ckpt02_block_{li:02d}_input", h[0, -1, :])

                if not is_g:
                    # ── State Block ──
                    norm_w = state_dict[f"{prefix}mixer.norm.weight"].to(device)
                    # Mixer RMSNorm
                    var = h.float().pow(2).mean(-1, keepdim=True)
                    h_normed = (h.float() / torch.sqrt(var + 1e-5) * norm_w.float()).to(h.dtype)
                    if is_detailed:
                        prompt_ckpts[f"ckpt03_block_{li:02d}_state_norm"] = dump_vector(p_dir, f"ckpt03_block_{li:02d}_state_norm", h_normed[0, -1, :])

                    # In projection
                    in_w = state_dict[f"{prefix}mixer.in_proj.weight"].to(device)
                    in_proj = F.linear(h_normed, in_w) # [1, S, 5120]
                    if is_detailed:
                        prompt_ckpts[f"ckpt04_block_{li:02d}_state_in_proj"] = dump_vector(p_dir, f"ckpt04_block_{li:02d}_state_in_proj", in_proj[0, -1, :])

                    gate, val = in_proj.chunk(2, dim=-1) # [1, S, 2560]
                    if is_detailed:
                        prompt_ckpts[f"ckpt05a_block_{li:02d}_state_gate"] = dump_vector(p_dir, f"ckpt05a_block_{li:02d}_state_gate", gate[0, -1, :])
                        prompt_ckpts[f"ckpt05b_block_{li:02d}_state_value"] = dump_vector(p_dir, f"ckpt05b_block_{li:02d}_state_value", val[0, -1, :])

                    # Conv1D
                    conv_w = state_dict[f"{prefix}mixer.conv1d.weight"].to(device)
                    conv_b = state_dict[f"{prefix}mixer.conv1d.bias"].to(device)
                    # Causal conv1d on value stream
                    # val shape: [1, S, D] -> [1, D, S]
                    val_t = val.transpose(1, 2)
                    conv_out = F.conv1d(val_t, conv_w, conv_b, padding=3, groups=D)[:, :, :S].transpose(1, 2)
                    if is_detailed:
                        prompt_ckpts[f"ckpt06_block_{li:02d}_state_conv"] = dump_vector(p_dir, f"ckpt06_block_{li:02d}_state_conv", conv_out[0, -1, :])

                    silu_gate = F.silu(gate)
                    if is_detailed:
                        prompt_ckpts[f"ckpt07_block_{li:02d}_state_silu"] = dump_vector(p_dir, f"ckpt07_block_{li:02d}_state_silu", silu_gate[0, -1, :])

                    gated = silu_gate * conv_out
                    if is_detailed:
                        prompt_ckpts[f"ckpt08_block_{li:02d}_state_gated"] = dump_vector(p_dir, f"ckpt08_block_{li:02d}_state_gated", gated[0, -1, :])

                    out_w = state_dict[f"{prefix}mixer.out_proj.weight"].to(device)
                    state_out = F.linear(gated, out_w)
                    if is_detailed:
                        prompt_ckpts[f"ckpt09_block_{li:02d}_state_out_proj"] = dump_vector(p_dir, f"ckpt09_block_{li:02d}_state_out_proj", state_out[0, -1, :])

                    h = h + state_out
                    if is_detailed:
                        prompt_ckpts[f"ckpt10_block_{li:02d}_state_residual"] = dump_vector(p_dir, f"ckpt10_block_{li:02d}_state_residual", h[0, -1, :])

                else:
                    # ── GQA Block ──
                    norm_w = state_dict[f"{prefix}mixer.norm.weight"].to(device)
                    var = h.float().pow(2).mean(-1, keepdim=True)
                    h_normed = (h.float() / torch.sqrt(var + 1e-5) * norm_w.float()).to(h.dtype)
                    if is_detailed:
                        prompt_ckpts[f"ckpt11_block_{li:02d}_gqa_norm"] = dump_vector(p_dir, f"ckpt11_block_{li:02d}_gqa_norm", h_normed[0, -1, :])

                    q_w = state_dict[f"{prefix}mixer.q_proj.weight"].to(device)
                    k_w = state_dict[f"{prefix}mixer.k_proj.weight"].to(device)
                    v_w = state_dict[f"{prefix}mixer.v_proj.weight"].to(device)
                    out_w = state_dict[f"{prefix}mixer.out_proj.weight"].to(device)

                    q = F.linear(h_normed, q_w).view(1, S, NQ, DH).transpose(1, 2)   # [1, NQ, S, DH]
                    k = F.linear(h_normed, k_w).view(1, S, NKV, DH).transpose(1, 2)  # [1, NKV, S, DH]
                    v = F.linear(h_normed, v_w).view(1, S, NKV, DH).transpose(1, 2)  # [1, NKV, S, DH]

                    if is_detailed:
                        prompt_ckpts[f"ckpt12a_block_{li:02d}_gqa_q"] = dump_vector(p_dir, f"ckpt12a_block_{li:02d}_gqa_q", q[0, :, -1, :].reshape(-1))
                        prompt_ckpts[f"ckpt12b_block_{li:02d}_gqa_k"] = dump_vector(p_dir, f"ckpt12b_block_{li:02d}_gqa_k", k[0, :, -1, :].reshape(-1))
                        prompt_ckpts[f"ckpt12c_block_{li:02d}_gqa_v"] = dump_vector(p_dir, f"ckpt12c_block_{li:02d}_gqa_v", v[0, :, -1, :].reshape(-1))

                    # GQA KV head repeat
                    repeat_factor = NQ // NKV
                    k_exp = k.repeat_interleave(repeat_factor, dim=1) # [1, NQ, S, DH]
                    v_exp = v.repeat_interleave(repeat_factor, dim=1) # [1, NQ, S, DH]

                    scores = torch.matmul(q, k_exp.transpose(-1, -2)) * scale_attn # [1, NQ, S, S]
                    causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=device, dtype=scores.dtype), diagonal=1)
                    scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
                    attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=v_exp.dtype)
                    context = torch.matmul(attn_w, v_exp).transpose(1, 2).contiguous().view(1, S, D)

                    if is_detailed:
                        prompt_ckpts[f"ckpt13_block_{li:02d}_gqa_attention"] = dump_vector(p_dir, f"ckpt13_block_{li:02d}_gqa_attention", context[0, -1, :])

                    gqa_out = F.linear(context, out_w)
                    if is_detailed:
                        prompt_ckpts[f"ckpt14_block_{li:02d}_gqa_out_proj"] = dump_vector(p_dir, f"ckpt14_block_{li:02d}_gqa_out_proj", gqa_out[0, -1, :])

                    h = h + gqa_out
                    if is_detailed:
                        prompt_ckpts[f"ckpt15_block_{li:02d}_gqa_residual"] = dump_vector(p_dir, f"ckpt15_block_{li:02d}_gqa_residual", h[0, -1, :])

                # ── FFN Block (All 24 blocks) ──
                ffn_norm_w = state_dict[f"{prefix}ffn.norm.weight"].to(device)
                var = h.float().pow(2).mean(-1, keepdim=True)
                h_ffn_norm = (h.float() / torch.sqrt(var + 1e-5) * ffn_norm_w.float()).to(h.dtype)
                if is_detailed:
                    prompt_ckpts[f"ckpt16_block_{li:02d}_ffn_norm"] = dump_vector(p_dir, f"ckpt16_block_{li:02d}_ffn_norm", h_ffn_norm[0, -1, :])

                g_w = state_dict[f"{prefix}ffn.gate_proj.weight"].to(device)
                u_w = state_dict[f"{prefix}ffn.up_proj.weight"].to(device)
                d_w = state_dict[f"{prefix}ffn.down_proj.weight"].to(device)

                gate_f = F.linear(h_ffn_norm, g_w)
                up_f   = F.linear(h_ffn_norm, u_w)
                if is_detailed:
                    prompt_ckpts[f"ckpt17_block_{li:02d}_ffn_gate"] = dump_vector(p_dir, f"ckpt17_block_{li:02d}_ffn_gate", gate_f[0, -1, :])
                    prompt_ckpts[f"ckpt18_block_{li:02d}_ffn_up"] = dump_vector(p_dir, f"ckpt18_block_{li:02d}_ffn_up", up_f[0, -1, :])

                swiglu = F.silu(gate_f) * up_f
                if is_detailed:
                    prompt_ckpts[f"ckpt19_block_{li:02d}_ffn_activation"] = dump_vector(p_dir, f"ckpt19_block_{li:02d}_ffn_activation", swiglu[0, -1, :])

                ffn_out = F.linear(swiglu, d_w)
                if is_detailed:
                    prompt_ckpts[f"ckpt20_block_{li:02d}_ffn_down"] = dump_vector(p_dir, f"ckpt20_block_{li:02d}_ffn_down", ffn_out[0, -1, :])

                h = h + ffn_out
                prompt_ckpts[f"ckpt21_block_{li:02d}_ffn_residual"] = dump_vector(p_dir, f"ckpt21_block_{li:02d}_ffn_residual", h[0, -1, :])

            # End of 24 blocks -> Final Norm
            var = h.float().pow(2).mean(-1, keepdim=True)
            h_final = (h.float() / torch.sqrt(var + 1e-5) * final_norm_w.float()).to(h.dtype)
            prompt_ckpts["ckpt22_final_norm"] = dump_vector(p_dir, "ckpt22_final_norm", h_final[0, -1, :])
            prompt_ckpts["ckpt23_lm_head_input"] = dump_vector(p_dir, "ckpt23_lm_head_input", h_final[0, -1, :])

            # LM Head
            logits = F.linear(h_final[0, -1, :], lm_head_w) # [65536]
            prompt_ckpts["ckpt24_logits"] = dump_vector(p_dir, "ckpt24_logits", logits)

            am = int(torch.argmax(logits).item())
            top5 = torch.topk(logits, 5).indices.tolist()
            print(f"  Final Token Logits: Argmax={am}, Top5={top5}")
            print(f"  Checkpoints written: {len(prompt_ckpts)} files in {p_dir}")

            all_prompt_records[label] = {
                "label": label,
                "prompt": prompt_text,
                "token_ids": token_ids,
                "argmax": am,
                "top5": top5,
                "checkpoints": prompt_ckpts,
            }

    # Re-verify Checkpoint SHA AFTER all inference
    print("\n[CHECKPOINT INTEGRITY AUDIT — AFTER RUN]")
    sha_after = sha256_file(CHECKPOINT_PATH)
    print(f"  SHA256 after run: {sha_after}")
    assert sha_after == EXPECTED_SHA, "CRITICAL ERROR: CHECKPOINT MUTATED!"
    print("  Checkpoint Immutability: 100% VERIFIED (SHA256 identical)")

    meta_file = OUT_DIR / "reference_a_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(all_prompt_records, f, indent=2)
    print(f"\nMetadata written to {meta_file}")
    print("ALL REFERENCE-A CHECKPOINTS SUCCESSFULLY GENERATED.")

if __name__ == "__main__":
    main()
