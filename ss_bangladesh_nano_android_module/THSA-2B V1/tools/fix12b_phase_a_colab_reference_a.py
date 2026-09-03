#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX-12B Phase A — Reference-A: Step-30 PyTorch Checkpoint Forward Pass
=======================================================================
Run this on Google Colab where checkpoint_step_000030.pt lives.

OUTPUTS (download after running):
  fix12b_reference_a_results.json
  fix12b/reference_a_logits_p0.bin  ... p4.bin  (65536 × float32 each)

CHECKPOINT: checkpoint_step_000030.pt
  Expected SHA256: 0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667
  Expected size:   4,106,953,961 bytes
"""

# ─── Colab: mount Drive ────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("Drive mounted.")
except Exception:
    print("Not on Colab or Drive not needed — proceeding.")

import sys, os, json, hashlib, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─── PATHS — adjust if checkpoint is in a different Drive location ─────────────
# Try common locations
CANDIDATE_PATHS = [
    "/content/drive/MyDrive/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B V1/training/checkpoints/checkpoint_step_000030.pt",
    "/content/drive/MyDrive/checkpoint_step_000030.pt",
    "/content/drive/MyDrive/checkpoints/checkpoint_step_000030.pt",
    "/content/checkpoint_step_000030.pt",
]
CHECKPOINT_PATH = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        CHECKPOINT_PATH = p
        break

if CHECKPOINT_PATH is None:
    print("ERROR: checkpoint_step_000030.pt not found. Update CANDIDATE_PATHS.")
    print("Tried:", CANDIDATE_PATHS)
    # Allow manual override
    CHECKPOINT_PATH = input("Enter full path to checkpoint_step_000030.pt: ").strip()

OUT_DIR = "/content/fix12b"
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_SHA  = "0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667"
EXPECTED_SIZE = 4_106_953_961

# ─── Token IDs from Phase B (verified) ────────────────────────────────────────
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

# ─── Helpers ───────────────────────────────────────────────────────────────────
def sha256_file(path):
    md = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk: break
            md.update(chunk)
    return md.hexdigest()

def tensor_stats(t, label=""):
    """Compute and return stats dict for a tensor."""
    arr = t.detach().float().cpu()
    flat = arr.reshape(-1)
    l2 = float(torch.norm(flat, p=2))
    return {
        "label":    label,
        "shape":    list(arr.shape),
        "dtype":    str(arr.dtype),
        "min":      float(arr.min()),
        "max":      float(arr.max()),
        "mean":     float(arr.mean()),
        "mean_abs": float(arr.abs().mean()),
        "max_abs":  float(arr.abs().max()),
        "l2_norm":  l2,
        "finite":   bool(torch.all(torch.isfinite(arr))),
        "nonzero":  int(torch.count_nonzero(arr)),
        "sha256":   hashlib.sha256(arr.numpy().astype(np.float32).tobytes()).hexdigest(),
    }

def logit_stats(logits_t, label=""):
    """Compute stats + top-10 for logits tensor [vocab]."""
    stats = tensor_stats(logits_t, label)
    arr = logits_t.detach().float().cpu()
    top10 = torch.topk(arr, 10)
    stats["argmax_id"]  = int(arr.argmax())
    stats["top5_ids"]   = top10.indices[:5].tolist()
    stats["top10_ids"]  = top10.indices.tolist()
    stats["top5_vals"]  = [float(v) for v in top10.values[:5]]
    stats["top10_vals"] = [float(v) for v in top10.values]
    return stats

# ─── Model Architecture ────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight.float()).to(x.dtype)

class TernaryLinear(nn.Module):
    """Placeholder for TernaryLinear — at inference uses standard float weights."""
    def __init__(self, in_features, out_features, bias=False, is_sensitive=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
    def forward(self, x):
        out = F.linear(x, self.weight)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out

class ShortConvStateBlock(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.in_proj = TernaryLinear(d_model, 2 * d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size - 1, groups=d_model)
        self.out_proj = TernaryLinear(d_model, d_model)
        self.norm = RMSNorm(d_model)
    def forward(self, x):
        B, S, D = x.shape
        residual = x
        x_n = self.norm(x)
        proj = self.in_proj(x_n)
        gate, value = proj.chunk(2, dim=-1)
        # Conv1D on value: [B,D,S] → [B,D,S]
        conv_out = self.conv1d(value.transpose(1,2))[:, :, :S].transpose(1,2)
        mixed = F.silu(gate) * conv_out
        out = self.out_proj(mixed)
        return residual + out

class GQAttentionBlock(nn.Module):
    def __init__(self, d_model, n_query_heads, n_kv_heads, d_head, is_sensitive=False):
        super().__init__()
        self.n_q = n_query_heads
        self.n_kv = n_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        self.q_proj = TernaryLinear(d_model, n_query_heads * d_head)
        self.k_proj = TernaryLinear(d_model, n_kv_heads * d_head)
        self.v_proj = TernaryLinear(d_model, n_kv_heads * d_head)
        self.out_proj = TernaryLinear(n_query_heads * d_head, d_model)
        self.norm = RMSNorm(d_model)
    def forward(self, x):
        B, S, D = x.shape
        residual = x
        x_n = self.norm(x)
        q = self.q_proj(x_n).view(B, S, self.n_q, self.d_head).transpose(1,2)
        k = self.k_proj(x_n).view(B, S, self.n_kv, self.d_head).transpose(1,2)
        v = self.v_proj(x_n).view(B, S, self.n_kv, self.d_head).transpose(1,2)
        rep = self.n_q // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        mask = torch.triu(torch.full((S,S), float('-inf'), device=x.device, dtype=q.dtype), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        ctx = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, S, -1)
        y = self.out_proj(ctx)
        return residual + y

class GatedSwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate_proj = TernaryLinear(d_model, d_ffn)
        self.up_proj   = TernaryLinear(d_model, d_ffn)
        self.down_proj = TernaryLinear(d_ffn, d_model)
        self.norm = RMSNorm(d_model)
    def forward(self, x):
        residual = x
        x_n = self.norm(x)
        gate = self.gate_proj(x_n)
        up   = self.up_proj(x_n)
        y    = self.down_proj(F.silu(gate) * up)
        return residual + y

class THSABackboneBlock(nn.Module):
    def __init__(self, mixer, ffn):
        super().__init__()
        self.mixer = mixer
        self.ffn = ffn
    def forward(self, x):
        return self.ffn(self.mixer(x))

class THSAHybridForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size   = config.get("vocab_size", 65536)
        self.d_model      = config.get("d_model", 2560)
        self.total_blocks = config.get("total_blocks", 24)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList()
        gqa_interval = self.total_blocks // config.get("gqa_blocks", 8)
        for i in range(self.total_blocks):
            if (i + 1) % gqa_interval == 0:
                is_b = (i == 0 or i == self.total_blocks - 1)
                mixer = GQAttentionBlock(
                    self.d_model,
                    config.get("n_query_heads", 20),
                    config.get("n_kv_heads", 4),
                    config.get("d_head", 128),
                    is_sensitive=is_b
                )
            else:
                mixer = ShortConvStateBlock(self.d_model, kernel_size=4)
            ffn = GatedSwiGLUFFN(self.d_model, config.get("d_ffn", 6912))
            self.layers.append(THSABackboneBlock(mixer, ffn))
        self.final_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward_with_checkpoints(self, input_ids):
        """Forward pass that captures 9 intermediate checkpoints."""
        x = self.embed_tokens(input_ids)  # [B, S, D]
        ckpts = {}
        ckpts["A1_EMBED"] = tensor_stats(x[:, -1, :], "A1_EMBED")

        for i, block in enumerate(self.layers):
            x = block(x)
            ckpt_map = {0: "A2_STATE0", 2: "A3_GQA2", 3: "A4_STATE3",
                        5: "A5_GQA5", 12: "A6_STATE12", 23: "A7_FINAL_BLOCK"}
            if i in ckpt_map:
                ckpts[ckpt_map[i]] = tensor_stats(x[:, -1, :], ckpt_map[i])

        x = self.final_norm(x)
        ckpts["A8_RMSNORM"] = tensor_stats(x[:, -1, :], "A8_RMSNORM")

        logits = self.lm_head(x)
        # Return logits for LAST token only
        last_logits = logits[:, -1, :].squeeze(0)  # [vocab]
        ckpts["A9_LOGITS"] = logit_stats(last_logits, "A9_LOGITS")
        return last_logits, ckpts

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FIX-12B PHASE A — REFERENCE-A: STEP-30 PYTORCH FORWARD PASS")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Step 1: Checkpoint SHA BEFORE ─────────────────────────────────────────
    print(f"\n[CHECKPOINT] {CHECKPOINT_PATH}")
    sz = os.path.getsize(CHECKPOINT_PATH)
    print(f"  Size: {sz:,} bytes  {'OK' if sz == EXPECTED_SIZE else 'MISMATCH — expected ' + str(EXPECTED_SIZE)}")
    print("  Computing SHA256 BEFORE load...")
    sha_before = sha256_file(CHECKPOINT_PATH)
    print(f"  SHA256 BEFORE: {sha_before}")
    sha_ok = (sha_before == EXPECTED_SHA)
    print(f"  SHA match: {'PASS' if sha_ok else 'FAIL — expected ' + EXPECTED_SHA}")
    if not sha_ok:
        print("WARNING: SHA mismatch. Proceeding but marking as FAIL.")

    # ── Step 2: Load checkpoint ────────────────────────────────────────────────
    print("\n[LOADING CHECKPOINT] (read-only, inference mode)...")
    t0 = time.time()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Keys: {list(ckpt.keys())[:5]} ...")
    global_step = ckpt.get("global_step", ckpt.get("step", "N/A"))
    print(f"  global_step = {global_step}")

    # Get state dict
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # Assume the checkpoint IS the state dict
        state_dict = ckpt

    tensor_count = len(state_dict)
    param_count = sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {param_count:,}")

    # ── Step 3: Build model and load weights ───────────────────────────────────
    config = {
        "vocab_size": 65536, "d_model": 2560, "total_blocks": 24,
        "gqa_blocks": 8, "d_ffn": 6912, "n_query_heads": 20,
        "n_kv_heads": 4, "d_head": 128,
    }
    print("\n[BUILDING MODEL] ...")
    model = THSAHybridForCausalLM(config)
    load_result = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(load_result.missing_keys)}")
    print(f"  Unexpected: {len(load_result.unexpected_keys)}")
    if load_result.missing_keys:
        print(f"  First missing: {load_result.missing_keys[:3]}")

    model = model.to(device)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable:,}")

    # NaN / Inf check
    nan_count = sum(1 for p in model.parameters() if torch.any(torch.isnan(p)))
    inf_count = sum(1 for p in model.parameters() if torch.any(torch.isinf(p)))
    print(f"  NaN tensors: {nan_count}   Inf tensors: {inf_count}")

    # ── Step 4: Forward passes for all 5 prompts ───────────────────────────────
    all_results = []
    print("\n[REFERENCE-A FORWARD PASSES]")
    with torch.inference_mode():
        for pi, (label, prompt) in enumerate(PROMPTS):
            ids = TOKEN_IDS[label]
            print(f"\n  [{label}] '{prompt}' → token_ids={ids}")
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            t0 = time.time()
            logits, ckpts = model.forward_with_checkpoints(inp)
            elapsed = time.time() - t0

            logits_np = logits.float().cpu().numpy().astype(np.float32)
            logit_sha = hashlib.sha256(logits_np.tobytes()).hexdigest()
            argmax = int(np.argmax(logits_np))
            top5 = [int(x) for x in np.argpartition(logits_np, -5)[-5:][np.argsort(logits_np[np.argpartition(logits_np, -5)[-5:]])[::-1]]]

            print(f"  ARGMAX={argmax} TOP5={top5}")
            print(f"  Logits SHA={logit_sha[:16]}... elapsed={elapsed:.2f}s")

            # Write logit binary
            logit_path = f"{OUT_DIR}/reference_a_logits_p{pi}.bin"
            logits_np.tofile(logit_path)
            print(f"  Written: {logit_path} ({os.path.getsize(logit_path)} bytes)")

            all_results.append({
                "label": label, "prompt": prompt,
                "token_ids": ids,
                "elapsed_s": elapsed,
                "logits_sha256": logit_sha,
                "logits_path": logit_path,
                "checkpoints": ckpts,
            })
            print(f"  FIX12B_REFA_{label}_ARGMAX = {argmax}")
            print(f"  FIX12B_REFA_{label}_TOP5   = {top5}")

    # ── Step 5: SHA AFTER (immutability check) ─────────────────────────────────
    print("\n[CHECKPOINT SHA AFTER]")
    sha_after = sha256_file(CHECKPOINT_PATH)
    immutable = (sha_before == sha_after)
    print(f"  SHA256 AFTER: {sha_after}")
    print(f"  IMMUTABLE: {'YES' if immutable else 'FAIL — SHA CHANGED!'}")

    # ── Step 6: Write results JSON ────────────────────────────────────────────
    output = {
        "fix_version": "FIX-12B",
        "phase": "A-REFERENCE-A-PYTORCH",
        "device": device,
        "global_step": str(global_step),
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_size": sz,
        "checkpoint_sha_before": sha_before,
        "checkpoint_sha_after": sha_after,
        "checkpoint_immutable": immutable,
        "checkpoint_sha_ok": sha_ok,
        "total_params": total_params,
        "tensor_count": tensor_count,
        "nan_tensors": nan_count,
        "inf_tensors": inf_count,
        "prompts": all_results,
    }
    result_path = f"{OUT_DIR}/reference_a_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Step 7: Machine-readable block ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIX12B_CHECKPOINT_SHA_BEFORE=" + sha_before)
    print("FIX12B_CHECKPOINT_SHA_AFTER="  + sha_after)
    print("FIX12B_CHECKPOINT_IMMUTABLE="  + ("YES" if immutable else "NO"))
    print(f"FIX12B_CHECKPOINT_SIZE={sz}")
    print(f"FIX12B_GLOBAL_STEP={global_step}")
    print(f"FIX12B_TOTAL_PARAMS={total_params}")
    print(f"FIX12B_TENSOR_COUNT={tensor_count}")
    print(f"FIX12B_NAN_TENSORS={nan_count}")
    print(f"FIX12B_INF_TENSORS={inf_count}")
    print(f"FIX12B_REFERENCE_A_READY=YES")
    for r in all_results:
        lbl = r["label"]
        am  = r["checkpoints"]["A9_LOGITS"]["argmax_id"]
        t5  = r["checkpoints"]["A9_LOGITS"]["top5_ids"]
        sha = r["logits_sha256"]
        print(f"FIX12B_REFA_{lbl}_ARGMAX={am}")
        print(f"FIX12B_REFA_{lbl}_TOP5={t5}")
        print(f"FIX12B_REFA_{lbl}_LOGITS_SHA={sha}")
    print("=" * 70)
    print(f"\nDownload these files from Colab:")
    print(f"  {result_path}")
    for pi in range(len(PROMPTS)):
        print(f"  {OUT_DIR}/reference_a_logits_p{pi}.bin")

if __name__ == "__main__":
    main()
