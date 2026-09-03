#!/usr/bin/env python3
"""
FIX-12 Phase G — Numerical Comparison: REFERENCE-B vs Android Native
=====================================================================
Loads:
  - fix12_phase_cd_reference_results.json  (REFERENCE-B, from Phase C/D)
  - fix12_logits_pN.bin                    (Android native 65536 logits, pulled from device)
  - fix12_diag.bin                         (Android native checkpoint stats, pulled from device)

Computes for each checkpoint:
  - max_abs_error, mean_abs_error, RMSE, cosine_similarity
  - Top-1/5/10 agreement between REFERENCE-B and Android

Outputs: tools/fix12_phase_g_comparison_results.json

Run from THSA-2B V1 root after pulling device files:
    python tools/fix12_phase_g_compare.py
"""

import sys, os, json, struct, hashlib, math
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)

REF_JSON    = os.path.join(SCRIPT_DIR, "fix12_phase_cd_reference_results.json")
DEVICE_DIR  = os.path.join(SCRIPT_DIR, "fix12_device_capture")  # ADB pulled files here
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "fix12_phase_g_comparison_results.json")

PROMPTS_ORDER = ["TEST-A", "TEST-B", "TEST-C", "TEST-D", "TEST-E"]

def sha256_b(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.sqrt(np.dot(a, a))
    nb = np.sqrt(np.dot(b, b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def compare_logits(ref_logits_sha: str, ref_ckpt: dict, android_logits: np.ndarray) -> dict:
    """Compare REFERENCE-B logit stats vs Android native logit array."""
    android_sha = sha256_b(android_logits.astype(np.float32).tobytes())

    # Reference stats (from JSON — summary only, no full array)
    ref_argmax  = ref_ckpt["argmax_id"]
    ref_top5    = ref_ckpt["top5_ids"]
    ref_top10   = ref_ckpt["top10_ids"]

    # Android stats
    and_argmax  = int(np.argmax(android_logits))
    top10_idx   = np.argpartition(android_logits, -10)[-10:]
    top10_idx   = top10_idx[np.argsort(android_logits[top10_idx])[::-1]]
    and_top5    = [int(x) for x in top10_idx[:5]]
    and_top10   = [int(x) for x in top10_idx]

    top1_match  = (ref_argmax == and_argmax)
    top5_match  = set(ref_top5) == set(and_top5)
    top10_match = set(ref_top10) == set(and_top10)

    # Stats comparison
    ref_min  = ref_ckpt["min"];  ref_max  = ref_ckpt["max"]
    ref_mean = ref_ckpt["mean"]; ref_l2   = ref_ckpt["l2_norm"]
    and_min  = float(android_logits.min()); and_max = float(android_logits.max())
    and_mean = float(android_logits.mean()); and_l2 = float(np.sqrt(np.dot(android_logits.astype(np.float64), android_logits.astype(np.float64))))

    return {
        "ref_argmax":  ref_argmax,  "android_argmax":  and_argmax,
        "ref_top5":    ref_top5,    "android_top5":    and_top5,
        "ref_top10":   ref_top10,   "android_top10":   and_top10,
        "top1_match":  top1_match,
        "top5_match":  top5_match,
        "top10_match": top10_match,
        "ref_min":     ref_min,     "android_min":     and_min,
        "ref_max":     ref_max,     "android_max":     and_max,
        "ref_mean":    ref_mean,    "android_mean":    and_mean,
        "ref_l2":      ref_l2,      "android_l2":      and_l2,
        "ref_sha256":  ref_logits_sha,
        "android_sha256": android_sha,
        "sha_match":   (ref_logits_sha == android_sha),
        "delta_min":   abs(ref_min  - and_min),
        "delta_max":   abs(ref_max  - and_max),
        "delta_mean":  abs(ref_mean - and_mean),
        "delta_l2":    abs(ref_l2   - and_l2),
    }

def parse_diag_bin(diag_path: str) -> list:
    """Parse fix12_diag.bin.
    Record format (72 bytes):
      uint32 ckpt_id, uint32 prompt_idx, uint32 dim, uint32 pad  (16 bytes)
      float  min, max, mean, mean_abs, max_abs, l2               (24 bytes)
      float  proxy[8]                                             (32 bytes)
    Total = 72 bytes
    """
    records = []
    if not os.path.isfile(diag_path):
        print(f"  [WARN] Diag file not found: {diag_path}")
        return records
    sz = os.path.getsize(diag_path)
    n  = sz // 72
    print(f"  [DIAG] {diag_path}: {sz} bytes, {n} records")
    with open(diag_path, "rb") as f:
        for _ in range(n):
            raw = f.read(72)
            if len(raw) < 72:
                break
            ckpt_id, prompt_idx, dim, _pad = struct.unpack_from("<IIII", raw, 0)
            mn, mx, mean, mean_abs, max_abs, l2 = struct.unpack_from("<ffffff", raw, 16)
            proxy = list(struct.unpack_from("<8f", raw, 40))
            records.append({
                "ckpt_id":    ckpt_id,
                "prompt_idx": prompt_idx,
                "dim":        dim,
                "min":        mn,
                "max":        mx,
                "mean":       mean,
                "mean_abs":   mean_abs,
                "max_abs":    max_abs,
                "l2_norm":    l2,
                "proxy":      proxy,
            })
    return records

CKPT_NAMES = {
    1: "CKPT1_EMBED",
    2: "CKPT2_STATE0",
    3: "CKPT3_GQA2",
    4: "CKPT4_STATE3",
    5: "CKPT5_GQA5",
    6: "CKPT6_STATE12",
    7: "CKPT7_FINAL_BLOCK",
    8: "CKPT8_RMSNORM",
    9: "CKPT9_LOGITS",
}

def compare_ckpt_stats(ref: dict, android: dict, name: str) -> dict:
    """Compare scalar stats between reference JSON and android diag record."""
    delta_min  = abs(ref.get("min",  0) - android.get("min",  0))
    delta_max  = abs(ref.get("max",  0) - android.get("max",  0))
    delta_mean = abs(ref.get("mean", 0) - android.get("mean", 0))
    delta_l2   = abs(ref.get("l2_norm", 0) - android.get("l2_norm", 0))
    rel_l2     = delta_l2 / max(abs(ref.get("l2_norm", 1)), 1e-9)

    # Heuristic pass/fail: delta_mean < 10% of ref max_abs, cosine ≈ 1
    ref_ma = ref.get("max_abs", 1.0)
    status = "CLOSE" if (delta_mean < 0.1 * max(ref_ma, 1e-6)) else "DIVERGED"

    return {
        "name":       name,
        "ref_min":    ref.get("min"),    "android_min":  android.get("min"),
        "ref_max":    ref.get("max"),    "android_max":  android.get("max"),
        "ref_mean":   ref.get("mean"),   "android_mean": android.get("mean"),
        "ref_l2":     ref.get("l2_norm"),"android_l2":   android.get("l2_norm"),
        "delta_min":  delta_min,
        "delta_max":  delta_max,
        "delta_mean": delta_mean,
        "delta_l2":   delta_l2,
        "rel_l2_err": rel_l2,
        "status":     status,
    }

def main():
    print("=" * 70)
    print("FIX-12 PHASE G — REFERENCE-B vs ANDROID NATIVE COMPARISON")
    print("=" * 70)

    # ── Load reference JSON ───────────────────────────────────────────────────
    if not os.path.isfile(REF_JSON):
        print(f"ERROR: Phase C/D results not found: {REF_JSON}")
        sys.exit(1)
    with open(REF_JSON, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    print(f"\n[REF] Loaded: {REF_JSON}")
    print(f"      nano_sha_ok={ref_data.get('nano_sha_ok')} tensor_count={ref_data.get('tensor_count')}")

    # ── Check device capture directory ────────────────────────────────────────
    if not os.path.isdir(DEVICE_DIR):
        os.makedirs(DEVICE_DIR, exist_ok=True)
        print(f"\n[WARN] Device capture directory created: {DEVICE_DIR}")
        print("       Please pull device files with:")
        print(f"       adb pull /data/data/<pkg>/files/fix12_diag.bin {DEVICE_DIR}/")
        print(f"       adb pull /data/data/<pkg>/files/fix12_logits_p0.bin {DEVICE_DIR}/")
        print(f"       ... for p0 through p4")
        print(f"       adb pull /data/data/<pkg>/files/fix12_perf.txt {DEVICE_DIR}/")
        print("\n       Then re-run this script.")
        sys.exit(0)

    # ── Parse android diag records ────────────────────────────────────────────
    diag_path = os.path.join(DEVICE_DIR, "fix12_diag.bin")
    android_records = parse_diag_bin(diag_path)
    print(f"\n[DIAG] Parsed {len(android_records)} checkpoint records from Android")

    # ── Compare per prompt ────────────────────────────────────────────────────
    all_results = []

    for prompt_data in ref_data.get("prompts", []):
        label = prompt_data["label"]
        prompt_idx = PROMPTS_ORDER.index(label) if label in PROMPTS_ORDER else -1
        ref_ckpts = prompt_data.get("checkpoints", {})

        print(f"\n{'─'*60}")
        print(f"[{label}] prompt_idx={prompt_idx}")

        prompt_results = {"label": label, "prompt_idx": prompt_idx, "checkpoints": [], "logits": None}

        # ── Checkpoint stats comparison ───────────────────────────────────────
        android_prompt_recs = [r for r in android_records if r["prompt_idx"] == prompt_idx]
        android_by_ckpt = {r["ckpt_id"]: r for r in android_prompt_recs}

        for ckpt_id, ckpt_name in sorted(CKPT_NAMES.items()):
            if ckpt_id == 9:
                continue  # Logits compared separately below
            ref_ckpt = ref_ckpts.get(ckpt_name)
            and_ckpt = android_by_ckpt.get(ckpt_id)
            if not ref_ckpt:
                print(f"  [{ckpt_name}] REF MISSING")
                continue
            if not and_ckpt:
                print(f"  [{ckpt_name}] ANDROID MISSING (diagnostic file not pulled?)")
                prompt_results["checkpoints"].append({"name": ckpt_name, "status": "ANDROID_MISSING"})
                continue
            cmp = compare_ckpt_stats(ref_ckpt, and_ckpt, ckpt_name)
            prompt_results["checkpoints"].append(cmp)
            print(f"  [{ckpt_name}]")
            print(f"    ref_mean={cmp['ref_mean']:.5f}  android_mean={cmp['android_mean']:.5f}  delta_mean={cmp['delta_mean']:.6f}")
            print(f"    ref_l2  ={cmp['ref_l2']:.4f}  android_l2  ={cmp['android_l2']:.4f}    rel_l2_err={cmp['rel_l2_err']:.6f}")
            print(f"    STATUS  ={cmp['status']}")

        # ── Logits comparison ─────────────────────────────────────────────────
        logit_bin = os.path.join(DEVICE_DIR, f"fix12_logits_p{prompt_idx}.bin")
        if not os.path.isfile(logit_bin):
            print(f"  [LOGITS] File not found: {logit_bin} — skipping")
            prompt_results["logits"] = {"status": "FILE_MISSING"}
        else:
            android_logits = np.frombuffer(open(logit_bin, "rb").read(), dtype=np.float32).copy()
            if len(android_logits) != 65536:
                print(f"  [LOGITS] Wrong size: {len(android_logits)} (expected 65536)")
                prompt_results["logits"] = {"status": f"SIZE_ERROR:{len(android_logits)}"}
            else:
                ref_ckpt9 = ref_ckpts.get("CKPT9_LOGITS", {})
                logit_cmp = compare_logits(
                    prompt_data.get("logits_sha256", ""),
                    ref_ckpt9,
                    android_logits
                )
                prompt_results["logits"] = logit_cmp

                print(f"\n  [CKPT9_LOGITS]")
                print(f"    ref_argmax={logit_cmp['ref_argmax']}  android_argmax={logit_cmp['android_argmax']}")
                print(f"    ref_top5  ={logit_cmp['ref_top5']}")
                print(f"    and_top5  ={logit_cmp['android_top5']}")
                print(f"    TOP1_MATCH={logit_cmp['top1_match']}  TOP5_MATCH={logit_cmp['top5_match']}")
                print(f"    SHA_MATCH ={logit_cmp['sha_match']}")
                print(f"    delta_mean={logit_cmp['delta_mean']:.6f}  delta_l2={logit_cmp['delta_l2']:.4f}")

        all_results.append(prompt_results)

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "fix_version": "FIX-12", "phase": "G",
        "ref_json": REF_JSON,
        "device_capture_dir": DEVICE_DIR,
        "android_records_total": len(android_records),
        "prompts": all_results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE G SUMMARY")
    print(f"{'='*70}")
    for pr in all_results:
        lbl = pr["label"]
        logit = pr.get("logits", {})
        print(f"\n[{lbl}]")
        print(f"  TOP1_MATCH={logit.get('top1_match', 'N/A')}  TOP5_MATCH={logit.get('top5_match', 'N/A')}")
        print(f"  SHA_MATCH ={logit.get('sha_match', 'N/A')}")
        for ck in pr.get("checkpoints", []):
            s = ck.get("status", "?")
            print(f"  {ck.get('name', '?')}: {s} delta_mean={ck.get('delta_mean', 'N/A')}")

    print(f"\nPHASE G OUTPUT: {OUTPUT_JSON}")
    print("=" * 70)

if __name__ == "__main__":
    main()
