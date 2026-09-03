# FIX-06 — REAL CHECKPOINT → REAL model.nano EXPORT FORENSIC REPORT

**FIX ID:** `FIX-06-REAL-CHECKPOINT-TO-NANO-EXPORT`  
**Target Module:** `ss_bangladesh_nano_android_module / THSA-2B V1`  
**Date:** September 2, 2026  
**Status:** **BLOCKED — REAL TRAINED CHECKPOINT REQUIRED**  

---

## 1. Executive Verdict

A comprehensive, clean-room filesystem and tensor inspection was conducted across the entire repository to locate the real trained PyTorch checkpoint for **THSA-2B V1** (`d_model=2560`, `d_ffn=6912`, `total_blocks=24`, `vocab_size=65536`).

### Critical Finding:
1. **No 2.0-Billion Parameter Checkpoint Exists in the Workspace:**
   The workspace contains a 350M proxy student checkpoint (`thsa_distilled_student.pt`, `d_model=1024`, `total_blocks=14`, $1.18\text{ GB}$), two 70M dense LLaMA checkpoints (`model.safetensors`, $286\text{ MB}$ / $217\text{ MB}$), and an invalid 24KB stub (`thsa_trained_model.pt`). **The genuine 2.0B production checkpoint (`d_model=2560`, 24 layers) physically does not exist in the repository.**
2. **Exporter Repaired & Synthetic Fallbacks Eliminated:**
   `export_to_nano.py` has been completely repaired. All synthetic generation routines (`bytes(sz)`, `i % 127`, dummy tensor filling) were excised. The exporter now strictly requires a valid checkpoint matching the target architecture and fails hard with non-zero exit codes on any missing tensor or shape mismatch.
3. **Tokenizer Runtime Cleaned:**
   The placeholder string formatting (`[tok_%d]`) has been eliminated from `nano_engine.cpp` and `nano_engine_jni.cpp`. `bpe_trie_runtime.cpp` has been updated to dynamically parse the full 65,536-entry vocabulary table.
4. **Hard Stop Under Rule 24:**
   In accordance with Mandatory Section 24 (*Critical Checkpoint Absence Rule*), we do not fabricate, tile, interpolate, or synthetic-fill weights. FIX-06 is declared **BLOCKED** until the real trained 2.0B PyTorch checkpoint is provided.

```
====================================================================================================
                                      FIX-06 FINAL VERDICT
====================================================================================================
  CHECKPOINT / EXPORT GATE                               STATUS      EVIDENCE
----------------------------------------------------------------------------------------------------
  1. Full Workspace Checkpoint Search                    COMPLETE    8 candidate files evaluated
  2. Candidate Checkpoint Loading & Inspection           COMPLETE    1 corrupt, 2 dense 70M, 1 proxy 350M
  3. Architecture Match vs THSA-2B Specification         MISMATCH    d_model=1024 vs required 2560
  4. Exporter Repair (Zero-Fill & Modulo Removal)        PASS        Synthetic fallbacks excised
  5. Tokenizer [tok_%d] String Fallback Elimination       PASS        Removed from native decode path
  6. Real Vocabulary Loading Support (65,536 tokens)     PASS        bpe_trie_runtime.cpp updated
  7. Production model.nano Generation                    BLOCKED     Awaiting real 2B checkpoint
====================================================================================================
  OVERALL VERDICT:                                        BLOCKED — REAL TRAINED CHECKPOINT REQUIRED
====================================================================================================
```

---

## 2. Comprehensive Candidate Checkpoint Inventory

A thorough search across all subdirectories was executed:

| Candidate File Path | Size (Bytes) | SHA-256 | Format / Framework | Evaluation & Rejection Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `checkpoints/thsa_trained_model.pt` | $24,576$ | `f1d5a8d630e6...` | Corrupt Stub | **REJECTED:** Incomplete 24KB binary fragment; `torch.load` throws `[Errno 22] Invalid argument`. |
| `models/core/ss_bangladesh/model/model.safetensors` | $286,123,304$ | `bb2f9e7cd79e...` | SafeTensors (70M) | **REJECTED:** Dense LLaMA architecture (`d_model=576`, `vocab=16000`, 93 tensors). Not THSA ternary hybrid. |
| `models/sstutor_bengali_70m_edu/model.safetensors` | $217,339,624$ | `a44215f9bc3e...` | SafeTensors (70M) | **REJECTED:** Dense LLaMA architecture (`d_model=576`, `vocab=1073`, 93 tensors). Not THSA ternary hybrid. |
| `ss_bangladesh/model/model.safetensors` | $286,123,304$ | `bb2f9e7cd79e...` | SafeTensors (70M) | **REJECTED:** Identical to core model above (`bb2f9e7cd79e...`). |
| `ss_bangladesh_nano_android_module/THSA-2B V1/training/checkpoints/thsa_distilled_student.pt` | $1,180,513,254$ | `d6ad65b87dda...` | PyTorch Dict (129 keys) | **REJECTED FOR 2B TARGET:** Valid trained weights, but architecture is `THSA-350M-PROXY-PILOT` (`d_model=1024`, 14 layers). Mismatches production 2.0B target (`d_model=2560`, 24 layers). |

---

## 3. Architecture Mismatch Analysis (350M Proxy vs 2.0B Target)

| Architectural Parameter | Target THSA-2B Specification | Available Student Checkpoint (`thsa_distilled_student.pt`) | Status |
| :--- | :--- | :--- | :--- |
| **Model ID** | `THSA-2B-V1-PRODUCTION` | `THSA-350M-PROXY-PILOT` | **MISMATCH** |
| **d_model (Hidden Dimension)** | **2560** | **1024** | **MISMATCH** |
| **d_ffn (FFN Intermediate)** | **6912** | **2764** | **MISMATCH** |
| **Total Blocks (Layers)** | **24** | **14** | **MISMATCH** |
| **State Blocks (1D Conv)** | **16** | **10** | **MISMATCH** |
| **GQA Blocks (Attention)** | **8** | **4** | **MISMATCH** |
| **Query Heads ($n_q$)** | **20** | **8** | **MISMATCH** |
| **KV Heads ($n_{kv}$)** | **4** | **2** | **MISMATCH** |
| **Head Dimension ($d_{head}$)** | 128 | 128 | MATCH |
| **Vocabulary Size** | 65,536 | 65,536 | MATCH |
| **Total Tensors** | **123** | **129** | **MISMATCH** |

Under Section 4 and Section 24, weights must never be tiled, truncated, or interpolated to artificially force dimensions. The exporter requires exact parameter alignment.

---

## 4. Exporter Root Cause & Permanent Repairs

### Historical Root Cause:
In commit `c3d0247`, `export_to_nano.py` was created with fallback clauses intended for dry-run testing. When executed without a 2.0B checkpoint, it wrote:
- `embed_tokens`: `bytes([i % 127 for i in range(...)])`
- `layer_*_attn_*`: `bytes(sz)` (100% all-zero bytes)
- `layer_*_ffn_*`: `bytes(sz)` (100% all-zero bytes)
- `lm_head`: `bytes([i % 127 for i in range(...)])`

This created the synthetic 686MB scaffold binary (`638d51bd...`) analyzed in FIX-05B.

### Changes Made in [`export_to_nano.py`](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B%20V1/tools/export_to_nano.py):
1. **Mandatory Checkpoint Requirement:** `checkpoint_path` is now a required CLI argument. Missing paths throw fatal `FileNotFoundError`.
2. **Zero-Tolerance Tensor Validation:** If any of the 123 tensors are missing from the checkpoint, `KeyError` is thrown immediately.
3. **Strict Dimension Verification:** Tensor shapes are validated against configuration dimensions (`[vocab_size, d_model]`, `[out_dim, in_dim]`, etc.). Dimension mismatches throw `ValueError`.
4. **Excised All Fake Data:** Zero bytes, modulo patterns, and synthetic fill code have been deleted.

---

## 5. Tokenizer Remediation

1. **Eliminated Placeholder Text Fallback:**
   Removed `snprintf(..., "[tok_%d]", ...)` from:
   - [`nano_engine.cpp:819`](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B%20V1/src/engine/nano_engine.cpp#L815-L822)
   - [`nano_engine_jni.cpp:147`](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B%20V1/jni/nano_engine_jni.cpp#L143-L150)
2. **Standardized Token ID Offsets in [`bpe_trie_runtime.cpp`](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/THSA-2B%20V1/src/tokenizer/bpe_trie_runtime.cpp#L75-L125):**
   - Special Tokens: $0 = \text{<pad>}$, $1 = \text{<unk>}$, $2 = \text{<s>}$, $3 = \text{</s>}$.
   - Byte Fallbacks: $4..259$ represent bytes `0x00` through `0xFF` (where byte $b = 4 + b$).
   - Full Vocabulary Loading: Added disk parsing for `thsa_tokenizer.vocab` ($65,536$ tokens).

---

## 6. Generated JSON Artifacts

The following JSON artifacts have been created in `artifacts/`:

1. [**`artifacts/fix06_checkpoint_manifest.json`**](file:///c:/Users/User/Desktop/SS_module_BD/artifacts/fix06_checkpoint_manifest.json): Full 129-tensor inventory, statistics, and SHA-256 hashes of `thsa_distilled_student.pt`.
2. [**`artifacts/fix06_nano_manifest.json`**](file:///c:/Users/User/Desktop/SS_module_BD/artifacts/fix06_nano_manifest.json): Header and tensor descriptor inspection of `model.nano`.
3. [**`artifacts/fix06_tensor_mapping.json`**](file:///c:/Users/User/Desktop/SS_module_BD/artifacts/fix06_tensor_mapping.json): Explicit architectural comparison between target 2B and checkpoint 350M models.
4. [**`artifacts/fix06_quantization_report.json`**](file:///c:/Users/User/Desktop/SS_module_BD/artifacts/fix06_quantization_report.json): Mathematical definitions and sample quantization error metrics.
5. [**`artifacts/fix06_tokenizer_report.json`**](file:///c:/Users/User/Desktop/SS_module_BD/artifacts/fix06_tokenizer_report.json): Complete breakdown of the 65,536-entry SentencePiece vocabulary.
6. [**`artifacts/fix06_numerical_sensitivity.json`**](file:///c:/Users/User/Desktop/SS_module_BD/artifacts/fix06_numerical_sensitivity.json): Testbed status documentation and blocking criteria.

---

## 7. Required Artifact for Unblocking

To unblock FIX-06 and proceed with model export and causal inference testing:

- **Required File:** `thsa_2b_production_checkpoint.pt` (or `.safetensors`)
- **Required Architecture:**
  - Hidden Dimension (`d_model`): **2560**
  - FFN Dimension (`d_ffn`): **6912**
  - Layer Count: **24 blocks** (16 State blocks + 8 GQA blocks)
  - Attention Heads: **20 Query heads, 4 KV heads** (Head Dim = 128)
  - Vocabulary Size: **65,536**
