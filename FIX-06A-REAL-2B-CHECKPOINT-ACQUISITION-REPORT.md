# FIX-06A — REAL 2B CHECKPOINT ACQUISITION & PROVENANCE FORENSIC REPORT

**FIX ID:** `FIX-06A-REAL-2B-CHECKPOINT-ACQUISITION`  
**Target Repository:** `ss_bangladesh_nano_android_module / THSA-2B V1`  
**Mirror Repository:** `ss_bangladesh_nano_android_module / THSA-2B_V2_helper`  
**Date:** September 2, 2026  
**Status:** **BLOCKED — REAL TRAINED THSA-2B CHECKPOINT NOT FOUND**  

---

## 1. Executive Verdict & Core Finding

A rigorous, clean-room forensic audit of the entire accessible workspace (`c:\Users\User\Desktop\SS_module_BD`), Git object database, training pipelines, and local directories was conducted to determine whether a genuine, trained **THSA-2B** ($d_{model}=2560$, $24\text{ layers}$) checkpoint exists.

### Primary Forensic Conclusion:
**No genuine 2.0-billion parameter (`d_model=2560`, `total_blocks=24`) trained PyTorch checkpoint physically exists anywhere in the accessible environment.**

1. The workspace contains a genuine trained **350M proxy student checkpoint** (`thsa_distilled_student.pt`, $1.18\text{ GB}$), but its architecture is strictly $d_{model}=1024$ with $14\text{ layers}$.
2. Under **Mandatory Section 0 and Section 18**, we do not resize, tile, interpolate, pad, or synthesize weights to manufacture a fake 2B model.
3. Under **Mandatory Section 1 and Section 17**, because no genuine compatible 2B checkpoint can be acquired, the project is declared **BLOCKED** at this acquisition gate.

```
====================================================================================================
                                      FIX-06A FINAL VERDICT
====================================================================================================
  ACQUISITION GATE CHECK                                 STATUS      EVIDENCE
----------------------------------------------------------------------------------------------------
  1. Full Workspace Filesystem Search                    COMPLETE    All weight/archive files scanned
  2. Git History & Object Database Scan                  COMPLETE    Zero 2B weight commits in Git history
  3. Training Pipeline & Configuration Audit             COMPLETE    Pipeline targets 350M proxy model
  4. Candidate Checkpoint Integrity Evaluation           COMPLETE    1 corrupt, 2 dense 70M, 1 proxy 350M
  5. 2.0B Architecture Compatibility Check               FAIL        d_model=1024 vs required 2560
  6. Prevention of Fake/Synthetic Weight Generation      PASS        Zero fake weights or tiling applied
====================================================================================================
  OVERALL VERDICT:                                        BLOCKED — REAL TRAINED THSA-2B CHECKPOINT NOT FOUND
====================================================================================================
```

---

## 2. Answers to Mandatory Forensic Questions (Section 16)

| # | Forensic Question | Definitive Answer & Evidence |
| :--- | :--- | :--- |
| **1** | **Does a genuine trained THSA-2B 2B checkpoint exist anywhere accessible?** | **NO.** No 2.0B trained checkpoint exists in the workspace or git history. |
| **2** | **If yes, exact path?** | N/A (None exists). |
| **3** | **Exact filename?** | N/A (None exists). |
| **4** | **Exact size?** | N/A (None exists). |
| **5** | **Exact SHA-256?** | N/A (None exists). |
| **6** | **Exact framework / format?** | Target specification requires PyTorch `model_state_dict` dictionary (`.pt`) or SafeTensors (`.safetensors`). |
| **7** | **Exact parameter count?** | Required: **~2,000,000,000 parameters** ($2.0\text{B}$). Available proxy: **~350,000,000 parameters** ($350\text{M}$). |
| **8** | **Does it contain the required 24-layer / d_model=2560 architecture?** | **NO.** The only trained hybrid checkpoint has 14 layers and $d_{model}=1024$. |
| **9** | **Does it contain the required 65536 vocabulary?** | **YES.** Both the proxy checkpoint and `thsa_tokenizer.vocab` have $V=65,536$. |
| **10** | **Is there evidence that a 2B model was actually trained?** | **NO.** Training logs and scripts in `training/` only document training the 350M proxy student. |
| **11** | **Is its provenance traceable?** | The 350M proxy student provenance is traceable to `train_qat.py` (distilled from `sarvam-1` & `Qwen2.5-7B-Instruct`). No 2B run is recorded. |
| **12** | **Is it compatible with the repaired `export_to_nano.py`?** | **NO.** Attempting to export the 350M checkpoint against `thsa_2b_config.json` raises a fatal `ValueError` on shape mismatch. |
| **13** | **Can it safely become the source for the next production `model.nano` export?** | **NO.** A genuine 2.0B checkpoint must be supplied to generate a valid production `model.nano`. |

---

## 3. Discovered Checkpoint Candidates & Rejection Analysis

| Candidate File Path | Size | SHA-256 | Architecture Found | Rejection Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `checkpoints/thsa_trained_model.pt` | $24\text{ KB}$ | `f1d5a8d630e6...` | Unknown | **Corrupt Stub:** File truncated; unreadable by PyTorch. |
| `models/core/ss_bangladesh/model/model.safetensors` | $286\text{ MB}$ | `bb2f9e7cd79e...` | Dense LLaMA 70M ($d_{model}=576$, $V=16000$) | **Architecture Mismatch:** 70M dense model; not THSA hybrid. |
| `models/sstutor_bengali_70m_edu/model.safetensors` | $217\text{ MB}$ | `a44215f9bc3e...` | Dense LLaMA 70M ($d_{model}=576$, $V=1073$) | **Architecture Mismatch:** 70M dense model, small vocabulary. |
| `THSA-2B V1/training/checkpoints/thsa_distilled_student.pt` | $1.18\text{ GB}$ | `d6ad65b87dda...` | THSA-350M Proxy ($d_{model}=1024$, $14\text{ layers}$) | **Architecture Mismatch:** Genuine trained hybrid model, but 350M proxy student rather than 2.0B production target. |
| `THSA-2B V1/android/src/main/assets/model_trained.nano` | $166.7\text{ MB}$ | `966a2ecf3ca4...` | THSA-350M Binary ($71\text{ tensors}$) | **Architecture Mismatch:** Export of 350M proxy student. |
| `THSA-2B V1/models/model.nano` | $686.1\text{ MB}$ | `638d51bd6813...` | Synthetic Scaffold ($123\text{ tensors}$) | **Synthetic Invalid:** 121 zero tensors; invalid placeholder. |

---

## 4. Parameter Count Comparison by Tensor Family

| Component Family | Target THSA-2B Production Parameters | Available 350M Proxy Student Parameters |
| :--- | :--- | :--- |
| **Token Embeddings** ($V \times d_{model}$) | $65,536 \times 2560 = \mathbf{167,772,160}$ | $65,536 \times 1024 = 67,108,864$ |
| **Attention Projections (Q, K, V, Out)** | $8 \text{ layers} \times 4 \text{ projs} \approx \mathbf{209,715,200}$ | $4 \text{ layers} \times 4 \text{ projs} \approx 16,777,216$ |
| **1D State Convolutions** | $16 \text{ layers} \times (4 \times 2560) = \mathbf{163,840}$ | $10 \text{ layers} \times (4 \times 1024) = 40,960$ |
| **FFN Projections (Gate, Up, Down)** | $24 \text{ layers} \times (3 \times 2560 \times 6912) \approx \mathbf{1,273,758,720}$ | $14 \text{ layers} \times (3 \times 1024 \times 2764) \approx 119,007,744$ |
| **RMSNorm Gammas** | $25 \times 2560 = \mathbf{64,000}$ | $15 \times 1024 = 15,360$ |
| **LM Head** ($V \times d_{model}$) | $65,536 \times 2560 = \mathbf{167,772,160}$ | $65,536 \times 1024 = 67,108,864$ |
| **Total Estimated Parameters** | $\mathbf{\approx 1,819,246,080 \ (\sim 2.0B)}$ | $\approx 270,058,944 \ (\sim 350M)$ |

---

## 5. Required Output Artifacts Generated

1. [**`FIX-06A-CHECKPOINT-INVENTORY.json`**](file:///c:/Users/User/Desktop/SS_module_BD/FIX-06A-CHECKPOINT-INVENTORY.json): Full metadata of all 6 discovered weight candidates.
2. [**`FIX-06A-CHECKPOINT-PROVENANCE-MATRIX.json`**](file:///c:/Users/User/Desktop/SS_module_BD/FIX-06A-CHECKPOINT-PROVENANCE-MATRIX.json): Provenance and rejection reasons for each candidate.
3. [**`FIX-06A-GIT-HISTORY-FORENSIC.md`**](file:///c:/Users/User/Desktop/SS_module_BD/FIX-06A-GIT-HISTORY-FORENSIC.md): Comprehensive analysis of all 1,321 Git objects and commit timeline.
4. [**`FIX-06A-TRAINING-PIPELINE-FORENSIC.md`**](file:///c:/Users/User/Desktop/SS_module_BD/FIX-06A-TRAINING-PIPELINE-FORENSIC.md): Detailed audit of `train_qat.py` and distillation mechanics.
5. [**`FIX-06A-REAL-2B-CHECKPOINT-ACQUISITION-REPORT.md`**](file:///c:/Users/User/Desktop/SS_module_BD/FIX-06A-REAL-2B-CHECKPOINT-ACQUISITION-REPORT.md): This master forensic report.

---

## 6. Actionable Next Steps to Unblock FIX-06 & FIX-07

The project cannot proceed with production 2B `.nano` binary generation until the real 2.0B trained checkpoint is supplied:

1. **Provide Genuine 2.0B Checkpoint:**
   Place the trained PyTorch checkpoint (`thsa_2b_production.pt` or `.safetensors`) into `training/checkpoints/` with dimensions:
   - $d_{model} = 2560$, $d_{ffn} = 6912$, $24\text{ layers}$, $V = 65536$.
2. **Execute Strict Exporter:**
   Run `export_to_nano.py --config training/config/thsa_2b_config.json --checkpoint <path> --output models/model.nano`.
3. **Execute Clean-Room Verification:**
   Run the independent numerical verification suite on physical hardware.
