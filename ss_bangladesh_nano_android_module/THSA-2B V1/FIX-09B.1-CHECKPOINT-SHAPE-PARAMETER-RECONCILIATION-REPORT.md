# FIX-09B.1: CHECKPOINT SHAPE & PARAMETER RECONCILIATION REPORT
## FORENSIC CHECKPOINT-LEVEL SHAPE / PARAMETER AUDIT

## 1. Executive Summary
- **FIX ID**: `FIX-09B.1-CHECKPOINT-SHAPE-PARAMETER-RECONCILIATION`
- **Authoritative Baseline**: Step-30 continuation checkpoint `checkpoint_step_000030.pt`
- **Expected SHA-256**: `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`
- **Expected File Size**: `4,106,953,961` bytes
- **Total Tensors Verified**: `219` (100% Bijection Match)
- **Total Parameters Verified**: `2,050,296,320` (Exact Match)
- **Primary Discrepancy Resolved**:
  - `layers.X.mixer.in_proj.weight` shape is conclusively proven to be **`[5120, 2560]`** (13,107,200 parameters per State layer, 209,715,200 parameters total across 16 State layers).
  - The PyTorch architecture (`ShortConvStateBlock`), the trained checkpoint weights, the exporter (`tools/export_to_nano.py`), and the native C++ engine (`src/engine/nano_engine.cpp`) are **100% mutually consistent** and all utilize `[5120, 2560]`.
  - The reference in the FIX-09B report prose mentioning `16 * (2560 * 2560) = 104,857,600` was an isolated typographical transcription error in the report markdown narrative.
- **Final Status**: `FIX-09B.1-PASS-CHECKPOINT-RECONCILED`

---

## 2. Checkpoint Identity & Immutability Verification
- **Authoritative Checkpoint**: `checkpoint_step_000030.pt`
- **SHA-256 Before Audit**: `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`
- **SHA-256 After Audit**:  `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`
- **File Size Before**: `4,106,953,961` bytes
- **File Size After**:  `4,106,953,961` bytes
- **Cryptographic Immutability**: **PASS** (Zero bytes modified, strictly read-only access).

---

## 3. 219-Key Bijection Audit
- **State Dict Key Count**: `219`
- **Expected Key Count**: `219`
- **Missing Keys**: `0`
- **Extra Keys**: `0`
- **Key-Set Bijection Verdict**: **PASS** (1-to-1 exact bijection with PyTorch `named_parameters()`).

---

## 4. Complete 219-Tensor Shape & Parameter Table

| Idx | State Dict Key | Layer | Block Type | Tensor Role | Actual Shape | Numel | Dtype | Shape Match | Numel Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `embed_tokens.weight` | ROOT | root | token_embeddings | `[65536, 2560]` | 167,772,160 | bfloat16 | PASS | PASS |
| 1 | `layers.0.mixer.conv1d.weight` | 0 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 2 | `layers.0.mixer.conv1d.bias` | 0 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 3 | `layers.0.mixer.in_proj.weight` | 0 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 4 | `layers.0.mixer.out_proj.weight` | 0 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 5 | `layers.0.mixer.norm.weight` | 0 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 6 | `layers.0.ffn.gate_proj.weight` | 0 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 7 | `layers.0.ffn.up_proj.weight` | 0 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 8 | `layers.0.ffn.down_proj.weight` | 0 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 9 | `layers.0.ffn.norm.weight` | 0 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 10 | `layers.1.mixer.conv1d.weight` | 1 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 11 | `layers.1.mixer.conv1d.bias` | 1 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 12 | `layers.1.mixer.in_proj.weight` | 1 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 13 | `layers.1.mixer.out_proj.weight` | 1 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 14 | `layers.1.mixer.norm.weight` | 1 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 15 | `layers.1.ffn.gate_proj.weight` | 1 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 16 | `layers.1.ffn.up_proj.weight` | 1 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 17 | `layers.1.ffn.down_proj.weight` | 1 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 18 | `layers.1.ffn.norm.weight` | 1 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 19 | `layers.2.mixer.q_proj.weight` | 2 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 20 | `layers.2.mixer.k_proj.weight` | 2 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 21 | `layers.2.mixer.v_proj.weight` | 2 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 22 | `layers.2.mixer.out_proj.weight` | 2 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 23 | `layers.2.mixer.norm.weight` | 2 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 24 | `layers.2.ffn.gate_proj.weight` | 2 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 25 | `layers.2.ffn.up_proj.weight` | 2 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 26 | `layers.2.ffn.down_proj.weight` | 2 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 27 | `layers.2.ffn.norm.weight` | 2 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 28 | `layers.3.mixer.conv1d.weight` | 3 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 29 | `layers.3.mixer.conv1d.bias` | 3 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 30 | `layers.3.mixer.in_proj.weight` | 3 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 31 | `layers.3.mixer.out_proj.weight` | 3 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 32 | `layers.3.mixer.norm.weight` | 3 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 33 | `layers.3.ffn.gate_proj.weight` | 3 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 34 | `layers.3.ffn.up_proj.weight` | 3 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 35 | `layers.3.ffn.down_proj.weight` | 3 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 36 | `layers.3.ffn.norm.weight` | 3 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 37 | `layers.4.mixer.conv1d.weight` | 4 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 38 | `layers.4.mixer.conv1d.bias` | 4 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 39 | `layers.4.mixer.in_proj.weight` | 4 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 40 | `layers.4.mixer.out_proj.weight` | 4 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 41 | `layers.4.mixer.norm.weight` | 4 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 42 | `layers.4.ffn.gate_proj.weight` | 4 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 43 | `layers.4.ffn.up_proj.weight` | 4 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 44 | `layers.4.ffn.down_proj.weight` | 4 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 45 | `layers.4.ffn.norm.weight` | 4 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 46 | `layers.5.mixer.q_proj.weight` | 5 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 47 | `layers.5.mixer.k_proj.weight` | 5 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 48 | `layers.5.mixer.v_proj.weight` | 5 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 49 | `layers.5.mixer.out_proj.weight` | 5 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 50 | `layers.5.mixer.norm.weight` | 5 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 51 | `layers.5.ffn.gate_proj.weight` | 5 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 52 | `layers.5.ffn.up_proj.weight` | 5 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 53 | `layers.5.ffn.down_proj.weight` | 5 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 54 | `layers.5.ffn.norm.weight` | 5 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 55 | `layers.6.mixer.conv1d.weight` | 6 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 56 | `layers.6.mixer.conv1d.bias` | 6 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 57 | `layers.6.mixer.in_proj.weight` | 6 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 58 | `layers.6.mixer.out_proj.weight` | 6 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 59 | `layers.6.mixer.norm.weight` | 6 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 60 | `layers.6.ffn.gate_proj.weight` | 6 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 61 | `layers.6.ffn.up_proj.weight` | 6 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 62 | `layers.6.ffn.down_proj.weight` | 6 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 63 | `layers.6.ffn.norm.weight` | 6 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 64 | `layers.7.mixer.conv1d.weight` | 7 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 65 | `layers.7.mixer.conv1d.bias` | 7 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 66 | `layers.7.mixer.in_proj.weight` | 7 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 67 | `layers.7.mixer.out_proj.weight` | 7 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 68 | `layers.7.mixer.norm.weight` | 7 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 69 | `layers.7.ffn.gate_proj.weight` | 7 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 70 | `layers.7.ffn.up_proj.weight` | 7 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 71 | `layers.7.ffn.down_proj.weight` | 7 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 72 | `layers.7.ffn.norm.weight` | 7 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 73 | `layers.8.mixer.q_proj.weight` | 8 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 74 | `layers.8.mixer.k_proj.weight` | 8 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 75 | `layers.8.mixer.v_proj.weight` | 8 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 76 | `layers.8.mixer.out_proj.weight` | 8 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 77 | `layers.8.mixer.norm.weight` | 8 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 78 | `layers.8.ffn.gate_proj.weight` | 8 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 79 | `layers.8.ffn.up_proj.weight` | 8 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 80 | `layers.8.ffn.down_proj.weight` | 8 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 81 | `layers.8.ffn.norm.weight` | 8 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 82 | `layers.9.mixer.conv1d.weight` | 9 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 83 | `layers.9.mixer.conv1d.bias` | 9 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 84 | `layers.9.mixer.in_proj.weight` | 9 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 85 | `layers.9.mixer.out_proj.weight` | 9 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 86 | `layers.9.mixer.norm.weight` | 9 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 87 | `layers.9.ffn.gate_proj.weight` | 9 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 88 | `layers.9.ffn.up_proj.weight` | 9 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 89 | `layers.9.ffn.down_proj.weight` | 9 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 90 | `layers.9.ffn.norm.weight` | 9 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 91 | `layers.10.mixer.conv1d.weight` | 10 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 92 | `layers.10.mixer.conv1d.bias` | 10 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 93 | `layers.10.mixer.in_proj.weight` | 10 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 94 | `layers.10.mixer.out_proj.weight` | 10 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 95 | `layers.10.mixer.norm.weight` | 10 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 96 | `layers.10.ffn.gate_proj.weight` | 10 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 97 | `layers.10.ffn.up_proj.weight` | 10 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 98 | `layers.10.ffn.down_proj.weight` | 10 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 99 | `layers.10.ffn.norm.weight` | 10 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 100 | `layers.11.mixer.q_proj.weight` | 11 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 101 | `layers.11.mixer.k_proj.weight` | 11 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 102 | `layers.11.mixer.v_proj.weight` | 11 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 103 | `layers.11.mixer.out_proj.weight` | 11 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 104 | `layers.11.mixer.norm.weight` | 11 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 105 | `layers.11.ffn.gate_proj.weight` | 11 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 106 | `layers.11.ffn.up_proj.weight` | 11 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 107 | `layers.11.ffn.down_proj.weight` | 11 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 108 | `layers.11.ffn.norm.weight` | 11 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 109 | `layers.12.mixer.conv1d.weight` | 12 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 110 | `layers.12.mixer.conv1d.bias` | 12 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 111 | `layers.12.mixer.in_proj.weight` | 12 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 112 | `layers.12.mixer.out_proj.weight` | 12 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 113 | `layers.12.mixer.norm.weight` | 12 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 114 | `layers.12.ffn.gate_proj.weight` | 12 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 115 | `layers.12.ffn.up_proj.weight` | 12 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 116 | `layers.12.ffn.down_proj.weight` | 12 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 117 | `layers.12.ffn.norm.weight` | 12 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 118 | `layers.13.mixer.conv1d.weight` | 13 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 119 | `layers.13.mixer.conv1d.bias` | 13 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 120 | `layers.13.mixer.in_proj.weight` | 13 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 121 | `layers.13.mixer.out_proj.weight` | 13 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 122 | `layers.13.mixer.norm.weight` | 13 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 123 | `layers.13.ffn.gate_proj.weight` | 13 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 124 | `layers.13.ffn.up_proj.weight` | 13 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 125 | `layers.13.ffn.down_proj.weight` | 13 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 126 | `layers.13.ffn.norm.weight` | 13 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 127 | `layers.14.mixer.q_proj.weight` | 14 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 128 | `layers.14.mixer.k_proj.weight` | 14 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 129 | `layers.14.mixer.v_proj.weight` | 14 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 130 | `layers.14.mixer.out_proj.weight` | 14 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 131 | `layers.14.mixer.norm.weight` | 14 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 132 | `layers.14.ffn.gate_proj.weight` | 14 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 133 | `layers.14.ffn.up_proj.weight` | 14 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 134 | `layers.14.ffn.down_proj.weight` | 14 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 135 | `layers.14.ffn.norm.weight` | 14 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 136 | `layers.15.mixer.conv1d.weight` | 15 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 137 | `layers.15.mixer.conv1d.bias` | 15 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 138 | `layers.15.mixer.in_proj.weight` | 15 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 139 | `layers.15.mixer.out_proj.weight` | 15 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 140 | `layers.15.mixer.norm.weight` | 15 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 141 | `layers.15.ffn.gate_proj.weight` | 15 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 142 | `layers.15.ffn.up_proj.weight` | 15 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 143 | `layers.15.ffn.down_proj.weight` | 15 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 144 | `layers.15.ffn.norm.weight` | 15 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 145 | `layers.16.mixer.conv1d.weight` | 16 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 146 | `layers.16.mixer.conv1d.bias` | 16 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 147 | `layers.16.mixer.in_proj.weight` | 16 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 148 | `layers.16.mixer.out_proj.weight` | 16 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 149 | `layers.16.mixer.norm.weight` | 16 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 150 | `layers.16.ffn.gate_proj.weight` | 16 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 151 | `layers.16.ffn.up_proj.weight` | 16 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 152 | `layers.16.ffn.down_proj.weight` | 16 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 153 | `layers.16.ffn.norm.weight` | 16 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 154 | `layers.17.mixer.q_proj.weight` | 17 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 155 | `layers.17.mixer.k_proj.weight` | 17 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 156 | `layers.17.mixer.v_proj.weight` | 17 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 157 | `layers.17.mixer.out_proj.weight` | 17 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 158 | `layers.17.mixer.norm.weight` | 17 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 159 | `layers.17.ffn.gate_proj.weight` | 17 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 160 | `layers.17.ffn.up_proj.weight` | 17 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 161 | `layers.17.ffn.down_proj.weight` | 17 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 162 | `layers.17.ffn.norm.weight` | 17 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 163 | `layers.18.mixer.conv1d.weight` | 18 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 164 | `layers.18.mixer.conv1d.bias` | 18 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 165 | `layers.18.mixer.in_proj.weight` | 18 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 166 | `layers.18.mixer.out_proj.weight` | 18 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 167 | `layers.18.mixer.norm.weight` | 18 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 168 | `layers.18.ffn.gate_proj.weight` | 18 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 169 | `layers.18.ffn.up_proj.weight` | 18 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 170 | `layers.18.ffn.down_proj.weight` | 18 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 171 | `layers.18.ffn.norm.weight` | 18 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 172 | `layers.19.mixer.conv1d.weight` | 19 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 173 | `layers.19.mixer.conv1d.bias` | 19 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 174 | `layers.19.mixer.in_proj.weight` | 19 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 175 | `layers.19.mixer.out_proj.weight` | 19 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 176 | `layers.19.mixer.norm.weight` | 19 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 177 | `layers.19.ffn.gate_proj.weight` | 19 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 178 | `layers.19.ffn.up_proj.weight` | 19 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 179 | `layers.19.ffn.down_proj.weight` | 19 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 180 | `layers.19.ffn.norm.weight` | 19 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 181 | `layers.20.mixer.q_proj.weight` | 20 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 182 | `layers.20.mixer.k_proj.weight` | 20 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 183 | `layers.20.mixer.v_proj.weight` | 20 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 184 | `layers.20.mixer.out_proj.weight` | 20 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 185 | `layers.20.mixer.norm.weight` | 20 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 186 | `layers.20.ffn.gate_proj.weight` | 20 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 187 | `layers.20.ffn.up_proj.weight` | 20 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 188 | `layers.20.ffn.down_proj.weight` | 20 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 189 | `layers.20.ffn.norm.weight` | 20 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 190 | `layers.21.mixer.conv1d.weight` | 21 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 191 | `layers.21.mixer.conv1d.bias` | 21 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 192 | `layers.21.mixer.in_proj.weight` | 21 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 193 | `layers.21.mixer.out_proj.weight` | 21 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 194 | `layers.21.mixer.norm.weight` | 21 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 195 | `layers.21.ffn.gate_proj.weight` | 21 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 196 | `layers.21.ffn.up_proj.weight` | 21 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 197 | `layers.21.ffn.down_proj.weight` | 21 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 198 | `layers.21.ffn.norm.weight` | 21 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 199 | `layers.22.mixer.conv1d.weight` | 22 | state | state_depthwise_conv_filter | `[2560, 1, 4]` | 10,240 | bfloat16 | PASS | PASS |
| 200 | `layers.22.mixer.conv1d.bias` | 22 | state | state_depthwise_conv_bias | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 201 | `layers.22.mixer.in_proj.weight` | 22 | state | state_in_projection_gate_val | `[5120, 2560]` | 13,107,200 | bfloat16 | PASS | PASS |
| 202 | `layers.22.mixer.out_proj.weight` | 22 | state | state_out_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 203 | `layers.22.mixer.norm.weight` | 22 | state | state_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 204 | `layers.22.ffn.gate_proj.weight` | 22 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 205 | `layers.22.ffn.up_proj.weight` | 22 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 206 | `layers.22.ffn.down_proj.weight` | 22 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 207 | `layers.22.ffn.norm.weight` | 22 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 208 | `layers.23.mixer.q_proj.weight` | 23 | gqa | gqa_query_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 209 | `layers.23.mixer.k_proj.weight` | 23 | gqa | gqa_key_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 210 | `layers.23.mixer.v_proj.weight` | 23 | gqa | gqa_value_projection | `[512, 2560]` | 1,310,720 | bfloat16 | PASS | PASS |
| 211 | `layers.23.mixer.out_proj.weight` | 23 | gqa | gqa_output_projection | `[2560, 2560]` | 6,553,600 | bfloat16 | PASS | PASS |
| 212 | `layers.23.mixer.norm.weight` | 23 | gqa | gqa_mixer_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 213 | `layers.23.ffn.gate_proj.weight` | 23 | ffn | swiglu_gate_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 214 | `layers.23.ffn.up_proj.weight` | 23 | ffn | swiglu_up_projection | `[6912, 2560]` | 17,694,720 | bfloat16 | PASS | PASS |
| 215 | `layers.23.ffn.down_proj.weight` | 23 | ffn | swiglu_down_projection | `[2560, 6912]` | 17,694,720 | bfloat16 | PASS | PASS |
| 216 | `layers.23.ffn.norm.weight` | 23 | ffn | ffn_pre_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 217 | `final_norm.weight` | ROOT | root | final_rmsnorm | `[2560]` | 2,560 | bfloat16 | PASS | PASS |
| 218 | `lm_head.weight` | ROOT | root | causal_lm_head | `[65536, 2560]` | 167,772,160 | bfloat16 | PASS | PASS |

---

## 5. State Block Critical Shape Audit & In-Projection Resolution
For all 16 State layers (`0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22`):
- `layers.X.mixer.norm.weight`: shape = `[2560]`, numel = `2,560` (FP32 RMSNorm)
- `layers.X.mixer.conv1d.weight`: shape = `[2560, 1, 4]`, numel = `10,240` (FP32 Depthwise Causal Conv)
- `layers.X.mixer.conv1d.bias`: shape = `[2560]`, numel = `2,560` (FP32 Conv Bias)
- `layers.X.mixer.in_proj.weight`: shape = **`[5120, 2560]`**, numel = **`13,107,200`** (Ternary 2-bit)
- `layers.X.mixer.out_proj.weight`: shape = `[2560, 2560]`, numel = `6,553,600` (Ternary 2-bit)

### Mathematical Resolution of `[5120, 2560]` vs `[2560, 2560]`:
1. `ShortConvStateBlock` in `training/models/state_conv_block.py` defines:
   ```python
   self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
   ```
   Linear projection maps $d_{model} (2560) \to 2 \times d_{model} (5120)$.
   PyTorch `weight` shape is `[out_features, in_features] = [5120, 2560]`.
2. In the forward pass:
   ```python
   projected = self.in_proj(x_norm) # [B, S, 5120]
   gate, value = projected.chunk(2, dim=-1) # Split into 2560 (gate) and 2560 (value)
   ```
3. The native C++ engine (`src/engine/nano_engine.cpp`) explicitly matches this dataflow:
   ```cpp
   nano_neon_gemv_ternary_int8(ctx->state_in_proj_act, lp.w_state_in_proj, ..., 5120, 2560);
   const float* gate_stream = ctx->state_in_proj_act;
   const float* value_stream = ctx->state_in_proj_act + 2560;
   ```
4. Each State block requires $5120 \times 2560 = 13,107,200$ parameters.
   Across 16 State blocks, total `in_proj` parameters = $16 \times 13,107,200 = \mathbf{209,715,200}$.

---

## 6. GQA Block Exact Shape Audit
For all 8 GQA layers (`2, 5, 8, 11, 14, 17, 20, 23`):
- `layers.X.mixer.q_proj.weight`: shape = `[2560, 2560]`, numel = `6,553,600`
- `layers.X.mixer.k_proj.weight`: shape = `[512, 2560]`, numel = `1,310,720` (4 heads * 128 head_dim)
- `layers.X.mixer.v_proj.weight`: shape = `[512, 2560]`, numel = `1,310,720` (4 heads * 128 head_dim)
- `layers.X.mixer.out_proj.weight`: shape = `[2560, 2560]`, numel = `6,553,600`
- `layers.X.mixer.norm.weight`: shape = `[2560]`, numel = `2,560`
Total GQA Mixer parameters across 8 blocks: `8 * 15,731,200 = 125,849,600` parameters.

---

## 7. FFN Block Exact Shape Audit
For all 24 backbone layers:
- `layers.X.ffn.gate_proj.weight`: shape = `[6912, 2560]`, numel = `17,694,720`
- `layers.X.ffn.up_proj.weight`: shape = `[6912, 2560]`, numel = `17,694,720`
- `layers.X.ffn.down_proj.weight`: shape = `[2560, 6912]`, numel = `17,694,720`
- `layers.X.ffn.norm.weight`: shape = `[2560]`, numel = `2,560`
Total SwiGLU FFN parameters across 24 blocks: `24 * 53,086,720 = 1,274,081,280` parameters.

---

## 8. Root Embeddings & LM Head Shape Audit
- `embed_tokens.weight`: shape = `[65536, 2560]`, numel = `167,772,160` (INT8)
- `lm_head.weight`: shape = `[65536, 2560]`, numel = `167,772,160` (INT8)
- `final_norm.weight`: shape = `[2560]`, numel = `2,560` (FP32)
Total Global Root parameters: `335,546,880` parameters.

---

## 9. Comprehensive Parameter Accounting

$$\begin{aligned}
\text{State Mixers (16)} &= 16 \times [13,107,200 + 6,553,600 + 10,240 + 2,560 + 2,560] = 314,818,560 \\
\text{GQA Mixers (8)} &= 8 \times [6,553,600 + 1,310,720 + 1,310,720 + 6,553,600 + 2,560] = 125,849,600 \\
\text{SwiGLU FFN (24)} &= 24 \times [17,694,720 + 17,694,720 + 17,694,720 + 2,560] = 1,274,081,280 \\
\text{Global Root (3)} &= 167,772,160 + 167,772,160 + 2,560 = 335,546,880 \\
\mathbf{\text{TOTAL}} &= 314,818,560 + 125,849,600 + 1,274,081,280 + 335,546,880 = \mathbf{2,050,296,320}
\end{aligned}$$

### Subtotal Summary by Functional Component:
- **State In-Projection Subtotal**: `209,715,200` parameters
- **State Out-Projection Subtotal**: `104,857,600` parameters
- **State Conv1D Weights Subtotal**: `163,840` parameters
- **State Conv1D Bias Subtotal**: `40,960` parameters
- **State Norm Subtotal**: `40,960` parameters
- **GQA Projections Subtotal**: `125,829,120` parameters
- **GQA Norm Subtotal**: `20,480` parameters
- **FFN Projections Subtotal**: `1,274,019,840` parameters
- **FFN Norm Subtotal**: `61,440` parameters
- **Embeddings Subtotal**: `167,772,160` parameters
- **LM Head Subtotal**: `167,772,160` parameters
- **Final Norm Subtotal**: `2,560` parameters
- **All Norms Combined (81 tensors with conv)**: `330,240` parameters
- **GRAND TOTAL**: **`2,050,296,320`** parameters.

---

## 10. Reconcile FIX-09B Report vs. Actual Checkpoint Facts

| Metric / Feature | FIX-09B Report Text | Actual Checkpoint Architecture | Status | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| Tensor Count | 219 | 219 | **MATCH** | Exact match |
| Total Parameter Count | 2,050,296,320 | 2,050,296,320 | **MATCH** | Exact match |
| State in_proj shape | [2560, 2560] (Typo) | [5120, 2560] | **MISMATCH (PROSE TYPO)** | Code uses `[5120, 2560]`; report text had formula typo |
| State out_proj shape | [2560, 2560] | [2560, 2560] | **MATCH** | Exact match |
| State conv shape | [2560, 1, 4] | [2560, 1, 4] | **MATCH** | Exact match |
| GQA q_proj shape | [2560, 2560] | [2560, 2560] | **MATCH** | Exact match |
| GQA k_proj shape | [512, 2560] | [512, 2560] | **MATCH** | Exact match |
| GQA v_proj shape | [512, 2560] | [512, 2560] | **MATCH** | Exact match |
| GQA out_proj shape | [2560, 2560] | [2560, 2560] | **MATCH** | Exact match |
| FFN gate_proj shape | [6912, 2560] | [6912, 2560] | **MATCH** | Exact match |
| FFN up_proj shape | [6912, 2560] | [6912, 2560] | **MATCH** | Exact match |
| FFN down_proj shape | [2560, 6912] | [2560, 6912] | **MATCH** | Exact match |
| RMSNorm shapes | [2560] | [2560] | **MATCH** | Exact match |
| Embedding shape | [65536, 2560] | [65536, 2560] | **MATCH** | Exact match |
| LM Head shape | [65536, 2560] | [65536, 2560] | **MATCH** | Exact match |

---

## 11. Reconcile Exporter Contract
Inspecting `tools/export_to_nano.py`:
- Line 229: `if list(in_t.shape) != [2 * d_model, d_model]:`
  Expects `[5120, 2560]`. Matches checkpoint: **YES**.
- Line 235: `if list(out_t.shape) != [d_model, d_model]:`
  Expects `[2560, 2560]`. Matches checkpoint: **YES**.
- Line 151: `total_params = sum(t.numel() for t in state_dict.values())`
  Requires `total_params == 2050296320`. Matches checkpoint: **YES**.
- Verdict: **EXPORTER_CHECKPOINT_MATCH = MATCH**.

---

## 12. Reconcile Native C++ Execution Graph
Inspecting `src/engine/nano_engine.cpp`:
- Line 86: `const uint8_t* w_state_in_proj; // [5120, 2560] (Ternary 2-bit packed)`
- Lines 283-291:
  ```cpp
  nano_neon_gemv_ternary_int8(ctx->state_in_proj_act, lp.w_state_in_proj, ctx->h_state_int8, &alpha_state_in, nullptr, 5120, 2560);
  ```
  Executes matrix multiplication producing 5120 output values.
- Lines 294-295:
  ```cpp
  const float* gate_stream = ctx->state_in_proj_act;
  const float* value_stream = ctx->state_in_proj_act + 2560;
  ```
  Splits 5120 into gate (2560) and value (2560). Matches `ShortConvStateBlock` `projected.chunk(2, dim=-1)` bit-for-bit.
- Lines 318-320:
  ```cpp
  nano_neon_gemv_ternary_int8(ctx->h_state_res, lp.w_state_out_proj, ctx->state_gated_int8, &alpha_state_out, nullptr, 2560, 2560);
  ```
  Projects 2560 gated features back to 2560 hidden state.
- Verdict: **NATIVE_CHECKPOINT_MATCH = MATCH**.

---

## 13. Dtype Audit
- Training Source Dtype: `bfloat16` (219 tensors)
- Zero NaN tensors, Zero Inf tensors verified in Step-30 post-persistence audit.

---

## 14. Final Verdict

**`FIX-09B.1-PASS-CHECKPOINT-RECONCILED`**

---

## 15. Required Machine-Readable Output Block

```text
FIX-09B.1-BEGIN

CHECKPOINT_SHA_BEFORE=0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667
CHECKPOINT_SHA_AFTER=0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667
CHECKPOINT_SIZE_BEFORE=4106953961
CHECKPOINT_SIZE_AFTER=4106953961

CHECKPOINT_TENSORS=219
CHECKPOINT_PARAMS=2050296320

EXPECTED_TENSORS=219
EXPECTED_PARAMS=2050296320

KEY_BIJECTION=PASS
SHAPE_RECONCILIATION=PASS
PARAMETER_RECONCILIATION=PASS

STATE_IN_PROJ_SHAPE=[5120, 2560]
STATE_IN_PROJ_PARAMS_PER_LAYER=13107200
STATE_IN_PROJ_TOTAL_PARAMS=209715200

STATE_OUT_PROJ_SHAPE=[2560, 2560]
STATE_CONV_WEIGHT_SHAPE=[2560, 1, 4]
STATE_CONV_BIAS_SHAPE=[2560]

GQA_Q_SHAPE=[2560, 2560]
GQA_K_SHAPE=[512, 2560]
GQA_V_SHAPE=[512, 2560]
GQA_OUT_SHAPE=[2560, 2560]

FFN_GATE_SHAPE=[6912, 2560]
FFN_UP_SHAPE=[6912, 2560]
FFN_DOWN_SHAPE=[2560, 6912]

EMBED_SHAPE=[65536, 2560]
FINAL_NORM_SHAPE=[2560]
LM_HEAD_SHAPE=[65536, 2560]

EXPORTER_CHECKPOINT_MATCH=MATCH
NATIVE_CHECKPOINT_MATCH=MATCH

CHECKPOINT_IMMUTABILITY=PASS

FIX-09B.1-END
```