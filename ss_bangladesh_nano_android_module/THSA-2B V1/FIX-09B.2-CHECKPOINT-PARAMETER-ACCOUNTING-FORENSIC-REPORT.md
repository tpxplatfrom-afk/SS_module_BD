# FIX-09B.2: CHECKPOINT PARAMETER ARITHMETIC & STATE-DICT ACCOUNTING FORENSIC REPORT
## COMPREHENSIVE PARAMETER ACCOUNTING, FOUR-WAY AUDIT & 40,960 DISCREPANCY RESOLUTION

## 1. Executive Verdict
- **FIX ID**: `FIX-09B.2-CHECKPOINT-PARAMETER-ACCOUNTING-FORENSIC`
- **Authoritative Checkpoint**: Step-30 continuation checkpoint `checkpoint_step_000030.pt`
- **Expected SHA-256**: `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`
- **Expected File Size**: `4,106,953,961` bytes
- **Claimed Parameter Count**: `2,050,296,320`
- **Actual State Dict Parameter Sum**: `2,050,296,320`
- **Actual Named Parameters Sum**: `2,050,296,320`
- **Actual Trainable Parameters Sum**: `2,050,296,320` (`requires_grad == True` on 219/219 tensors)
- **Actual Frozen Parameters Sum**: `0` (`requires_grad == False` on 0/219 tensors)
- **Actual Buffers Sum**: `0` (0 buffers)
- **Final Verdict**: **`FIX-09B.2-PASS-PARAMETER-ACCOUNTING-RECONCILED`**

---

## 2. Scope & Isolation Proof
- Operating strictly within: `ss_bangladesh_nano_android_module/THSA-2B V1`.
- `ss_bangladesh/` and external master repositories remain untouched, uninspected, and unmodified.
- Zero retraining, zero export of `model.nano`, and zero parameter mutations occurred.

---

## 3. Checkpoint Identity & Immutability Verification
- **Checkpoint File**: `checkpoint_step_000030.pt`
- **SHA-256 Before Audit**: `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`
- **SHA-256 After Audit**:  `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`
- **Size Before Audit**:    `4,106,953,961` bytes
- **Size After Audit**:     `4,106,953,961` bytes
- **Checkpoint Immutability**: **PASS** (Byte-for-byte identical, strictly read-only access).

---

## 4. The 40,960 Discrepancy Forensic Investigation
### The Forensic Question:
Why did an independent arithmetic sum produce `2,050,337,280`, differing from the claimed `2,050,296,320` by exactly `40,960` ($16 \times 2560$)?

### Direct Forensic Proof:
1. **The 16 State Conv1D Bias Tensors are fully present in the checkpoint and PyTorch model**:
   - Names: `layers.{0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22}.mixer.conv1d.bias`
   - Shape: `[2560]` each
   - Numel: `2,560` parameters each
   - Total Bias Parameters: $16 \times 2,560 = \mathbf{40{,}960}$ parameters.
2. **The claimed architecture count of `2,050,296,320` ALREADY INCLUDES these 40,960 Conv1D biases**:
   - If the 16 Conv1D biases are excluded, the model has only 203 tensors and **2,050,255,360** parameters ($2,050,296,320 - 40,960$).
   - With the 16 Conv1D biases included, the model has 219 tensors and **2,050,296,320** parameters.
3. **Root Cause of the `2,050,337,280` Figure**:
   - The figure `2,050,337,280` was obtained by taking `2,050,296,320` and ADDING `40,960` to it ($2,050,296,320 + 40,960 = 2,050,337,280$).
   - This represents an inadvertent **double-counting** of the 16 State Conv1D biases, based on a mistaken assumption that the base count of 2,050,296,320 was weights-only.
   - **Conclusion**: The true, mathematically verified total of all 219 tensors (including all weights and biases) is **`2,050,296,320`**.

### 10-Point Conv1D Bias Audit Questionnaire:
1. **Are the 16 biases in `state_dict`?** YES.
2. **Are they `nn.Parameter` objects?** YES (instantiated by `nn.Conv1d(..., bias=True)`).
3. **Are they in `named_parameters()`?** YES (all 16 appear in iteration).
4. **Do they have `requires_grad=True`?** YES (all 16 are trainable).
5. **Were they included in training?** YES (optimized during Step 1-30).
6. **Are they included in the 219 tensor count?** YES (without them, tensor count would be 203).
7. **Are they included in the claimed 2,050,296,320?** YES (sum with biases is 2,050,296,320).
8. **Are they serialized in the planned .nano format?** YES (`layer_{l}_state_conv_b` in Format V2).
9. **Does native execution consume them?** YES (`nano_neon_short_conv_step` consumes `lp.conv_bias`).
10. **Is 2,050,296,320 a trainable-only total?** YES, and it is ALSO the total parameter count (frozen = 0).

---

## 5. Four-Way Accounting Identity Verification

$$\begin{aligned}
\text{ACCOUNTING A (State Dict Sum)} &= 219\text{ tensors} = \mathbf{2{,}050{,}296{,}320} \\
\text{ACCOUNTING B (All Named Parameters)} &= 219\text{ parameters} = \mathbf{2{,}050{,}296{,}320} \\
\text{ACCOUNTING C (Trainable Parameters)} &= 219\text{ parameters} = \mathbf{2{,}050{,}296{,}320} \\
\text{ACCOUNTING D (Frozen Parameters)} &= 0\text{ parameters} = \mathbf{0} \\
\text{MODEL BUFFERS} &= 0\text{ buffers} = \mathbf{0} \\
\text{Identity 1: } A &= B + \text{Buffers} \iff 2{,}050{,}296{,}320 = 2{,}050{,}296{,}320 + 0 \quad (\mathbf{PASS}) \\
\text{Identity 2: } B &= C + D \iff 2{,}050{,}296{,}320 = 2{,}050{,}296{,}320 + 0 \quad (\mathbf{PASS})
\end{aligned}$$

---

## 6. Weight Tying & Storage Aliasing Audit
- **Total Instantiated Parameters**: `219`
- **Unique Untyped Storage Pointers**: `219`
- **Storage Aliasing Detected**: `False`
- **`embed_tokens.weight` is `lm_head.weight`**: `False` (Independent weights, untied).
- **Verdict**: **PASS** (Zero storage sharing, zero double-counting, 100% parameter uniqueness).

---

## 7. 17 Architectural Group Totals

| # | Group Name | Tensor Count | Shape | Parameters/Tensor | Aggregate Parameters | Quant Class | Trainable | Frozen |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1. Token embedding | 1 | `[65536, 2560]` | 167,772,160 | 167,772,160 | INT8 | 1 | 0 |
| 2 | 2. State mixer RMSNorm | 16 | `[2560]` | 2,560 | 40,960 | FP32 | 16 | 0 |
| 3 | 3. State Conv1D weights | 16 | `[2560, 1, 4]` | 10,240 | 163,840 | FP32 | 16 | 0 |
| 4 | 4. State Conv1D biases | 16 | `[2560]` | 2,560 | 40,960 | FP32 | 16 | 0 |
| 5 | 5. State in-projection | 16 | `[5120, 2560]` | 13,107,200 | 209,715,200 | TERNARY | 16 | 0 |
| 6 | 6. State out-projection | 16 | `[2560, 2560]` | 6,553,600 | 104,857,600 | TERNARY | 16 | 0 |
| 7 | 7. GQA mixer RMSNorm | 8 | `[2560]` | 2,560 | 20,480 | FP32 | 8 | 0 |
| 8 | 8. GQA Q projection | 8 | `[2560, 2560]` | 6,553,600 | 52,428,800 | TERNARY | 8 | 0 |
| 9 | 9. GQA K projection | 8 | `[512, 2560]` | 1,310,720 | 10,485,760 | TERNARY | 8 | 0 |
| 10 | 10. GQA V projection | 8 | `[512, 2560]` | 1,310,720 | 10,485,760 | TERNARY | 8 | 0 |
| 11 | 11. GQA out-projection | 8 | `[2560, 2560]` | 6,553,600 | 52,428,800 | TERNARY | 8 | 0 |
| 12 | 12. FFN RMSNorm | 24 | `[2560]` | 2,560 | 61,440 | FP32 | 24 | 0 |
| 13 | 13. FFN gate projection | 24 | `[6912, 2560]` | 17,694,720 | 424,673,280 | TERNARY | 24 | 0 |
| 14 | 14. FFN up projection | 24 | `[6912, 2560]` | 17,694,720 | 424,673,280 | TERNARY | 24 | 0 |
| 15 | 15. FFN down projection | 24 | `[2560, 6912]` | 17,694,720 | 424,673,280 | TERNARY | 24 | 0 |
| 16 | 16. Final RMSNorm | 1 | `[2560]` | 2,560 | 2,560 | FP32 | 1 | 0 |
| 17 | 17. LM head | 1 | `[65536, 2560]` | 167,772,160 | 167,772,160 | INT8 | 1 | 0 |
| **TOTAL** | **17 Groups** | **219** | - | - | **2,050,296,320** | - | **219** | **0** |

---

## 8. Complete 219-Tensor Machine-Generated Inventory Table

| ID | State Dict Key | Category | Shape | Dtype | Numel | Requires Grad | Cumulative Numel | Quant Class | Descriptor ID | Native Graph Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `embed_tokens.weight` | root | `[65536, 2560]` | bfloat16 | 167,772,160 | True | 167,772,160 | INT8 | 0 | `token_embeddings` |
| 1 | `layers.0.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 167,782,400 | FP32 | 1 | `state_depthwise_conv_filter` |
| 2 | `layers.0.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 167,784,960 | FP32 | 2 | `state_depthwise_conv_bias` |
| 3 | `layers.0.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 180,892,160 | TERNARY | 3 | `state_in_projection_gate_val` |
| 4 | `layers.0.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 187,445,760 | TERNARY | 4 | `state_out_projection` |
| 5 | `layers.0.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 187,448,320 | FP32 | 5 | `state_mixer_rmsnorm` |
| 6 | `layers.0.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 205,143,040 | TERNARY | 6 | `swiglu_gate_projection` |
| 7 | `layers.0.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 222,837,760 | TERNARY | 7 | `swiglu_up_projection` |
| 8 | `layers.0.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 240,532,480 | TERNARY | 8 | `swiglu_down_projection` |
| 9 | `layers.0.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 240,535,040 | FP32 | 9 | `ffn_pre_rmsnorm` |
| 10 | `layers.1.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 240,545,280 | FP32 | 10 | `state_depthwise_conv_filter` |
| 11 | `layers.1.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 240,547,840 | FP32 | 11 | `state_depthwise_conv_bias` |
| 12 | `layers.1.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 253,655,040 | TERNARY | 12 | `state_in_projection_gate_val` |
| 13 | `layers.1.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 260,208,640 | TERNARY | 13 | `state_out_projection` |
| 14 | `layers.1.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 260,211,200 | FP32 | 14 | `state_mixer_rmsnorm` |
| 15 | `layers.1.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 277,905,920 | TERNARY | 15 | `swiglu_gate_projection` |
| 16 | `layers.1.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 295,600,640 | TERNARY | 16 | `swiglu_up_projection` |
| 17 | `layers.1.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 313,295,360 | TERNARY | 17 | `swiglu_down_projection` |
| 18 | `layers.1.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 313,297,920 | FP32 | 18 | `ffn_pre_rmsnorm` |
| 19 | `layers.2.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 319,851,520 | TERNARY | 19 | `gqa_query_projection` |
| 20 | `layers.2.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 321,162,240 | TERNARY | 20 | `gqa_key_projection` |
| 21 | `layers.2.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 322,472,960 | TERNARY | 21 | `gqa_value_projection` |
| 22 | `layers.2.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 329,026,560 | TERNARY | 22 | `gqa_output_projection` |
| 23 | `layers.2.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 329,029,120 | FP32 | 23 | `gqa_mixer_rmsnorm` |
| 24 | `layers.2.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 346,723,840 | TERNARY | 24 | `swiglu_gate_projection` |
| 25 | `layers.2.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 364,418,560 | TERNARY | 25 | `swiglu_up_projection` |
| 26 | `layers.2.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 382,113,280 | TERNARY | 26 | `swiglu_down_projection` |
| 27 | `layers.2.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 382,115,840 | FP32 | 27 | `ffn_pre_rmsnorm` |
| 28 | `layers.3.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 382,126,080 | FP32 | 28 | `state_depthwise_conv_filter` |
| 29 | `layers.3.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 382,128,640 | FP32 | 29 | `state_depthwise_conv_bias` |
| 30 | `layers.3.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 395,235,840 | TERNARY | 30 | `state_in_projection_gate_val` |
| 31 | `layers.3.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 401,789,440 | TERNARY | 31 | `state_out_projection` |
| 32 | `layers.3.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 401,792,000 | FP32 | 32 | `state_mixer_rmsnorm` |
| 33 | `layers.3.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 419,486,720 | TERNARY | 33 | `swiglu_gate_projection` |
| 34 | `layers.3.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 437,181,440 | TERNARY | 34 | `swiglu_up_projection` |
| 35 | `layers.3.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 454,876,160 | TERNARY | 35 | `swiglu_down_projection` |
| 36 | `layers.3.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 454,878,720 | FP32 | 36 | `ffn_pre_rmsnorm` |
| 37 | `layers.4.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 454,888,960 | FP32 | 37 | `state_depthwise_conv_filter` |
| 38 | `layers.4.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 454,891,520 | FP32 | 38 | `state_depthwise_conv_bias` |
| 39 | `layers.4.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 467,998,720 | TERNARY | 39 | `state_in_projection_gate_val` |
| 40 | `layers.4.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 474,552,320 | TERNARY | 40 | `state_out_projection` |
| 41 | `layers.4.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 474,554,880 | FP32 | 41 | `state_mixer_rmsnorm` |
| 42 | `layers.4.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 492,249,600 | TERNARY | 42 | `swiglu_gate_projection` |
| 43 | `layers.4.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 509,944,320 | TERNARY | 43 | `swiglu_up_projection` |
| 44 | `layers.4.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 527,639,040 | TERNARY | 44 | `swiglu_down_projection` |
| 45 | `layers.4.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 527,641,600 | FP32 | 45 | `ffn_pre_rmsnorm` |
| 46 | `layers.5.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 534,195,200 | TERNARY | 46 | `gqa_query_projection` |
| 47 | `layers.5.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 535,505,920 | TERNARY | 47 | `gqa_key_projection` |
| 48 | `layers.5.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 536,816,640 | TERNARY | 48 | `gqa_value_projection` |
| 49 | `layers.5.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 543,370,240 | TERNARY | 49 | `gqa_output_projection` |
| 50 | `layers.5.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 543,372,800 | FP32 | 50 | `gqa_mixer_rmsnorm` |
| 51 | `layers.5.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 561,067,520 | TERNARY | 51 | `swiglu_gate_projection` |
| 52 | `layers.5.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 578,762,240 | TERNARY | 52 | `swiglu_up_projection` |
| 53 | `layers.5.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 596,456,960 | TERNARY | 53 | `swiglu_down_projection` |
| 54 | `layers.5.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 596,459,520 | FP32 | 54 | `ffn_pre_rmsnorm` |
| 55 | `layers.6.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 596,469,760 | FP32 | 55 | `state_depthwise_conv_filter` |
| 56 | `layers.6.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 596,472,320 | FP32 | 56 | `state_depthwise_conv_bias` |
| 57 | `layers.6.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 609,579,520 | TERNARY | 57 | `state_in_projection_gate_val` |
| 58 | `layers.6.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 616,133,120 | TERNARY | 58 | `state_out_projection` |
| 59 | `layers.6.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 616,135,680 | FP32 | 59 | `state_mixer_rmsnorm` |
| 60 | `layers.6.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 633,830,400 | TERNARY | 60 | `swiglu_gate_projection` |
| 61 | `layers.6.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 651,525,120 | TERNARY | 61 | `swiglu_up_projection` |
| 62 | `layers.6.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 669,219,840 | TERNARY | 62 | `swiglu_down_projection` |
| 63 | `layers.6.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 669,222,400 | FP32 | 63 | `ffn_pre_rmsnorm` |
| 64 | `layers.7.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 669,232,640 | FP32 | 64 | `state_depthwise_conv_filter` |
| 65 | `layers.7.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 669,235,200 | FP32 | 65 | `state_depthwise_conv_bias` |
| 66 | `layers.7.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 682,342,400 | TERNARY | 66 | `state_in_projection_gate_val` |
| 67 | `layers.7.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 688,896,000 | TERNARY | 67 | `state_out_projection` |
| 68 | `layers.7.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 688,898,560 | FP32 | 68 | `state_mixer_rmsnorm` |
| 69 | `layers.7.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 706,593,280 | TERNARY | 69 | `swiglu_gate_projection` |
| 70 | `layers.7.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 724,288,000 | TERNARY | 70 | `swiglu_up_projection` |
| 71 | `layers.7.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 741,982,720 | TERNARY | 71 | `swiglu_down_projection` |
| 72 | `layers.7.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 741,985,280 | FP32 | 72 | `ffn_pre_rmsnorm` |
| 73 | `layers.8.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 748,538,880 | TERNARY | 73 | `gqa_query_projection` |
| 74 | `layers.8.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 749,849,600 | TERNARY | 74 | `gqa_key_projection` |
| 75 | `layers.8.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 751,160,320 | TERNARY | 75 | `gqa_value_projection` |
| 76 | `layers.8.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 757,713,920 | TERNARY | 76 | `gqa_output_projection` |
| 77 | `layers.8.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 757,716,480 | FP32 | 77 | `gqa_mixer_rmsnorm` |
| 78 | `layers.8.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 775,411,200 | TERNARY | 78 | `swiglu_gate_projection` |
| 79 | `layers.8.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 793,105,920 | TERNARY | 79 | `swiglu_up_projection` |
| 80 | `layers.8.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 810,800,640 | TERNARY | 80 | `swiglu_down_projection` |
| 81 | `layers.8.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 810,803,200 | FP32 | 81 | `ffn_pre_rmsnorm` |
| 82 | `layers.9.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 810,813,440 | FP32 | 82 | `state_depthwise_conv_filter` |
| 83 | `layers.9.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 810,816,000 | FP32 | 83 | `state_depthwise_conv_bias` |
| 84 | `layers.9.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 823,923,200 | TERNARY | 84 | `state_in_projection_gate_val` |
| 85 | `layers.9.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 830,476,800 | TERNARY | 85 | `state_out_projection` |
| 86 | `layers.9.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 830,479,360 | FP32 | 86 | `state_mixer_rmsnorm` |
| 87 | `layers.9.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 848,174,080 | TERNARY | 87 | `swiglu_gate_projection` |
| 88 | `layers.9.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 865,868,800 | TERNARY | 88 | `swiglu_up_projection` |
| 89 | `layers.9.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 883,563,520 | TERNARY | 89 | `swiglu_down_projection` |
| 90 | `layers.9.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 883,566,080 | FP32 | 90 | `ffn_pre_rmsnorm` |
| 91 | `layers.10.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 883,576,320 | FP32 | 91 | `state_depthwise_conv_filter` |
| 92 | `layers.10.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 883,578,880 | FP32 | 92 | `state_depthwise_conv_bias` |
| 93 | `layers.10.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 896,686,080 | TERNARY | 93 | `state_in_projection_gate_val` |
| 94 | `layers.10.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 903,239,680 | TERNARY | 94 | `state_out_projection` |
| 95 | `layers.10.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 903,242,240 | FP32 | 95 | `state_mixer_rmsnorm` |
| 96 | `layers.10.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 920,936,960 | TERNARY | 96 | `swiglu_gate_projection` |
| 97 | `layers.10.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 938,631,680 | TERNARY | 97 | `swiglu_up_projection` |
| 98 | `layers.10.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 956,326,400 | TERNARY | 98 | `swiglu_down_projection` |
| 99 | `layers.10.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 956,328,960 | FP32 | 99 | `ffn_pre_rmsnorm` |
| 100 | `layers.11.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 962,882,560 | TERNARY | 100 | `gqa_query_projection` |
| 101 | `layers.11.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 964,193,280 | TERNARY | 101 | `gqa_key_projection` |
| 102 | `layers.11.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 965,504,000 | TERNARY | 102 | `gqa_value_projection` |
| 103 | `layers.11.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 972,057,600 | TERNARY | 103 | `gqa_output_projection` |
| 104 | `layers.11.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 972,060,160 | FP32 | 104 | `gqa_mixer_rmsnorm` |
| 105 | `layers.11.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 989,754,880 | TERNARY | 105 | `swiglu_gate_projection` |
| 106 | `layers.11.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,007,449,600 | TERNARY | 106 | `swiglu_up_projection` |
| 107 | `layers.11.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,025,144,320 | TERNARY | 107 | `swiglu_down_projection` |
| 108 | `layers.11.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,025,146,880 | FP32 | 108 | `ffn_pre_rmsnorm` |
| 109 | `layers.12.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,025,157,120 | FP32 | 109 | `state_depthwise_conv_filter` |
| 110 | `layers.12.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,025,159,680 | FP32 | 110 | `state_depthwise_conv_bias` |
| 111 | `layers.12.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,038,266,880 | TERNARY | 111 | `state_in_projection_gate_val` |
| 112 | `layers.12.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,044,820,480 | TERNARY | 112 | `state_out_projection` |
| 113 | `layers.12.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,044,823,040 | FP32 | 113 | `state_mixer_rmsnorm` |
| 114 | `layers.12.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,062,517,760 | TERNARY | 114 | `swiglu_gate_projection` |
| 115 | `layers.12.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,080,212,480 | TERNARY | 115 | `swiglu_up_projection` |
| 116 | `layers.12.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,097,907,200 | TERNARY | 116 | `swiglu_down_projection` |
| 117 | `layers.12.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,097,909,760 | FP32 | 117 | `ffn_pre_rmsnorm` |
| 118 | `layers.13.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,097,920,000 | FP32 | 118 | `state_depthwise_conv_filter` |
| 119 | `layers.13.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,097,922,560 | FP32 | 119 | `state_depthwise_conv_bias` |
| 120 | `layers.13.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,111,029,760 | TERNARY | 120 | `state_in_projection_gate_val` |
| 121 | `layers.13.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,117,583,360 | TERNARY | 121 | `state_out_projection` |
| 122 | `layers.13.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,117,585,920 | FP32 | 122 | `state_mixer_rmsnorm` |
| 123 | `layers.13.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,135,280,640 | TERNARY | 123 | `swiglu_gate_projection` |
| 124 | `layers.13.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,152,975,360 | TERNARY | 124 | `swiglu_up_projection` |
| 125 | `layers.13.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,170,670,080 | TERNARY | 125 | `swiglu_down_projection` |
| 126 | `layers.13.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,170,672,640 | FP32 | 126 | `ffn_pre_rmsnorm` |
| 127 | `layers.14.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,177,226,240 | TERNARY | 127 | `gqa_query_projection` |
| 128 | `layers.14.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,178,536,960 | TERNARY | 128 | `gqa_key_projection` |
| 129 | `layers.14.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,179,847,680 | TERNARY | 129 | `gqa_value_projection` |
| 130 | `layers.14.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,186,401,280 | TERNARY | 130 | `gqa_output_projection` |
| 131 | `layers.14.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 1,186,403,840 | FP32 | 131 | `gqa_mixer_rmsnorm` |
| 132 | `layers.14.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,204,098,560 | TERNARY | 132 | `swiglu_gate_projection` |
| 133 | `layers.14.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,221,793,280 | TERNARY | 133 | `swiglu_up_projection` |
| 134 | `layers.14.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,239,488,000 | TERNARY | 134 | `swiglu_down_projection` |
| 135 | `layers.14.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,239,490,560 | FP32 | 135 | `ffn_pre_rmsnorm` |
| 136 | `layers.15.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,239,500,800 | FP32 | 136 | `state_depthwise_conv_filter` |
| 137 | `layers.15.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,239,503,360 | FP32 | 137 | `state_depthwise_conv_bias` |
| 138 | `layers.15.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,252,610,560 | TERNARY | 138 | `state_in_projection_gate_val` |
| 139 | `layers.15.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,259,164,160 | TERNARY | 139 | `state_out_projection` |
| 140 | `layers.15.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,259,166,720 | FP32 | 140 | `state_mixer_rmsnorm` |
| 141 | `layers.15.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,276,861,440 | TERNARY | 141 | `swiglu_gate_projection` |
| 142 | `layers.15.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,294,556,160 | TERNARY | 142 | `swiglu_up_projection` |
| 143 | `layers.15.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,312,250,880 | TERNARY | 143 | `swiglu_down_projection` |
| 144 | `layers.15.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,312,253,440 | FP32 | 144 | `ffn_pre_rmsnorm` |
| 145 | `layers.16.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,312,263,680 | FP32 | 145 | `state_depthwise_conv_filter` |
| 146 | `layers.16.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,312,266,240 | FP32 | 146 | `state_depthwise_conv_bias` |
| 147 | `layers.16.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,325,373,440 | TERNARY | 147 | `state_in_projection_gate_val` |
| 148 | `layers.16.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,331,927,040 | TERNARY | 148 | `state_out_projection` |
| 149 | `layers.16.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,331,929,600 | FP32 | 149 | `state_mixer_rmsnorm` |
| 150 | `layers.16.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,349,624,320 | TERNARY | 150 | `swiglu_gate_projection` |
| 151 | `layers.16.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,367,319,040 | TERNARY | 151 | `swiglu_up_projection` |
| 152 | `layers.16.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,385,013,760 | TERNARY | 152 | `swiglu_down_projection` |
| 153 | `layers.16.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,385,016,320 | FP32 | 153 | `ffn_pre_rmsnorm` |
| 154 | `layers.17.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,391,569,920 | TERNARY | 154 | `gqa_query_projection` |
| 155 | `layers.17.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,392,880,640 | TERNARY | 155 | `gqa_key_projection` |
| 156 | `layers.17.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,394,191,360 | TERNARY | 156 | `gqa_value_projection` |
| 157 | `layers.17.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,400,744,960 | TERNARY | 157 | `gqa_output_projection` |
| 158 | `layers.17.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 1,400,747,520 | FP32 | 158 | `gqa_mixer_rmsnorm` |
| 159 | `layers.17.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,418,442,240 | TERNARY | 159 | `swiglu_gate_projection` |
| 160 | `layers.17.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,436,136,960 | TERNARY | 160 | `swiglu_up_projection` |
| 161 | `layers.17.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,453,831,680 | TERNARY | 161 | `swiglu_down_projection` |
| 162 | `layers.17.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,453,834,240 | FP32 | 162 | `ffn_pre_rmsnorm` |
| 163 | `layers.18.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,453,844,480 | FP32 | 163 | `state_depthwise_conv_filter` |
| 164 | `layers.18.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,453,847,040 | FP32 | 164 | `state_depthwise_conv_bias` |
| 165 | `layers.18.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,466,954,240 | TERNARY | 165 | `state_in_projection_gate_val` |
| 166 | `layers.18.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,473,507,840 | TERNARY | 166 | `state_out_projection` |
| 167 | `layers.18.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,473,510,400 | FP32 | 167 | `state_mixer_rmsnorm` |
| 168 | `layers.18.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,491,205,120 | TERNARY | 168 | `swiglu_gate_projection` |
| 169 | `layers.18.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,508,899,840 | TERNARY | 169 | `swiglu_up_projection` |
| 170 | `layers.18.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,526,594,560 | TERNARY | 170 | `swiglu_down_projection` |
| 171 | `layers.18.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,526,597,120 | FP32 | 171 | `ffn_pre_rmsnorm` |
| 172 | `layers.19.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,526,607,360 | FP32 | 172 | `state_depthwise_conv_filter` |
| 173 | `layers.19.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,526,609,920 | FP32 | 173 | `state_depthwise_conv_bias` |
| 174 | `layers.19.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,539,717,120 | TERNARY | 174 | `state_in_projection_gate_val` |
| 175 | `layers.19.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,546,270,720 | TERNARY | 175 | `state_out_projection` |
| 176 | `layers.19.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,546,273,280 | FP32 | 176 | `state_mixer_rmsnorm` |
| 177 | `layers.19.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,563,968,000 | TERNARY | 177 | `swiglu_gate_projection` |
| 178 | `layers.19.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,581,662,720 | TERNARY | 178 | `swiglu_up_projection` |
| 179 | `layers.19.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,599,357,440 | TERNARY | 179 | `swiglu_down_projection` |
| 180 | `layers.19.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,599,360,000 | FP32 | 180 | `ffn_pre_rmsnorm` |
| 181 | `layers.20.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,605,913,600 | TERNARY | 181 | `gqa_query_projection` |
| 182 | `layers.20.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,607,224,320 | TERNARY | 182 | `gqa_key_projection` |
| 183 | `layers.20.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,608,535,040 | TERNARY | 183 | `gqa_value_projection` |
| 184 | `layers.20.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,615,088,640 | TERNARY | 184 | `gqa_output_projection` |
| 185 | `layers.20.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 1,615,091,200 | FP32 | 185 | `gqa_mixer_rmsnorm` |
| 186 | `layers.20.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,632,785,920 | TERNARY | 186 | `swiglu_gate_projection` |
| 187 | `layers.20.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,650,480,640 | TERNARY | 187 | `swiglu_up_projection` |
| 188 | `layers.20.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,668,175,360 | TERNARY | 188 | `swiglu_down_projection` |
| 189 | `layers.20.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,668,177,920 | FP32 | 189 | `ffn_pre_rmsnorm` |
| 190 | `layers.21.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,668,188,160 | FP32 | 190 | `state_depthwise_conv_filter` |
| 191 | `layers.21.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,668,190,720 | FP32 | 191 | `state_depthwise_conv_bias` |
| 192 | `layers.21.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,681,297,920 | TERNARY | 192 | `state_in_projection_gate_val` |
| 193 | `layers.21.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,687,851,520 | TERNARY | 193 | `state_out_projection` |
| 194 | `layers.21.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,687,854,080 | FP32 | 194 | `state_mixer_rmsnorm` |
| 195 | `layers.21.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,705,548,800 | TERNARY | 195 | `swiglu_gate_projection` |
| 196 | `layers.21.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,723,243,520 | TERNARY | 196 | `swiglu_up_projection` |
| 197 | `layers.21.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,740,938,240 | TERNARY | 197 | `swiglu_down_projection` |
| 198 | `layers.21.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,740,940,800 | FP32 | 198 | `ffn_pre_rmsnorm` |
| 199 | `layers.22.mixer.conv1d.weight` | state | `[2560, 1, 4]` | bfloat16 | 10,240 | True | 1,740,951,040 | FP32 | 199 | `state_depthwise_conv_filter` |
| 200 | `layers.22.mixer.conv1d.bias` | state | `[2560]` | bfloat16 | 2,560 | True | 1,740,953,600 | FP32 | 200 | `state_depthwise_conv_bias` |
| 201 | `layers.22.mixer.in_proj.weight` | state | `[5120, 2560]` | bfloat16 | 13,107,200 | True | 1,754,060,800 | TERNARY | 201 | `state_in_projection_gate_val` |
| 202 | `layers.22.mixer.out_proj.weight` | state | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,760,614,400 | TERNARY | 202 | `state_out_projection` |
| 203 | `layers.22.mixer.norm.weight` | state | `[2560]` | bfloat16 | 2,560 | True | 1,760,616,960 | FP32 | 203 | `state_mixer_rmsnorm` |
| 204 | `layers.22.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,778,311,680 | TERNARY | 204 | `swiglu_gate_projection` |
| 205 | `layers.22.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,796,006,400 | TERNARY | 205 | `swiglu_up_projection` |
| 206 | `layers.22.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,813,701,120 | TERNARY | 206 | `swiglu_down_projection` |
| 207 | `layers.22.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,813,703,680 | FP32 | 207 | `ffn_pre_rmsnorm` |
| 208 | `layers.23.mixer.q_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,820,257,280 | TERNARY | 208 | `gqa_query_projection` |
| 209 | `layers.23.mixer.k_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,821,568,000 | TERNARY | 209 | `gqa_key_projection` |
| 210 | `layers.23.mixer.v_proj.weight` | gqa | `[512, 2560]` | bfloat16 | 1,310,720 | True | 1,822,878,720 | TERNARY | 210 | `gqa_value_projection` |
| 211 | `layers.23.mixer.out_proj.weight` | gqa | `[2560, 2560]` | bfloat16 | 6,553,600 | True | 1,829,432,320 | TERNARY | 211 | `gqa_output_projection` |
| 212 | `layers.23.mixer.norm.weight` | gqa | `[2560]` | bfloat16 | 2,560 | True | 1,829,434,880 | FP32 | 212 | `gqa_mixer_rmsnorm` |
| 213 | `layers.23.ffn.gate_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,847,129,600 | TERNARY | 213 | `swiglu_gate_projection` |
| 214 | `layers.23.ffn.up_proj.weight` | ffn | `[6912, 2560]` | bfloat16 | 17,694,720 | True | 1,864,824,320 | TERNARY | 214 | `swiglu_up_projection` |
| 215 | `layers.23.ffn.down_proj.weight` | ffn | `[2560, 6912]` | bfloat16 | 17,694,720 | True | 1,882,519,040 | TERNARY | 215 | `swiglu_down_projection` |
| 216 | `layers.23.ffn.norm.weight` | ffn | `[2560]` | bfloat16 | 2,560 | True | 1,882,521,600 | FP32 | 216 | `ffn_pre_rmsnorm` |
| 217 | `final_norm.weight` | root | `[2560]` | bfloat16 | 2,560 | True | 1,882,524,160 | FP32 | 217 | `final_rmsnorm` |
| 218 | `lm_head.weight` | root | `[65536, 2560]` | bfloat16 | 167,772,160 | True | 2,050,296,320 | INT8 | 218 | `causal_lm_head` |

---

## 9. Native Engine & Format V2 Representation Implications
Inspecting `include/nano_types.h` and `src/engine/nano_engine.cpp`:
- Format V2 requires `tensor_count == 219`.
- Exactly 16 descriptors (`layer_{l}_state_conv_b`) represent the 16 State Conv1D bias tensors.
- Native engine execution explicitly passes `lp.conv_bias` to `nano_neon_short_conv_step`:
  ```cpp
  nano_neon_short_conv_step(value_stream, lp.conv_weights, lp.conv_bias, &ctx->state_contexts[l], 2560, ctx->state_conv_out);
  ```
- Zero parameters are omitted, folded, or implicit.

---

## 10. Final Status

**`FIX-09B.2-PASS-PARAMETER-ACCOUNTING-RECONCILED`**

FIX-09B.2-FINAL-STATUS: FIX-09B.2-PASS-PARAMETER-ACCOUNTING-RECONCILED