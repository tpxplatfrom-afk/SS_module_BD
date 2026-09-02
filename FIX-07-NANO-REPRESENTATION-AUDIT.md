# FIX-07 — THSA-2B V1 CHECKPOINT → NANO REPRESENTATION RECONCILIATION AUDIT

**Fix Identifier:** `FIX-07-THSA-2B-NANO-RECONCILIATION`
**Date / Timestamp:** `2026-09-02T21:40:00+06:00`
**Repository Scope:** `ss_bangladesh_nano_android_module / THSA-2B V1`
**Authoritative Target Checkpoint:** Step-30 checkpoint (`checkpoint_step_000030.pt`, SHA-256: `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`)
**Architecture Base:** `THSAHybridForCausalLM` (2,050,296,320 parameters, 219 trainable tensors)
**Final Status Verdict:** `FIX-07-BLOCKED-FORMAT-INCOMPATIBLE`

---

## 1. Executive Summary & Verdict

This forensic audit inspected all **219 PyTorch trainable tensors** of the authoritative THSA-2B V1 model against the native binary model format (`.nano`) defined in `include/nano_types.h`, serialized by `tools/export_to_nano.py`, and executed by `src/engine/nano_engine.cpp`.

### Forensic Findings:
1. **Critical Tensor Discrepancy:**
   - PyTorch Model: **219 trainable tensors** (2,050,296,320 parameters).
   - Native Exporter / Header: **123 descriptors** declared.
   - **96 PyTorch tensors (314,736,640 parameters, or ~15.35% of the model)** are completely omitted from the binary export and absent from the native engine execution graph!
2. **Missing Components in Current Native Implementation:**
   - **State Blocks (16 layers):** The exporter only extracts `conv1d.weight` (1 tensor per block). It completely discards:
     - `mixer.norm.weight` (16 tensors, RMSNorm)
     - `mixer.in_proj.weight` (16 tensors, $5120 \times 2560 = 209,715,200$ parameters)
     - `mixer.conv1d.bias` (16 tensors, depthwise bias)
     - `mixer.out_proj.weight` (16 tensors, $2560 \times 2560 = 104,857,600$ parameters)
     - `ffn.norm.weight` (16 tensors, RMSNorm)
   - **GQA Attention Blocks (8 layers):** The exporter extracts Q, K, V, Out and FFN Gate, Up, Down, but completely discards:
     - `mixer.norm.weight` (8 tensors, RMSNorm pre-attention)
     - `ffn.norm.weight` (8 tensors, RMSNorm pre-FFN)
3. **Mathematical Proof of Non-Equivalence:**
   The omitted State block operations ($\text{RMSNorm}$, dense cross-channel projection $W_{\text{in}}$, non-linear $\text{SiLU}$, channel-wise Hadamard gating, dense projection $W_{\text{out}}$, and depthwise bias $b$) **cannot be mathematically folded** into the single linear 1D depthwise convolution $\sum w_k x_{t-k}$ executed by `nano_neon_short_conv_step`.
4. **Deterministic Micro-Reference Test Results:**
   - State Block Output Cosine Similarity: **0.9067** (Catastrophic divergence; expected > 0.9999).
   - Max Absolute Error: **2.1186**; Mean Relative Error: **168.57%**.
5. **Final Verdict:**
   **`FORMAT_INCOMPATIBLE_WITH_CURRENT_PYTORCH_GRAPH`**
   **`FIX-07-BLOCKED-FORMAT-INCOMPATIBLE`**
   Exporting `model.nano` with 123 descriptors is strictly BLOCKED. Doing so would produce an invalid model missing over 314 million trained parameters and generating nonsense outputs.

---

## 2. Mathematical Proof of Non-Equivalence (Folding Impossibility)

In PyTorch, the authoritative forward formulation for `ShortConvStateBlock` (`state_conv_block.py`) is:

$$\tilde{x} = \text{RMSNorm}(x, \gamma_{\text{norm}}) = \frac{x}{\sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2 + \epsilon}} \odot \gamma_{\text{norm}}$$

$$p = W_{\text{in}} \tilde{x} \in \mathbb{R}^{2D}, \quad \text{where } p = [g; v], \quad g \in \mathbb{R}^D, \; v \in \mathbb{R}^D$$

$$c_t = \sum_{k=0}^{K-1} w_k \odot v_{t-k} + b \in \mathbb{R}^D$$

$$m_t = \text{SiLU}(g_t) \odot c_t = \left(\frac{g_t}{1 + e^{-g_t}}\right) \odot c_t$$

$$y_t = W_{\text{out}} m_t \in \mathbb{R}^D$$

$$\text{Output}_t = x_t + y_t$$

In contrast, the native C++ implementation `nano_neon_short_conv_step` (`neon_state_update.cpp`) computes:

$$y_t^{\text{native}} = \sum_{k=0}^{K-1} w_k^{\text{native}} \odot x_{t-k}$$

$$\text{Output}_t^{\text{native}} = x_t + y_t^{\text{native}}$$

### Why Folding is Mathematically Impossible:
1. **Bilinear & Non-Linear Gating:** The term $\text{SiLU}(g_t) \odot c_t$ is quadratic in the projected inputs and non-linear. The native kernel computes a strictly linear combination of past $x$ vectors. No linear depthwise convolution can represent non-linear $\text{SiLU}$ modulation.
2. **Dense Channel Rank vs Diagonal Rank:** $W_{\text{in}} \in \mathbb{R}^{5120 \times 2560}$ and $W_{\text{out}} \in \mathbb{R}^{2560 \times 2560}$ perform full rank-2560 cross-channel transformation ($19,660,800$ weights per layer). The depthwise convolution filter has only $4 \times 2560 = 10,240$ weights with zero cross-channel mixing (diagonal rank 1 per channel). A full-rank linear map cannot be compressed into a rank-1 diagonal filter without losing $\ge 99.94\%$ of the channel transformation capacity.
3. **Degree-0 Homogeneity of RMSNorm:** $\text{RMSNorm}(\alpha x) = \text{sign}(\alpha) \text{RMSNorm}(x)$ is scale-invariant, whereas linear convolution is degree-1 homogeneous ($\text{conv}(\alpha x) = \alpha \text{conv}(x)$).

**Conclusion:** The claim that $W_{\text{in}}$, $W_{\text{out}}$, $\gamma_{\text{norm}}$, and $b$ are "folded" into $w_k$ is mathematically false. These weights were simply neglected during initial native engine scaffolding.

---

## 3. Deterministic Micro-Reference Test Results

Executing `scratch/run_fix07_comprehensive_audit.py` with deterministic input vectors $x \in \mathbb{R}^{1 \times 4 \times 2560}$:

| Test Component | Target Operation | Max Abs Error | Mean Abs Error | Cosine Similarity | Forensic Finding |
|---|---|---|---|---|---|
| **State Block** | PyTorch Full Block vs Current Native Engine | **2.118581** | **0.336306** | **0.906690** | **Catastrophic Divergence** |
| State Block RMSNorm | $x - \text{RMSNorm}(x)$ | 0.082104 | 0.009620 | 0.999812 | Unnormalized scale drift |
| State Block In-Proj | Channel expansion $2560 \to 5120$ | N/A | N/A | N/A | **20,480 omitted activations/token** |
| State Block Out-Proj | Channel compression $5120 \to 2560$ | N/A | N/A | N/A | **6,553,600 omitted weights/layer** |
| State Block Conv Bias | $b \in \mathbb{R}^{2560}$ | 0.691230 | 0.249020 | N/A | Bias systematically ignored |
| **FFN Block** | PyTorch FFN vs Native (omitted RMSNorm) | **0.010095** | **0.000963** | 0.999999 | Input scale degradation |
| **GQA Block** | PyTorch GQA vs Native (omitted RMSNorm) | **0.015904** | **0.002216** | 0.999996 | Attention score drift |

---

## 4. Quantization Audit

All linear weights in THSA-2B are trained in full precision (`bfloat16`) and mapped to runtime formats. An audit of the quantization routines in `tools/export_to_nano.py` demonstrates:

### A. Ternary Weight Quantization (2-bit Packed {-1, 0, +1})
- **Evaluated Tensor:** $W \in \mathbb{R}^{2560 \times 2560}$ (6,553,600 weights)
- **Original FP32 Distribution:** $\text{min} = -0.5469, \; \text{max} = 0.5409, \; \mu = -0.0000, \; \sigma = 0.1000$
- **Scale Factor $\gamma$ ($\text{mean}(|W|)$):** `0.079798`
- **Dequantized Distribution:** $\text{min} = -0.0798, \; \text{max} = 0.0798, \; \mu = 0.0000, \; \sigma = 0.0663$
- **Max Quantization Error:** `0.467093`
- **Mean Quantization Error:** `0.035721`
- **Integrity Status:** Ternary quantization is numerically valid and matches the 1.58-bit ternary architecture specification.

### B. Symmetric INT8 Quantization ([-127, +127])
- **Evaluated Tensor:** $W_{\text{embed}} \in \mathbb{R}^{1000 \times 2560}$ (2,560,000 weights)
- **Original FP32 Distribution:** $\text{min} = -0.2643, \; \text{max} = 0.2613, \; \mu = 0.0000, \; \sigma = 0.0500$
- **Dequantization Scale Factor:** `0.002081`
- **Dequantized Distribution:** $\text{min} = -0.2643, \; \text{max} = 0.2622, \; \mu = 0.0000, \; \sigma = 0.0500$
- **Max Quantization Error:** `0.001040`
- **Mean Quantization Error:** `0.000520`
- **Integrity Status:** High-precision INT8 representation with $< 0.1\%$ relative quantization noise.

---

## 5. Binary Format Audit (`.nano` v1)

| Parameter | Specification in `nano_types.h` | Exporter Implementation | Status |
|---|---|---|---|
| **Magic Header** | `NANO` (`0x4E414E4F`) | `MAGIC_NANO = b"NANO"` | PASS |
| **Header Byte Size** | Exactly 64 bytes (`#pragma pack(1)`) | Exactly 64 bytes (`struct.pack`) | PASS |
| **Descriptor Entry Size** | Exactly 32 bytes (`NanoTensorDescriptor`) | Exactly 32 bytes (`struct.pack("<IIQQfI")`) | PASS |
| **Payload Alignment** | 64-byte aligned (`NANO_ALIGN64`) | `align_to(..., 64)` | PASS |
| **Checksum Verification** | CRC32 over descriptors + payload | `zlib.crc32(desc + pad + payload)` | PASS |
| **Declared Tensor Count** | `header.tensor_count == 123` | Emits 123 descriptors | **BLOCKED (OOB for 219 graph)** |

---

## 6. Complete 219-Tensor Reconciliation Table

*Note: The table below classifies all 219 PyTorch tensors across the 24 backbone layers and root modules.*

| PyTorch Tensor Identifier | Shape | Dtype | Numel | .nano Representation | Current Nano ID | Native Pointer | Native Execution Site | Quantization | Status |
|---|---|---|---|---|---|---|---|---|---|
| `embed_tokens.weight` | `[65536, 2560]` | BF16/FP32 | 167,772,160 | `embed_tokens` | `0` | `ctx->embed_tokens_ptr` | `nano_engine.cpp:180` | INT8 (scale) | **SERIALIZED_DIRECT** |
| **Layer 0 (State)** | | | | | | | | | |
| `layers.0.mixer.norm.weight` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.0.mixer.in_proj.weight` | `[5120, 2560]` | BF16/FP32 | 13,107,200 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.0.mixer.conv1d.weight` | `[2560, 1, 4]` | BF16/FP32 | 10,240 | `layer_0_state_conv_w` | `1` | `ctx->layers[0].conv_weights` | `nano_engine.cpp:249` | FP32 | **SERIALIZED_DIRECT** |
| `layers.0.mixer.conv1d.bias` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None (passes nullptr) | None | **UNACCOUNTED_MISSING** |
| `layers.0.mixer.out_proj.weight` | `[2560, 2560]` | BF16/FP32 | 6,553,600 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.0.ffn.norm.weight` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.0.ffn.gate_proj.weight` | `[6912, 2560]` | BF16/FP32 | 17,694,720 | `layer_0_ffn_gate` | `2` | `ctx->layers[0].w_gate_packed` | `nano_engine.cpp:267` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.0.ffn.up_proj.weight` | `[6912, 2560]` | BF16/FP32 | 17,694,720 | `layer_0_ffn_up` | `3` | `ctx->layers[0].w_up_packed` | `nano_engine.cpp:268` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.0.ffn.down_proj.weight` | `[2560, 6912]` | BF16/FP32 | 17,694,720 | `layer_0_ffn_down` | `4` | `ctx->layers[0].w_down_packed` | `nano_engine.cpp:276` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| **Layer 1 (State)** | | | | | | | | | |
| `layers.1.mixer.norm.weight` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.1.mixer.in_proj.weight` | `[5120, 2560]` | BF16/FP32 | 13,107,200 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.1.mixer.conv1d.weight` | `[2560, 1, 4]` | BF16/FP32 | 10,240 | `layer_1_state_conv_w` | `5` | `ctx->layers[1].conv_weights` | `nano_engine.cpp:249` | FP32 | **SERIALIZED_DIRECT** |
| `layers.1.mixer.conv1d.bias` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.1.mixer.out_proj.weight` | `[2560, 2560]` | BF16/FP32 | 6,553,600 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.1.ffn.norm.weight` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.1.ffn.gate_proj.weight` | `[6912, 2560]` | BF16/FP32 | 17,694,720 | `layer_1_ffn_gate` | `6` | `ctx->layers[1].w_gate_packed` | `nano_engine.cpp:267` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.1.ffn.up_proj.weight` | `[6912, 2560]` | BF16/FP32 | 17,694,720 | `layer_1_ffn_up` | `7` | `ctx->layers[1].w_up_packed` | `nano_engine.cpp:268` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.1.ffn.down_proj.weight` | `[2560, 6912]` | BF16/FP32 | 17,694,720 | `layer_1_ffn_down` | `8` | `ctx->layers[1].w_down_packed` | `nano_engine.cpp:276` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| **Layer 2 (GQA)** | | | | | | | | | |
| `layers.2.mixer.norm.weight` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.2.mixer.q_proj.weight` | `[2560, 2560]` | BF16/FP32 | 6,553,600 | `layer_2_attn_q` | `9` | `ctx->layers[2].w_q_packed` | `nano_engine.cpp:203` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.2.mixer.k_proj.weight` | `[512, 2560]` | BF16/FP32 | 1,310,720 | `layer_2_attn_k` | `10` | `ctx->layers[2].w_k_packed` | `nano_engine.cpp:204` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.2.mixer.v_proj.weight` | `[512, 2560]` | BF16/FP32 | 1,310,720 | `layer_2_attn_v` | `11` | `ctx->layers[2].w_v_packed` | `nano_engine.cpp:205` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.2.mixer.out_proj.weight` | `[2560, 2560]` | BF16/FP32 | 6,553,600 | `layer_2_attn_out` | `12` | `ctx->layers[2].w_out_packed` | `nano_engine.cpp:239` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.2.ffn.norm.weight` | `[2560]` | BF16/FP32 | 2,560 | Missing | None | None | None | None | **UNACCOUNTED_MISSING** |
| `layers.2.ffn.gate_proj.weight` | `[6912, 2560]` | BF16/FP32 | 17,694,720 | `layer_2_ffn_gate` | `13` | `ctx->layers[2].w_gate_packed` | `nano_engine.cpp:267` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.2.ffn.up_proj.weight` | `[6912, 2560]` | BF16/FP32 | 17,694,720 | `layer_2_ffn_up` | `14` | `ctx->layers[2].w_up_packed` | `nano_engine.cpp:268` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| `layers.2.ffn.down_proj.weight` | `[2560, 6912]` | BF16/FP32 | 17,694,720 | `layer_2_ffn_down` | `15` | `ctx->layers[2].w_down_packed` | `nano_engine.cpp:276` | Ternary 2-bit | **SERIALIZED_DIRECT** |
| *(Layers 3 to 23 follow identical patterns)* | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **Root Modules** | | | | | | | | | |
| `final_norm.weight` | `[2560]` | BF16/FP32 | 2,560 | `final_norm` | `121` | `ctx->final_norm_gamma` | `nano_engine.cpp:291` | FP32 | **SERIALIZED_DIRECT** |
| `lm_head.weight` | `[65536, 2560]` | BF16/FP32 | 167,772,160 | `lm_head` | `122` | `ctx->lm_head_ptr` | `nano_engine.cpp:302` | INT8 (scale) | **SERIALIZED_DIRECT** |

### Accounting Summary Across All 219 Tensors:
- **`SERIALIZED_DIRECT`**: **123 tensors** (1,735,559,680 parameters, 84.65%)
- **`UNACCOUNTED_MISSING`**: **96 tensors** (314,736,640 parameters, 15.35%)
- **`FOLDED_WITH_PROOF`**: **0** (Folding proven mathematically impossible)
- **`DERIVED_WITH_PROOF`**: **0**
- **`INTENTIONALLY_UNUSED_WITH_PROOF`**: **0**

Because 96 tensors end in `UNACCOUNTED_MISSING`, the audit produces **FAIL** according to mandatory accounting rules.

---

## 7. Compilation Verification

Both host and ARM64 target compilations were verified after correcting CMake definitions:
1. **ARM64 Native Compilation (Android NDK r27 + Clang 18 for `arm64-v8a`):**
   - Output: `libnano_engine.so`, `libnano_engine_static.a`, `test_neon_kernels`, `test_native_model_loader`, `test_neural_forward_pass`
   - **Result: PASS (Exit Code 0)**
2. **Host Native Compilation (Windows MSVC x64 + Ninja):**
   - Output: `nano_engine.dll`, `nano_engine_static.lib`, `test_neon_kernels.exe`, `test_native_model_loader.exe`, `test_neural_forward_pass.exe`
   - **Result: PASS (Exit Code 0)**
   - Test execution `test_neon_kernels.exe`: `5/5 tests passed (100% verified)`

---

## 8. Required Architectural Remediation Plan (To Unblock FIX-07)

To achieve true bitwise and structural fidelity between PyTorch and `.nano`, the format and runtime must be updated:

### 1. Upgrade `.nano` Format to 219 Descriptors (`format_version = 2`)
- Update `tensor_count` from 123 to **219**.
- Add the missing 96 descriptors to `tools/export_to_nano.py`:
  - 16 State Block RMSNorm weights (`layer_{l}_state_norm`, FP32)
  - 16 State Block In-Projection weights (`layer_{l}_state_in_proj`, Ternary)
  - 16 State Block Conv1D Biases (`layer_{l}_state_conv_b`, FP32)
  - 16 State Block Out-Projection weights (`layer_{l}_state_out_proj`, Ternary)
  - 8 GQA Block RMSNorm weights (`layer_{l}_attn_norm`, FP32)
  - 24 FFN Block RMSNorm weights (`layer_{l}_ffn_norm`, FP32)

### 2. Update `struct NanoLayerPointers` in `nano_engine.cpp`
```cpp
struct NanoLayerPointers {
    bool           is_gqa;
    // GQA Block
    const float*   gamma_attn;       // RMSNorm
    const uint8_t* w_q_packed;
    float          scale_q;
    const uint8_t* w_k_packed;
    float          scale_k;
    const uint8_t* w_v_packed;
    float          scale_v;
    const uint8_t* w_out_packed;
    float          scale_out;

    // State Block
    const float*   gamma_state;      // RMSNorm
    const uint8_t* w_state_in_proj;  // In-projection [5120, 2560]
    float          scale_state_in;
    const float*   conv_weights;     // [4, 2560]
    const float*   conv_bias;        // [2560]
    const uint8_t* w_state_out_proj; // Out-projection [2560, 2560]
    float          scale_state_out;

    // FFN Block
    const float*   gamma_ffn;        // RMSNorm
    const uint8_t* w_gate_packed;
    float          scale_gate;
    const uint8_t* w_up_packed;
    float          scale_up;
    const uint8_t* w_down_packed;
    float          scale_down;
};
```

### 3. Implement the Complete `ShortConvStateBlock` Forward Pass in Native C++
In `nano_forward_pass_single_token`:
1. Execute `nano_neon_rmsnorm(ctx->h_state, lp.gamma_state, 2560, ctx->norm_out)`.
2. Compute `in_proj`: $2560 \to 5120$ via `nano_neon_gemv_ternary_int8`.
3. Chunk into `gate` [2560] and `value` [2560].
4. Run depthwise convolution on `value` with `conv_weights` and `conv_bias`.
5. Compute gated SiLU activation: $\text{SiLU}(\text{gate}) \odot \text{conv\_out}$.
6. Project back: $5120 \to 2560$ via `out_proj`.
7. Add to residual `h_state`.
8. Apply `gamma_ffn` RMSNorm before FFN.

---

## 9. Final Forensic Verdict

```
================================================================================
FIX-07 FORENSIC RECONCILIATION VERDICT
================================================================================

FORMAT COMPATIBILITY STATUS:
FORMAT_INCOMPATIBLE_WITH_CURRENT_PYTORCH_GRAPH

TOTAL CHECKPOINT TRAINABLE TENSORS:
219 / 219 (2,050,296,320 parameters)

CURRENT .NANO FORMAT DESCRIPTORS:
123 / 219 (1,735,559,680 parameters)

MISSING TENSORS / PARAMETERS:
96 tensors / 314,736,640 parameters (15.35% of model omitted)

NUMERICAL DIVERGENCE (STATE BLOCK):
Cosine Similarity: 0.906690 (Catastrophic Error, Expected >= 0.9999)

FINAL AUDIT VERDICT:
FIX-07-BLOCKED-FORMAT-INCOMPATIBLE

================================================================================
```
