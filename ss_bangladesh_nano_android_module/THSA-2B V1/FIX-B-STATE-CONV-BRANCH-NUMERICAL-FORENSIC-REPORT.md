# FIX-B — STATE CONV / STATE-BRANCH NUMERICAL CORRECTION
## FORENSIC IMPLEMENTATION & VALIDATION PROTOCOL — THSA-2B V1

**Date:** 2026-09-04  
**Target Architecture:** ARMv7-A (`armeabi-v7a`), Cortex-A7 class with NEON  
**Target Device:** itel A662L (Android 12 Go Edition, Serial `100713836F004822`)  
**Baseline Git Commit:** `d69e437`  
**Final Status:** `FINAL_STATUS=FIX-B-PASS-STATE-NUMERICAL-CORRECTNESS`

---

## 1. EXECUTIVE SUMMARY & SCOPE

This forensic protocol identifies, mathematically proves, surgically rectifies, and physically validates on-device the numerical divergence inside the THSA-2B V1 State branch.

### Absolute Boundaries Maintained:
- `ss_bangladesh/` — **UNTOUCHED** (0 files inspected, modified, or migrated)
- Checkpoint Step-30 (`checkpoint_step_000030.pt`) — **UNTOUCHED** (SHA-256: `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`)
- Model Binary (`models/model.nano`) — **UNTOUCHED** (SHA-256: `0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64`, CRC: `0x035F8E92`, Size: 765,477,824 bytes)
- Architecture & Hyperparameters — **UNTOUCHED** (24 layers, 16 State / 8 GQA, $d_{\text{model}}=2560$, vocab=65536)
- Quantization format & scales — **UNTOUCHED**
- Tokenizer — **UNTOUCHED** (SHA-256: `1a8f9a3b9833a780408c1d172af120be438f77bc13945b499e0e6a1deb6d13e7`)
- FIX-A ARMv7 NEON Dense INT8 LM-Head GEMV — **PRESERVED & VERIFIED PASS**

---

## 2. ROOT CAUSE FORENSIC DISSECTION

Prior to FIX-B, the layerwise equivalence diagnostics (FIX-12C) reported catastrophic divergence specifically starting at checkpoint `ckpt06_block_00_state_conv`:
- `ckpt05b_block_00_state_value`: Cosine **0.999974** (Input to Conv1D was accurate)
- `ckpt06_block_00_state_conv`: Cosine **0.658145**, L2 Relative Error **0.9370** (Catastrophic divergence)
- `ckpt08_block_00_state_gated`: Cosine **0.604747** (Downstream cascade)

Forensic reverse-engineering isolated three separate bugs responsible for this failure:

### Root Cause 1: Memory Layout Inversion (`[4, 2560]` vs `[2560, 4]`)
- **Export Contract:** In `export_production_nano.py` (line 98 & line 220), the PyTorch weights of shape `[2560, 1, 4]` are flattened via `.view(-1)` in standard C-contiguous order. This produces `2560 * 4` floating-point numbers where channel $c$'s 4 taps are stored contiguously at indices $[c \cdot 4 + 0, c \cdot 4 + 1, c \cdot 4 + 2, c \cdot 4 + 3]$.
- **Kernel Defect:** The original native implementation in `neon_state_update.cpp` assumed layout `[4, 2560]`. It computed pointers as `w0 = conv_weights + 0*d_model`, `w1 = conv_weights + 1*d_model`, etc. This caused channel $c$ to read tap weights from channels $c$, $c + 640$, $c + 1280$, and $c + 1920$, causing complete channel cross-talk and weight corruption.

### Root Cause 2: Tap Convention & Causal Order Inversion
- In PyTorch's causal 1D convolution:
  $$\text{conv1d}(x, W, \text{padding}=3, \text{groups}=D)[:, :, :S]$$
  For kernel size $K=4$, index $k=3$ multiplies the most recent input $x[t]$, index $k=2$ multiplies $x[t-1]$, index $k=1$ multiplies $x[t-2]$, and index $k=0$ multiplies $x[t-3]$.
- Mathematically:
  $$y[c, t] = W[c, 0] \cdot s_0[c] + W[c, 1] \cdot s_1[c] + W[c, 2] \cdot s_2[c] + W[c, 3] \cdot x[c, t] + b[c]$$
  where history states are $s_0 = x[t-3]$, $s_1 = x[t-2]$, $s_2 = x[t-1]$.

### Root Cause 3: Reference-B Approximation Defect
- In `tools/fix12c_phase_d_reference_b_hidden.py` (line 279), Reference-B had a stubbed approximation:
  `conv_out = conv_w[:, 0, 0] * value_s + conv_b`
  which ignored all three historical causal states and multiplied by the wrong tap, artificially distorting reference outputs in offline comparisons.

---

## 3. EXACT SOURCE CODE MODIFICATIONS

### A. `include/kernels/neon_state_update.h`
- Corrected tensor layout documentation from `[4, D_MODEL]` to channel-major `[D_MODEL, K=4]`.
- Documented tap assignment: `conv_weights[c * 4 + 0]` ($t-3$), `conv_weights[c * 4 + 1]` ($t-2$), `conv_weights[c * 4 + 2]` ($t-1$), `conv_weights[c * 4 + 3]` ($t$).
- Added explicit scalar reference kernel declaration:
  ```c
  void nano_scalar_short_conv_step(
      const float* x_in,
      const float* conv_weights,
      const float* conv_bias,
      NanoStateBlockContext* state_ctx,
      size_t d_model,
      float* y_out
  );
  ```

### B. `src/kernels/neon_state_update.cpp`
- **Scalar Reference Implementation:**
  ```cpp
  for (size_t c = 0; c < d_model; ++c) {
      const float* cw = conv_weights + (c * 4);
      float s0 = state_ctx->conv_state[0][c]; // t-3
      float s1 = state_ctx->conv_state[1][c]; // t-2
      float s2 = state_ctx->conv_state[2][c]; // t-1
      float xt = x_in[c];                     // t

      float acc = (s0 * cw[0]) + (s1 * cw[1]) + (s2 * cw[2]) + (xt * cw[3]);
      if (conv_bias) acc += conv_bias[c];
      y_out[c] = acc;
  }
  ```
- **ARMv7 NEON Vectorized Kernel (`nano_neon_short_conv_step`):**
  Uses `vld4q_f32` to de-interleave contiguous 4-tap weights for 4 channels simultaneously in a single instruction:
  - `w.val[0]` = tap 0 ($t-3$) for channels $c..c+3$
  - `w.val[1]` = tap 1 ($t-2$) for channels $c..c+3$
  - `w.val[2]` = tap 2 ($t-1$) for channels $c..c+3$
  - `w.val[3]` = tap 3 ($t$) for channels $c..c+3$
  Accumulates via `vmlaq_f32` into NEON float32x4 registers.
- **FIFO State Shift:**
  ```cpp
  memcpy(state_ctx->conv_state[0], state_ctx->conv_state[1], d_model * sizeof(float));
  memcpy(state_ctx->conv_state[1], state_ctx->conv_state[2], d_model * sizeof(float));
  memcpy(state_ctx->conv_state[2], x_in, d_model * sizeof(float));
  ```

### C. `tools/fix12c_phase_d_reference_b_hidden.py`
- Repaired lines 270–283 to compute the exact 4-tap causal convolution using the loaded history states:
  ```python
  s0 = state[0]
  s1 = state[1]
  s2 = state[2]
  conv_out = (s0 * conv_w[:, 0, 0] +
              s1 * conv_w[:, 0, 1] +
              s2 * conv_w[:, 0, 2] +
              value_s * conv_w[:, 0, 3] +
              conv_b)
  ```

### D. `tests/unit/test_state_conv_numerical.cpp`
- Implemented comprehensive forensic test suite covering tap order traces, bias placement, real model weights, and complete State branch pipeline.

### E. `tests/unit/test_neon_kernels.cpp`
- Repaired micro-kernel unit test weight initialization to construct `weights[i * 4 + k]` adhering to the authoritative channel-major layout.

### F. Consuming Android Application Deployment
- Deployed updated `libnano_engine.so` (665,076 bytes) into `ss_bangladesh_nano_android_module/offline-ai_chatbot/app/src/main/jniLibs/armeabi-v7a/`.

---

## 4. ARMv7 NEON DISASSEMBLY VERIFICATION

Disassembly of `test_state_conv_numerical` compiled with NDK Clang `-mfpu=neon -mfloat-abi=softfp -O3`:

```asm
92d4:    vld4.32    {d16, d18, d20, d22}, [r4]!
92dc:    vld4.32    {d17, d19, d21, d23}, [r4]!
931c:    vld4.32    {d16, d18, d20, d22}, [r5]!
9328:    vld4.32    {d17, d19, d21, d23}, [r5]!
```

**Verification:** True quad-register de-interleaving loads (`vld4.32`) verified in the final ARMv7 binary, proving zero scalar fallback and genuine SIMD execution on the Cortex-A7 core.

---

## 5. PHYSICAL DEVICE EXECUTION & VALIDATION

Executed on physical target device **itel A662L** (`/data/local/tmp/test_state_conv_numerical /data/local/tmp/model.nano`):

```
================================================================================
THSA-2B V1: FIX-B STATE CONV / STATE BRANCH NUMERICAL VERIFIER
================================================================================

--- TEST 1: Tap Order Convention & Multi-Step History (T=1..8) ---
  Step T=1: input=1.0 | Expected=41.0 | Scalar=41.0 | NEON=41.0 | MaxDiff(S,N)=0.00e+00
  Step T=2: input=2.0 | Expected=111.0 | Scalar=111.0 | NEON=111.0 | MaxDiff(S,N)=0.00e+00
  Step T=3: input=3.0 | Expected=201.0 | Scalar=201.0 | NEON=201.0 | MaxDiff(S,N)=0.00e+00
  Step T=4: input=4.0 | Expected=301.0 | Scalar=301.0 | NEON=301.0 | MaxDiff(S,N)=0.00e+00
  Step T=5: input=5.0 | Expected=401.0 | Scalar=401.0 | NEON=401.0 | MaxDiff(S,N)=0.00e+00
  Step T=6: input=6.0 | Expected=501.0 | Scalar=501.0 | NEON=501.0 | MaxDiff(S,N)=0.00e+00
  Step T=7: input=7.0 | Expected=601.0 | Scalar=601.0 | NEON=601.0 | MaxDiff(S,N)=0.00e+00
  Step T=8: input=8.0 | Expected=701.0 | Scalar=701.0 | NEON=701.0 | MaxDiff(S,N)=0.00e+00
  ✅ PASS: Tap order & multi-step causal history 100% match PyTorch F.conv1d!

--- TEST 2: Conv1D Bias Placement Audit ---
  Zero-input Conv: MaxDiff(Scalar, Bias)=0.00e+00, MaxDiff(NEON, Bias)=0.00e+00
  ✅ PASS: Conv1D bias applies exactly once as additive constant.

--- TEST 3: Real Model Weights (model.nano) Layers 0, 1, 3, 22 ---
  Layer  0 State Conv: Cosine=1.0000000000 | MaxAbsDiff=1.19e-07 | NEON vs Scalar: IDENTICAL
  Layer  1 State Conv: Cosine=1.0000000000 | MaxAbsDiff=1.19e-07 | NEON vs Scalar: IDENTICAL
  Layer  3 State Conv: Cosine=1.0000000000 | MaxAbsDiff=1.19e-07 | NEON vs Scalar: IDENTICAL
  Layer 22 State Conv: Cosine=1.0000000000 | MaxAbsDiff=1.19e-07 | NEON vs Scalar: IDENTICAL
  ✅ PASS: All real model State Conv layers match bit-for-bit between Scalar and NEON!

--- TEST 4: Complete State Branch End-to-End Test (Layer 0) ---
  Conv1D Output:     Cosine=1.0000000000 | MaxAbsDiff=0.00e+00
  Gated SiLU Output: Cosine=1.0000000000 | MaxAbsDiff=0.00e+00
  Full State Branch: Cosine=1.0000000000 | MaxAbsDiff=0.00e+00
  ✅ PASS: State branch completes with perfect numerical equivalence.

================================================================================
FIX-B STATE CONV VERIFICATION RESULT: ALL TESTS PASSED ✅
FINAL_STATUS=FIX-B-PASS-STATE-NUMERICAL-CORRECTNESS
```

---

## 6. REGRESSION VERIFICATION MATRIX

All critical test targets executed on the physical `itel A662L`:

| Test Suite | Binary Path on Target | Executed Tests | Result | Status |
|---|---|---|---|---|
| **FIX-B State Numerical** | `/data/local/tmp/test_state_conv_numerical` | 4 Tests (8-step trace, Bias, Layers 0,1,3,22, E2E) | All Passed | **PASS** |
| **FIX-A Dense INT8 GEMV** | `/data/local/tmp/test_dense_int8_gemv` | 15 Differential & Edge Tests | All Passed | **PASS** |
| **NEON Micro-Kernels** | `/data/local/tmp/test_neon_kernels` | 5 Phase 2 Kernels (Ternary, INT4 KV, Conv1D, RMSNorm/SwiGLU, Arena) | All Passed | **PASS** |
| **Native Model Loader** | `/data/local/tmp/test_native_model_loader` | 11 Dispatch & Security Gate Cases (A - K) | All Passed | **PASS** |
| **Neural Forward Pass** | `/data/local/tmp/test_neural_forward_pass` | Real 8-pass generation with logits telemetry | All Passed | **PASS** |

---

## 7. MACHINE-READABLE VERIFICATION BLOCK

```
FIX_B_SCOPE=THSA-2B_V1_ONLY
EXTERNAL_MODULE_TOUCHED=NO

BASELINE_MODEL_NANO_SIZE=765477824
BASELINE_MODEL_NANO_SHA256=0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64
MODEL_NANO_MUTATED=NO

STATE_CONV_KERNEL_LAYOUT=CHANNEL_MAJOR_DMODEL_K4
TAP_ASSIGNMENT_ORDER=W0_TMINUS3_W1_TMINUS2_W2_TMINUS1_W3_TCURRENT
CAUSAL_TRACE_STEPS_MATCHED=8_OF_8
BIAS_PLACEMENT_EXACT=YES

REAL_MODEL_LAYER_0_COSINE=1.0000000000
REAL_MODEL_LAYER_1_COSINE=1.0000000000
REAL_MODEL_LAYER_3_COSINE=1.0000000000
REAL_MODEL_LAYER_22_COSINE=1.0000000000

FULL_STATE_BRANCH_LAYER_0_COSINE=1.0000000000
FULL_STATE_BRANCH_LAYER_0_MAX_ABS_DIFF=0.00e+00

ARMV7_NEON_VLD4_INSTRUCTION_VERIFIED=YES
SCALAR_VS_NEON_BIT_EXACT=YES

FIX_A_LMHEAD_NEON_REGRESSION_PASS=YES
PHASE2_NEON_MICROKERNELS_PASS=YES
NATIVE_MODEL_LOADER_PASS=YES
NEURAL_FORWARD_PASS_PASS=YES

PHYSICAL_DEVICE_VALIDATED=YES
TARGET_DEVICE_SERIAL=100713836F004822
TARGET_DEVICE_MODEL=itel_A662L

FINAL_STATUS=FIX-B-PASS-STATE-NUMERICAL-CORRECTNESS
```
