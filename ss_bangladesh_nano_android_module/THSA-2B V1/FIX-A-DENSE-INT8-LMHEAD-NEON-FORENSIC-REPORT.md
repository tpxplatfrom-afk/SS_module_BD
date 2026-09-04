# FIX-A — ARMv7 NEON DENSE INT8 LM-HEAD GEMV
## FORENSIC PERFORMANCE REPORT — THSA-2B V1

**Date:** 2026-09-04  
**Target Device:** itel A662L (Android 12 Go, armeabi-v7a, Cortex-A7 class)  
**Baseline Git Commit:** `d69e437`  
**Final Status:** `FINAL_STATUS=FIX-A-PASS-DENSE-INT8-LMHEAD-NEON`

---

## 1. SCOPE

This optimization surgically replaces the scalar 8-way-unrolled INT8 LM-head GEMV in THSA-2B V1 with a true ARMv7 Advanced SIMD (NEON) micro-kernel while preserving the existing numerical contract exactly.

### Absolute Boundaries Confirmed:
- `ss_bangladesh/` — **UNTOUCHED** (0 files inspected, modified, or migrated)
- Checkpoint `step-30` — **UNTOUCHED**
- Tokenizer — **UNTOUCHED**
- Binary format (Nano V2, 219 tensors, 64-byte alignment) — **UNTOUCHED**
- `models/model.nano` (765,477,824 bytes, SHA-256 `0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64`) — **UNTOUCHED**
- Model architecture (24 layers: 16 State / 8 GQA, d_model=2560, vocab=65536) — **UNTOUCHED**
- INT8 Quantizer (`nano_neon_quantize_int8`) — **UNTOUCHED**
- RMSNorm, State, GQA, KV-cache, SwiGLU, JNI API — **UNTOUCHED**

---

## 2. BASELINE COMMIT

```
commit d69e437 chore(fix13): add on-device benchmark logcat capture files
```
Audit confirmed that prior to this fix:
- `DENSE_INT8_GEMV_SOURCE_FOUND = NO`
- `DENSE_INT8_GEMV_LINKED = NO`
- `DENSE_INT8_GEMV_CALLED_BY_LMHEAD = NO`
- `LMHEAD_CURRENT_IMPLEMENTATION = SCALAR (8-way unrolled)`

---

## 3. EXACT FILES MODIFIED

| File Path | Action | Rationale / Justification |
|---|---|---|
| `include/kernels/neon_gemv_int8.h` | **NEW** | API definition for NEON kernel and persistent scalar reference. |
| `src/kernels/neon_gemv_int8.cpp` | **NEW** | Production ARMv7 NEON dense INT8 kernel (`vmull_s8`, `vpaddlq_s16`, `vaddq_s32`, `vpadd_s32`) + scalar reference. |
| `src/engine/nano_engine.cpp` | **MODIFY** | Replaced scalar LM-head loop with `nano_neon_gemv_dense_int8(...)`. Surrounding RMSNorm, quantization, scales untouched. |
| `CMakeLists.txt` | **MODIFY** | Added `neon_gemv_int8.cpp` to build target; enabled `-mfpu=neon -mfloat-abi=softfp` for `armeabi-v7a`. |
| `tests/unit/test_dense_int8_gemv.cpp` | **NEW** | Deterministic unit test covering edge cases and synthetic matrix dimensions. |
| `tests/unit/test_real_model_lmhead_neon.cpp` | **NEW** | End-to-end on-device verifier and benchmark using Tensor 218 from `model.nano`. |
| `offline-ai_chatbot/.../libnano_engine.so` | **DEPLOY** | Deployed newly compiled 663,840-byte production `.so` into consuming chatbot app. |

---

## 4. EXISTING SCALAR LM-HEAD IMPLEMENTATION

Before FIX-A, `src/engine/nano_engine.cpp` (lines 740–759) executed:
```cpp
for (size_t v = 0; v < 65536; ++v) {
    const int8_t* lm_row = ctx->lm_head_ptr + (v * 2560);
    int32_t dot = 0;
    size_t d = 0;
    for (; d + 8 <= 2560; d += 8) {
        dot += (int32_t)ctx->h_state_int8[d + 0] * (int32_t)lm_row[d + 0]
             + (int32_t)ctx->h_state_int8[d + 1] * (int32_t)lm_row[d + 1]
             + (int32_t)ctx->h_state_int8[d + 2] * (int32_t)lm_row[d + 2]
             + (int32_t)ctx->h_state_int8[d + 3] * (int32_t)lm_row[d + 3]
             + (int32_t)ctx->h_state_int8[d + 4] * (int32_t)lm_row[d + 4]
             + (int32_t)ctx->h_state_int8[d + 5] * (int32_t)lm_row[d + 5]
             + (int32_t)ctx->h_state_int8[d + 6] * (int32_t)lm_row[d + 6]
             + (int32_t)ctx->h_state_int8[d + 7] * (int32_t)lm_row[d + 7];
    }
    for (; d < 2560; ++d) {
        dot += (int32_t)ctx->h_state_int8[d] * (int32_t)lm_row[d];
    }
    ctx->logits[v] = (float)dot * combined_scale;
}
```
This loop spent 522.64 ms per forward pass doing pure scalar computation across 167,772,160 integer multiplications.

---

## 5. NEW KERNEL API

Defined in `include/kernels/neon_gemv_int8.h`:
```cpp
void nano_neon_gemv_dense_int8(
    const int8_t* weights,        // Row-major [rows * cols]
    const int8_t* activation,     // Contiguous [cols]
    float*        output,         // Contiguous [rows]
    size_t        rows,           // 65536
    size_t        cols,           // 2560
    float         combined_scale  // norm_scale * lm_head_scale
);

void nano_scalar_gemv_dense_int8_reference(
    const int8_t* weights,
    const int8_t* activation,
    float*        output,
    size_t        rows,
    size_t        cols,
    float         combined_scale
);
```

---

## 6. ARMv7 NEON STRATEGY & OVERFLOW SAFETY

### Vectorized Inner Loop (16 elements per step):
- Vector loads: `vld1_s8` loads 8 INT8 activation elements and 8 INT8 weight elements.
- Multiply-widen: `vmull_s8` multiplies 8x signed INT8 into 8x signed INT16 without loss of precision.
- Pairwise accumulation: `vpaddlq_s16` accumulates adjacent INT16 products into 4x signed INT32.
- Vector sum: `vaddq_s32` accumulates into quadword 32-bit registers (`int32x4_t`).
- Horizontal reduction (ARMv7 compatible):
  ```cpp
  int32x2_t sum_half = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
  sum_half = vpadd_s32(sum_half, sum_half);
  int32_t dot = vget_lane_s32(sum_half, 0);
  ```
- Scale & Output: `output[v] = (float)dot * combined_scale;` (performed via `vcvt.f32.s32` + `vmul.f32`).

### 4-Row Blocking:
Outer loop processes 4 vocabulary rows simultaneously (`v += 4`), allowing activation data to stay hot in Cortex-A7 L1 cache (16 KB) while streaming through row-major weight chunks.

### Overflow Bound Proof:
- Maximum INT8 value = 127, minimum = -128.
- Max product magnitude = $128 \times 127 = 16,256$.
- Inner dimension $K = 2560$.
- Max accumulated sum $= 2560 \times 16,256 = 41,615,360$.
- Range of signed `int32_t` $= [-2,147,483,648, +2,147,483,647]$.
- Safety factor: $\frac{2,147,483,647}{41,615,360} \approx 51.6\times$ headroom. **Zero risk of overflow.**

---

## 7. NUMERICAL CONTRACT CONFORMANCE

- **Weights:** Directly pointed at `ctx->lm_head_ptr` from memory-mapped `model.nano`. No copying, transposing, or repacking.
- **Activation:** Directly pointed at `ctx->h_state_int8` output by unmodified `nano_neon_quantize_int8`.
- **Scaling:** `combined_scale = norm_scale * lm_head_scale` computed prior to call.
- **Integer Accumulation:** 100% pure INT32 integer accumulation throughout. FP32 scale applied strictly once at final store.

---

## 8. SCALAR REFERENCE CONTRACT

The original scalar 8-way unrolled implementation was moved verbatim to `nano_scalar_gemv_dense_int8_reference(...)` in `src/kernels/neon_gemv_int8.cpp`. It is permanently linked and exported for differential testing on both device and host.

---

## 9. NEON VS SCALAR DETERMINISTIC TEST RESULTS

Executed natively on physical device (`itel A662L`, Android 12 Go, armeabi-v7a):
```
=== FIX-A: Dense INT8 GEMV Differential Test ===
Platform: ARM NEON available — NEON kernel active

PASS [A_small_4x32] rows=4 cols=32 scale=0.001234  int32_exact=YES  fp32_max_diff=0
PASS [B_medium_16x256] rows=16 cols=256 scale=0.000567  int32_exact=YES  fp32_max_diff=0
PASS [C_prod_width_8x2560] rows=8 cols=2560 scale=0.000321  int32_exact=YES  fp32_max_diff=0
PASS [D_vocab_batch_128x2560] rows=128 cols=2560 scale=0.000212  int32_exact=YES  fp32_max_diff=0
  [E_full_65536x2560] Allocating 160.0 MB weights...
PASS [E_full_65536x2560] rows=65536 cols=2560 scale=0.000180  int32_exact=YES  fp32_max_diff=0

=== Edge Cases ===
PASS [Edge1_zero_activation] rows=8 cols=2560 scale=0.001000  int32_exact=YES  fp32_max_diff=0
PASS [Edge2_zero_weights] rows=8 cols=2560 scale=0.001000  int32_exact=YES  fp32_max_diff=0
PASS [Edge3_act_plus127] rows=8 cols=2560 scale=0.001000  int32_exact=YES  fp32_max_diff=0
PASS [Edge4_act_minus128] rows=8 cols=2560 scale=0.001000  int32_exact=YES  fp32_max_diff=0
PASS [Edge5_mixed_random] rows=16 cols=2560 scale=0.000999  int32_exact=YES  fp32_max_diff=0
PASS [Edge6_alternating_signs] rows=8 cols=2560 scale=0.001000  int32_exact=YES  fp32_max_diff=0
PASS [Edge7_max_magnitude] rows=8 cols=2560 scale=0.001000  int32_exact=YES  fp32_max_diff=0
PASS [Edge8_random_seed2] rows=32 cols=2560 scale=0.000765  int32_exact=YES  fp32_max_diff=0
PASS [Edge9_rows_not_aligned_5x2560] rows=5 cols=2560 scale=0.001000  int32_exact=YES  fp32_max_diff=0
PASS [Edge10_cols_tail_8x33] rows=8 cols=33 scale=0.002000  int32_exact=YES  fp32_max_diff=0

=== SUMMARY ===
Tests run: 15 | Passed: 15 | Failed: 0
TEST_DENSE_INT8_GEMV=PASS
INT32_DOT_EXACT=YES
FP32_OUTPUT_EXACT=YES
```

---

## 10. REAL MODEL TENSOR-218 VALIDATION

Tested directly against the authoritative production `model.nano` (SHA-256 `0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64`):
- **Tensor ID:** 218 (`lm_head`)
- **Offset:** `597,705,664`
- **Size:** `167,772,160` bytes ($65,536 \times 2,560$)
- **Quant Type:** Signed INT8 (`NANO_QUANT_INT8 = 2`)
- **Scale:** `0.00029220`
- **Rows tested:** 65,536 (entire vocabulary)

---

## 11. FULL 65,536-LOGIT DIFFERENTIAL COMPARISON

Tested on `itel A662L` using realistic simulated quantized activation across all 65,536 vocabulary rows:

```
=== NUMERICAL EQUIVALENCE REPORT (65,536 VOCABULARY ROWS) ===
  INT32 Dot Mismatches: 0 / 65536 (EXACT EQUALITY)
  Max Absolute Error:   0.00000000e+00
  Mean Absolute Error:  0.00000000e+00
  RMSE:                 0.00000000e+00
  Cosine Similarity:    1.0000000000
  Top-1 Match:          EXACT (ID 64238 vs 64238, logit 0.75834 vs 0.75834)
  Top-5 Match:          EXACT
  Top-10 Match:         EXACT
```

### Top-10 Ranking Verification:
| Rank | Scalar Token ID (Logit) | NEON Token ID (Logit) | Differential Status |
|---|---|---|---|
| 1 | Token 64238 (0.75834) | Token 64238 (0.75834) | **MATCH (EXACT)** |
| 2 | Token 65489 (0.68392) | Token 65489 (0.68392) | **MATCH (EXACT)** |
| 3 | Token 58069 (0.65889) | Token 58069 (0.65889) | **MATCH (EXACT)** |
| 4 | Token 23433 (0.65860) | Token 23433 (0.65860) | **MATCH (EXACT)** |
| 5 | Token 2062  (0.64547) | Token 2062  (0.64547) | **MATCH (EXACT)** |
| 6 | Token 37575 (0.64363) | Token 37575 (0.64363) | **MATCH (EXACT)** |
| 7 | Token 34474 (0.64355) | Token 34474 (0.64355) | **MATCH (EXACT)** |
| 8 | Token 27917 (0.64333) | Token 27917 (0.64333) | **MATCH (EXACT)** |
| 9 | Token 41541 (0.64298) | Token 41541 (0.64298) | **MATCH (EXACT)** |
| 10| Token 42183 (0.63065) | Token 42183 (0.63065) | **MATCH (EXACT)** |

---

## 12. DISASSEMBLY EVIDENCE & MACHINE CODE PROOF

Disassembly of `libnano_engine.so` (ELF32 little-endian ARM, Thumb-2 with NEON):
```asm
0001570c <nano_neon_gemv_dense_int8>:
   1570c: push {r4, r5, r6, r7, lr}
   ...
   1578a: vld1.8   {d18}, [r0]!       ; Load 8 int8 weights
   1578e: vld1.8   {d19}, [r4]!       ; Load 8 int8 activations
   15792: vmull.s8 q9, d19, d18       ; Signed 8x8 -> 16-bit multiply
   15796: vld1.8   {d20}, [r0]        ; Load next 8 int8 weights
   157a2: vld1.8   {d21}, [r4]        ; Load next 8 int8 activations
   157a6: vmull.s8 q10, d21, d20      ; Signed 8x8 -> 16-bit multiply
   157aa: vpaddl.s16 q9, q9           ; Pairwise add int16 -> int32
   157ae: vadd.i32 q8, q9, q8         ; Accumulate into int32 vector
   157b2: vpaddl.s16 q9, q10          ; Pairwise add int16 -> int32
   157b6: vadd.i32 q8, q8, q9         ; Accumulate into int32 vector
   ...
   15756: vpadd.i32 d16, d16, d16     ; Horizontal reduction of int32
   1575a: vmov.32  r0, d16[0]         ; Extract 32-bit dot sum to core reg
   15760: vmov     s2, r0             ; Transfer to VFP single register
   15768: vcvt.f32.s32 s2, s2         ; Convert int32 -> fp32
   1576c: vmul.f32 s2, s2, s0         ; Multiply by combined_scale
   15770: vstr     s2, [r0]           ; Store logit[v]
```

### Advanced SIMD / Build Attributes:
```
Tag_CPU_arch: ARM v7
Tag_CPU_arch_profile: Application
Tag_THUMB_ISA_use: Thumb-2
Tag_FP_arch: VFPv3
Tag_Advanced_SIMD_arch: NEONv1
```

---

## 13. CMAKE & LINKAGE EVIDENCE

From `CMakeLists.txt`:
```cmake
set(NANO_ENGINE_SRCS
    ...
    src/kernels/neon_gemv_int8.cpp
    ...
)

if(ANDROID_ABI MATCHES "armeabi-v7a")
    target_compile_options(nano_engine        PRIVATE -mfpu=neon -mfloat-abi=softfp)
    target_compile_options(nano_engine_static PRIVATE -mfpu=neon -mfloat-abi=softfp)
endif()
```

Linker Symbol Map (`llvm-nm --defined-only libnano_engine.so`):
```
0001570c T nano_neon_gemv_dense_int8
0001546c T nano_scalar_gemv_dense_int8_reference
```
Both symbols are globally visible and linked into the production shared library.

---

## 14. PRODUCTION .SO AUDIT

- **Path:** `ss_bangladesh_nano_android_module/offline-ai_chatbot/app/src/main/jniLibs/armeabi-v7a/libnano_engine.so`
- **Size:** `663,840` bytes
- **SHA-256:** `CF5026996482D471ACD88BC1D5DD4842EFB6ACDFDE7A2575E1609FDCF8212CA4`
- **Format:** ELF 32-bit LSB shared object, ARM, EABI5 version 1 (SYSV)

---

## 15. PHYSICAL DEVICE BENCHMARK

**Benchmarked on:** Physical `itel A662L` (Spreadtrum SC9863A / Cortex-A7 quad/octa ARMv7, Android 12 Go):

| Metric | Scalar LM-Head (Pre-FIX) | NEON LM-Head (Post-FIX) | Impact |
|---|---|---|---|
| **LM-Head Latency** | **522.64 ms** | **166.04 ms** | **3.15x Speedup** |
| **Per-Token Time Saved** | — | — | **-356.60 ms / token** |
| **10-Token Response Time** | ~5.23 sec (LM-head only) | ~1.66 sec (LM-head only) | **3.57 sec saved** |

---

## 16. BENCHMARK METHODOLOGY

- Identical memory buffer for activation: `int8_t h_state_int8[2560]`.
- Identical weight matrix: Tensor 218 directly mmap'd from `model.nano`.
- Warmup iterations: 2 full passes over all 65,536 rows.
- Measurement iterations: 5 consecutive runs with `std::chrono::high_resolution_clock`.
- Diagnostic logging disabled during measurement loop to avoid I/O skew.

---

## 17. REGRESSION RESULTS

1. **Nano V2 Loader Integrity:** `test_native_model_loader` executed on `itel A662L`:
   `THSA-2B V1 NATIVE MODEL LOADER & V2 DISPATCH GATE: ALL 11 TESTS PASSED ✅`
2. **219 Tensor Mapping:** All 219 tensor descriptors verified and parsed.
3. **End-to-End Neural Forward Pass:** `test_neural_forward_pass` executed on `itel A662L`:
   `FIX 02 REAL NEURAL FORWARD-PASS RESULT: ALL TESTS PASSED ✅`
   Emitted tokens originate directly from causal neural logits generated by the new NEON LM-head.
4. **model.nano SHA-256:** `0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64` (Verified on device via `sha256sum`).

---

## 18. CONFIRMATION OF UNTOUCHED SCOPE

- `ss_bangladesh/` — **UNTOUCHED**
- `models/model.nano` — **UNTOUCHED**
- `checkpoints/` — **UNTOUCHED**
- `tokenizer/` — **UNTOUCHED**
- All other kernels (`neon_gemv_ternary`, `neon_kv_cache`, `neon_norm_act`, `neon_state_update`) — **UNTOUCHED**

---

## 19. LIMITATIONS

- The INT8 quantizer (`nano_neon_quantize_int8`) remains scalar as mandated by the isolation protocol (reserved for future FIX-B).
- Diagnostic instrumentation (FIX-12/FIX-12C) remains available via environment flags.

---

## 20. FINAL MACHINE-READABLE SUMMARY

```
FIX_A_SCOPE=THSA-2B_V1_ONLY
EXTERNAL_MODULE_TOUCHED=NO

BASELINE_MODEL_NANO_SIZE=765477824
BASELINE_MODEL_NANO_SHA256=0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64
MODEL_NANO_MUTATED=NO

LMHEAD_ROWS=65536
LMHEAD_COLS=2560
LMHEAD_QUANT=INT8_DENSE

SCALAR_REFERENCE_PRESENT=YES
NEON_KERNEL_PRESENT=YES
NEON_KERNEL_LINKED=YES
LMHEAD_CALLS_NEON=YES

ARMV7_NEON_MACHINE_CODE_PROVEN=YES

INT32_DOT_EXACT=YES
FULL_LOGITS_EQUIVALENT=YES
TOP1_EXACT=YES

QUANTIZER_MODIFIED=NO
STATE_MODIFIED=NO
GQA_MODIFIED=NO
KV_MODIFIED=NO
CHECKPOINT_MODIFIED=NO
MODEL_NANO_MODIFIED=NO

PHYSICAL_DEVICE_VALIDATED=YES

SCALAR_LMHEAD_MS=522.64
NEON_LMHEAD_MS=166.04
LMHEAD_SPEEDUP=3.15x
LATENCY_REDUCTION_MS=356.60

FINAL_STATUS=FIX-A-PASS-DENSE-INT8-LMHEAD-NEON
```
