/**
 * @file neon_gemv_int8.h
 * @brief FIX-A: ARMv7 NEON Dense INT8 GEMV kernel for THSA-2B V1 LM-Head.
 *
 * Numerical contract:
 *   dot[v]    = SUM_{d=0}^{cols-1}  int32(activation[d]) * int32(weights[v*cols + d])
 *   output[v] = (float)dot[v] * combined_scale
 *
 * Accumulation: INT8 × INT8 → INT32 (never FP32 mid-stream).
 *
 * Overflow bound:
 *   Max per-element product magnitude = 127 * 128 = 16 256
 *   Max full-row sum (cols=2560)      = 2560 * 16256 = 41 615 360
 *   INT32 max                         = 2 147 483 647
 *   Safety margin                     = ~51×  (well within INT32)
 *
 * Memory contract:
 *   weights   — row-major, weights[v * cols + d], read-only, no copy/transpose.
 *   activation — contiguous [cols], read-only.
 *   output     — contiguous [rows], written once per call.
 *
 * ABI: armeabi-v7a (ARMv7 + NEON), also compiles clean on arm64-v8a.
 *      Scalar fallback compiled when __ARM_NEON is not defined (host builds).
 */

#ifndef NEON_GEMV_INT8_H
#define NEON_GEMV_INT8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Production ARMv7 NEON dense INT8 GEMV for the LM-Head.
 *
 * When __ARM_NEON is defined, executes real NEON intrinsics using
 * vmull_s8, vpaddlq_s16, vaddq_s32 for INT8×INT8→INT32 accumulation.
 * Falls back to the scalar reference implementation on non-ARM hosts.
 *
 * @param weights      Row-major INT8 weight matrix [rows × cols].
 *                     Pointer to lm_head_ptr in the mmap'd model.nano.
 *                     Never copied, transposed, or repacked.
 * @param activation   INT8 activation vector [cols].
 *                     Pointer to h_state_int8 (output of nano_neon_quantize_int8).
 * @param output       FP32 output logit vector [rows].  Written in order.
 * @param rows         Number of vocabulary rows (production: 65536).
 * @param cols         Number of model dimensions (production: 2560).
 * @param combined_scale  Scalar = norm_scale * lm_head_scale.  Applied once per row.
 */
void nano_neon_gemv_dense_int8(
    const int8_t* weights,
    const int8_t* activation,
    float*        output,
    size_t        rows,
    size_t        cols,
    float         combined_scale
);

/**
 * @brief Scalar reference implementation — always available, never removed.
 *
 * Used for:
 *   - Differential testing against the NEON kernel (int32 dot must be bit-exact).
 *   - Host-side validation without NEON.
 *   - Regression baseline.
 *
 * Implements the original 8-way-unrolled scalar logic moved here verbatim,
 * preserving the exact numerical contract.
 */
void nano_scalar_gemv_dense_int8_reference(
    const int8_t* weights,
    const int8_t* activation,
    float*        output,
    size_t        rows,
    size_t        cols,
    float         combined_scale
);

#ifdef __cplusplus
}
#endif

#endif /* NEON_GEMV_INT8_H */
