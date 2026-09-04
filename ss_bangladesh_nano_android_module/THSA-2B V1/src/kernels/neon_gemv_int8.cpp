/**
 * @file neon_gemv_int8.cpp
 * @brief FIX-A: ARMv7 NEON Dense INT8 GEMV kernel — THSA-2B V1 LM-Head.
 *
 * ============================================================
 * NUMERICAL CONTRACT (immutable)
 * ============================================================
 *   For each vocabulary row v (0 .. rows-1):
 *
 *     dot[v] = SUM_{d=0}^{cols-1}
 *                  int32(activation[d]) * int32(weights[v*cols + d])
 *
 *     output[v] = (float)dot[v] * combined_scale
 *
 *   Accumulation: INT8 × INT8 → INT32.  Never FP32 mid-stream.
 *   Scale applied once per row after full INT32 reduction.
 *
 * ============================================================
 * OVERFLOW ANALYSIS (cols = 2560)
 * ============================================================
 *   Max |activation[d]| = 127   (signed INT8, quantizer clips to [-128,127])
 *   Max |weights[v,d]|  = 128   (signed INT8 range)
 *   Max |product|       = 127 * 128 = 16,256
 *   Max |full-row sum|  = 2560 * 16,256 = 41,615,360
 *   INT32 max           = 2,147,483,647
 *   Safety margin       ≈ 51×  → No overflow possible.
 *
 * ============================================================
 * ARMv7 NEON STRATEGY
 * ============================================================
 *   Inner dimension vectorized in blocks of 16 INT8 elements:
 *     - vld1_s8 × 2: load 8+8 activation elements
 *     - vld1_s8 × 2: load 8+8 weight elements
 *     - vmull_s8   : INT8×INT8 → INT16 (8-wide)
 *     - vpaddlq_s16: INT16 pairwise-add → INT32 (4-wide), accumulate
 *     - vaddq_s32  : accumulate across blocks
 *
 *   Horizontal reduction (ARMv7 lacks vaddvq_s32):
 *     - vget_low_s32 / vget_high_s32 + vadd_s32 + vpadd_s32
 *
 *   Outer loop processes 4 rows at a time for better cache utilisation
 *   of the weight matrix (4 × 2560 bytes = 10 KB fits in L1 scratch).
 *
 *   2560 / 16 = 160 exact iterations — no tail in production.
 *   Tail loop included defensively for arbitrary cols.
 *
 * ============================================================
 * TARGET: armeabi-v7a (Cortex-A7, itel A662L Android 12 Go)
 * Also compiles on arm64-v8a (superset of ARMv7 NEON).
 * Scalar fallback for host x86/x86_64 builds.
 * ============================================================
 */

#include "../../include/kernels/neon_gemv_int8.h"
#include <string.h>

/* Bring in NEON intrinsics on any ARM target that has NEON */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define NANO_HAS_NEON 1
#else
#  define NANO_HAS_NEON 0
#endif


/* ============================================================
 * SCALAR REFERENCE — always compiled, never removed.
 *
 * This is the original production scalar logic preserved verbatim
 * from nano_engine.cpp lines 740-758 (pre-FIX-A), refactored into
 * a standalone function for differential testing.
 * ============================================================ */
void nano_scalar_gemv_dense_int8_reference(
    const int8_t* weights,
    const int8_t* activation,
    float*        output,
    size_t        rows,
    size_t        cols,
    float         combined_scale
) {
    for (size_t v = 0; v < rows; ++v) {
        const int8_t* w_row = weights + (v * cols);
        int32_t dot = 0;
        size_t d = 0;

        /* 8-way unrolled — mirrors the original scalar loop exactly */
        for (; d + 8 <= cols; d += 8) {
            dot += (int32_t)activation[d + 0] * (int32_t)w_row[d + 0]
                 + (int32_t)activation[d + 1] * (int32_t)w_row[d + 1]
                 + (int32_t)activation[d + 2] * (int32_t)w_row[d + 2]
                 + (int32_t)activation[d + 3] * (int32_t)w_row[d + 3]
                 + (int32_t)activation[d + 4] * (int32_t)w_row[d + 4]
                 + (int32_t)activation[d + 5] * (int32_t)w_row[d + 5]
                 + (int32_t)activation[d + 6] * (int32_t)w_row[d + 6]
                 + (int32_t)activation[d + 7] * (int32_t)w_row[d + 7];
        }
        /* scalar tail */
        for (; d < cols; ++d) {
            dot += (int32_t)activation[d] * (int32_t)w_row[d];
        }
        output[v] = (float)dot * combined_scale;
    }
}


/* ============================================================
 * ARMv7 NEON PRODUCTION KERNEL
 * ============================================================ */

#if NANO_HAS_NEON

/**
 * @brief Compute one row's INT8 dot product using NEON intrinsics.
 *
 * Processes 16 INT8 elements per iteration via:
 *   vmull_s8  (8-wide INT8×INT8 → INT16)
 *   vpaddlq_s16 (widen & pairwise-add → INT32 accumulator)
 *
 * @param w_row     Pointer to weight row [cols].
 * @param act       Activation vector [cols].
 * @param cols      Number of elements (must be > 0).
 * @return          INT32 dot product.
 */
static inline int32_t neon_dot_int8_row(
    const int8_t* __restrict__ w_row,
    const int8_t* __restrict__ act,
    size_t cols
) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t d = 0;

    /* Main loop: 16 elements per iteration */
    for (; d + 16 <= cols; d += 16) {
        /* Load 8 activation + 8 weight elements (first half) */
        int8x8_t a0 = vld1_s8(act   + d);
        int8x8_t w0 = vld1_s8(w_row + d);

        /* Load 8 activation + 8 weight elements (second half) */
        int8x8_t a1 = vld1_s8(act   + d + 8);
        int8x8_t w1 = vld1_s8(w_row + d + 8);

        /* INT8 × INT8 → INT16 (8-wide multiply) */
        int16x8_t p0 = vmull_s8(a0, w0);
        int16x8_t p1 = vmull_s8(a1, w1);

        /* Widen INT16 → INT32 via pairwise-add, accumulate */
        acc = vaddq_s32(acc, vpaddlq_s16(p0));
        acc = vaddq_s32(acc, vpaddlq_s16(p1));
    }

    /* Scalar tail (handles cols not divisible by 16) */
    /* Production: 2560 / 16 = 160 exact — tail never executes in practice */
    int32_t tail = 0;
    for (; d < cols; ++d) {
        tail += (int32_t)act[d] * (int32_t)w_row[d];
    }

    /* Horizontal reduction of int32x4_t → scalar int32.
     * ARMv7 NEON does NOT have vaddvq_s32 (that is AArch64-only).
     * Use: pairwise-add of low/high halves, then vpadd on the resulting pair. */
    int32x2_t sum_half = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    sum_half = vpadd_s32(sum_half, sum_half);   /* [a+b, a+b] */
    int32_t dot = vget_lane_s32(sum_half, 0);

    return dot + tail;
}

/**
 * @brief NEON dense INT8 GEMV — 4-rows-at-a-time outer blocking.
 *
 * Processing 4 rows simultaneously improves weight-cache reuse:
 * 4 rows × 2560 bytes = 10 KB — fits in Cortex-A7 L1 data cache (16 KB typical).
 *
 * Each row's dot product is computed independently via neon_dot_int8_row().
 * The final scale multiply is a single FP32 op per row.
 */
static void neon_gemv_dense_int8_4row(
    const int8_t* weights,
    const int8_t* activation,
    float*        output,
    size_t        rows,
    size_t        cols,
    float         combined_scale
) {
    size_t v = 0;

    /* 4-row block loop */
    for (; v + 4 <= rows; v += 4) {
        const int8_t* w0 = weights + (v + 0) * cols;
        const int8_t* w1 = weights + (v + 1) * cols;
        const int8_t* w2 = weights + (v + 2) * cols;
        const int8_t* w3 = weights + (v + 3) * cols;

        int32_t d0 = neon_dot_int8_row(w0, activation, cols);
        int32_t d1 = neon_dot_int8_row(w1, activation, cols);
        int32_t d2 = neon_dot_int8_row(w2, activation, cols);
        int32_t d3 = neon_dot_int8_row(w3, activation, cols);

        output[v + 0] = (float)d0 * combined_scale;
        output[v + 1] = (float)d1 * combined_scale;
        output[v + 2] = (float)d2 * combined_scale;
        output[v + 3] = (float)d3 * combined_scale;
    }

    /* Scalar tail for rows not divisible by 4 */
    for (; v < rows; ++v) {
        const int8_t* wr = weights + v * cols;
        int32_t dot = neon_dot_int8_row(wr, activation, cols);
        output[v] = (float)dot * combined_scale;
    }
}

#endif /* NANO_HAS_NEON */


/* ============================================================
 * PUBLIC ENTRY POINT
 * ============================================================ */

void nano_neon_gemv_dense_int8(
    const int8_t* weights,
    const int8_t* activation,
    float*        output,
    size_t        rows,
    size_t        cols,
    float         combined_scale
) {
#if NANO_HAS_NEON
    /*
     * Real ARMv7/AArch64 NEON path.
     * The generated machine code WILL contain vmull.s8, vpaddl.s16,
     * vadd.i32, vld1.8 instructions — verifiable via objdump.
     */
    neon_gemv_dense_int8_4row(
        weights, activation, output, rows, cols, combined_scale
    );
#else
    /*
     * Host scalar fallback (x86/x86_64 builds for unit-test compilation).
     * This path is NEVER executed on the production armeabi-v7a device.
     * On Android armeabi-v7a, __ARM_NEON is always defined when -mfpu=neon
     * is active, so the NEON path above is taken exclusively.
     */
    nano_scalar_gemv_dense_int8_reference(
        weights, activation, output, rows, cols, combined_scale
    );
#endif
}
