/**
 * @file test_dense_int8_gemv.cpp
 * @brief FIX-A: Deterministic differential test for dense INT8 GEMV kernel.
 *
 * Tests the ARMv7 NEON nano_neon_gemv_dense_int8() against the scalar
 * nano_scalar_gemv_dense_int8_reference() for:
 *
 *   A. Small synthetic matrix  (rows=4,   cols=32)
 *   B. Medium matrix           (rows=16,  cols=256)
 *   C. Production-width        (rows=8,   cols=2560)
 *   D. Large vocabulary batch  (rows=128, cols=2560)
 *   E. Full production shape   (rows=65536, cols=2560)  [if memory permits]
 *
 * Edge cases:
 *   1. All activation zeros
 *   2. All weights zeros
 *   3. All activation +127
 *   4. Activation -128
 *   5. Mixed positive/negative
 *   6. Alternating signs
 *   7. Maximum-magnitude values (+127 activation, ±127 weights)
 *   8. Random deterministic seed (LCG, seed=0xDEADBEEF)
 *   9. Rows not aligned to 4 (tail handling)
 *  10. Cols not aligned to 16 (tail handling — synthetic)
 *
 * Acceptance criteria:
 *   - INT32 dot product: EXACT equality (bit-exact).
 *   - FP32 output: EXACT equality (same arithmetic sequence both paths).
 *   - Any non-zero difference → FAIL, do not suppress.
 *
 * On host builds (x86/x86_64) without __ARM_NEON, the NEON path falls
 * back to scalar, so both paths produce identical results by construction.
 * The test still validates the API contract and scalar reference correctness.
 * On-device (armeabi-v7a) execution is required to prove real NEON code.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neon_gemv_int8.h"

/* ============================================================
 * Utilities
 * ============================================================ */

/* Minimal LCG random number generator — deterministic, seed-based */
static uint32_t lcg_state = 0xDEADBEEFu;
static uint32_t lcg_next() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}
static void lcg_seed(uint32_t seed) { lcg_state = seed; }
static int8_t lcg_int8() {
    return (int8_t)(lcg_next() & 0xFF);
}

/* Fill buffer with constant int8 value */
static void fill_const(int8_t* buf, size_t n, int8_t val) {
    for (size_t i = 0; i < n; ++i) buf[i] = val;
}

/* Fill buffer with alternating +v / -v */
static void fill_alternating(int8_t* buf, size_t n, int8_t v) {
    for (size_t i = 0; i < n; ++i)
        buf[i] = (i % 2 == 0) ? v : (int8_t)(-v);
}

/* Fill buffer with LCG random int8 */
static void fill_random(int8_t* buf, size_t n) {
    for (size_t i = 0; i < n; ++i) buf[i] = lcg_int8();
}

/* ============================================================
 * Single test runner
 * ============================================================ */
typedef struct {
    const char* name;
    size_t      rows;
    size_t      cols;
    int8_t*     weights;    /* [rows * cols] */
    int8_t*     activation; /* [cols]        */
    float       scale;
    int         pass;
    float       max_fp32_diff;
    int         int32_exact;
} TestCase;

static int run_test(TestCase* tc) {
    size_t rows    = tc->rows;
    size_t cols    = tc->cols;
    float  scale   = tc->scale;

    /* Allocate output buffers */
    float* out_scalar = (float*)malloc(rows * sizeof(float));
    float* out_neon   = (float*)malloc(rows * sizeof(float));
    int32_t* dot_scalar = (int32_t*)malloc(rows * sizeof(int32_t));
    int32_t* dot_neon   = (int32_t*)malloc(rows * sizeof(int32_t));

    if (!out_scalar || !out_neon || !dot_scalar || !dot_neon) {
        printf("FAIL [%s] OOM\n", tc->name);
        free(out_scalar); free(out_neon); free(dot_scalar); free(dot_neon);
        tc->pass = 0;
        return 0;
    }

    /* --- Scalar reference --- */
    nano_scalar_gemv_dense_int8_reference(
        tc->weights, tc->activation, out_scalar, rows, cols, scale
    );

    /* --- NEON kernel --- */
    nano_neon_gemv_dense_int8(
        tc->weights, tc->activation, out_neon, rows, cols, scale
    );

    /* --- Compare --- */
    int int32_exact = 1;
    float max_diff  = 0.0f;
    int pass        = 1;

    /* Reconstruct int32 dots from FP32 outputs for exact comparison */
    /* dot = output / scale  (scale != 0 guaranteed by test setup) */
    for (size_t v = 0; v < rows; ++v) {
        /* Compare FP32 outputs directly */
        float diff = fabsf(out_scalar[v] - out_neon[v]);
        if (diff > max_diff) max_diff = diff;

        /* Recover int32 dot from scalar output */
        /* dot_recovered = round(out / scale) */
        if (scale != 0.0f) {
            int64_t ds = (int64_t)(out_scalar[v] / scale + (out_scalar[v] >= 0 ? 0.5f : -0.5f));
            int64_t dn = (int64_t)(out_neon[v]   / scale + (out_neon[v]   >= 0 ? 0.5f : -0.5f));
            if (ds != dn) {
                int32_exact = 0;
                pass = 0;
                printf("  MISMATCH row=%zu: scalar_dot=%lld neon_dot=%lld\n",
                       v, (long long)ds, (long long)dn);
                if (v < 5) { /* Print first few only */
                    printf("    scalar_out=%.8f neon_out=%.8f diff=%.8f\n",
                           out_scalar[v], out_neon[v], diff);
                }
            }
        }

        /* FP32 exact equality required (same computation path) */
        if (diff != 0.0f) {
            pass = 0;
        }
    }

    tc->pass         = pass;
    tc->max_fp32_diff = max_diff;
    tc->int32_exact  = int32_exact;

    if (pass) {
        printf("PASS [%s] rows=%zu cols=%zu scale=%.6f  int32_exact=YES  fp32_max_diff=0\n",
               tc->name, rows, cols, scale);
    } else {
        printf("FAIL [%s] rows=%zu cols=%zu  int32_exact=%s  fp32_max_diff=%.8e\n",
               tc->name, rows, cols,
               int32_exact ? "YES" : "NO",
               max_diff);
    }

    free(out_scalar); free(out_neon); free(dot_scalar); free(dot_neon);
    return pass;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void) {
    int total = 0, passed = 0;

    printf("=== FIX-A: Dense INT8 GEMV Differential Test ===\n");
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    printf("Platform: ARM NEON available — NEON kernel active\n");
#else
    printf("Platform: No ARM NEON — scalar fallback path (host build)\n");
    printf("NOTE: Run on armeabi-v7a device to prove real NEON code.\n");
#endif
    printf("\n");

    /* --------------------------------------------------------
     * A. Small synthetic: rows=4, cols=32
     * -------------------------------------------------------- */
    {
        const size_t R = 4, C = 32;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0xDEADBEEF);
        fill_random(W, R * C);
        fill_random(A, C);
        TestCase tc = { "A_small_4x32", R, C, W, A, 0.001234f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * B. Medium: rows=16, cols=256
     * -------------------------------------------------------- */
    {
        const size_t R = 16, C = 256;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0xCAFEBABE);
        fill_random(W, R * C);
        fill_random(A, C);
        TestCase tc = { "B_medium_16x256", R, C, W, A, 0.000567f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * C. Production-width: rows=8, cols=2560
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0x12345678);
        fill_random(W, R * C);
        fill_random(A, C);
        TestCase tc = { "C_prod_width_8x2560", R, C, W, A, 0.000321f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * D. Large vocabulary batch: rows=128, cols=2560
     * -------------------------------------------------------- */
    {
        const size_t R = 128, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0xABCDEF01);
        fill_random(W, R * C);
        fill_random(A, C);
        TestCase tc = { "D_vocab_batch_128x2560", R, C, W, A, 0.000212f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * E. Full production shape: rows=65536, cols=2560
     * Memory: 65536 * 2560 = 167 MB weights + 256 KB logits
     * Only run if we can allocate.
     * -------------------------------------------------------- */
    {
        const size_t R = 65536, C = 2560;
        const size_t wsz = R * C;
        printf("  [E_full_65536x2560] Allocating %.1f MB weights...\n",
               (double)wsz / (1024.0*1024.0));
        int8_t* W = (int8_t*)malloc(wsz);
        int8_t* A = (int8_t*)malloc(C);
        if (W && A) {
            lcg_seed(0xDEADC0DE);
            fill_random(W, wsz);
            fill_random(A, C);
            TestCase tc = { "E_full_65536x2560", R, C, W, A, 0.000180f, 0, 0.0f, 0 };
            total++; passed += run_test(&tc);
        } else {
            printf("  SKIP [E_full_65536x2560] — insufficient memory\n");
        }
        free(W); free(A);
    }

    printf("\n=== Edge Cases ===\n");

    /* --------------------------------------------------------
     * Edge 1: All activation zeros
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)calloc(C, 1);
        lcg_seed(0x11111111); fill_random(W, R * C);
        TestCase tc = { "Edge1_zero_activation", R, C, W, A, 0.001f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 2: All weights zeros
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 2560;
        int8_t* W = (int8_t*)calloc(R * C, 1);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0x22222222); fill_random(A, C);
        TestCase tc = { "Edge2_zero_weights", R, C, W, A, 0.001f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 3: All activation +127
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0x33333333); fill_random(W, R * C);
        fill_const(A, C, 127);
        TestCase tc = { "Edge3_act_plus127", R, C, W, A, 0.001f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 4: Activation -128
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0x44444444); fill_random(W, R * C);
        fill_const(A, C, (int8_t)-128);
        TestCase tc = { "Edge4_act_minus128", R, C, W, A, 0.001f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 5: Mixed positive/negative (random)
     * -------------------------------------------------------- */
    {
        const size_t R = 16, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0x55555555); fill_random(W, R * C); fill_random(A, C);
        TestCase tc = { "Edge5_mixed_random", R, C, W, A, 0.000999f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 6: Alternating signs
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        fill_alternating(W, R * C, 64);
        fill_alternating(A, C, 64);
        TestCase tc = { "Edge6_alternating_signs", R, C, W, A, 0.001f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 7: Maximum-magnitude values (+127 act, ±127 weights)
     * Overflow bound check: 2560 * 127 * 127 = 41,289,280 < INT32_MAX
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        fill_const(A, C, 127);
        /* Rows alternate +127 / -127 weights */
        for (size_t v = 0; v < R; ++v)
            fill_const(W + v * C, C, (v % 2 == 0) ? (int8_t)127 : (int8_t)-127);
        TestCase tc = { "Edge7_max_magnitude", R, C, W, A, 0.001f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 8: Random deterministic seed (different seed)
     * -------------------------------------------------------- */
    {
        const size_t R = 32, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0x9A3F7C1Bu); fill_random(W, R * C); fill_random(A, C);
        TestCase tc = { "Edge8_random_seed2", R, C, W, A, 0.000765f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 9: Rows not aligned to 4 (tail path in outer loop)
     * rows=5 → 1 block of 4 + tail of 1
     * -------------------------------------------------------- */
    {
        const size_t R = 5, C = 2560;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0xBBBBBBBBu); fill_random(W, R * C); fill_random(A, C);
        TestCase tc = { "Edge9_rows_not_aligned_5x2560", R, C, W, A, 0.001f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Edge 10: Cols not aligned to 16 (tail path in inner loop)
     * Synthetic: cols=33 (2 full 16-element blocks + tail of 1)
     * -------------------------------------------------------- */
    {
        const size_t R = 8, C = 33;
        int8_t* W = (int8_t*)malloc(R * C);
        int8_t* A = (int8_t*)malloc(C);
        lcg_seed(0xCCCCCCCCu); fill_random(W, R * C); fill_random(A, C);
        TestCase tc = { "Edge10_cols_tail_8x33", R, C, W, A, 0.002f, 0, 0.0f, 0 };
        total++; passed += run_test(&tc);
        free(W); free(A);
    }

    /* --------------------------------------------------------
     * Summary
     * -------------------------------------------------------- */
    printf("\n=== SUMMARY ===\n");
    printf("Tests run: %d\n", total);
    printf("Passed:    %d\n", passed);
    printf("Failed:    %d\n", total - passed);
    printf("\n");

    if (passed == total) {
        printf("TEST_DENSE_INT8_GEMV=PASS\n");
        printf("INT32_DOT_EXACT=YES\n");
        printf("FP32_OUTPUT_EXACT=YES\n");
        return 0;
    } else {
        printf("TEST_DENSE_INT8_GEMV=FAIL\n");
        printf("ACTION: Investigate all FAIL lines above before declaring PASS.\n");
        return 1;
    }
}
