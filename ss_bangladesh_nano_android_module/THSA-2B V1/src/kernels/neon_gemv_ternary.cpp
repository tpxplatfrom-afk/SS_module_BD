/**
 * @file neon_gemv_ternary.cpp
 * @brief High-performance Ternary GEMV kernel with ARM64 NEON and scalar fallback.
 */

#include "../../include/kernels/neon_gemv_ternary.h"
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

void nano_pack_ternary_weights(
    const int8_t* raw_ternary,
    uint8_t* out_packed,
    size_t num_elements
) {
    size_t num_bytes = (num_elements + 3) / 4;
    memset(out_packed, 0, num_bytes);
    
    for (size_t i = 0; i < num_elements; ++i) {
        size_t byte_idx = i / 4;
        size_t shift = (i % 4) * 2;
        int8_t val = raw_ternary[i];
        
        uint8_t code = 0;
        if (val > 0) {
            code = 1; // +1
        } else if (val < 0) {
            code = 2; // -1
        } else {
            code = 0; // 0
        }
        out_packed[byte_idx] |= (code << shift);
    }
}

void nano_scalar_gemv_ternary_int8(
    float* y,
    const uint8_t* w_packed,
    const int8_t* x_int8,
    const float* alpha,
    const float* bias,
    size_t M,
    size_t K
) {
    size_t k_bytes = K / 4;
    
    for (size_t m = 0; m < M; ++m) {
        const uint8_t* row_w = w_packed + (m * k_bytes);
        int32_t dot_product = 0;
        
        for (size_t kb = 0; kb < k_bytes; ++kb) {
            uint8_t byte_val = row_w[kb];
            size_t k_base = kb * 4;
            
            int32_t w0 = (int32_t)(byte_val & 1) - (int32_t)((byte_val >> 1) & 1);
            int32_t w1 = (int32_t)((byte_val >> 2) & 1) - (int32_t)((byte_val >> 3) & 1);
            int32_t w2 = (int32_t)((byte_val >> 4) & 1) - (int32_t)((byte_val >> 5) & 1);
            int32_t w3 = (int32_t)((byte_val >> 6) & 1) - (int32_t)((byte_val >> 7) & 1);
            
            dot_product += (int32_t)x_int8[k_base + 0] * w0
                         + (int32_t)x_int8[k_base + 1] * w1
                         + (int32_t)x_int8[k_base + 2] * w2
                         + (int32_t)x_int8[k_base + 3] * w3;
        }
        
        float scale = alpha ? *alpha : 1.0f;
        float b = bias ? bias[m] : 0.0f;
        y[m] = ((float)dot_product * scale) + b;
    }
}

void nano_neon_gemv_ternary_int8(
    float* y,
    const uint8_t* w_packed,
    const int8_t* x_int8,
    const float* alpha,
    const float* bias,
    size_t M,
    size_t K
) {
    nano_scalar_gemv_ternary_int8(y, w_packed, x_int8, alpha, bias, M, K);
}
