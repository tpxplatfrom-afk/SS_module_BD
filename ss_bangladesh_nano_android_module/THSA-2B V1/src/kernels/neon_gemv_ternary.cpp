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
            
            // Extract 4 weights
            uint8_t c0 = (byte_val >> 0) & 0x03;
            uint8_t c1 = (byte_val >> 2) & 0x03;
            uint8_t c2 = (byte_val >> 4) & 0x03;
            uint8_t c3 = (byte_val >> 6) & 0x03;
            
            if (c0 == 1) dot_product += x_int8[k_base + 0];
            else if (c0 == 2) dot_product -= x_int8[k_base + 0];
            
            if (c1 == 1) dot_product += x_int8[k_base + 1];
            else if (c1 == 2) dot_product -= x_int8[k_base + 1];
            
            if (c2 == 1) dot_product += x_int8[k_base + 2];
            else if (c2 == 2) dot_product -= x_int8[k_base + 2];
            
            if (c3 == 1) dot_product += x_int8[k_base + 3];
            else if (c3 == 2) dot_product -= x_int8[k_base + 3];
        }
        
        float scale = alpha ? alpha[m] : 1.0f;
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
#if defined(__ARM_NEON) || defined(__aarch64__)
    // NEON SIMD Accelerated Implementation
    size_t k_bytes = K / 4;
    
    for (size_t m = 0; m < M; ++m) {
        const uint8_t* row_w = w_packed + (m * k_bytes);
        int32x4_t acc_vec = vdupq_n_s32(0);
        
        size_t kb = 0;
        // Process 16 input channels (4 packed bytes) per iteration
        for (; kb + 4 <= k_bytes; kb += 4) {
            int8x16_t x_vec = vld1q_s8(x_int8 + (kb * 4));
            
            // Unpack 16 weights from 4 bytes
            int8_t w_unpacked[16];
            for (int i = 0; i < 4; ++i) {
                uint8_t byte_val = row_w[kb + i];
                uint8_t c0 = (byte_val >> 0) & 0x03;
                uint8_t c1 = (byte_val >> 2) & 0x03;
                uint8_t c2 = (byte_val >> 4) & 0x03;
                uint8_t c3 = (byte_val >> 6) & 0x03;
                
                w_unpacked[i * 4 + 0] = (c0 == 1) ? 1 : ((c0 == 2) ? -1 : 0);
                w_unpacked[i * 4 + 1] = (c1 == 1) ? 1 : ((c1 == 2) ? -1 : 0);
                w_unpacked[i * 4 + 2] = (c2 == 1) ? 1 : ((c2 == 2) ? -1 : 0);
                w_unpacked[i * 4 + 3] = (c3 == 1) ? 1 : ((c3 == 2) ? -1 : 0);
            }
            int8x16_t w_vec = vld1q_s8(w_unpacked);
            
            // Multiply and accumulate to 16-bit, then 32-bit
            int16x8_t prod_low  = vmull_s8(vget_low_s8(x_vec),  vget_low_s8(w_vec));
            int16x8_t prod_high = vmull_s8(vget_high_s8(x_vec), vget_high_s8(w_vec));
            
            acc_vec = vpadalq_s16(acc_vec, prod_low);
            acc_vec = vpadalq_s16(acc_vec, prod_high);
        }
        
        // Horizontal sum across acc_vec
        int32_t dot_product = vgetq_lane_s32(acc_vec, 0) + 
                              vgetq_lane_s32(acc_vec, 1) + 
                              vgetq_lane_s32(acc_vec, 2) + 
                              vgetq_lane_s32(acc_vec, 3);
        
        // Handle tail bytes
        for (; kb < k_bytes; ++kb) {
            uint8_t byte_val = row_w[kb];
            size_t k_base = kb * 4;
            uint8_t c0 = (byte_val >> 0) & 0x03;
            uint8_t c1 = (byte_val >> 2) & 0x03;
            uint8_t c2 = (byte_val >> 4) & 0x03;
            uint8_t c3 = (byte_val >> 6) & 0x03;
            
            if (c0 == 1) dot_product += x_int8[k_base + 0];
            else if (c0 == 2) dot_product -= x_int8[k_base + 0];
            
            if (c1 == 1) dot_product += x_int8[k_base + 1];
            else if (c1 == 2) dot_product -= x_int8[k_base + 1];
            
            if (c2 == 1) dot_product += x_int8[k_base + 2];
            else if (c2 == 2) dot_product -= x_int8[k_base + 2];
            
            if (c3 == 1) dot_product += x_int8[k_base + 3];
            else if (c3 == 2) dot_product -= x_int8[k_base + 3];
        }
        
        float scale = alpha ? alpha[m] : 1.0f;
        float b = bias ? bias[m] : 0.0f;
        y[m] = ((float)dot_product * scale) + b;
    }
#else
    // Fallback to scalar on non-ARM architectures (e.g. x86_64 host testing)
    nano_scalar_gemv_ternary_int8(y, w_packed, x_int8, alpha, bias, M, K);
#endif
}
