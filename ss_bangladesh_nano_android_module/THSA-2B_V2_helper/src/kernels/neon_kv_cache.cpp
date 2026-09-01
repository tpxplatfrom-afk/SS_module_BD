/**
 * @file neon_kv_cache.cpp
 * @brief INT4 Grouped KV-Cache SIMD kernels and GQA attention computation.
 */

#include "../../include/kernels/neon_kv_cache.h"
#include <string.h>
#include <math.h>
#include <float.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

void nano_neon_kv_quantize_int4(
    const float* src_fp,
    uint8_t* out_int4,
    float* out_scale,
    size_t d_head
) {
    // Find absolute maximum
    float max_val = 1e-6f;
    for (size_t i = 0; i < d_head; ++i) {
        float abs_v = fabsf(src_fp[i]);
        if (abs_v > max_val) max_val = abs_v;
    }
    
    // Scale factor: dynamic range [-7, +7] mapped into 4 bits (symmetric)
    float scale = max_val / 7.0f;
    float inv_scale = 1.0f / scale;
    *out_scale = scale;
    
    size_t half_d = d_head / 2;
    for (size_t i = 0; i < half_d; ++i) {
        float v0 = src_fp[i * 2 + 0];
        float v1 = src_fp[i * 2 + 1];
        
        int q0 = (int)roundf(v0 * inv_scale) + 7; // range [0, 14]
        int q1 = (int)roundf(v1 * inv_scale) + 7;
        
        if (q0 < 0) q0 = 0;
        if (q0 > 15) q0 = 15;
        if (q1 < 0) q1 = 0;
        if (q1 > 15) q1 = 15;
        
        out_int4[i] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

void nano_neon_kv_dequantize_int4(
    const uint8_t* src_int4,
    float scale,
    float* out_fp,
    size_t d_head
) {
    size_t half_d = d_head / 2;
    for (size_t i = 0; i < half_d; ++i) {
        uint8_t b = src_int4[i];
        int q0 = (b & 0x0F) - 7;
        int q1 = ((b >> 4) & 0x0F) - 7;
        
        out_fp[i * 2 + 0] = (float)q0 * scale;
        out_fp[i * 2 + 1] = (float)q1 * scale;
    }
}

void nano_neon_gqa_attention_int4(
    const float* q,
    const uint8_t* k_cache,
    const float* k_scales,
    const uint8_t* v_cache,
    const float* v_scales,
    size_t seq_len,
    size_t n_q,
    size_t n_kv,
    size_t d_head,
    float* out_attn
) {
    if (seq_len == 0) return;
    
    size_t gqa_group_size = n_q / n_kv; // e.g. 20 / 4 = 5 Query heads per KV head
    float inv_sqrt_d = 1.0f / sqrtf((float)d_head);
    size_t half_d = d_head / 2;
    
    // Allocate temporary score buffer
    float scores[10000];
    float dequant_k[256];
    float dequant_v[256];
    
    for (size_t q_head = 0; q_head < n_q; ++q_head) {
        size_t kv_head = q_head / gqa_group_size;
        const float* q_vec = q + (q_head * d_head);
        float* out_head = out_attn + (q_head * d_head);
        
        // 1. Compute Dot Products (Q @ K^T)
        float max_score = -1e9f;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t k_offset = (kv_head * 10000 + t) * half_d;
            float k_scale = k_scales[kv_head * 10000 + t];
            
            nano_neon_kv_dequantize_int4(k_cache + k_offset, k_scale, dequant_k, d_head);
            
            float dot = 0.0f;
            for (size_t d = 0; d < d_head; ++d) {
                dot += q_vec[d] * dequant_k[d];
            }
            float score = dot * inv_sqrt_d;
            scores[t] = score;
            if (score > max_score) max_score = score;
        }
        
        // 2. Softmax Normalization
        float exp_sum = 0.0f;
        for (size_t t = 0; t < seq_len; ++t) {
            scores[t] = expf(scores[t] - max_score);
            exp_sum += scores[t];
        }
        float inv_exp_sum = 1.0f / (exp_sum + 1e-9f);
        for (size_t t = 0; t < seq_len; ++t) {
            scores[t] *= inv_exp_sum;
        }
        
        // 3. Weighted Value Accumulation (Attn @ V)
        memset(out_head, 0, sizeof(float) * d_head);
        for (size_t t = 0; t < seq_len; ++t) {
            float weight = scores[t];
            if (weight < 1e-7f) continue;
            
            size_t v_offset = (kv_head * 10000 + t) * half_d;
            float v_scale = v_scales[kv_head * 10000 + t];
            
            nano_neon_kv_dequantize_int4(v_cache + v_offset, v_scale, dequant_v, d_head);
            
            for (size_t d = 0; d < d_head; ++d) {
                out_head[d] += weight * dequant_v[d];
            }
        }
    }
}
