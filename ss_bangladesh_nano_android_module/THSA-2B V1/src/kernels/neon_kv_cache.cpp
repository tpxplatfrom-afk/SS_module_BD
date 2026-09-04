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

static inline float scalar_dot_f32(const float* a, const float* b, size_t n) {
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

static inline void scalar_accum_weighted_v(float* out, const float* v, float weight, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] += weight * v[i];
    }
}

#if defined(__ARM_NEON) || defined(__aarch64__)
static inline float neon_dot_f32(const float* a, const float* b, size_t n) {
    if (n == 128) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        for (size_t i = 0; i < 128; i += 16) {
            float32x4_t a0 = vld1q_f32(a + i);
            float32x4_t b0 = vld1q_f32(b + i);
            sum0 = vmlaq_f32(sum0, a0, b0);

            float32x4_t a1 = vld1q_f32(a + i + 4);
            float32x4_t b1 = vld1q_f32(b + i + 4);
            sum1 = vmlaq_f32(sum1, a1, b1);

            float32x4_t a2 = vld1q_f32(a + i + 8);
            float32x4_t b2 = vld1q_f32(b + i + 8);
            sum2 = vmlaq_f32(sum2, a2, b2);

            float32x4_t a3 = vld1q_f32(a + i + 12);
            float32x4_t b3 = vld1q_f32(b + i + 12);
            sum3 = vmlaq_f32(sum3, a3, b3);
        }
        float32x4_t sum_all = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        float32x2_t low = vget_low_f32(sum_all);
        float32x2_t high = vget_high_f32(sum_all);
        float32x2_t pair = vadd_f32(low, high);
        return vget_lane_f32(pair, 0) + vget_lane_f32(pair, 1);
    } else {
        return scalar_dot_f32(a, b, n);
    }
}

static inline void neon_accum_weighted_v(float* out, const float* v, float weight, size_t n) {
    if (n == 128) {
        float32x4_t w = vdupq_n_f32(weight);
        for (size_t i = 0; i < 128; i += 16) {
            float32x4_t o0 = vld1q_f32(out + i);
            float32x4_t v0 = vld1q_f32(v + i);
            vst1q_f32(out + i, vmlaq_f32(o0, w, v0));

            float32x4_t o1 = vld1q_f32(out + i + 4);
            float32x4_t v1 = vld1q_f32(v + i + 4);
            vst1q_f32(out + i + 4, vmlaq_f32(o1, w, v1));

            float32x4_t o2 = vld1q_f32(out + i + 8);
            float32x4_t v2 = vld1q_f32(v + i + 8);
            vst1q_f32(out + i + 8, vmlaq_f32(o2, w, v2));

            float32x4_t o3 = vld1q_f32(out + i + 12);
            float32x4_t v3 = vld1q_f32(v + i + 12);
            vst1q_f32(out + i + 12, vmlaq_f32(o3, w, v3));
        }
    } else {
        scalar_accum_weighted_v(out, v, weight, n);
    }
}
#else
static inline float neon_dot_f32(const float* a, const float* b, size_t n) {
    return scalar_dot_f32(a, b, n);
}
static inline void neon_accum_weighted_v(float* out, const float* v, float weight, size_t n) {
    scalar_accum_weighted_v(out, v, weight, n);
}
#endif

void nano_scalar_gqa_attention_fp32(
    const float* q,
    const float* k_cache_fp32,
    const float* v_cache_fp32,
    size_t seq_len,
    size_t max_seq,
    size_t n_q,
    size_t n_kv,
    size_t d_head,
    float* out_attn
) {
    if (seq_len == 0) return;
    
    size_t gqa_group_size = n_q / n_kv; // e.g. 20 / 4 = 5 Query heads per KV head
    float inv_sqrt_d = 1.0f / sqrtf((float)d_head);
    
    // Allocate temporary score buffer
    float scores[10000];
    
    for (size_t q_head = 0; q_head < n_q; ++q_head) {
        size_t kv_head = q_head / gqa_group_size;
        const float* q_vec = q + (q_head * d_head);
        float* out_head = out_attn + (q_head * d_head);
        
        // 1. Compute Dot Products (Q @ K^T)
        float max_score = -1e9f;
        for (size_t t = 0; t < seq_len; ++t) {
            const float* k_vec = k_cache_fp32 + (kv_head * max_seq + t) * d_head;
            float dot = scalar_dot_f32(q_vec, k_vec, d_head);
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
        float inv_exp_sum = 1.0f / exp_sum;
        for (size_t t = 0; t < seq_len; ++t) {
            scores[t] *= inv_exp_sum;
        }
        
        // 3. Weighted Value Accumulation (Attn @ V)
        memset(out_head, 0, sizeof(float) * d_head);
        for (size_t t = 0; t < seq_len; ++t) {
            float weight = scores[t];
            if (weight < 1e-7f) continue;
            
            const float* v_vec = v_cache_fp32 + (kv_head * max_seq + t) * d_head;
            scalar_accum_weighted_v(out_head, v_vec, weight, d_head);
        }
    }
}

void nano_neon_gqa_attention_fp32(
    const float* q,
    const float* k_cache_fp32,
    const float* v_cache_fp32,
    size_t seq_len,
    size_t max_seq,
    size_t n_q,
    size_t n_kv,
    size_t d_head,
    float* out_attn
) {
    if (seq_len == 0) return;
    
    size_t gqa_group_size = n_q / n_kv;
    float inv_sqrt_d = 1.0f / sqrtf((float)d_head);
    
    float scores[10000];
    
    for (size_t q_head = 0; q_head < n_q; ++q_head) {
        size_t kv_head = q_head / gqa_group_size;
        const float* q_vec = q + (q_head * d_head);
        float* out_head = out_attn + (q_head * d_head);
        
        // 1. Compute Dot Products (Q @ K^T) using NEON
        float max_score = -1e9f;
        for (size_t t = 0; t < seq_len; ++t) {
            const float* k_vec = k_cache_fp32 + (kv_head * max_seq + t) * d_head;
            float dot = neon_dot_f32(q_vec, k_vec, d_head);
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
        float inv_exp_sum = 1.0f / exp_sum;
        for (size_t t = 0; t < seq_len; ++t) {
            scores[t] *= inv_exp_sum;
        }
        
        // 3. Weighted Value Accumulation (Attn @ V) using NEON
        memset(out_head, 0, sizeof(float) * d_head);
        for (size_t t = 0; t < seq_len; ++t) {
            float weight = scores[t];
            if (weight < 1e-7f) continue;
            
            const float* v_vec = v_cache_fp32 + (kv_head * max_seq + t) * d_head;
            neon_accum_weighted_v(out_head, v_vec, weight, d_head);
        }
    }
}

void nano_scalar_gqa_attention_int4(
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
    
    size_t gqa_group_size = n_q / n_kv;
    float inv_sqrt_d = 1.0f / sqrtf((float)d_head);
    size_t half_d = d_head / 2;
    
    float scores[10000];
    float dequant_k[256];
    float dequant_v[256];
    
    for (size_t q_head = 0; q_head < n_q; ++q_head) {
        size_t kv_head = q_head / gqa_group_size;
        const float* q_vec = q + (q_head * d_head);
        float* out_head = out_attn + (q_head * d_head);
        
        // 1. Dot Products with scalar dequantized Key
        float max_score = -1e9f;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t k_offset = (kv_head * 10000 + t) * half_d;
            float k_scale = k_scales[kv_head * 10000 + t];
            
            nano_neon_kv_dequantize_int4(k_cache + k_offset, k_scale, dequant_k, d_head);
            
            float dot = scalar_dot_f32(q_vec, dequant_k, d_head);
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
        float inv_exp_sum = 1.0f / exp_sum;
        for (size_t t = 0; t < seq_len; ++t) {
            scores[t] *= inv_exp_sum;
        }
        
        // 3. Weighted Value Accumulation with scalar
        memset(out_head, 0, sizeof(float) * d_head);
        for (size_t t = 0; t < seq_len; ++t) {
            float weight = scores[t];
            if (weight < 1e-7f) continue;
            
            size_t v_offset = (kv_head * 10000 + t) * half_d;
            float v_scale = v_scales[kv_head * 10000 + t];
            
            nano_neon_kv_dequantize_int4(v_cache + v_offset, v_scale, dequant_v, d_head);
            scalar_accum_weighted_v(out_head, dequant_v, weight, d_head);
        }
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
    
    size_t gqa_group_size = n_q / n_kv;
    float inv_sqrt_d = 1.0f / sqrtf((float)d_head);
    size_t half_d = d_head / 2;
    
    float scores[10000];
    float dequant_k[256];
    float dequant_v[256];
    
    for (size_t q_head = 0; q_head < n_q; ++q_head) {
        size_t kv_head = q_head / gqa_group_size;
        const float* q_vec = q + (q_head * d_head);
        float* out_head = out_attn + (q_head * d_head);
        
        // 1. Dot Products with NEON on dequantized Key
        float max_score = -1e9f;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t k_offset = (kv_head * 10000 + t) * half_d;
            float k_scale = k_scales[kv_head * 10000 + t];
            
            nano_neon_kv_dequantize_int4(k_cache + k_offset, k_scale, dequant_k, d_head);
            
            float dot = neon_dot_f32(q_vec, dequant_k, d_head);
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
        float inv_exp_sum = 1.0f / exp_sum;
        for (size_t t = 0; t < seq_len; ++t) {
            scores[t] *= inv_exp_sum;
        }
        
        // 3. Weighted Value Accumulation with NEON
        memset(out_head, 0, sizeof(float) * d_head);
        for (size_t t = 0; t < seq_len; ++t) {
            float weight = scores[t];
            if (weight < 1e-7f) continue;
            
            size_t v_offset = (kv_head * 10000 + t) * half_d;
            float v_scale = v_scales[kv_head * 10000 + t];
            
            nano_neon_kv_dequantize_int4(v_cache + v_offset, v_scale, dequant_v, d_head);
            neon_accum_weighted_v(out_head, dequant_v, weight, d_head);
        }
    }
}
