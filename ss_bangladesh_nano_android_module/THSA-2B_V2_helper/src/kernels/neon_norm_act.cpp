/**
 * @file neon_norm_act.cpp
 * @brief Vectorized RMSNorm and SwiGLU activation functions.
 */

#include "../../include/kernels/neon_norm_act.h"
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

void nano_neon_rmsnorm(
    const float* x,
    const float* gamma,
    size_t N,
    float* y_out
) {
    // 1. Calculate sum of squares
    float sum_sq = 0.0f;
    size_t i = 0;
    
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 4 <= N; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        sum_vec = vmlaq_f32(sum_vec, v, v);
    }
    sum_sq = vgetq_lane_f32(sum_vec, 0) + 
             vgetq_lane_f32(sum_vec, 1) + 
             vgetq_lane_f32(sum_vec, 2) + 
             vgetq_lane_f32(sum_vec, 3);
#endif

    for (; i < N; ++i) {
        sum_sq += x[i] * x[i];
    }
    
    // 2. Compute scale = 1.0 / sqrt(mean + eps)
    float mean_sq = sum_sq / (float)N;
    float rsqrt_val = 1.0f / sqrtf(mean_sq + NANO_RMSNORM_EPSILON);
    
    // 3. Scale and apply gamma (applies gamma weight, defaulting to 1.0 if null or zero-weight)
    for (size_t k = 0; k < N; ++k) {
        float g = (gamma && gamma[k] != 0.0f) ? gamma[k] : 1.0f;
        y_out[k] = (x[k] * rsqrt_val) * g;
    }
}

static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

void nano_neon_swiglu(
    const float* gate_in,
    const float* up_in,
    size_t N,
    float* y_out
) {
    for (size_t i = 0; i < N; ++i) {
        float g = gate_in[i];
        float u = up_in[i];
        float silu_g = fast_silu(g);
        y_out[i] = silu_g * u;
    }
}

void nano_neon_quantize_int8(
    const float* src_fp,
    int8_t* out_int8,
    float* out_scale,
    size_t N
) {
    float max_abs = 1e-6f;
    for (size_t i = 0; i < N; ++i) {
        float a = fabsf(src_fp[i]);
        if (a > max_abs) max_abs = a;
    }
    
    float scale = max_abs / 127.0f;
    float inv_scale = 1.0f / scale;
    *out_scale = scale;
    
    for (size_t i = 0; i < N; ++i) {
        int val = (int)roundf(src_fp[i] * inv_scale);
        if (val < -128) val = -128;
        if (val > 127) val = 127;
        out_int8[i] = (int8_t)val;
    }
}
