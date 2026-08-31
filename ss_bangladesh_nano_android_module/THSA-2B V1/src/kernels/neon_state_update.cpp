/**
 * @file neon_state_update.cpp
 * @brief 1D Causal Short-Convolution and Linear State Update Kernel.
 */

#include "../../include/kernels/neon_state_update.h"
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

void nano_state_block_reset(NanoStateBlockContext* state, size_t d_model) {
    if (!state) return;
    memset(state->conv_state, 0, sizeof(state->conv_state));
    state->step_count = 0;
}

void nano_neon_short_conv_step(
    const float* x_in,
    const float* conv_weights,
    const float* conv_bias,
    NanoStateBlockContext* state,
    size_t d_model,
    float* y_out
) {
    // 1D Causal Depthwise Convolution across K=4 window
    // conv_weights layout: [K=4, D_MODEL]
    // History states: state->conv_state[0] (t-3), state->conv_state[1] (t-2), state->conv_state[2] (t-1)
    
    const float* w0 = conv_weights + (0 * d_model); // t-3 weight
    const float* w1 = conv_weights + (1 * d_model); // t-2 weight
    const float* w2 = conv_weights + (2 * d_model); // t-1 weight
    const float* w3 = conv_weights + (3 * d_model); // current token t weight
    
    const float* s0 = state->conv_state[0];
    const float* s1 = state->conv_state[1];
    const float* s2 = state->conv_state[2];
    
    size_t i = 0;
    
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 4 <= d_model; i += 4) {
        float32x4_t x_cur = vld1q_f32(x_in + i);
        float32x4_t h0    = vld1q_f32(s0 + i);
        float32x4_t h1    = vld1q_f32(s1 + i);
        float32x4_t h2    = vld1q_f32(s2 + i);
        
        float32x4_t kw0   = vld1q_f32(w0 + i);
        float32x4_t kw1   = vld1q_f32(w1 + i);
        float32x4_t kw2   = vld1q_f32(w2 + i);
        float32x4_t kw3   = vld1q_f32(w3 + i);
        
        float32x4_t acc   = conv_bias ? vld1q_f32(conv_bias + i) : vdupq_n_f32(0.0f);
        
        acc = vmlaq_f32(acc, h0, kw0);
        acc = vmlaq_f32(acc, h1, kw1);
        acc = vmlaq_f32(acc, h2, kw2);
        acc = vmlaq_f32(acc, x_cur, kw3);
        
        vst1q_f32(y_out + i, acc);
    }
#endif

    for (; i < d_model; ++i) {
        float b = conv_bias ? conv_bias[i] : 0.0f;
        float conv_val = (s0[i] * w0[i]) + 
                         (s1[i] * w1[i]) + 
                         (s2[i] * w2[i]) + 
                         (x_in[i] * w3[i]) + b;
        y_out[i] = conv_val;
    }
    
    // Shift state history FIFO: t-3 <= t-2, t-2 <= t-1, t-1 <= current x_in
    memcpy(state->conv_state[0], state->conv_state[1], sizeof(float) * d_model);
    memcpy(state->conv_state[1], state->conv_state[2], sizeof(float) * d_model);
    memcpy(state->conv_state[2], x_in, sizeof(float) * d_model);
    state->step_count++;
}
