/**
 * @file neon_state_update.cpp
 * @brief FIX-B: 1D Causal Short-Convolution and Linear State Update Kernel.
 *
 * Implements exact mathematical equivalence with PyTorch Conv1D:
 *   nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
 *
 * Weight Layout in model.nano (serialized from PyTorch [d_model, 1, 4]):
 *   For each channel c (0 <= c < d_model):
 *     conv_weights[c * 4 + 0] = W_0 (weight for t-3, state->conv_state[0])
 *     conv_weights[c * 4 + 1] = W_1 (weight for t-2, state->conv_state[1])
 *     conv_weights[c * 4 + 2] = W_2 (weight for t-1, state->conv_state[2])
 *     conv_weights[c * 4 + 3] = W_3 (weight for current token t, x_in)
 *
 * Tap Order Proof:
 *   For sequence x at time t with causal padding=3:
 *     y[c, t] = W_3 * x[c, t] + W_2 * x[c, t-1] + W_1 * x[c, t-2] + W_0 * x[c, t-3] + bias[c]
 *   Where x[c, tau] = 0 for tau < 0.
 */

#include "../../include/kernels/neon_state_update.h"
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

void nano_state_block_reset(NanoStateBlockContext* state, size_t d_model) {
    (void)d_model;
    if (!state) return;
    memset(state->conv_state, 0, sizeof(state->conv_state));
    state->step_count = 0;
}

/**
 * @brief Scalar Reference Implementation of 1D Causal Short-Convolution
 * Bit-exact reference matching PyTorch Conv1D execution.
 */
void nano_scalar_short_conv_step(
    const float* x_in,
    const float* conv_weights,
    const float* conv_bias,
    NanoStateBlockContext* state,
    size_t d_model,
    float* y_out
) {
    const float* s0 = state->conv_state[0]; // t-3
    const float* s1 = state->conv_state[1]; // t-2
    const float* s2 = state->conv_state[2]; // t-1

    for (size_t c = 0; c < d_model; ++c) {
        const float* cw = conv_weights + (c * 4);
        float b = conv_bias ? conv_bias[c] : 0.0f;
        float conv_val = (s0[c] * cw[0]) +
                         (s1[c] * cw[1]) +
                         (s2[c] * cw[2]) +
                         (x_in[c] * cw[3]) + b;
        y_out[c] = conv_val;
    }

    // Shift state history FIFO: t-3 <= t-2, t-2 <= t-1, t-1 <= current x_in
    memmove(state->conv_state[0], state->conv_state[1], sizeof(float) * d_model);
    memmove(state->conv_state[1], state->conv_state[2], sizeof(float) * d_model);
    memcpy(state->conv_state[2], x_in, sizeof(float) * d_model);
    state->step_count++;
}

/**
 * @brief ARMv7 NEON Vectorized 1D Causal Short-Convolution
 * Uses vld4q_f32 to de-interleave 4 channels of 4 taps into vector registers.
 */
void nano_neon_short_conv_step(
    const float* x_in,
    const float* conv_weights,
    const float* conv_bias,
    NanoStateBlockContext* state,
    size_t d_model,
    float* y_out
) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float* s0 = state->conv_state[0]; // t-3
    const float* s1 = state->conv_state[1]; // t-2
    const float* s2 = state->conv_state[2]; // t-1

    size_t c = 0;
    // Process 4 channels per iteration
    for (; c + 4 <= d_model; c += 4) {
        // vld4q_f32 loads 16 floats for 4 channels (4 taps each) and de-interleaves:
        //   w.val[0] = [W_0[c], W_0[c+1], W_0[c+2], W_0[c+3]]  (t-3 tap)
        //   w.val[1] = [W_1[c], W_1[c+1], W_1[c+2], W_1[c+3]]  (t-2 tap)
        //   w.val[2] = [W_2[c], W_2[c+1], W_2[c+2], W_2[c+3]]  (t-1 tap)
        //   w.val[3] = [W_3[c], W_3[c+1], W_3[c+2], W_3[c+3]]  (current t tap)
        float32x4x4_t w = vld4q_f32(conv_weights + (c * 4));

        float32x4_t h0    = vld1q_f32(s0 + c);
        float32x4_t h1    = vld1q_f32(s1 + c);
        float32x4_t h2    = vld1q_f32(s2 + c);
        float32x4_t x_cur = vld1q_f32(x_in + c);

        float32x4_t acc = conv_bias ? vld1q_f32(conv_bias + c) : vdupq_n_f32(0.0f);

        acc = vmlaq_f32(acc, h0, w.val[0]);
        acc = vmlaq_f32(acc, h1, w.val[1]);
        acc = vmlaq_f32(acc, h2, w.val[2]);
        acc = vmlaq_f32(acc, x_cur, w.val[3]);

        vst1q_f32(y_out + c, acc);
    }

    // Scalar tail for any remaining channels
    for (; c < d_model; ++c) {
        const float* cw = conv_weights + (c * 4);
        float b = conv_bias ? conv_bias[c] : 0.0f;
        float conv_val = (s0[c] * cw[0]) +
                         (s1[c] * cw[1]) +
                         (s2[c] * cw[2]) +
                         (x_in[c] * cw[3]) + b;
        y_out[c] = conv_val;
    }

    // Shift state history FIFO: t-3 <= t-2, t-2 <= t-1, t-1 <= current x_in
    memmove(state->conv_state[0], state->conv_state[1], sizeof(float) * d_model);
    memmove(state->conv_state[1], state->conv_state[2], sizeof(float) * d_model);
    memcpy(state->conv_state[2], x_in, sizeof(float) * d_model);
    state->step_count++;
#else
    nano_scalar_short_conv_step(x_in, conv_weights, conv_bias, state, d_model, y_out);
#endif
}
