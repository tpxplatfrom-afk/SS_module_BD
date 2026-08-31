/**
 * @file neon_state_update.h
 * @brief Phase 2C: 16-Block Linear State / Short-Conv Vector Kernel.
 * Implements O(1) state memory sequence transformation across 16 non-attention blocks.
 */

#ifndef NEON_STATE_UPDATE_H
#define NEON_STATE_UPDATE_H

#include "../nano_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NANO_SHORT_CONV_KERNEL_SIZE 4  /**< Temporal convolution window K=4 */

/**
 * @brief State buffer for a single Short-Conv / State block
 */
typedef struct {
    float conv_state[NANO_SHORT_CONV_KERNEL_SIZE - 1][2560]; /**< Previous K-1 token vectors */
    uint32_t step_count;                                     /**< Sequence step index */
} NanoStateBlockContext;

/**
 * @brief 1D Causal Depthwise Short-Convolution with Gated Non-Linearity
 * @param x_in Input token activation [D_MODEL = 2560]
 * @param conv_weights Depthwise 1D conv kernel weights [K=4, D_MODEL = 2560]
 * @param conv_bias Depthwise bias vector [D_MODEL]
 * @param state State buffer holding history tokens for this block
 * @param d_model Dimension of hidden state (2560)
 * @param y_out Output vector [D_MODEL]
 */
void nano_neon_short_conv_step(
    const float* x_in,
    const float* conv_weights,
    const float* conv_bias,
    NanoStateBlockContext* state,
    size_t d_model,
    float* y_out
);

/**
 * @brief Reset state memory for a state block to zero
 */
void nano_state_block_reset(NanoStateBlockContext* state, size_t d_model);

#ifdef __cplusplus
}
#endif

#endif /* NEON_STATE_UPDATE_H */
