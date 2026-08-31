/**
 * @file neon_norm_act.h
 * @brief Phase 2D: Vectorized RMSNorm and SwiGLU Activation Pipelines.
 * Fast vector math using ARM64 NEON with reciprocal sqrt and in-place scaling.
 */

#ifndef NEON_NORM_ACT_H
#define NEON_NORM_ACT_H

#include "../nano_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NANO_RMSNORM_EPSILON 1e-5f  /**< Defensive RMSNorm numerical stability epsilon */

/**
 * @brief Vectorized Root Mean Square Normalization (RMSNorm)
 * Formulation: y = (x / sqrt(mean(x^2) + eps)) * gamma
 * @param x Input vector [N]
 * @param gamma Scaling weight vector [N]
 * @param N Dimension (must be multiple of 4, e.g. 2560)
 * @param y_out Output normalized vector [N]
 */
void nano_neon_rmsnorm(
    const float* x,
    const float* gamma,
    size_t N,
    float* y_out
);

/**
 * @brief Vectorized SwiGLU Gated Feed-Forward Activation
 * Formulation: y = (gate * sigmoid(gate)) * up
 * @param gate_in Gate activation vector [D_FFN = 6912]
 * @param up_in Up-projection activation vector [D_FFN = 6912]
 * @param N Dimension (6912)
 * @param y_out Output fused activation vector [N]
 */
void nano_neon_swiglu(
    const float* gate_in,
    const float* up_in,
    size_t N,
    float* y_out
);

/**
 * @brief Quantize FP32 activation vector to INT8 dynamically
 * Formula: x_int8 = round(x * 127.0 / max(|x|)), scale = max(|x|) / 127.0
 * @param src_fp Input FP32 vector [N]
 * @param out_int8 Output INT8 vector [N]
 * @param out_scale Output FP32 scale factor
 * @param N Dimension
 */
void nano_neon_quantize_int8(
    const float* src_fp,
    int8_t* out_int8,
    float* out_scale,
    size_t N
);

#ifdef __cplusplus
}
#endif

#endif /* NEON_NORM_ACT_H */
