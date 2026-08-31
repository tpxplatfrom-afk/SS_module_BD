/**
 * @file neon_gemv_ternary.h
 * @brief Phase 2A: High-Performance Ternary GEMV / GEMM Micro-Kernel.
 * INT8 activations x Ternary {-1, 0, +1} packed 2-bit weights with ARM64 NEON acceleration.
 */

#ifndef NEON_GEMV_TERNARY_H
#define NEON_GEMV_TERNARY_H

#include "../nano_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 2-Bit Ternary Weight Packing Format:
 * 2 bits per weight value:
 *   00 (0) ->  0
 *   01 (1) -> +1
 *   10 (2) -> -1
 *   11 (3) -> Reserved / 0
 * 4 weights per byte (uint8_t), 64 weights per 16-byte block.
 */

/**
 * @brief High-performance Ternary GEMV: y = alpha * (W_ternary * x_int8) + bias
 * @param y Output vector [M] in FP32
 * @param w_packed Packed 2-bit ternary weight matrix [M x (K / 4)]
 * @param x_int8 Quantized input activation vector [K] in INT8
 * @param alpha Per-tensor / per-row FP32 scaling factors [M]
 * @param bias Optional FP32 bias vector [M] (can be NULL)
 * @param M Number of output channels (e.g. 2560 or 6912)
 * @param K Number of input channels (e.g. 2560 or 6912, must be multiple of 16)
 */
void nano_neon_gemv_ternary_int8(
    float* y,
    const uint8_t* w_packed,
    const int8_t* x_int8,
    const float* alpha,
    const float* bias,
    size_t M,
    size_t K
);

/**
 * @brief Scalar Reference Implementation for Bit-Exact Differential Testing
 */
void nano_scalar_gemv_ternary_int8(
    float* y,
    const uint8_t* w_packed,
    const int8_t* x_int8,
    const float* alpha,
    const float* bias,
    size_t M,
    size_t K
);

/**
 * @brief Helper: Pack an array of scalar ternary values {-1, 0, +1} into 2-bit format
 */
void nano_pack_ternary_weights(
    const int8_t* raw_ternary,
    uint8_t* out_packed,
    size_t num_elements
);

#ifdef __cplusplus
}
#endif

#endif /* NEON_GEMV_TERNARY_H */
