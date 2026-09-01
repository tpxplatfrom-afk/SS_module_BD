/**
 * @file neon_kv_cache.h
 * @brief Phase 2B: INT4 Grouped KV-Cache SIMD & Attention-Sink Rolling Window.
 * 20 Query heads share 4 KV heads (5:1 GQA ratio), packed at 4 bits per element.
 */

#ifndef NEON_KV_CACHE_H
#define NEON_KV_CACHE_H

#include "../nano_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Quantize FP32/FP16 Key/Value vector into grouped INT4 representation
 * @param src_fp Input vector [D_HEAD = 128]
 * @param out_int4 Output packed 4-bit buffer [D_HEAD / 2 = 64 bytes]
 * @param out_scale Output FP32 scale factor
 * @param d_head Dimension of head (must be multiple of 16, e.g. 128)
 */
void nano_neon_kv_quantize_int4(
    const float* src_fp,
    uint8_t* out_int4,
    float* out_scale,
    size_t d_head
);

/**
 * @brief Dequantize packed INT4 Key/Value vector into FP32
 * @param src_int4 Packed 4-bit buffer [D_HEAD / 2 bytes]
 * @param scale FP32 scale factor
 * @param out_fp Output vector [D_HEAD]
 * @param d_head Dimension of head
 */
void nano_neon_kv_dequantize_int4(
    const uint8_t* src_int4,
    float scale,
    float* out_fp,
    size_t d_head
);

/**
 * @brief Compute GQA Attention Scores over INT4 KV-cache with Attention Sinks
 * @param q Query vector [N_Q = 20, D_HEAD = 128]
 * @param k_cache Packed INT4 Key cache [N_KV = 4, MAX_SEQ, D_HEAD / 2]
 * @param k_scales Key scale factors [N_KV = 4, MAX_SEQ]
 * @param v_cache Packed INT4 Value cache [N_KV = 4, MAX_SEQ, D_HEAD / 2]
 * @param v_scales Value scale factors [N_KV = 4, MAX_SEQ]
 * @param seq_len Current valid sequence length (up to 10000)
 * @param n_q Number of Query heads (20)
 * @param n_kv Number of KV heads (4)
 * @param d_head Dimension per head (128)
 * @param out_attn Output attention vector [N_Q * D_HEAD]
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* NEON_KV_CACHE_H */
