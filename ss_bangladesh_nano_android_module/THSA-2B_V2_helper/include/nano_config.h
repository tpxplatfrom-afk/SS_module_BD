/**
 * @file nano_config.h
 * @brief Versioned Model Scale Configuration struct and presets for THSA-2B V1.
 * Standard Compliance: C99 / C++17 compatible header.
 */

#ifndef NANO_CONFIG_H
#define NANO_CONFIG_H

#include "nano_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Model Scale Configuration Struct (Section 3.1)
 * Drives all architectural dimensions dynamically without kernel recompilation.
 */
typedef struct {
    uint16_t  format_version;      /**< File format version (e.g. 0x0001 for V1.0) */
    uint32_t  total_blocks;        /**< Total backbone blocks (24 for 2B, 28 for 3B) */
    uint32_t  state_blocks;        /**< Number of State/Short-Conv blocks (16 for 2B) */
    uint32_t  gqa_blocks;          /**< Number of GQA attention blocks (8 for 2B) */
    uint32_t  d_model;             /**< Hidden dimension (2560 for 2B) */
    uint32_t  d_ffn;               /**< Intermediate FFN dimension (6912 for 2B) */
    uint32_t  n_query_heads;       /**< Attention query heads (20 for 2B) */
    uint32_t  n_kv_heads;          /**< Attention KV heads (4 for all tiers) */
    uint32_t  d_head;              /**< Dimension per head (128 for all tiers) */
    uint32_t  vocab_size;          /**< Vocabulary size (65536 for all tiers) */
    uint32_t  max_context_tokens;  /**< Runtime context horizon (e.g. 10000) */
    uint32_t  chunk_size;          /**< Chunked prefill micro-chunk (256 tokens) */
    char      model_id[32];        /**< Identifier string (e.g. "THSA-2B-V1") */
} NanoModelConfig;

/**
 * @brief Generation Sampling Parameters
 */
typedef struct {
    float     temperature;         /**< Sampling temperature (0.0 = greedy argmax, default: 0.7) */
    float     top_p;               /**< Nucleus sampling threshold (default: 0.9) */
    int32_t   top_k;               /**< Top-K candidates limit (default: 40) */
    float     repetition_penalty;  /**< Repetition penalty factor (default: 1.1) */
    int32_t   max_output_tokens;   /**< Maximum tokens to generate per turn (default: 2048) */
} NanoGenerationConfig;

/**
 * @brief Initialize a NanoModelConfig with default THSA-2B V1 parameters
 */
static inline NanoModelConfig nano_config_default_2b(void) {
    NanoModelConfig cfg;
    cfg.format_version      = 0x0001;
    cfg.total_blocks        = 24;
    cfg.state_blocks        = 16;
    cfg.gqa_blocks          = 8;
    cfg.d_model             = 2560;
    cfg.d_ffn               = 6912;
    cfg.n_query_heads       = 20;
    cfg.n_kv_heads          = 4;
    cfg.d_head              = 128;
    cfg.vocab_size          = 65536;
    cfg.max_context_tokens  = 10000;
    cfg.chunk_size          = 256;
    cfg.model_id[0]         = '\0';
    return cfg;
}

/**
 * @brief Initialize a NanoGenerationConfig with default production parameters
 */
static inline NanoGenerationConfig nano_gen_config_default(void) {
    NanoGenerationConfig cfg;
    cfg.temperature         = 0.7f;
    cfg.top_p               = 0.9f;
    cfg.top_k               = 40;
    cfg.repetition_penalty  = 1.1f;
    cfg.max_output_tokens   = 2048;
    return cfg;
}

#ifdef __cplusplus
}
#endif

#endif /* NANO_CONFIG_H */
