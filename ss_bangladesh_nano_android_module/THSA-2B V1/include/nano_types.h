/**
 * @file nano_types.h
 * @brief Core type definitions, error codes, and quantization enums for THSA-2B V1.
 * Standard Compliance: C99 / C++17 compatible header.
 */

#ifndef NANO_TYPES_H
#define NANO_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 64-byte Cache Line Alignment Attribute (ARM Cortex-A) */
#if defined(_MSC_VER)
    #define NANO_ALIGN64 __declspec(align(64))
#else
    #define NANO_ALIGN64 __attribute__((aligned(64)))
#endif

/* Export / Visibility Macros */
#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef NANO_BUILD_DLL
        #define NANO_API __declspec(dllexport)
    #else
        #define NANO_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define NANO_API __attribute__((visibility("default")))
    #else
        #define NANO_API
    #endif
#endif

/**
 * @brief Engine Status and Error Codes (Section 9.6)
 */
typedef enum {
    NANO_SUCCESS             =  0,  /**< Operation completed successfully */
    NANO_ERR_INVALID_PARAM   = -1,  /**< Null handle, bad pointer or out-of-bounds parameter */
    NANO_ERR_OOM             = -2,  /**< Memory allocation exceeded static arena ceiling */
    NANO_ERR_CANCELLED       = -3,  /**< Inference was cancelled asynchronously */
    NANO_ERR_CORRUPT_MODEL   = -4,  /**< Magic header or CRC32 checksum mismatch */
    NANO_ERR_INVALID_TOKEN   = -5,  /**< Token ID is out of vocabulary range (0 <= id < V) */
    NANO_ERR_BUSY            = -6,  /**< Engine context is currently busy generating */
    NANO_ERR_UNSUPPORTED     = -7   /**< Requested operation or ABI is unsupported */
} NanoStatus;

/**
 * @brief Supported Quantization Precision Tiers (Section 5.0 & 11.1)
 */
typedef enum {
    NANO_QUANT_FP32          = 0,   /**< Standard 32-bit floating point */
    NANO_QUANT_FP16          = 1,   /**< Standard 16-bit half precision floating point */
    NANO_QUANT_INT8          = 2,   /**< Signed 8-bit integer quantization [-128, +127] */
    NANO_QUANT_INT4          = 3,   /**< Grouped 4-bit integer quantization [0, 15] */
    NANO_QUANT_TERNARY_2BIT  = 4    /**< 1.58-bit Ternary {-1, 0, +1} packed 2 bits per weight */
} NanoQuantType;

/**
 * @brief Engine Lifecycle State Machine (Section 9.3)
 */
typedef enum {
    NANO_STATE_UNINITIALIZED = 0,   /**< No model loaded; no memory allocated */
    NANO_STATE_INITIALIZING  = 1,   /**< Allocating static arena & memory-mapping model */
    NANO_STATE_READY         = 2,   /**< Model loaded, ready for prompt ingestion */
    NANO_STATE_PREFILLING    = 3,   /**< Chunked prefill processing prompt tokens */
    NANO_STATE_GENERATING    = 4,   /**< Autoregressive single-token decode loop active */
    NANO_STATE_PAUSED        = 5,   /**< Suspended on Android onPause() */
    NANO_STATE_ERROR         = 6    /**< Unrecoverable error encountered */
} NanoEngineState;

/**
 * @brief Token ID type definition
 */
typedef int32_t NanoTokenId;

/**
 * @brief Special Control Tokens (Section 11.2)
 */
#define NANO_TOKEN_BOS        1
#define NANO_TOKEN_EOS        2
#define NANO_TOKEN_UNK        3
#define NANO_TOKEN_PAD        4
#define NANO_TOKEN_IM_START   5
#define NANO_TOKEN_IM_END     6

#ifdef __cplusplus
}
#endif

#endif /* NANO_TYPES_H */
