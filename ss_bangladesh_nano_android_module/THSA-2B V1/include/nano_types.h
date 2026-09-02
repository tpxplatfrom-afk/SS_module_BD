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
    NANO_ERR_UNSUPPORTED     = -7,  /**< Requested operation or ABI is unsupported */
    NANO_ERR_FILE_NOT_FOUND  = -8,  /**< Model file does not exist or cannot be opened */
    NANO_ERR_INVALID_HEADER  = -9,  /**< Invalid NANO magic, version, or dimensions */
    NANO_ERR_CHECKSUM_MISMATCH = -10, /**< CRC32 integrity check failed */
    NANO_ERR_TRUNCATED_FILE  = -11  /**< File size smaller than header or declared tensors */
} NanoStatus;

/**
 * @brief 64-byte Header structure for .nano binary model distribution packages.
 */
#pragma pack(push, 1)
typedef struct {
    char      magic[4];            /**< "NANO" = 0x4E414E4F */
    uint16_t  version;             /**< Format version (0x0002 for V2 production, 0x0001 for legacy V1) */
    uint16_t  total_blocks;        /**< 24 */
    uint16_t  state_blocks;        /**< 16 */
    uint16_t  gqa_blocks;          /**< 8 */
    uint32_t  d_model;             /**< 2560 */
    uint32_t  d_ffn;               /**< 6912 */
    uint16_t  n_q;                 /**< 20 */
    uint16_t  n_kv;                /**< 4 */
    uint16_t  d_head;              /**< 128 */
    uint16_t  pad;                 /**< Alignment pad */
    uint32_t  vocab_size;          /**< 65536 */
    uint32_t  max_context;         /**< 10000 */
    uint32_t  crc32;               /**< Stored CRC32 checksum of descriptors + payload */
    uint32_t  tensor_count;        /**< 219 for V2 production, 123 for legacy V1 */
    uint8_t   reserved[20];        /**< Reserved padding */
} NanoBinaryHeader;

/**
 * @brief 32-byte Tensor Descriptor entry in .nano manifest table.
 */
typedef struct {
    uint32_t  tensor_id;           /**< Numerical tensor ID (0..218) */
    uint32_t  quant_type;          /**< NanoQuantType enum */
    uint64_t  offset;              /**< Absolute 64-byte aligned file offset */
    uint64_t  size_bytes;          /**< Byte size of raw tensor data */
    float     scale;               /**< Dequantization scale factor */
    uint32_t  pad;                 /**< Alignment pad */
} NanoTensorDescriptor;
#pragma pack(pop)

#if defined(__cplusplus)
static_assert(sizeof(NanoBinaryHeader) == 64, "NanoBinaryHeader ABI size must be exactly 64 bytes");
static_assert(sizeof(NanoTensorDescriptor) == 32, "NanoTensorDescriptor ABI size must be exactly 32 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(NanoBinaryHeader) == 64, "NanoBinaryHeader ABI size must be exactly 64 bytes");
_Static_assert(sizeof(NanoTensorDescriptor) == 32, "NanoTensorDescriptor ABI size must be exactly 32 bytes");
#endif

/**
 * @brief Native Model State retained after successful loading.
 */
typedef struct {
    char                    model_path[512];
    size_t                  file_size;
    NanoBinaryHeader        header;
    NanoTensorDescriptor*   descriptors;     /**< Pointer to descriptor array */
    const uint8_t*          mmap_ptr;        /**< Pointer to mapped binary memory */
    size_t                  mmap_size;
    bool                    is_mmap;
    uint32_t                tensor_count;
    uint32_t                computed_crc;
    bool                    integrity_verified;
    void*                   platform_file_handle;
    void*                   platform_map_handle;
} NanoModelState;

/**
 * @brief Real Forward Pass Execution Telemetry & Counters (Section 9.8)
 */
typedef struct {
    uint64_t forward_pass_count;
    uint64_t embedding_execution_count;
    uint64_t attention_execution_count;
    uint64_t ffn_execution_count;
    uint64_t norm_execution_count;
    uint64_t logits_generation_count;
    uint64_t sampling_count;
    int32_t  last_selected_token_id;
    float    last_max_logit;
} NanoForwardPassStats;

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
