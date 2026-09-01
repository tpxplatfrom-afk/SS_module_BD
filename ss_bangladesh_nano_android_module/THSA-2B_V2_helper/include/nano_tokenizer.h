/**
 * @file nano_tokenizer.h
 * @brief Public C Interface for Compact Multilingual Trie Tokenizer Runtime (V=65,536).
 * English + Bengali native support with Unicode NFC, conjunct preservation, and UTF-8 stream buffer.
 */

#ifndef NANO_TOKENIZER_H
#define NANO_TOKENIZER_H

#include "nano_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Tokenizer Handle */
typedef struct NanoTokenizer NanoTokenizer;

/* Opaque UTF-8 Accumulation Buffer Handle */
typedef struct NanoUtf8RingBuffer NanoUtf8RingBuffer;

/**
 * @brief Create and initialize the BPE Trie Tokenizer runtime.
 * Memory footprint <= 8.0 MB.
 * @param vocab_file_path Path to the serialized vocabulary / trie binary
 * @param out_tokenizer Pointer to receive the allocated tokenizer handle
 * @return NANO_SUCCESS on success, or error code
 */
NANO_API NanoStatus nano_tokenizer_create(
    const char* vocab_file_path,
    NanoTokenizer** out_tokenizer
);

/**
 * @brief Encode UTF-8 text into an array of token IDs.
 * Automatically applies Unicode NFC normalization before tokenization.
 * @param tok Valid tokenizer handle
 * @param text Input UTF-8 string
 * @param text_len Length of input string in bytes
 * @param out_tokens Buffer to receive encoded token IDs
 * @param max_tokens Maximum capacity of out_tokens buffer
 * @param out_num_tokens Pointer to receive actual number of tokens written
 * @return NANO_SUCCESS on success
 */
NANO_API NanoStatus nano_tokenizer_encode(
    const NanoTokenizer* tok,
    const char* text,
    size_t text_len,
    NanoTokenId* out_tokens,
    size_t max_tokens,
    size_t* out_num_tokens
);

/**
 * @brief Decode a single token ID into a UTF-8 character string.
 * @param tok Valid tokenizer handle
 * @param token_id Token ID to decode
 * @param out_buf Buffer to receive decoded string
 * @param buf_capacity Capacity of out_buf in bytes
 * @param out_bytes_written Pointer to receive number of bytes written
 * @return NANO_SUCCESS on success
 */
NANO_API NanoStatus nano_tokenizer_decode_token(
    const NanoTokenizer* tok,
    NanoTokenId token_id,
    char* out_buf,
    size_t buf_capacity,
    size_t* out_bytes_written
);

/**
 * @brief Free and destroy the tokenizer runtime.
 * @param tok Tokenizer handle to free
 */
NANO_API void nano_tokenizer_destroy(NanoTokenizer* tok);

/* ========================================================================= */
/* Streaming UTF-8 Accumulation Buffer (Section 11.2)                         */
/* ========================================================================= */

/**
 * @brief Create a 16-byte UTF-8 accumulation ring buffer for streaming.
 */
NANO_API NanoUtf8RingBuffer* nano_utf8_buffer_create(void);

/**
 * @brief Feed raw token bytes into buffer and extract complete valid UTF-8 characters.
 * Prevents bisected multi-byte Bengali character corruption on streaming displays.
 * @param ring_buf Buffer handle
 * @param incoming_bytes Raw bytes from decoded token
 * @param incoming_len Number of incoming bytes
 * @param out_valid_utf8 Buffer to receive fully validated UTF-8 string
 * @param out_capacity Capacity of out_valid_utf8 buffer
 * @param out_valid_len Number of complete bytes written
 * @return NANO_SUCCESS on success
 */
NANO_API NanoStatus nano_utf8_buffer_feed(
    NanoUtf8RingBuffer* ring_buf,
    const char* incoming_bytes,
    size_t incoming_len,
    char* out_valid_utf8,
    size_t out_capacity,
    size_t* out_valid_len
);

/**
 * @brief Flush any remaining bytes in ring buffer on sequence completion.
 */
NANO_API void nano_utf8_buffer_flush(
    NanoUtf8RingBuffer* ring_buf,
    char* out_buf,
    size_t out_capacity,
    size_t* out_len
);

/**
 * @brief Destroy the UTF-8 ring buffer.
 */
NANO_API void nano_utf8_buffer_destroy(NanoUtf8RingBuffer* ring_buf);

#ifdef __cplusplus
}
#endif

#endif /* NANO_TOKENIZER_H */
