/**
 * @file utf8_ring_buffer.cpp
 * @brief 16-Byte UTF-8 Streaming Accumulation Ring Buffer.
 * Prevents broken multi-byte sequences from rendering on client screens during token emission.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "../../include/nano_tokenizer.h"

struct NanoUtf8RingBuffer {
    char   pending_bytes[16];
    size_t pending_count;
};

static inline int utf8_sequence_length(uint8_t lead_byte) {
    if ((lead_byte & 0x80) == 0x00) return 1; // 0xxxxxxx: ASCII (1 byte)
    if ((lead_byte & 0xE0) == 0xC0) return 2; // 110xxxxx: 2 bytes
    if ((lead_byte & 0xF0) == 0xE0) return 3; // 1110xxxx: 3 bytes (Bengali)
    if ((lead_byte & 0xF8) == 0xF0) return 4; // 11110xxx: 4 bytes (Emoji)
    return 1; // Fallback / invalid continuation byte
}

NanoUtf8RingBuffer* nano_utf8_buffer_create(void) {
    NanoUtf8RingBuffer* buf = (NanoUtf8RingBuffer*)malloc(sizeof(NanoUtf8RingBuffer));
    if (buf) {
        buf->pending_count = 0;
        memset(buf->pending_bytes, 0, sizeof(buf->pending_bytes));
    }
    return buf;
}

NanoStatus nano_utf8_buffer_feed(
    NanoUtf8RingBuffer* ring_buf,
    const char* incoming_bytes,
    size_t incoming_len,
    char* out_valid_utf8,
    size_t out_capacity,
    size_t* out_valid_len
) {
    if (!ring_buf || !incoming_bytes || !out_valid_utf8 || !out_valid_len) {
        return NANO_ERR_INVALID_PARAM;
    }
    
    // Combine pending bytes with incoming bytes
    char temp_stream[64];
    size_t total_in = ring_buf->pending_count + incoming_len;
    if (total_in > 60) total_in = 60; // Defensive cap
    
    memcpy(temp_stream, ring_buf->pending_bytes, ring_buf->pending_count);
    memcpy(temp_stream + ring_buf->pending_count, incoming_bytes, incoming_len);
    
    size_t read_idx = 0;
    size_t write_idx = 0;
    
    while (read_idx < total_in) {
        uint8_t lead = (uint8_t)temp_stream[read_idx];
        int seq_len = utf8_sequence_length(lead);
        
        if (read_idx + seq_len <= total_in) {
            // Full complete character available
            if (write_idx + seq_len < out_capacity) {
                memcpy(out_valid_utf8 + write_idx, temp_stream + read_idx, seq_len);
                write_idx += seq_len;
                read_idx += seq_len;
            } else {
                break;
            }
        } else {
            // Incomplete multi-byte sequence at the tail — hold in pending buffer
            break;
        }
    }
    
    out_valid_utf8[write_idx] = '\0';
    *out_valid_len = write_idx;
    
    // Store remaining incomplete bytes into ring_buf
    size_t remaining = total_in - read_idx;
    if (remaining > 0 && remaining < 16) {
        memcpy(ring_buf->pending_bytes, temp_stream + read_idx, remaining);
        ring_buf->pending_count = remaining;
    } else {
        ring_buf->pending_count = 0;
    }
    
    return NANO_SUCCESS;
}

void nano_utf8_buffer_flush(
    NanoUtf8RingBuffer* ring_buf,
    char* out_buf,
    size_t out_capacity,
    size_t* out_len
) {
    if (!ring_buf || !out_buf || !out_len) return;
    size_t to_flush = ring_buf->pending_count;
    if (to_flush >= out_capacity) to_flush = out_capacity - 1;
    
    memcpy(out_buf, ring_buf->pending_bytes, to_flush);
    out_buf[to_flush] = '\0';
    *out_len = to_flush;
    ring_buf->pending_count = 0;
}

void nano_utf8_buffer_destroy(NanoUtf8RingBuffer* ring_buf) {
    if (ring_buf) {
        free(ring_buf);
    }
}
