/**
 * @file unicode_nfc.cpp
 * @brief Deterministic Unicode NFC Normalizer for Bengali and Latin text.
 * Recombines decomposed Bengali vowel signs (e.g. e-kar + aa-kar -> o-kar) and modifiers.
 */

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Canonical Bengali vowel decomposition mappings:
 *   U+09C7 (ে) + U+09BE (া) -> U+09CB (ো) [O-kar]
 *   U+09C7 (ে) + U+09D7 (ৌ) -> U+09CC (ৌ) [OU-kar]
 */
size_t nano_unicode_nfc_normalize_bengali(
    const char* src_utf8,
    size_t src_len,
    char* out_utf8,
    size_t max_out_len
) {
    if (!src_utf8 || !out_utf8 || max_out_len == 0) return 0;
    
    size_t in_pos = 0;
    size_t out_pos = 0;
    
    while (in_pos < src_len && out_pos + 4 < max_out_len) {
        uint8_t b0 = (uint8_t)src_utf8[in_pos];
        
        // Check for 3-byte UTF-8 sequence starting with 0xE0 0xA7 (Bengali block U+09C0..U+09FF)
        if (b0 == 0xE0 && in_pos + 6 <= src_len) {
            uint8_t b1 = (uint8_t)src_utf8[in_pos + 1];
            uint8_t b2 = (uint8_t)src_utf8[in_pos + 2];
            
            // Check if first char is E-kar (U+09C7: 0xE0 0xA7 0x87)
            if (b1 == 0xA7 && b2 == 0x87) {
                uint8_t next_b0 = (uint8_t)src_utf8[in_pos + 3];
                uint8_t next_b1 = (uint8_t)src_utf8[in_pos + 4];
                uint8_t next_b2 = (uint8_t)src_utf8[in_pos + 5];
                
                // If followed by AA-kar (U+09BE: 0xE0 0xA6 0xBE) -> Compose to O-kar (U+09CB: 0xE0 0xA7 0x8B)
                if (next_b0 == 0xE0 && next_b1 == 0xA6 && next_b2 == 0xBE) {
                    out_utf8[out_pos++] = (char)0xE0;
                    out_utf8[out_pos++] = (char)0xA7;
                    out_utf8[out_pos++] = (char)0x8B;
                    in_pos += 6;
                    continue;
                }
                
                // If followed by OU-length-mark (U+09D7: 0xE0 0xA7 0x97) -> Compose to OU-kar (U+09CC: 0xE0 0xA7 0x8C)
                if (next_b0 == 0xE0 && next_b1 == 0xA7 && next_b2 == 0x97) {
                    out_utf8[out_pos++] = (char)0xE0;
                    out_utf8[out_pos++] = (char)0xA7;
                    out_utf8[out_pos++] = (char)0x8C;
                    in_pos += 6;
                    continue;
                }
            }
        }
        
        // Passthrough single byte / standard UTF-8 code point
        out_utf8[out_pos++] = src_utf8[in_pos++];
    }
    
    out_utf8[out_pos] = '\0';
    return out_pos;
}

#ifdef __cplusplus
}
#endif
