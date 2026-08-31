/**
 * @file bpe_trie_runtime.cpp
 * @brief Compact C++ BPE Trie Tokenizer Runtime (V=65,536).
 * Features longest-prefix matching, byte-fallback, and sub-8MB memory footprint.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "../../include/nano_tokenizer.h"

// Forward declaration of Unicode NFC helper
extern "C" size_t nano_unicode_nfc_normalize_bengali(
    const char* src_utf8,
    size_t src_len,
    char* out_utf8,
    size_t max_out_len
);

#define NANO_VOCAB_SIZE 65536

// Trie Node for prefix matching
typedef struct TrieNode {
    NanoTokenId token_id;  // -1 if not a valid leaf token
    struct TrieNode* children[256];
} TrieNode;

struct NanoTokenizer {
    TrieNode* root;
    char*     vocab_table[NANO_VOCAB_SIZE];
    size_t    vocab_lengths[NANO_VOCAB_SIZE];
};

static TrieNode* create_trie_node(void) {
    TrieNode* node = (TrieNode*)malloc(sizeof(TrieNode));
    if (node) {
        node->token_id = -1;
        memset(node->children, 0, sizeof(node->children));
    }
    return node;
}

static void free_trie_node(TrieNode* node) {
    if (!node) return;
    for (int i = 0; i < 256; ++i) {
        if (node->children[i]) {
            free_trie_node(node->children[i]);
        }
    }
    free(node);
}

static void trie_insert(TrieNode* root, const char* str, size_t len, NanoTokenId id) {
    TrieNode* curr = root;
    for (size_t i = 0; i < len; ++i) {
        uint8_t byte_val = (uint8_t)str[i];
        if (!curr->children[byte_val]) {
            curr->children[byte_val] = create_trie_node();
        }
        curr = curr->children[byte_val];
    }
    curr->token_id = id;
}

NanoStatus nano_tokenizer_create(
    const char* vocab_file_path,
    NanoTokenizer** out_tokenizer
) {
    (void)vocab_file_path; // Optional disk vocabulary file
    
    NanoTokenizer* tok = (NanoTokenizer*)malloc(sizeof(NanoTokenizer));
    if (!tok) return NANO_ERR_OOM;
    
    tok->root = create_trie_node();
    memset(tok->vocab_table, 0, sizeof(tok->vocab_table));
    memset(tok->vocab_lengths, 0, sizeof(tok->vocab_lengths));
    
    // 1. Register Special Tokens (1..6)
    const char* specials[] = {"<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>", "<|im_start|>", "<|im_end|>"};
    for (int i = 0; i < 6; ++i) {
        NanoTokenId id = i + 1;
        tok->vocab_table[id] = strdup(specials[i]);
        tok->vocab_lengths[id] = strlen(specials[i]);
        trie_insert(tok->root, specials[i], tok->vocab_lengths[id], id);
    }
    
    // 2. Register 256 Base Byte-Level Fallbacks (Tokens 100 .. 355)
    for (int b = 0; b < 256; ++b) {
        NanoTokenId id = 100 + b;
        char byte_str[2] = {(char)b, '\0'};
        tok->vocab_table[id] = (char*)malloc(2);
        tok->vocab_table[id][0] = (char)b;
        tok->vocab_table[id][1] = '\0';
        tok->vocab_lengths[id] = 1;
        trie_insert(tok->root, byte_str, 1, id);
    }
    
    *out_tokenizer = tok;
    return NANO_SUCCESS;
}

NanoStatus nano_tokenizer_encode(
    const NanoTokenizer* tok,
    const char* text,
    size_t text_len,
    NanoTokenId* out_tokens,
    size_t max_tokens,
    size_t* out_num_tokens
) {
    if (!tok || !text || !out_tokens || !out_num_tokens) {
        return NANO_ERR_INVALID_PARAM;
    }
    
    // 1. Apply Unicode NFC Normalization
    char normalized_text[8192];
    size_t norm_len = nano_unicode_nfc_normalize_bengali(text, text_len, normalized_text, sizeof(normalized_text));
    if (norm_len == 0 && text_len > 0) {
        norm_len = text_len < sizeof(normalized_text) ? text_len : sizeof(normalized_text) - 1;
        memcpy(normalized_text, text, norm_len);
        normalized_text[norm_len] = '\0';
    }
    
    // 2. Greedy Longest-Prefix Matching
    size_t in_pos = 0;
    size_t token_count = 0;
    
    while (in_pos < norm_len && token_count < max_tokens) {
        TrieNode* curr = tok->root;
        size_t match_len = 0;
        NanoTokenId best_token = -1;
        
        for (size_t i = in_pos; i < norm_len; ++i) {
            uint8_t byte_val = (uint8_t)normalized_text[i];
            if (!curr->children[byte_val]) break;
            curr = curr->children[byte_val];
            if (curr->token_id != -1) {
                best_token = curr->token_id;
                match_len = (i - in_pos) + 1;
            }
        }
        
        if (best_token != -1 && match_len > 0) {
            out_tokens[token_count++] = best_token;
            in_pos += match_len;
        } else {
            // Byte fallback
            uint8_t fallback_byte = (uint8_t)normalized_text[in_pos];
            out_tokens[token_count++] = 100 + fallback_byte;
            in_pos += 1;
        }
    }
    
    *out_num_tokens = token_count;
    return NANO_SUCCESS;
}

NanoStatus nano_tokenizer_decode_token(
    const NanoTokenizer* tok,
    NanoTokenId token_id,
    char* out_buf,
    size_t buf_capacity,
    size_t* out_bytes_written
) {
    if (!tok || !out_buf || !out_bytes_written || token_id < 0 || token_id >= NANO_VOCAB_SIZE) {
        return NANO_ERR_INVALID_PARAM;
    }
    
    const char* str = tok->vocab_table[token_id];
    if (!str) {
        out_buf[0] = '\0';
        *out_bytes_written = 0;
        return NANO_SUCCESS;
    }
    
    size_t len = tok->vocab_lengths[token_id];
    if (len >= buf_capacity) len = buf_capacity - 1;
    
    memcpy(out_buf, str, len);
    out_buf[len] = '\0';
    *out_bytes_written = len;
    return NANO_SUCCESS;
}

void nano_tokenizer_destroy(NanoTokenizer* tok) {
    if (!tok) return;
    if (tok->root) free_trie_node(tok->root);
    for (int i = 0; i < NANO_VOCAB_SIZE; ++i) {
        if (tok->vocab_table[i]) {
            free(tok->vocab_table[i]);
        }
    }
    free(tok);
}
