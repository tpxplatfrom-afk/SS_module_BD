/**
 * @file memory_arena.cpp
 * @brief Static Monolithic Memory Arena Allocator for THSA-2B V1.
 * Guarantees O(1) allocation time, zero malloc during decode, and zero memory leaks.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "../../include/nano_types.h"
#include "../../include/nano_config.h"

#define NANO_ARENA_MAX_CAPACITY_BYTES (250 * 1024 * 1024) // 250 MB Hard Ceiling

typedef struct {
    uint8_t* base_ptr;
    size_t   total_capacity_bytes;
    size_t   allocated_bytes;
    
    // Sub-arena pointers
    uint8_t* kv_cache_ptr;
    size_t   kv_cache_size;
    
    uint8_t* activation_ptr;
    size_t   activation_size;
    
    uint8_t* workspace_ptr;
    size_t   workspace_size;
    
    uint8_t* runtime_meta_ptr;
    size_t   runtime_meta_size;
} NanoMemoryArena;

#ifdef __cplusplus
extern "C" {
#endif

NanoMemoryArena* nano_arena_create(const NanoModelConfig* config) {
    if (!config) return NULL;
    
    NanoMemoryArena* arena = (NanoMemoryArena*)malloc(sizeof(NanoMemoryArena));
    if (!arena) return NULL;
    
    // 1. Calculate required sub-arena capacities
    // KV-Cache: 2 * L * N_gqa * N_kv * D_head * 0.5 bytes
    size_t kv_bytes = (size_t)(2 * config->max_context_tokens * config->gqa_blocks * 
                               config->n_kv_heads * config->d_head * 0.5f);
    
    // Activation chunk buffer (256 tokens micro-chunk)
    size_t act_bytes = 25 * 1024 * 1024; // 25 MB
    
    // Workspace scratchpad
    size_t ws_bytes = 20 * 1024 * 1024;  // 20 MB
    
    // Runtime metadata & LayerNorm
    size_t meta_bytes = 15 * 1024 * 1024; // 15 MB
    
    size_t total_needed = kv_bytes + act_bytes + ws_bytes + meta_bytes;
    if (total_needed > NANO_ARENA_MAX_CAPACITY_BYTES) {
        free(arena);
        return NULL;
    }
    
    // 2. Allocate monolithic memory block (64-byte aligned)
#if defined(_WIN32)
    arena->base_ptr = (uint8_t*)_aligned_malloc(total_needed, 64);
#else
    if (posix_memalign((void**)&arena->base_ptr, 64, total_needed) != 0) {
        free(arena);
        return NULL;
    }
#endif

    if (!arena->base_ptr) {
        free(arena);
        return NULL;
    }
    
    memset(arena->base_ptr, 0, total_needed);
    arena->total_capacity_bytes = total_needed;
    arena->allocated_bytes      = total_needed;
    
    // 3. Slice sub-arenas
    uint8_t* curr = arena->base_ptr;
    
    arena->kv_cache_ptr  = curr;
    arena->kv_cache_size = kv_bytes;
    curr += kv_bytes;
    
    arena->activation_ptr  = curr;
    arena->activation_size = act_bytes;
    curr += act_bytes;
    
    arena->workspace_ptr  = curr;
    arena->workspace_size = ws_bytes;
    curr += ws_bytes;
    
    arena->runtime_meta_ptr  = curr;
    arena->runtime_meta_size = meta_bytes;
    
    return arena;
}

void nano_arena_reset_workspace(NanoMemoryArena* arena) {
    if (!arena || !arena->workspace_ptr) return;
    memset(arena->workspace_ptr, 0, arena->workspace_size);
}

void nano_arena_reset_kv_cache(NanoMemoryArena* arena) {
    if (!arena || !arena->kv_cache_ptr) return;
    memset(arena->kv_cache_ptr, 0, arena->kv_cache_size);
}

void nano_arena_destroy(NanoMemoryArena* arena) {
    if (!arena) return;
    if (arena->base_ptr) {
#if defined(_WIN32)
        _aligned_free(arena->base_ptr);
#else
        free(arena->base_ptr);
#endif
        arena->base_ptr = NULL;
    }
    free(arena);
}

#ifdef __cplusplus
}
#endif
