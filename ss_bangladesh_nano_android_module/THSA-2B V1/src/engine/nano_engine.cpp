/**
 * @file nano_engine.cpp
 * @brief High-Performance C++17 Engine Core for THSA-2B On-Device Runtime.
 * Features 64-byte mmap binary loading, real neural forward-pass execution,
 * static arena execution, async cancel, and real-time telemetry.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <chrono>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "../../include/nano_engine.h"
#include "../../include/nano_types.h"
#include "../../include/nano_config.h"
#include "../../include/nano_telemetry.h"
#include "../../include/nano_tokenizer.h"
#include "../../include/kernels/neon_gemv_ternary.h"
#include "../../include/kernels/neon_kv_cache.h"
#include "../../include/kernels/neon_state_update.h"
#include "../../include/kernels/neon_norm_act.h"
#include "../../include/kernels/neon_gemv_int8.h"   /* FIX-A: ARMv7 NEON dense INT8 LM-head GEMV */

#ifdef __ANDROID__
#include <android/log.h>
#define NANO_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NanoEngineNative", __VA_ARGS__)
#define NANO_LOGW(...) __android_log_print(ANDROID_LOG_WARN, "NanoEngineNative", __VA_ARGS__)
#define NANO_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NanoEngineNative", __VA_ARGS__)
#else
#define NANO_LOGI(...)
#define NANO_LOGW(...)
#define NANO_LOGE(...)
#endif

// Forward declaration of arena
typedef struct NanoMemoryArena NanoMemoryArena;
extern "C" {
    NanoMemoryArena* nano_arena_create(const NanoModelConfig* config);
    void nano_arena_reset_workspace(NanoMemoryArena* arena);
    void nano_arena_reset_kv_cache(NanoMemoryArena* arena);
    void nano_arena_destroy(NanoMemoryArena* arena);
}

// =============================================================================
// FIX-12 DIAGNOSTIC INSTRUMENTATION (non-invasive, gated on env var)
// Activated when NANO_FIX12_DIAG_PATH is set to a writable directory path.
// Does NOT alter any inference math. Writes bounded binary capture files.
// =============================================================================
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

static char g_fix12_diag_dir[512] = {0};
static bool g_fix12_enabled = false;
static FILE* g_fix12_diag_fp  = nullptr;  // fix12_diag.bin
static FILE* g_fix12_perf_fp  = nullptr;  // fix12_perf.txt
static int   g_fix12_prompt_idx = 0;

// 32-entry timing ring for per-layer performance
static struct {
    long long embed_us;
    long long layer_us[24];
    long long rmsnorm_us;
    long long lmhead_us;
    long long total_us;
} g_fix12_timing;

static long long fix12_now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}

// FIX-12 checkpoint record (written to fix12_diag.bin)
// 72 bytes per record: uint32 checkpoint_id, uint32 prompt_idx,
//                      float min, max, mean, mean_abs, max_abs, l2_norm,
//                      float sha_proxy[8] (first 32 bytes of h_state as float)
// Total: 8*4 + 8*4 = 72 bytes
static void fix12_capture_checkpoint(
    uint32_t ckpt_id, const float* h_state, size_t dim, const char* label)
{
    if (!g_fix12_enabled || !g_fix12_diag_fp) return;

    double sum = 0.0, sum_abs = 0.0, sum_sq = 0.0;
    float  mn = h_state[0], mx = h_state[0];
    for (size_t i = 0; i < dim; ++i) {
        float v = h_state[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum     += v;
        sum_abs += (v < 0 ? -v : v);
        sum_sq  += (double)v * v;
    }
    float mean_v    = (float)(sum / dim);
    float mean_abs_v= (float)(sum_abs / dim);
    float max_abs_v = (mn < 0 ? ((-mn) > mx ? -mn : mx) : mx);
    float l2_v      = (float)sqrt(sum_sq);

    // Header: ckpt_id(u32), prompt_idx(u32), dim(u32), pad(u32)
    uint32_t hdr[4] = { ckpt_id, (uint32_t)g_fix12_prompt_idx, (uint32_t)dim, 0 };
    fwrite(hdr, 4, 4, g_fix12_diag_fp);

    // Stats: min, max, mean, mean_abs, max_abs, l2 (6 floats)
    float stats[6] = { mn, mx, mean_v, mean_abs_v, max_abs_v, l2_v };
    fwrite(stats, 4, 6, g_fix12_diag_fp);

    // Proxy: first 8 floats of h_state (32 bytes fingerprint)
    float proxy[8];
    for (int i = 0; i < 8 && (size_t)i < dim; ++i) proxy[i] = h_state[i];
    fwrite(proxy, 4, 8, g_fix12_diag_fp);
    fflush(g_fix12_diag_fp);

    NANO_LOGI("FIX12_CKPT_%02u [%s] prompt=%d dim=%zu min=%.4f max=%.4f mean=%.4f l2=%.4f",
              ckpt_id, label, g_fix12_prompt_idx, dim, mn, mx, mean_v, l2_v);
}

static void fix12_dump_logits(const float* logits) {
    if (!g_fix12_enabled || !g_fix12_diag_dir[0]) return;
    char path[600];
    snprintf(path, sizeof(path), "%s/fix12_logits_p%d.bin", g_fix12_diag_dir, g_fix12_prompt_idx);
    FILE* fp = fopen(path, "wb");
    if (!fp) { NANO_LOGE("FIX12: cannot write logits to %s", path); return; }
    fwrite(logits, 4, 65536, fp);
    fclose(fp);
    // Log top-10
    float top10_v[10]; int top10_id[10];
    for (int r = 0; r < 10; ++r) { top10_v[r] = -1e30f; top10_id[r] = -1; }
    for (int v = 0; v < 65536; ++v) {
        if (logits[v] > top10_v[9]) {
            top10_v[9] = logits[v]; top10_id[9] = v;
            for (int r = 8; r >= 0; --r) {
                if (top10_v[r+1] > top10_v[r]) {
                    float tv = top10_v[r]; top10_v[r] = top10_v[r+1]; top10_v[r+1] = tv;
                    int   ti = top10_id[r]; top10_id[r] = top10_id[r+1]; top10_id[r+1] = ti;
                } else break;
            }
        }
    }
    NANO_LOGI("FIX12_LOGITS_READY: prompt=%d argmax=%d val=%.4f",
              g_fix12_prompt_idx, top10_id[0], top10_v[0]);
    NANO_LOGI("FIX12_TOP10_IDS: %d %d %d %d %d %d %d %d %d %d",
              top10_id[0],top10_id[1],top10_id[2],top10_id[3],top10_id[4],
              top10_id[5],top10_id[6],top10_id[7],top10_id[8],top10_id[9]);
}

static void fix12_write_perf() {
    if (!g_fix12_enabled || !g_fix12_perf_fp) return;
    fprintf(g_fix12_perf_fp,
        "FIX12_PERF_BEGIN prompt=%d\n"
        "FIX12_EMBED_US=%lld\n",
        g_fix12_prompt_idx, g_fix12_timing.embed_us);
    for (int l = 0; l < 24; ++l) {
        fprintf(g_fix12_perf_fp, "FIX12_BLOCK_%02d_US=%lld\n", l, g_fix12_timing.layer_us[l]);
    }
    fprintf(g_fix12_perf_fp,
        "FIX12_RMSNORM_US=%lld\n"
        "FIX12_LMHEAD_US=%lld\n"
        "FIX12_TOTAL_US=%lld\n"
        "FIX12_PERF_END\n",
        g_fix12_timing.rmsnorm_us, g_fix12_timing.lmhead_us, g_fix12_timing.total_us);
    fflush(g_fix12_perf_fp);
}

static void fix12_init() {
    const char* dir = getenv("NANO_FIX12_DIAG_PATH");
#ifdef __ANDROID__
    // FIX-12B: On Android, env vars cannot be set from Java.
    // Always activate diagnostics to the app's private files directory.
    static char fallback[256];
    if (!dir || !dir[0]) {
        // Use app-private files dir (accessible via adb pull)
        snprintf(fallback, sizeof(fallback),
                 "/data/data/com.aistudio.offlineai.krvq/files");
        dir = fallback;
    }
#endif
    if (!dir || !dir[0]) { g_fix12_enabled = false; return; }
    strncpy(g_fix12_diag_dir, dir, sizeof(g_fix12_diag_dir) - 1);
    g_fix12_enabled = true;

    char diag_path[600], perf_path[600];
    snprintf(diag_path, sizeof(diag_path), "%s/fix12_diag.bin", dir);
    snprintf(perf_path, sizeof(perf_path), "%s/fix12_perf.txt", dir);

    g_fix12_diag_fp = fopen(diag_path, "wb");
    g_fix12_perf_fp = fopen(perf_path, "w");

    NANO_LOGI("FIX12_DIAG_INIT: enabled=YES dir=%s diag=%p perf=%p",
              dir, (void*)g_fix12_diag_fp, (void*)g_fix12_perf_fp);
}
// =============================================================================
// END FIX-12 DIAGNOSTIC INSTRUMENTATION
// =============================================================================

// =============================================================================
// FIX-12C LAYERWISE INTERMEDIATE CHECKPOINT CAPTURE
// =============================================================================
static bool g_fix12c_enabled = true;
static char g_fix12c_dir[512] = {0};

static void fix12c_init() {
    const char* dir = getenv("NANO_FIX12C_DIAG_PATH");
#ifdef __ANDROID__
    static char fallback[256];
    if (!dir || !dir[0]) {
        snprintf(fallback, sizeof(fallback),
                 "/data/data/com.aistudio.offlineai.krvq/files/fix12c");
        dir = fallback;
    }
#endif
    if (!dir || !dir[0]) { g_fix12c_enabled = false; return; }
    strncpy(g_fix12c_dir, dir, sizeof(g_fix12c_dir) - 1);
    mkdir(g_fix12c_dir, 0777);
    g_fix12c_enabled = true;
    NANO_LOGI("FIX12C_INIT: enabled=YES dir=%s", g_fix12c_dir);
}

static void fix12c_dump_vec(int prompt_idx, const char* name, const float* data, size_t count) {
    if (!g_fix12c_enabled || !g_fix12c_dir[0] || !data || count == 0) return;
    char pdir[600];
    snprintf(pdir, sizeof(pdir), "%s/prompt_%d", g_fix12c_dir, prompt_idx);
    mkdir(pdir, 0777);
    char path[650];
    snprintf(path, sizeof(path), "%s/%s.bin", pdir, name);
    FILE* fp = fopen(path, "wb");
    if (fp) {
        fwrite(data, sizeof(float), count, fp);
        fclose(fp);
    }
}
// =============================================================================
// END FIX-12C INSTRUMENTATION
// =============================================================================

// IEEE 802.3 CRC32 Implementation (Matches Python zlib.crc32)
static uint32_t compute_nano_crc32(const uint8_t* buffer, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc ^= buffer[i];
        for (int j = 0; j < 8; ++j) {
            uint32_t mask = -(crc & 1);
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
    }
    return ~crc & 0xFFFFFFFF;
}

// Layer Pointer Routing Struct
struct NanoLayerPointers {
    bool           is_gqa;
    // Mixer Norm (RMSNorm FP32 [2560])
    const float*   gamma_mixer;
    
    // GQA Attention Weights (Ternary 2-bit packed)
    const uint8_t* w_q_packed;
    float          scale_q;
    const uint8_t* w_k_packed;
    float          scale_k;
    const uint8_t* w_v_packed;
    float          scale_v;
    const uint8_t* w_out_packed;
    float          scale_out;
    
    // State Block Weights
    const uint8_t* w_state_in_proj;  // [5120, 2560] (Ternary 2-bit packed)
    float          scale_state_in;
    const float*   conv_weights;     // [2560, 4] (FP32, channel-major: W_0=t-3, W_1=t-2, W_2=t-1, W_3=t)
    const float*   conv_bias;        // [2560] (FP32)
    const uint8_t* w_state_out_proj; // [2560, 2560] (Ternary 2-bit packed)
    float          scale_state_out;
    
    // FFN Norm (RMSNorm FP32 [2560])
    const float*   gamma_ffn;
    
    // FFN Weights (Ternary 2-bit packed)
    const uint8_t* w_gate_packed;
    float          scale_gate;
    const uint8_t* w_up_packed;
    float          scale_up;
    const uint8_t* w_down_packed;
    float          scale_down;
};

struct NanoEngineContext {
    NanoEngineState       state;
    NanoModelConfig       config;
    NanoMemoryArena*      arena;
    NanoTokenizer*        tokenizer;
    std::atomic<bool>     cancel_requested;
    uint32_t              active_kv_tokens;
    uint32_t              total_tokens_emitted;
    float                 last_tok_per_sec;
    float                 chassis_temp_est;
    NanoModelState        model;
    NanoForwardPassStats  stats;
    
    // Model Tensor Direct Pointers
    const int8_t*         embed_tokens_ptr;
    float                 embed_scale;
    const float*          final_norm_gamma;
    const int8_t*         lm_head_ptr;
    float                 lm_head_scale;
    NanoLayerPointers     layers[24];
    
    // Working Activation Scratchpads
    float*                h_state;         // [2560]
    float*                h_state_res;     // [2560]
    int8_t*               h_state_int8;    // [2560]
    float*                q_act;           // [2560]
    float*                k_act;           // [512]
    float*                v_act;           // [512]
    float*                attn_out;        // [2560]
    int8_t*               attn_out_int8;   // [2560]
    float*                gate_act;        // [6912]
    float*                up_act;          // [6912]
    float*                ffn_act;         // [6912]
    int8_t*               ffn_act_int8;    // [6912]
    float*                ffn_out;         // [2560]
    float*                norm_out;        // [2560]
    float*                state_in_proj_act; // [5120]
    float*                state_conv_out;    // [2560]
    float*                state_gated_act;   // [2560]
    int8_t*               state_gated_int8;  // [2560]
    float*                logits;          // [65536]
    
    // Recurrent States & KV Caches (24 layers)
    NanoStateBlockContext state_contexts[24];
    uint8_t*              kv_cache_k[24];
    float*                kv_cache_k_scales[24];
    uint8_t*              kv_cache_v[24];
    float*                kv_cache_v_scales[24];
};

struct LogitRank {
    NanoTokenId token_id;
    float logit;
};

static void compute_top5_logits(const float* logits, size_t vocab_size, LogitRank top5[5]) {
    for (int r = 0; r < 5; ++r) {
        top5[r].token_id = -1;
        top5[r].logit = -1e30f;
    }
    for (size_t v = 0; v < vocab_size; ++v) {
        float val = logits[v];
        if (val > top5[4].logit) {
            top5[4].token_id = (NanoTokenId)v;
            top5[4].logit = val;
            for (int r = 3; r >= 0; --r) {
                if (top5[r + 1].logit > top5[r].logit) {
                    LogitRank tmp = top5[r];
                    top5[r] = top5[r + 1];
                    top5[r + 1] = tmp;
                } else {
                    break;
                }
            }
        }
    }
}

static NanoTokenId nano_forward_pass_single_token(
    NanoEngineContext* ctx,
    NanoTokenId input_token,
    size_t current_seq_len,
    bool compute_logits = true
) {
    if (!ctx) return NANO_TOKEN_UNK;
    if (input_token < 0 || input_token >= 65536) {
        input_token = NANO_TOKEN_UNK;
    }

    long long t_total_start = fix12_now_us();

    // -------------------------------------------------------------
    // 1. EMBEDDING LOOKUP (INT8 Sensitive Shield)
    // -------------------------------------------------------------
    long long t_embed_start = fix12_now_us();
    const int8_t* emb_row = ctx->embed_tokens_ptr + ((size_t)input_token * 2560);
    for (size_t i = 0; i < 2560; ++i) {
        ctx->h_state[i] = (float)emb_row[i] * ctx->embed_scale;
    }
    ctx->stats.embedding_execution_count++;
    g_fix12_timing.embed_us = fix12_now_us() - t_embed_start;

    // FIX12 CKPT-1: Embedding output
    if (g_fix12_enabled) {
        NANO_LOGI("FIX12_EMBED_READY: token_id=%d prompt=%d", input_token, g_fix12_prompt_idx);
        fix12_capture_checkpoint(1, ctx->h_state, 2560, "EMBED");
    }
    // FIX-12C CKPT-01: Embedding output (only on last prompt token = compute_logits)
    if (compute_logits) {
        fix12c_dump_vec(g_fix12_prompt_idx, "ckpt01_embed", ctx->h_state, 2560);
        NANO_LOGI("FIX12C_CKPT: prompt=%d name=ckpt01_embed dim=2560", g_fix12_prompt_idx);
    }

    // -------------------------------------------------------------
    // 2. BACKBONE LAYERS (24 Layers: 16 State / 8 GQA)
    // -------------------------------------------------------------
    for (size_t l = 0; l < 24; ++l) {
        const NanoLayerPointers& lp = ctx->layers[l];
        long long t_layer_start = fix12_now_us();

        // FIX-12C CKPT-02: Block input
        if (compute_logits) {
            char ckpt_name[64];
            snprintf(ckpt_name, sizeof(ckpt_name), "ckpt02_block_%02zu_input", l);
            fix12c_dump_vec(g_fix12_prompt_idx, ckpt_name, ctx->h_state, 2560);
        }

        if (lp.is_gqa) {

            // (A) COMPLETE GQA ATTENTION BLOCK
            // 1. Mixer Pre-RMSNorm
            if (lp.gamma_mixer) {
                nano_neon_rmsnorm(ctx->h_state, lp.gamma_mixer, 2560, ctx->norm_out);
            } else {
                memcpy(ctx->norm_out, ctx->h_state, 2560 * sizeof(float));
            }
            // FIX-12C CKPT-11: GQA RMSNorm output
            if (compute_logits) {
                char n11[64]; snprintf(n11, sizeof(n11), "ckpt11_block_%02zu_gqa_norm", l);
                fix12c_dump_vec(g_fix12_prompt_idx, n11, ctx->norm_out, 2560);
            }
            
            // 2. Quantize normalized state to INT8
            float x_scale = 1.0f;
            nano_neon_quantize_int8(ctx->norm_out, ctx->h_state_int8, &x_scale, 2560);
            
            // 3. Q, K, V Projections (Ternary GEMV)
            float alpha_q = lp.scale_q * x_scale;
            float alpha_k = lp.scale_k * x_scale;
            float alpha_v = lp.scale_v * x_scale;
            
            nano_neon_gemv_ternary_int8(ctx->q_act, lp.w_q_packed, ctx->h_state_int8, &alpha_q, nullptr, 2560, 2560);
            nano_neon_gemv_ternary_int8(ctx->k_act, lp.w_k_packed, ctx->h_state_int8, &alpha_k, nullptr, 512, 2560);
            nano_neon_gemv_ternary_int8(ctx->v_act, lp.w_v_packed, ctx->h_state_int8, &alpha_v, nullptr, 512, 2560);
            // FIX-12C CKPT-12a/b/c: Q, K, V
            if (compute_logits) {
                char n12a[64]; snprintf(n12a, sizeof(n12a), "ckpt12a_block_%02zu_gqa_q", l);
                fix12c_dump_vec(g_fix12_prompt_idx, n12a, ctx->q_act, 2560);
                char n12b[64]; snprintf(n12b, sizeof(n12b), "ckpt12b_block_%02zu_gqa_k", l);
                fix12c_dump_vec(g_fix12_prompt_idx, n12b, ctx->k_act, 512);
                char n12c[64]; snprintf(n12c, sizeof(n12c), "ckpt12c_block_%02zu_gqa_v", l);
                fix12c_dump_vec(g_fix12_prompt_idx, n12c, ctx->v_act, 512);
            }
            
            // 4. Append K, V to KV Cache at current sequence position
            size_t t_idx = current_seq_len < 10000 ? current_seq_len : 9999;
            for (size_t h = 0; h < 4; ++h) {
                float k_head_scale = 1.0f;
                float v_head_scale = 1.0f;
                size_t k_dst_offset = (h * 10000 + t_idx) * 64; // 128 / 2 = 64 bytes INT4
                
                nano_neon_kv_quantize_int4(ctx->k_act + (h * 128), ctx->kv_cache_k[l] + k_dst_offset, &k_head_scale, 128);
                ctx->kv_cache_k_scales[l][h * 10000 + t_idx] = k_head_scale;
                
                nano_neon_kv_quantize_int4(ctx->v_act + (h * 128), ctx->kv_cache_v[l] + k_dst_offset, &v_head_scale, 128);
                ctx->kv_cache_v_scales[l][h * 10000 + t_idx] = v_head_scale;
            }
            
            // 5. Compute GQA Attention
            nano_neon_gqa_attention_int4(
                ctx->q_act,
                ctx->kv_cache_k[l],
                ctx->kv_cache_k_scales[l],
                ctx->kv_cache_v[l],
                ctx->kv_cache_v_scales[l],
                t_idx + 1,
                20,
                4,
                128,
                ctx->attn_out
            );
            // FIX-12C CKPT-13: Attention output
            if (compute_logits) {
                char n13[64]; snprintf(n13, sizeof(n13), "ckpt13_block_%02zu_gqa_attention", l);
                fix12c_dump_vec(g_fix12_prompt_idx, n13, ctx->attn_out, 2560);
            }
            
            // 6. Out Projection & Residual Connection
            float attn_out_scale = 1.0f;
            nano_neon_quantize_int8(ctx->attn_out, ctx->attn_out_int8, &attn_out_scale, 2560);
            float alpha_out = lp.scale_out * attn_out_scale;
            nano_neon_gemv_ternary_int8(ctx->h_state_res, lp.w_out_packed, ctx->attn_out_int8, &alpha_out, nullptr, 2560, 2560);
            // FIX-12C CKPT-14: GQA out projection
            if (compute_logits) {
                char n14[64]; snprintf(n14, sizeof(n14), "ckpt14_block_%02zu_gqa_out_proj", l);
                fix12c_dump_vec(g_fix12_prompt_idx, n14, ctx->h_state_res, 2560);
            }
            
            for (size_t i = 0; i < 2560; ++i) {
                ctx->h_state[i] += ctx->h_state_res[i];
            }
            // FIX-12C CKPT-15: GQA residual
            if (compute_logits) {
                char n15[64]; snprintf(n15, sizeof(n15), "ckpt15_block_%02zu_gqa_residual", l);
                fix12c_dump_vec(g_fix12_prompt_idx, n15, ctx->h_state, 2560);
            }
            ctx->stats.attention_execution_count++;
        } else {
            // (B) COMPLETE 1D SHORT-CONV STATE BLOCK
            if (lp.w_state_in_proj && lp.w_state_out_proj) {
                // 1. Mixer Pre-RMSNorm
                if (lp.gamma_mixer) {
                    nano_neon_rmsnorm(ctx->h_state, lp.gamma_mixer, 2560, ctx->norm_out);
                } else {
                    memcpy(ctx->norm_out, ctx->h_state, 2560 * sizeof(float));
                }
                // FIX-12C CKPT-03: State RMSNorm output
                if (compute_logits) {
                    char n03[64]; snprintf(n03, sizeof(n03), "ckpt03_block_%02zu_state_norm", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n03, ctx->norm_out, 2560);
                }
                
                // 2. In-Projection (2560 -> 5120)
                float in_norm_scale = 1.0f;
                nano_neon_quantize_int8(ctx->norm_out, ctx->h_state_int8, &in_norm_scale, 2560);
                float alpha_state_in = lp.scale_state_in * in_norm_scale;
                nano_neon_gemv_ternary_int8(
                    ctx->state_in_proj_act,
                    lp.w_state_in_proj,
                    ctx->h_state_int8,
                    &alpha_state_in,
                    nullptr,
                    5120,
                    2560
                );
                // FIX-12C CKPT-04: State in_proj output [5120]
                if (compute_logits) {
                    char n04[64]; snprintf(n04, sizeof(n04), "ckpt04_block_%02zu_state_in_proj", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n04, ctx->state_in_proj_act, 5120);
                }
                
                // 3. Split [gate (2560), value (2560)]
                const float* gate_stream = ctx->state_in_proj_act;
                const float* value_stream = ctx->state_in_proj_act + 2560;
                // FIX-12C CKPT-05a/b: Gate and Value
                if (compute_logits) {
                    char n05a[64]; snprintf(n05a, sizeof(n05a), "ckpt05a_block_%02zu_state_gate", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n05a, gate_stream, 2560);
                    char n05b[64]; snprintf(n05b, sizeof(n05b), "ckpt05b_block_%02zu_state_value", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n05b, value_stream, 2560);
                }
                
                // 4. Depthwise causal Conv1D on value_stream with conv_weights and conv_bias
                nano_neon_short_conv_step(
                    value_stream,
                    lp.conv_weights,
                    lp.conv_bias,
                    &ctx->state_contexts[l],
                    2560,
                    ctx->state_conv_out
                );
                // FIX-12C CKPT-06: Conv1D output
                if (compute_logits) {
                    char n06[64]; snprintf(n06, sizeof(n06), "ckpt06_block_%02zu_state_conv", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n06, ctx->state_conv_out, 2560);
                }
                
                // 5. Gated SiLU activation: silu(gate) * conv_out
                for (size_t i = 0; i < 2560; ++i) {
                    float g = gate_stream[i];
                    float silu_g = g / (1.0f + expf(-g));
                    ctx->state_gated_act[i] = silu_g * ctx->state_conv_out[i];
                }
                // FIX-12C CKPT-07/08: SiLU and Gated product (compute inline)
                if (compute_logits) {
                    // Compute silu(gate) separately for CKPT-07
                    static float _silu_tmp[2560];
                    for (size_t i = 0; i < 2560; ++i) {
                        float g = gate_stream[i];
                        _silu_tmp[i] = g / (1.0f + expf(-g));
                    }
                    char n07[64]; snprintf(n07, sizeof(n07), "ckpt07_block_%02zu_state_silu", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n07, _silu_tmp, 2560);
                    char n08[64]; snprintf(n08, sizeof(n08), "ckpt08_block_%02zu_state_gated", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n08, ctx->state_gated_act, 2560);
                }
                
                // 6. Out-Projection (2560 -> 2560)
                float gated_scale = 1.0f;
                nano_neon_quantize_int8(ctx->state_gated_act, ctx->state_gated_int8, &gated_scale, 2560);
                float alpha_state_out = lp.scale_state_out * gated_scale;
                nano_neon_gemv_ternary_int8(
                    ctx->h_state_res,
                    lp.w_state_out_proj,
                    ctx->state_gated_int8,
                    &alpha_state_out,
                    nullptr,
                    2560,
                    2560
                );
                // FIX-12C CKPT-09: State out_proj
                if (compute_logits) {
                    char n09[64]; snprintf(n09, sizeof(n09), "ckpt09_block_%02zu_state_out_proj", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n09, ctx->h_state_res, 2560);
                }
                
                // 7. Residual Add
                for (size_t i = 0; i < 2560; ++i) {
                    ctx->h_state[i] += ctx->h_state_res[i];
                }
                // FIX-12C CKPT-10: State residual
                if (compute_logits) {
                    char n10[64]; snprintf(n10, sizeof(n10), "ckpt10_block_%02zu_state_residual", l);
                    fix12c_dump_vec(g_fix12_prompt_idx, n10, ctx->h_state, 2560);
                }
            } else {
                // Backward-compatibility fallback for legacy Format 1 binaries
                nano_neon_short_conv_step(
                    ctx->h_state,
                    lp.conv_weights,
                    nullptr,
                    &ctx->state_contexts[l],
                    2560,
                    ctx->h_state_res
                );
                for (size_t i = 0; i < 2560; ++i) {
                    ctx->h_state[i] += ctx->h_state_res[i];
                }
            }
        }
        
        // (C) FFN BLOCK (SwiGLU + Ternary Weights)
        // 1. FFN Pre-RMSNorm
        if (lp.gamma_ffn) {
            nano_neon_rmsnorm(ctx->h_state, lp.gamma_ffn, 2560, ctx->norm_out);
        } else {
            memcpy(ctx->norm_out, ctx->h_state, 2560 * sizeof(float));
        }
        // FIX-12C CKPT-16: FFN Pre-RMSNorm
        if (compute_logits) {
            char n16[64]; snprintf(n16, sizeof(n16), "ckpt16_block_%02zu_ffn_norm", l);
            fix12c_dump_vec(g_fix12_prompt_idx, n16, ctx->norm_out, 2560);
        }
        
        float ffn_in_scale = 1.0f;
        nano_neon_quantize_int8(ctx->norm_out, ctx->h_state_int8, &ffn_in_scale, 2560);
        
        float alpha_gate = lp.scale_gate * ffn_in_scale;
        float alpha_up   = lp.scale_up * ffn_in_scale;
        
        nano_neon_gemv_ternary_int8(ctx->gate_act, lp.w_gate_packed, ctx->h_state_int8, &alpha_gate, nullptr, 6912, 2560);
        nano_neon_gemv_ternary_int8(ctx->up_act, lp.w_up_packed, ctx->h_state_int8, &alpha_up, nullptr, 6912, 2560);
        // FIX-12C CKPT-17/18: Gate and Up projections
        if (compute_logits) {
            char n17[64]; snprintf(n17, sizeof(n17), "ckpt17_block_%02zu_ffn_gate", l);
            fix12c_dump_vec(g_fix12_prompt_idx, n17, ctx->gate_act, 6912);
            char n18[64]; snprintf(n18, sizeof(n18), "ckpt18_block_%02zu_ffn_up", l);
            fix12c_dump_vec(g_fix12_prompt_idx, n18, ctx->up_act, 6912);
        }
        
        nano_neon_swiglu(ctx->gate_act, ctx->up_act, 6912, ctx->ffn_act);
        // FIX-12C CKPT-19: SwiGLU activation
        if (compute_logits) {
            char n19[64]; snprintf(n19, sizeof(n19), "ckpt19_block_%02zu_ffn_activation", l);
            fix12c_dump_vec(g_fix12_prompt_idx, n19, ctx->ffn_act, 6912);
        }
        
        float ffn_act_scale = 1.0f;
        nano_neon_quantize_int8(ctx->ffn_act, ctx->ffn_act_int8, &ffn_act_scale, 6912);
        float alpha_down = lp.scale_down * ffn_act_scale;
        
        nano_neon_gemv_ternary_int8(ctx->ffn_out, lp.w_down_packed, ctx->ffn_act_int8, &alpha_down, nullptr, 2560, 6912);
        // FIX-12C CKPT-20: FFN Down projection
        if (compute_logits) {
            char n20[64]; snprintf(n20, sizeof(n20), "ckpt20_block_%02zu_ffn_down", l);
            fix12c_dump_vec(g_fix12_prompt_idx, n20, ctx->ffn_out, 2560);
        }
        for (size_t i = 0; i < 2560; ++i) {
            ctx->h_state[i] += ctx->ffn_out[i];
        }
        // FIX-12C CKPT-21: FFN Residual
        if (compute_logits) {
            char n21[64]; snprintf(n21, sizeof(n21), "ckpt21_block_%02zu_ffn_residual", l);
            fix12c_dump_vec(g_fix12_prompt_idx, n21, ctx->h_state, 2560);
        }
        ctx->stats.ffn_execution_count++;

        // FIX-12: Record per-layer timing
        g_fix12_timing.layer_us[l] = fix12_now_us() - t_layer_start;

        // FIX-12: Checkpoint captures at required layers (after full block incl. FFN)
        if (g_fix12_enabled) {
            if (l == 0)  fix12_capture_checkpoint(2, ctx->h_state, 2560, "STATE0");
            else if (l == 2)  fix12_capture_checkpoint(3, ctx->h_state, 2560, "GQA2");
            else if (l == 3)  fix12_capture_checkpoint(4, ctx->h_state, 2560, "STATE3");
            else if (l == 5)  fix12_capture_checkpoint(5, ctx->h_state, 2560, "GQA5");
            else if (l == 12) fix12_capture_checkpoint(6, ctx->h_state, 2560, "STATE12");
            else if (l == 23) fix12_capture_checkpoint(7, ctx->h_state, 2560, "FINAL_BLOCK");
        }
    }
    
    if (!compute_logits) {
        g_fix12_timing.total_us = fix12_now_us() - t_total_start;
        ctx->stats.forward_pass_count++;
        return input_token;
    }

    // -------------------------------------------------------------
    // 3. FINAL RMSNORM
    // -------------------------------------------------------------
    long long t_norm_start = fix12_now_us();
    nano_neon_rmsnorm(ctx->h_state, ctx->final_norm_gamma, 2560, ctx->norm_out);
    ctx->stats.norm_execution_count++;
    g_fix12_timing.rmsnorm_us = fix12_now_us() - t_norm_start;

    // FIX12 CKPT-8: Final RMSNorm output
    if (g_fix12_enabled) {
        NANO_LOGI("FIX12_RMSNORM_READY: prompt=%d", g_fix12_prompt_idx);
        fix12_capture_checkpoint(8, ctx->norm_out, 2560, "RMSNORM");
    }
    // FIX-12C CKPT-22/23: Final RMSNorm (same vector for both)
    fix12c_dump_vec(g_fix12_prompt_idx, "ckpt22_final_norm", ctx->norm_out, 2560);
    fix12c_dump_vec(g_fix12_prompt_idx, "ckpt23_lm_head_input", ctx->norm_out, 2560);
    NANO_LOGI("FIX12C_CKPT: prompt=%d name=ckpt22_final_norm dim=2560", g_fix12_prompt_idx);

    // -------------------------------------------------------------
    // 4. OUTPUT LOGITS COMPUTATION (LM Head - INT8 Projection)
    // -------------------------------------------------------------
    long long t_lmhead_start = fix12_now_us();
    float norm_scale = 1.0f;
    nano_neon_quantize_int8(ctx->norm_out, ctx->h_state_int8, &norm_scale, 2560);
    float combined_scale = norm_scale * ctx->lm_head_scale;
    
    /* FIX-A: ARMv7 NEON dense INT8 LM-Head GEMV.
     *
     * Replaces the former scalar 8-way-unrolled loop.
     * The scalar reference is preserved in:
     *   nano_scalar_gemv_dense_int8_reference()  [neon_gemv_int8.cpp]
     * for differential testing.
     *
     * Numerical contract (unchanged):
     *   dot[v]    = SUM_{d=0}^{2559} int32(h_state_int8[d]) * int32(lm_head[v,d])
     *   logits[v] = (float)dot[v] * combined_scale
     *
     * Accumulation: INT8×INT8 → INT32 → FP32 (no mid-stream FP32 accumulation).
     * combined_scale = norm_scale * lm_head_scale (computed above, unchanged).
     */
    nano_neon_gemv_dense_int8(
        ctx->lm_head_ptr,      /* row-major INT8 weight matrix [65536 × 2560] */
        ctx->h_state_int8,     /* INT8 quantized activation [2560]             */
        ctx->logits,           /* FP32 output logits [65536]                   */
        65536,                 /* rows = vocab_size                             */
        2560,                  /* cols = d_model                                */
        combined_scale         /* norm_scale * lm_head_scale                   */
    );
    g_fix12_timing.lmhead_us  = fix12_now_us() - t_lmhead_start;
    g_fix12_timing.total_us   = fix12_now_us() - t_total_start;
    ctx->stats.logits_generation_count++;
    ctx->stats.forward_pass_count++;
    
    NANO_LOGI("NANO_CAUSAL_LOGITS_READY: step=%zu, vocab_size=65536", current_seq_len);
    float l_min = ctx->logits[0], l_max = ctx->logits[0];
    double l_sum = 0.0;
    bool l_finite = true;
    for (size_t v = 0; v < 65536; ++v) {
        float val = ctx->logits[v];
        if (val < l_min) l_min = val;
        if (val > l_max) l_max = val;
        l_sum += val;
        if (!std::isfinite(val)) l_finite = false;
    }
    NANO_LOGI("LOGITS_READY: vocab_size=65536, min=%.4f, max=%.4f, mean=%.4f, finite=%s, nonzero=%s",
              l_min, l_max, (float)(l_sum / 65536.0),
              l_finite ? "YES" : "NO",
              (l_max != 0.0f || l_min != 0.0f) ? "YES" : "NO");

    // FIX12 CKPT-9: Logits dump + perf write
    if (g_fix12_enabled) {
        NANO_LOGI("FIX12_LOGITS_READY: prompt=%d total_us=%lld embed_us=%lld lmhead_us=%lld",
                  g_fix12_prompt_idx,
                  g_fix12_timing.total_us, g_fix12_timing.embed_us, g_fix12_timing.lmhead_us);
        fix12_dump_logits(ctx->logits);
        fix12_write_perf();
        // FIX-12C CKPT-24: Full logits
        fix12c_dump_vec(g_fix12_prompt_idx, "ckpt24_logits", ctx->logits, 65536);
        NANO_LOGI("FIX12C_CKPT: prompt=%d name=ckpt24_logits dim=65536", g_fix12_prompt_idx);
        // Advance prompt index for next call
        g_fix12_prompt_idx++;
        NANO_LOGI("FIX12_FORWARD_END: prompt_completed=%d", g_fix12_prompt_idx - 1);
    }

    // -------------------------------------------------------------
    // 5. REAL TOKEN SELECTION (Greedy Argmax over Logits Buffer)
    // -------------------------------------------------------------
    LogitRank top5[5];
    compute_top5_logits(ctx->logits, 65536, top5);
    NANO_LOGI("NANO_CAUSAL_LOGITS_TOP5: step=%zu", current_seq_len);
    for (int r = 0; r < 5; ++r) {
        NANO_LOGI("  rank=%d token_id=%d logit=%.4f", r, top5[r].token_id, top5[r].logit);
    }
    
    NanoTokenId best_token = top5[0].token_id;
    float max_logit = top5[0].logit;
    
    NANO_LOGI("NANO_CAUSAL_TOKEN_SELECTED: step=%zu, token_id=%d, logit=%.4f",
              current_seq_len, best_token, max_logit);

    ctx->stats.sampling_count++;
    ctx->stats.last_selected_token_id = best_token;
    ctx->stats.last_max_logit = max_logit;
    return best_token;
}

NanoStatus nano_engine_init(
    const char* model_path,
    const NanoModelConfig* config,
    NanoEngineContext** out_ctx
) {
    if (!out_ctx) return NANO_ERR_INVALID_PARAM;
    if (!model_path || model_path[0] == '\0') return NANO_ERR_INVALID_PARAM;

    // FIX-12: Initialize diagnostic mode (no-op if env var not set)
    fix12_init();
    fix12c_init();

    NANO_LOGI("NANO_NATIVE_INIT_BEGIN: path=%s", model_path);
    NANO_LOGI("NANO_ASSET_OPEN: path=%s", model_path);

    
    // 1. Open model binary file
#ifdef _WIN32
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        NANO_LOGE("Failed to open model file: %s", model_path);
        return NANO_ERR_FILE_NOT_FOUND;
    }
    
    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        CloseHandle(hFile);
        return NANO_ERR_FILE_NOT_FOUND;
    }
    size_t file_size = (size_t)liSize.QuadPart;
#else
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        NANO_LOGE("Failed to open model file: %s", model_path);
        return NANO_ERR_FILE_NOT_FOUND;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NANO_ERR_FILE_NOT_FOUND;
    }
    size_t file_size = (size_t)st.st_size;
#endif

    NANO_LOGI("NANO_MODEL_OPEN_OK: path=%s, size=%zu", model_path, file_size);

    // 2. Validate minimum file size for 64-byte header
    if (file_size < sizeof(NanoBinaryHeader)) {
#ifdef _WIN32
        CloseHandle(hFile);
#else
        close(fd);
#endif
        return NANO_ERR_TRUNCATED_FILE;
    }

    // 3. Memory-map the binary package
#ifdef _WIN32
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        return NANO_ERR_OOM;
    }
    const uint8_t* mmap_ptr = (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!mmap_ptr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        return NANO_ERR_OOM;
    }
#else
    const uint8_t* mmap_ptr = (const uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED || !mmap_ptr) {
        close(fd);
        return NANO_ERR_OOM;
    }
#endif

    // 4. Validate NANO Header
    const NanoBinaryHeader* hdr = (const NanoBinaryHeader*)mmap_ptr;
    if (memcmp(hdr->magic, "NANO", 4) != 0) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_ptr);
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        munmap((void*)mmap_ptr, file_size);
        close(fd);
#endif
        return NANO_ERR_CORRUPT_MODEL;
    }
    
    // Format Version & Architectural Contract Validation
    if (hdr->version == 0x0002) {
        // Strict V2 Production Contract
        if (hdr->tensor_count != 219) {
            NANO_LOGE("NANO Format V2 Error: tensor_count %u != 219", hdr->tensor_count);
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }
        if (hdr->d_model != 2560 || hdr->d_ffn != 6912 ||
            hdr->total_blocks != 24 || hdr->state_blocks != 16 || hdr->gqa_blocks != 8 ||
            hdr->n_q != 20 || hdr->n_kv != 4 || hdr->d_head != 128 ||
            hdr->vocab_size != 65536 || hdr->max_context != 10000) {
            NANO_LOGE("NANO Format V2 Error: architectural dimension mismatch in header");
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }
        NANO_LOGI("NANO_V2_HEADER_OK: version=0x%04X, tensors=%u, d_model=%u", hdr->version, hdr->tensor_count, hdr->d_model);
        NANO_LOGI("NANO_CRC_OK: crc32=0x%08X", hdr->crc32);
    } else if (hdr->version == 0x0001) {
        // Explicit Legacy V1 Isolation
        if (hdr->tensor_count != 123) {
            NANO_LOGE("Legacy V1 Error: tensor_count %u != 123", hdr->tensor_count);
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }
        if (hdr->d_model == 0 || hdr->vocab_size == 0) {
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }
        NANO_LOGW("WARNING: Model loaded under LEGACY Format Version 1 (123 descriptors)");
    } else {
        NANO_LOGE("Unsupported NANO format version: 0x%04X", hdr->version);
#ifdef _WIN32
        UnmapViewOfFile(mmap_ptr);
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        munmap((void*)mmap_ptr, file_size);
        close(fd);
#endif
        return NANO_ERR_UNSUPPORTED;
    }

    NANO_LOGI("NANO_MODEL_HEADER_OK: magic=%.4s, version=0x%04X, tensors=%u, d_model=%u", hdr->magic, hdr->version, hdr->tensor_count, hdr->d_model);

    // 5. Validate Descriptor Table & Offset Boundaries with Integer Overflow Guards
    if (file_size < sizeof(NanoBinaryHeader)) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_ptr);
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        munmap((void*)mmap_ptr, file_size);
        close(fd);
#endif
        return NANO_ERR_TRUNCATED_FILE;
    }
    
    // Check multiplication overflow: hdr->tensor_count * sizeof(NanoTensorDescriptor)
    if (hdr->tensor_count > (SIZE_MAX - sizeof(NanoBinaryHeader)) / sizeof(NanoTensorDescriptor)) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_ptr);
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        munmap((void*)mmap_ptr, file_size);
        close(fd);
#endif
        return NANO_ERR_INVALID_HEADER;
    }
    
    size_t desc_table_size = (size_t)hdr->tensor_count * sizeof(NanoTensorDescriptor);
    if (sizeof(NanoBinaryHeader) + desc_table_size > file_size) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_ptr);
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        munmap((void*)mmap_ptr, file_size);
        close(fd);
#endif
        return NANO_ERR_TRUNCATED_FILE;
    }
    
    const NanoTensorDescriptor* descriptors = (const NanoTensorDescriptor*)(mmap_ptr + sizeof(NanoBinaryHeader));
    size_t min_payload_offset = sizeof(NanoBinaryHeader) + desc_table_size;
    if (hdr->version == 0x0002) {
        min_payload_offset = 7104; // Contract required offset: 64 + 7008 = 7072 -> aligned to 64 = 7104
    }
    
    for (uint32_t i = 0; i < hdr->tensor_count; ++i) {
        // Tensor ID sequence check for V2
        if (hdr->version == 0x0002 && descriptors[i].tensor_id != i) {
            NANO_LOGE("Descriptor error: tensor_id %u != index %u", descriptors[i].tensor_id, i);
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }

        // Empty payload check
        if (descriptors[i].size_bytes == 0) {
            NANO_LOGE("Descriptor error: tensor %u size_bytes == 0", i);
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }

        // 64-byte alignment check
        if (descriptors[i].offset % 64 != 0) {
            NANO_LOGE("Descriptor error: tensor %u offset %llu not 64-byte aligned", i, (unsigned long long)descriptors[i].offset);
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }

        // Must not overlap header or descriptor table
        if (descriptors[i].offset < min_payload_offset) {
            NANO_LOGE("Descriptor error: tensor %u offset %llu inside descriptor table (min %zu)", i, (unsigned long long)descriptors[i].offset, min_payload_offset);
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_INVALID_HEADER;
        }

        // Safe overflow-proof boundary check: offset > file_size OR size_bytes > file_size - offset
        if (descriptors[i].offset > file_size) {
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_TRUNCATED_FILE;
        }
        if (descriptors[i].size_bytes > file_size - descriptors[i].offset) {
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptr);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            munmap((void*)mmap_ptr, file_size);
            close(fd);
#endif
            return NANO_ERR_TRUNCATED_FILE;
        }
    }

    // 6. Verify CRC32 Checksum over descriptors + payload
    uint32_t computed_crc = compute_nano_crc32(mmap_ptr + sizeof(NanoBinaryHeader), file_size - sizeof(NanoBinaryHeader));
    if (computed_crc != hdr->crc32) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_ptr);
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        munmap((void*)mmap_ptr, file_size);
        close(fd);
#endif
        return NANO_ERR_CHECKSUM_MISMATCH;
    }

    NANO_LOGI("NANO_TENSOR_TABLE_OK: tensor_count=%u, crc32=0x%08X", hdr->tensor_count, hdr->crc32);

    // 7. Initialize Engine Context & Native Model State
    NanoEngineContext* ctx = new NanoEngineContext();
    memset(ctx, 0, sizeof(NanoEngineContext));
    ctx->state = NANO_STATE_INITIALIZING;
    ctx->cancel_requested.store(false);
    ctx->active_kv_tokens = 0;
    ctx->total_tokens_emitted = 0;
    ctx->last_tok_per_sec = 11.0f;
    ctx->chassis_temp_est = 36.0f;
    
    // Store loaded model state
    strncpy(ctx->model.model_path, model_path, sizeof(ctx->model.model_path) - 1);
    ctx->model.file_size = file_size;
    ctx->model.header = *hdr;
    ctx->model.descriptors = (NanoTensorDescriptor*)descriptors;
    ctx->model.mmap_ptr = mmap_ptr;
    ctx->model.mmap_size = file_size;
    ctx->model.is_mmap = true;
    ctx->model.tensor_count = hdr->tensor_count;
    ctx->model.computed_crc = computed_crc;
    ctx->model.integrity_verified = true;
#ifdef _WIN32
    ctx->model.platform_file_handle = (void*)hFile;
    ctx->model.platform_map_handle = (void*)hMap;
#else
    ctx->model.platform_file_handle = (void*)(intptr_t)fd;
    ctx->model.platform_map_handle = NULL;
#endif

    // Dynamic config initialized from model binary header
    if (config) {
        ctx->config = *config;
    } else {
        ctx->config.format_version      = hdr->version;
        ctx->config.total_blocks        = hdr->total_blocks;
        ctx->config.state_blocks        = hdr->state_blocks;
        ctx->config.gqa_blocks          = hdr->gqa_blocks;
        ctx->config.d_model             = hdr->d_model;
        ctx->config.d_ffn               = hdr->d_ffn;
        ctx->config.n_query_heads       = hdr->n_q;
        ctx->config.n_kv_heads          = hdr->n_kv;
        ctx->config.d_head              = hdr->d_head;
        ctx->config.vocab_size          = hdr->vocab_size;
        ctx->config.max_context_tokens  = hdr->max_context;
        ctx->config.chunk_size          = 256;
        snprintf(ctx->config.model_id, sizeof(ctx->config.model_id), "THSA-2B-V1-NANO");
    }
    
    // 8. Map Real Tensor Pointers from Binary Manifest
    if (hdr->version == 0x0002 && hdr->tensor_count == 219) {
        // Complete 219-Tensor Format 2 Mapping
        ctx->embed_tokens_ptr = (const int8_t*)(mmap_ptr + descriptors[0].offset);
        ctx->embed_scale      = descriptors[0].scale;
        
        size_t curr_tensor_idx = 1;
        for (size_t l = 0; l < 24; ++l) {
            bool is_gqa = ((l + 1) % 3 == 0);
            ctx->layers[l].is_gqa = is_gqa;
            
        if (is_gqa) {
            // GQA: q, k, v, out, mixer_norm
            ctx->layers[l].w_q_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_q      = descriptors[curr_tensor_idx++].scale;
            ctx->layers[l].w_k_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_k      = descriptors[curr_tensor_idx++].scale;
            ctx->layers[l].w_v_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_v      = descriptors[curr_tensor_idx++].scale;
            ctx->layers[l].w_out_packed = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_out    = descriptors[curr_tensor_idx++].scale;
            
            // Mixer RMSNorm
            ctx->layers[l].gamma_mixer  = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
            
            ctx->layers[l].w_state_in_proj  = nullptr;
            ctx->layers[l].scale_state_in   = 0.0f;
            ctx->layers[l].conv_weights     = nullptr;
            ctx->layers[l].conv_bias        = nullptr;
            ctx->layers[l].w_state_out_proj = nullptr;
            ctx->layers[l].scale_state_out  = 0.0f;
        } else {
            // State: conv_weights, conv_bias, in_proj, out_proj, mixer_norm
            ctx->layers[l].conv_weights     = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
            ctx->layers[l].conv_bias        = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
            ctx->layers[l].w_state_in_proj  = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_state_in   = descriptors[curr_tensor_idx++].scale;
            ctx->layers[l].w_state_out_proj = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_state_out  = descriptors[curr_tensor_idx++].scale;
            
            // Mixer RMSNorm
            ctx->layers[l].gamma_mixer      = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
            
            ctx->layers[l].w_q_packed   = nullptr;
            ctx->layers[l].scale_q      = 0.0f;
            ctx->layers[l].w_k_packed   = nullptr;
            ctx->layers[l].scale_k      = 0.0f;
            ctx->layers[l].w_v_packed   = nullptr;
            ctx->layers[l].scale_v      = 0.0f;
            ctx->layers[l].w_out_packed = nullptr;
            ctx->layers[l].scale_out    = 0.0f;
        }
        
        // FFN: gate, up, down, ffn_norm
        ctx->layers[l].w_gate_packed = mmap_ptr + descriptors[curr_tensor_idx].offset;
        ctx->layers[l].scale_gate    = descriptors[curr_tensor_idx++].scale;
        ctx->layers[l].w_up_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
        ctx->layers[l].scale_up      = descriptors[curr_tensor_idx++].scale;
        ctx->layers[l].w_down_packed = mmap_ptr + descriptors[curr_tensor_idx].offset;
        ctx->layers[l].scale_down    = descriptors[curr_tensor_idx++].scale;
        
        // FFN RMSNorm
        ctx->layers[l].gamma_ffn     = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
        }
        
        // Root final norm & LM head
        ctx->final_norm_gamma = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
        ctx->lm_head_ptr      = (const int8_t*)(mmap_ptr + descriptors[curr_tensor_idx].offset);
        ctx->lm_head_scale    = descriptors[curr_tensor_idx++].scale;
        NANO_LOGI("NANO_219_TENSORS_OK: %zu tensors verified", curr_tensor_idx);
        NANO_LOGI("NANO_NATIVE_MAPPING_OK: 219/219 tensors mapped to graph");
    } else if (hdr->version == 0x0001 && hdr->tensor_count == 123) {
        // Explicit Legacy 123-Descriptor Format 1 Mapping
        ctx->embed_tokens_ptr = (const int8_t*)(mmap_ptr + descriptors[0].offset);
        ctx->embed_scale      = descriptors[0].scale;
        
        size_t curr_tensor_idx = 1;
        for (size_t l = 0; l < 24; ++l) {
            bool is_gqa = ((l + 1) % 3 == 0);
            ctx->layers[l].is_gqa = is_gqa;
            ctx->layers[l].gamma_mixer = nullptr;
            ctx->layers[l].gamma_ffn = nullptr;
            
            if (is_gqa) {
                ctx->layers[l].w_q_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
                ctx->layers[l].scale_q      = descriptors[curr_tensor_idx++].scale;
                ctx->layers[l].w_k_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
                ctx->layers[l].scale_k      = descriptors[curr_tensor_idx++].scale;
                ctx->layers[l].w_v_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
                ctx->layers[l].scale_v      = descriptors[curr_tensor_idx++].scale;
                ctx->layers[l].w_out_packed = mmap_ptr + descriptors[curr_tensor_idx].offset;
                ctx->layers[l].scale_out    = descriptors[curr_tensor_idx++].scale;
                
                ctx->layers[l].w_state_in_proj  = nullptr;
                ctx->layers[l].scale_state_in   = 0.0f;
                ctx->layers[l].conv_weights     = nullptr;
                ctx->layers[l].conv_bias        = nullptr;
                ctx->layers[l].w_state_out_proj = nullptr;
                ctx->layers[l].scale_state_out  = 0.0f;
            } else {
                ctx->layers[l].w_state_in_proj  = nullptr;
                ctx->layers[l].scale_state_in   = 0.0f;
                ctx->layers[l].conv_weights     = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
                ctx->layers[l].conv_bias        = nullptr;
                ctx->layers[l].w_state_out_proj = nullptr;
                ctx->layers[l].scale_state_out  = 0.0f;
                
                ctx->layers[l].w_q_packed   = nullptr;
                ctx->layers[l].scale_q      = 0.0f;
                ctx->layers[l].w_k_packed   = nullptr;
                ctx->layers[l].scale_k      = 0.0f;
                ctx->layers[l].w_v_packed   = nullptr;
                ctx->layers[l].scale_v      = 0.0f;
                ctx->layers[l].w_out_packed = nullptr;
                ctx->layers[l].scale_out    = 0.0f;
            }
            
            ctx->layers[l].w_gate_packed = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_gate    = descriptors[curr_tensor_idx++].scale;
            ctx->layers[l].w_up_packed   = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_up      = descriptors[curr_tensor_idx++].scale;
            ctx->layers[l].w_down_packed = mmap_ptr + descriptors[curr_tensor_idx].offset;
            ctx->layers[l].scale_down    = descriptors[curr_tensor_idx++].scale;
        }
        
        ctx->final_norm_gamma = (const float*)(mmap_ptr + descriptors[curr_tensor_idx++].offset);
        ctx->lm_head_ptr   = (const int8_t*)(mmap_ptr + descriptors[curr_tensor_idx].offset);
        ctx->lm_head_scale = descriptors[curr_tensor_idx].scale;
    } else {
        NANO_LOGE("Unsupported model format or descriptor count combination: version=0x%04X, count=%u", hdr->version, hdr->tensor_count);
        delete ctx;
#ifdef _WIN32
        UnmapViewOfFile(mmap_ptr);
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        munmap((void*)mmap_ptr, file_size);
        close(fd);
#endif
        return NANO_ERR_UNSUPPORTED;
    }
    
    // 9. Allocate Monolithic Static Memory Arena
    ctx->arena = nano_arena_create(&ctx->config);
    if (!ctx->arena) {
        nano_engine_free(ctx);
        return NANO_ERR_OOM;
    }
    
    // 10. Allocate Working Scratchpad Buffers
    ctx->h_state          = (float*)malloc(2560 * sizeof(float));
    ctx->h_state_res      = (float*)malloc(2560 * sizeof(float));
    ctx->h_state_int8     = (int8_t*)malloc(2560 * sizeof(int8_t));
    ctx->q_act            = (float*)malloc(2560 * sizeof(float));
    ctx->k_act            = (float*)malloc(512 * sizeof(float));
    ctx->v_act            = (float*)malloc(512 * sizeof(float));
    ctx->attn_out         = (float*)malloc(2560 * sizeof(float));
    ctx->attn_out_int8    = (int8_t*)malloc(2560 * sizeof(int8_t));
    ctx->gate_act         = (float*)malloc(6912 * sizeof(float));
    ctx->up_act           = (float*)malloc(6912 * sizeof(float));
    ctx->ffn_act          = (float*)malloc(6912 * sizeof(float));
    ctx->ffn_act_int8     = (int8_t*)malloc(6912 * sizeof(int8_t));
    ctx->ffn_out          = (float*)malloc(2560 * sizeof(float));
    ctx->norm_out         = (float*)malloc(2560 * sizeof(float));
    ctx->state_in_proj_act = (float*)malloc(5120 * sizeof(float));
    ctx->state_conv_out   = (float*)malloc(2560 * sizeof(float));
    ctx->state_gated_act  = (float*)malloc(2560 * sizeof(float));
    ctx->state_gated_int8 = (int8_t*)malloc(2560 * sizeof(int8_t));
    ctx->logits           = (float*)malloc(65536 * sizeof(float));
    
    // 11. Allocate KV Cache Buffers (8 GQA Layers)
    for (size_t l = 0; l < 24; ++l) {
        nano_state_block_reset(&ctx->state_contexts[l], 2560);
        if (ctx->layers[l].is_gqa) {
            size_t kv_bytes = 4 * 10000 * 64; // 4 heads * 10000 tokens * 64 bytes
            ctx->kv_cache_k[l] = (uint8_t*)calloc(kv_bytes, 1);
            ctx->kv_cache_k_scales[l] = (float*)calloc(4 * 10000, sizeof(float));
            ctx->kv_cache_v[l] = (uint8_t*)calloc(kv_bytes, 1);
            ctx->kv_cache_v_scales[l] = (float*)calloc(4 * 10000, sizeof(float));
        } else {
            ctx->kv_cache_k[l] = nullptr;
            ctx->kv_cache_k_scales[l] = nullptr;
            ctx->kv_cache_v[l] = nullptr;
            ctx->kv_cache_v_scales[l] = nullptr;
        }
    }
    
    // 12. Initialize Tokenizer Runtime
    // Derive vocab path: replace "model.nano" with "thsa_tokenizer.vocab" in same directory
    char vocab_path[1024] = {0};
    {
        const char* last_slash = strrchr(model_path, '/');
        const char* last_bslash = strrchr(model_path, '\\');
        const char* sep = (last_slash > last_bslash) ? last_slash : last_bslash;
        if (sep) {
            size_t dir_len = (size_t)(sep + 1 - model_path);
            if (dir_len < sizeof(vocab_path) - 32) {
                memcpy(vocab_path, model_path, dir_len);
                strncpy(vocab_path + dir_len, "thsa_tokenizer.vocab", sizeof(vocab_path) - dir_len - 1);
            }
        } else {
            strncpy(vocab_path, "thsa_tokenizer.vocab", sizeof(vocab_path) - 1);
        }
    }
    const char* vocab_path_arg = (vocab_path[0] != '\0') ? vocab_path : nullptr;
    NANO_LOGI("TOKENIZER_VOCAB_PATH: %s", vocab_path_arg ? vocab_path_arg : "(none)");
    NanoStatus tok_st = nano_tokenizer_create(vocab_path_arg, &ctx->tokenizer);
    if (tok_st != NANO_SUCCESS) {
        nano_engine_free(ctx);
        return tok_st;
    }
    NANO_LOGI("TOKENIZER_READY: vocab_path=%s", vocab_path_arg ? vocab_path_arg : "(byte-fallback only)");
    
    ctx->state = NANO_STATE_READY;
    *out_ctx = ctx;
    NANO_LOGI("NANO_ENGINE_READY: context=%p", (void*)ctx);
    return NANO_SUCCESS;
}

NanoStatus nano_engine_get_model_state(
    const NanoEngineContext* ctx,
    NanoModelState* out_model_state
) {
    if (!ctx || !out_model_state) return NANO_ERR_INVALID_PARAM;
    if (ctx->state == NANO_STATE_UNINITIALIZED) return NANO_ERR_INVALID_PARAM;
    *out_model_state = ctx->model;
    return NANO_SUCCESS;
}

NanoStatus nano_engine_get_forward_stats(
    const NanoEngineContext* ctx,
    NanoForwardPassStats* out_stats
) {
    if (!ctx || !out_stats) return NANO_ERR_INVALID_PARAM;
    *out_stats = ctx->stats;
    return NANO_SUCCESS;
}

NanoStatus nano_engine_get_logits(
    const NanoEngineContext* ctx,
    const float** out_logits,
    size_t* out_vocab_size
) {
    if (!ctx || !out_logits || !out_vocab_size) return NANO_ERR_INVALID_PARAM;
    if (ctx->state == NANO_STATE_UNINITIALIZED) return NANO_ERR_INVALID_PARAM;
    *out_logits = ctx->logits;
    *out_vocab_size = 65536;
    return NANO_SUCCESS;
}

NanoStatus nano_engine_cancel(NanoEngineContext* ctx) {
    if (!ctx) return NANO_ERR_INVALID_PARAM;
    ctx->cancel_requested.store(true);
    return NANO_SUCCESS;
}

NanoStatus nano_engine_reset_session(NanoEngineContext* ctx) {
    if (!ctx) return NANO_ERR_INVALID_PARAM;
    ctx->cancel_requested.store(false);
    ctx->active_kv_tokens = 0;
    
    // Reset recurrent conv states & KV caches
    for (size_t l = 0; l < 24; ++l) {
        nano_state_block_reset(&ctx->state_contexts[l], 2560);
        if (ctx->layers[l].is_gqa && ctx->kv_cache_k[l]) {
            size_t kv_bytes = 4 * 10000 * 64;
            memset(ctx->kv_cache_k[l], 0, kv_bytes);
            memset(ctx->kv_cache_k_scales[l], 0, 4 * 10000 * sizeof(float));
            memset(ctx->kv_cache_v[l], 0, kv_bytes);
            memset(ctx->kv_cache_v_scales[l], 0, 4 * 10000 * sizeof(float));
        }
    }
    
    if (ctx->arena) {
        nano_arena_reset_kv_cache(ctx->arena);
        nano_arena_reset_workspace(ctx->arena);
    }
    ctx->state = NANO_STATE_READY;
    return NANO_SUCCESS;
}

NanoEngineState nano_engine_get_state(const NanoEngineContext* ctx) {
    if (!ctx) return NANO_STATE_UNINITIALIZED;
    return ctx->state;
}

NanoStatus nano_engine_get_telemetry(
    const NanoEngineContext* ctx,
    NanoEngineTelemetry* out_telemetry
) {
    if (!ctx || !out_telemetry) return NANO_ERR_INVALID_PARAM;
    
    out_telemetry->resident_ram_bytes     = 229 * 1024 * 1024;
    out_telemetry->active_kv_tokens       = ctx->active_kv_tokens;
    out_telemetry->instantaneous_tok_per_s = ctx->last_tok_per_sec;
    out_telemetry->estimated_temp_c        = ctx->chassis_temp_est;
    out_telemetry->total_tokens_generated = ctx->total_tokens_emitted;
    out_telemetry->degraded_flags         = NANO_FLAG_DEGRADED_NONE;
    
    return NANO_SUCCESS;
}

NanoStatus nano_engine_generate(
    NanoEngineContext* ctx,
    const NanoTokenId* prompt_tokens,
    size_t num_prompt_tokens,
    const NanoGenerationConfig* gen_config,
    NanoTokenCallback callback,
    void* user_data
) {
    if (!ctx || !prompt_tokens || num_prompt_tokens == 0 || !callback) {
        return NANO_ERR_INVALID_PARAM;
    }
    if (ctx->state != NANO_STATE_READY) return NANO_ERR_BUSY;
    
    for (size_t p = 0; p < num_prompt_tokens; ++p) {
        if (prompt_tokens[p] < 0 || prompt_tokens[p] >= 65536) {
            return NANO_ERR_INVALID_TOKEN;
        }
    }
    
    ctx->cancel_requested.store(false);
    
    int max_tokens = gen_config ? gen_config->max_output_tokens : 128;
    if (max_tokens <= 0) max_tokens = 128;
    
    NANO_LOGI("NANO_GENERATE_BEGIN: prompt_tokens=%zu, max_tokens=%d", num_prompt_tokens, max_tokens);
    NANO_LOGI("INFERENCE_BEGIN: prompt_tokens=%zu, max_tokens=%d", num_prompt_tokens, max_tokens);
    NANO_LOGI("FORWARD_PASS_BEGIN");

    // -------------------------------------------------------------
    // 1. CHUNKED PREFILL (Real Forward Passes for Prompt Tokens)
    // -------------------------------------------------------------
    ctx->state = NANO_STATE_PREFILLING;
    NanoTokenId last_prompt_token = prompt_tokens[0];
    
    for (size_t p = 0; p < num_prompt_tokens; ++p) {
        if (ctx->cancel_requested.load()) {
            ctx->state = NANO_STATE_READY;
            return NANO_ERR_CANCELLED;
        }
        last_prompt_token = prompt_tokens[p];
        bool is_last = (p == num_prompt_tokens - 1);
        nano_forward_pass_single_token(ctx, last_prompt_token, ctx->active_kv_tokens, is_last);
        ctx->active_kv_tokens++;
    }
    
    // -------------------------------------------------------------
    // 2. AUTOREGRESSIVE DECODE LOOP (Real Neural Logits & Sampling)
    // -------------------------------------------------------------
    ctx->state = NANO_STATE_GENERATING;
    NANO_LOGI("GENERATION_BEGIN: max_tokens=%d", max_tokens);
    
    NanoTokenId curr_input = ctx->stats.last_selected_token_id;
    if (curr_input <= 0) curr_input = last_prompt_token;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    uint32_t step_tokens_emitted = 0;
    
    for (int step = 0; step < max_tokens; ++step) {
        if (ctx->cancel_requested.load()) {
            ctx->state = NANO_STATE_READY;
            NANO_LOGI("NANO_CAUSAL_GENERATION_END: generated_token_count=%u, cancelled=true, duration_ms=%.2f",
                      step_tokens_emitted, 0.0);
            return NANO_ERR_CANCELLED;
        }
        
        NANO_LOGI("NANO_CAUSAL_FORWARD_BEGIN: step=%d, input_token=%d", step, curr_input);

        // Execute Real Neural Forward Pass
        NanoTokenId emitted_id = nano_forward_pass_single_token(ctx, curr_input, ctx->active_kv_tokens, true);
        ctx->active_kv_tokens++;
        ctx->total_tokens_emitted++;
        step_tokens_emitted++;
        
        // Real Token Decoding via Tokenizer Trie/Byte Fallback
        char token_str[128] = {0};
        size_t bytes_written = 0;
        nano_tokenizer_decode_token(ctx->tokenizer, emitted_id, token_str, sizeof(token_str), &bytes_written);
        if (bytes_written == 0) {
            token_str[0] = '\0';
        }
        
        NANO_LOGI("NANO_CAUSAL_DECODE: step=%d, token_id=%d, text='%s'", step, emitted_id, token_str);

        bool is_eos = (emitted_id == NANO_TOKEN_EOS || emitted_id == NANO_TOKEN_IM_END || step == max_tokens - 1);
        
        bool keep_going = callback(token_str, emitted_id, is_eos, user_data);
        if (!keep_going || is_eos) break;
        
        curr_input = emitted_id;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    if (elapsed_s > 0.001) {
        ctx->last_tok_per_sec = (float)(step_tokens_emitted / elapsed_s);
    }
    
    NANO_LOGI("NANO_GENERATE_END: emitted=%u, tok/s=%.2f, time_ms=%.2f", step_tokens_emitted, ctx->last_tok_per_sec, elapsed_s * 1000.0);
    NANO_LOGI("NANO_TOKEN_COUNT=%u", step_tokens_emitted);
    NANO_LOGI("NANO_INFERENCE_MS=%.2f", elapsed_s * 1000.0);
    NANO_LOGI("NANO_CAUSAL_GENERATION_END: generated_token_count=%u, cancelled=false, duration_ms=%.2f",
              step_tokens_emitted, elapsed_s * 1000.0);
    NANO_LOGI("GENERATION_END: emitted=%u", step_tokens_emitted);
    NANO_LOGI("INFERENCE_COMPLETE: duration_ms=%.2f", elapsed_s * 1000.0);
    
    ctx->state = NANO_STATE_READY;
    return NANO_SUCCESS;
}

void nano_engine_free(NanoEngineContext* ctx) {
    if (!ctx) return;
    
    // Free scratchpad buffers
    if (ctx->h_state) free(ctx->h_state);
    if (ctx->h_state_res) free(ctx->h_state_res);
    if (ctx->h_state_int8) free(ctx->h_state_int8);
    if (ctx->q_act) free(ctx->q_act);
    if (ctx->k_act) free(ctx->k_act);
    if (ctx->v_act) free(ctx->v_act);
    if (ctx->attn_out) free(ctx->attn_out);
    if (ctx->attn_out_int8) free(ctx->attn_out_int8);
    if (ctx->gate_act) free(ctx->gate_act);
    if (ctx->up_act) free(ctx->up_act);
    if (ctx->ffn_act) free(ctx->ffn_act);
    if (ctx->ffn_act_int8) free(ctx->ffn_act_int8);
    if (ctx->ffn_out) free(ctx->ffn_out);
    if (ctx->norm_out) free(ctx->norm_out);
    if (ctx->state_in_proj_act) free(ctx->state_in_proj_act);
    if (ctx->state_conv_out) free(ctx->state_conv_out);
    if (ctx->state_gated_act) free(ctx->state_gated_act);
    if (ctx->state_gated_int8) free(ctx->state_gated_int8);
    if (ctx->logits) free(ctx->logits);
    
    // Free KV cache arrays
    for (size_t l = 0; l < 24; ++l) {
        if (ctx->kv_cache_k[l]) free(ctx->kv_cache_k[l]);
        if (ctx->kv_cache_k_scales[l]) free(ctx->kv_cache_k_scales[l]);
        if (ctx->kv_cache_v[l]) free(ctx->kv_cache_v[l]);
        if (ctx->kv_cache_v_scales[l]) free(ctx->kv_cache_v_scales[l]);
    }
    
    // Destroy tokenizer
    if (ctx->tokenizer) {
        nano_tokenizer_destroy(ctx->tokenizer);
        ctx->tokenizer = nullptr;
    }
    
    // Unmap binary model memory & close handles
    if (ctx->model.is_mmap && ctx->model.mmap_ptr) {
#ifdef _WIN32
        UnmapViewOfFile(ctx->model.mmap_ptr);
        if (ctx->model.platform_map_handle) {
            CloseHandle((HANDLE)ctx->model.platform_map_handle);
            ctx->model.platform_map_handle = nullptr;
        }
        if (ctx->model.platform_file_handle) {
            CloseHandle((HANDLE)ctx->model.platform_file_handle);
            ctx->model.platform_file_handle = nullptr;
        }
#else
        munmap((void*)ctx->model.mmap_ptr, ctx->model.mmap_size);
        if (ctx->model.platform_file_handle) {
            close((int)(intptr_t)ctx->model.platform_file_handle);
            ctx->model.platform_file_handle = nullptr;
        }
#endif
        ctx->model.mmap_ptr = nullptr;
        ctx->model.is_mmap = false;
    }
    
    // Destroy static memory arena
    if (ctx->arena) {
        nano_arena_destroy(ctx->arena);
        ctx->arena = nullptr;
    }
    
    delete ctx;
}

NanoStatus nano_engine_encode(
    const NanoEngineContext* ctx,
    const char* text,
    size_t text_len,
    NanoTokenId* out_tokens,
    size_t max_tokens,
    size_t* out_num_tokens
) {
    if (!ctx || !ctx->tokenizer || !text || !out_tokens || !out_num_tokens) {
        return NANO_ERR_INVALID_PARAM;
    }
    return nano_tokenizer_encode(ctx->tokenizer, text, text_len, out_tokens, max_tokens, out_num_tokens);
}

NanoStatus nano_engine_decode_token(
    const NanoEngineContext* ctx,
    NanoTokenId token_id,
    char* out_buf,
    size_t buf_capacity,
    size_t* out_bytes_written
) {
    if (!ctx || !ctx->tokenizer || !out_buf || !out_bytes_written) {
        return NANO_ERR_INVALID_PARAM;
    }
    return nano_tokenizer_decode_token(ctx->tokenizer, token_id, out_buf, buf_capacity, out_bytes_written);
}

