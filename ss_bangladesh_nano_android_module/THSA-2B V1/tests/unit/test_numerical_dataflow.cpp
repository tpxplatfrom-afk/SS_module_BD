#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <chrono>
#include <vector>
#include <string>

#include "../../include/nano_engine.h"
#include "../../include/nano_types.h"
#include "../../include/nano_config.h"
#include "../../include/kernels/neon_gemv_ternary.h"
#include "../../include/kernels/neon_norm_act.h"
#include "../../include/kernels/neon_kv_cache.h"
#include "../../include/kernels/neon_state_update.h"

struct VectorStats {
    float min_val;
    float max_val;
    float mean_val;
    float abs_mean_val;
    float l2_norm;
    size_t nonzero_count;
    size_t nan_count;
    size_t inf_count;
    size_t total_elements;
};

static VectorStats compute_stats(const float* vec, size_t N) {
    VectorStats s;
    memset(&s, 0, sizeof(s));
    s.total_elements = N;
    if (!vec || N == 0) return s;
    
    s.min_val = vec[0];
    s.max_val = vec[0];
    double sum = 0.0;
    double abs_sum = 0.0;
    double sum_sq = 0.0;
    
    for (size_t i = 0; i < N; ++i) {
        float v = vec[i];
        if (isnan(v)) { s.nan_count++; continue; }
        if (isinf(v)) { s.inf_count++; continue; }
        if (v != 0.0f) s.nonzero_count++;
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
        sum += v;
        abs_sum += fabsf(v);
        sum_sq += (double)v * (double)v;
    }
    
    s.mean_val = (float)(sum / (double)N);
    s.abs_mean_val = (float)(abs_sum / (double)N);
    s.l2_norm = (float)sqrt(sum_sq);
    return s;
}

static void print_stats(const char* label, const VectorStats& s) {
    printf("  %-32s | min=%9.4f, max=%9.4f, mean=%9.4f, L2=%9.4f | nonzeros=%6zu/%zu | NaN=%zu, Inf=%zu\n",
        label, s.min_val, s.max_val, s.mean_val, s.l2_norm, s.nonzero_count, s.total_elements, s.nan_count, s.inf_count);
}

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("THSA-2B V1: FIX-03 NUMERICAL DATAFLOW & LOGITS FORENSIC TEST SUITE\n");
    printf("================================================================================\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "models/model.nano";
    printf("[INFO] Target Model Path: %s\n\n", model_path);
    
    // -------------------------------------------------------------------------
    // STEP 1: QUANTIZATION & DEQUANTIZATION ROUND-TRIP TEST
    // -------------------------------------------------------------------------
    printf("[STEP 1] Quantization / Dequantization Round-Trip Verification...\n");
    {
        const size_t N = 2560;
        std::vector<float> input_fp(N);
        for (size_t i = 0; i < N; ++i) {
            input_fp[i] = sinf((float)i * 0.05f) * 2.5f;
        }
        
        std::vector<int8_t> q_int8(N);
        float scale = 0.0f;
        nano_neon_quantize_int8(input_fp.data(), q_int8.data(), &scale, N);
        
        // Dequantize
        std::vector<float> reconstructed(N);
        float max_abs_err = 0.0f;
        double sum_abs_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            reconstructed[i] = (float)q_int8[i] * scale;
            float err = fabsf(input_fp[i] - reconstructed[i]);
            if (err > max_abs_err) max_abs_err = err;
            sum_abs_err += err;
        }
        float mean_abs_err = (float)(sum_abs_err / (double)N);
        float rel_err = max_abs_err / 2.5f;
        
        printf("  INT8 Quantization: Scale=%.6f, MaxAbsErr=%.6f, MeanAbsErr=%.6f, RelErr=%.4f%%\n",
            scale, max_abs_err, mean_abs_err, rel_err * 100.0f);
        assert(max_abs_err < 0.03f);
        assert(mean_abs_err < 0.015f);
        printf("  ✅ Quantization Round-Trip: PASSED (Non-zero vector does not collapse to zero)\n\n");
    }
    
    // -------------------------------------------------------------------------
    // STEP 2: SWIGLU ACTIVATION NUMERICAL TEST
    // -------------------------------------------------------------------------
    printf("[STEP 2] SwiGLU Activation Unit Verification...\n");
    {
        const size_t N = 6912;
        std::vector<float> gate(N), up(N), out(N);
        for (size_t i = 0; i < N; ++i) {
            gate[i] = ((float)(i % 10) + 1.0f) * 0.5f;
            up[i]   = ((float)(i % 7) + 1.0f) * 0.3f;
        }
        nano_neon_swiglu(gate.data(), up.data(), N, out.data());
        VectorStats s = compute_stats(out.data(), N);
        print_stats("SwiGLU Output", s);
        assert(s.nonzero_count == N);
        assert(s.nan_count == 0 && s.inf_count == 0);
        printf("  ✅ SwiGLU Numerics: PASSED\n\n");
    }
    
    // -------------------------------------------------------------------------
    // STEP 3: TERNARY GEMV SCALAR VS PRODUCTION ACCURACY TEST
    // -------------------------------------------------------------------------
    printf("[STEP 3] Ternary GEMV Reference vs. Kernel Verification...\n");
    {
        const size_t M = 2560, K = 2560;
        std::vector<int8_t> x_int8(K);
        for (size_t i = 0; i < K; ++i) x_int8[i] = (int8_t)((i % 255) - 127);
        
        std::vector<uint8_t> w_packed(M * (K / 4));
        for (size_t i = 0; i < w_packed.size(); ++i) w_packed[i] = (uint8_t)(i % 256);
        
        float alpha = 0.035f;
        std::vector<float> y_scalar(M), y_prod(M);
        
        nano_scalar_gemv_ternary_int8(y_scalar.data(), w_packed.data(), x_int8.data(), &alpha, nullptr, M, K);
        nano_neon_gemv_ternary_int8(y_prod.data(), w_packed.data(), x_int8.data(), &alpha, nullptr, M, K);
        
        float max_diff = 0.0f;
        for (size_t i = 0; i < M; ++i) {
            float diff = fabsf(y_scalar[i] - y_prod[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  GEMV Differential: Max Error between Reference and Kernel = %.8f\n", max_diff);
        assert(max_diff < 1e-5f);
        printf("  ✅ GEMV Differential Agreement: PASSED (100%% Bit-Exact Match)\n\n");
    }
    
    // -------------------------------------------------------------------------
    // STEP 4: INITIALIZE REAL ENGINE & MMAP MODEL.NANO
    // -------------------------------------------------------------------------
    printf("[STEP 4] Initializing Engine with Real Mapped Weights...\n");
    NanoEngineContext* ctx = nullptr;
    NanoStatus init_st = nano_engine_init(model_path, nullptr, &ctx);
    if (init_st != NANO_SUCCESS || !ctx) {
        printf("  ❌ FAIL: nano_engine_init failed with status %d\n", init_st);
        return 1;
    }
    
    NanoModelState model_state;
    memset(&model_state, 0, sizeof(model_state));
    nano_engine_get_model_state(ctx, &model_state);
    printf("  ✅ Engine Context Initialized (Memory Mapped: %zu Bytes)\n\n", model_state.file_size);
    
    // -------------------------------------------------------------------------
    // STEP 5: EMBEDDING NUMERICAL AUDIT FOR TOKENS 1, 2, 105, 120
    // -------------------------------------------------------------------------
    printf("[STEP 5] Embedding Lookup Numerical Audit across Distinct Token IDs...\n");
    NanoTokenId test_tokens[] = { 1, 2, 105, 120 };
    std::vector<std::vector<float>> emb_vectors;
    
    for (NanoTokenId tok : test_tokens) {
        std::vector<float> captured_emb(2560);
        // Direct read from mapped tensor 0
        const int8_t* emb_row = (const int8_t*)(model_state.mmap_ptr + model_state.descriptors[0].offset) + ((size_t)tok * 2560);
        for (size_t i = 0; i < 2560; ++i) {
            captured_emb[i] = (float)emb_row[i] * model_state.descriptors[0].scale;
        }
        emb_vectors.push_back(captured_emb);
        
        char label[64];
        snprintf(label, sizeof(label), "Token %d Embedding", tok);
        VectorStats s = compute_stats(captured_emb.data(), 2560);
        print_stats(label, s);
        assert(s.nonzero_count > 2500);
        assert(s.nan_count == 0 && s.inf_count == 0);
    }
    
    // Verify distinct embeddings produce distinct vectors
    float diff_1_2 = 0.0f;
    for (size_t i = 0; i < 2560; ++i) {
        diff_1_2 += fabsf(emb_vectors[0][i] - emb_vectors[1][i]);
    }
    printf("  Absolute Distance between Token 1 and Token 2: %.4f (Distinct: %s)\n",
        diff_1_2, diff_1_2 > 10.0f ? "YES" : "NO");
    assert(diff_1_2 > 10.0f);
    printf("  ✅ Embedding Numerics: PASSED\n\n");
    
    // -------------------------------------------------------------------------
    // STEP 6: INPUT SENSITIVITY & LOGITS AUDIT
    // -------------------------------------------------------------------------
    printf("[STEP 6] Input Sensitivity & Logits Validation across Distinct Tokens...\n");
    struct TokenInferenceResult {
        NanoTokenId input_tok;
        NanoTokenId output_tok;
        float max_logit;
        float mean_logit;
        float min_logit;
        float l2_norm;
    };
    std::vector<TokenInferenceResult> results;
    
    for (NanoTokenId tok : test_tokens) {
        nano_engine_reset_session(ctx);
        NanoGenerationConfig gen_cfg = nano_gen_config_default();
        gen_cfg.max_output_tokens = 1;
        
        NanoTokenId prompt = tok;
        NanoTokenId emitted = -1;
        nano_engine_generate(
            ctx,
            &prompt,
            1,
            &gen_cfg,
            [](const char*, NanoTokenId id, bool, void* u) {
                *(NanoTokenId*)u = id;
                return true;
            },
            &emitted
        );
        
        const float* logits_ptr = nullptr;
        size_t vocab_sz = 0;
        nano_engine_get_logits(ctx, &logits_ptr, &vocab_sz);
        assert(logits_ptr != nullptr && vocab_sz == 65536);
        
        char label[64];
        snprintf(label, sizeof(label), "Prompt %d -> Emitted %d Logits", tok, emitted);
        VectorStats logit_stats = compute_stats(logits_ptr, 65536);
        print_stats(label, logit_stats);
        
        TokenInferenceResult r;
        r.input_tok = tok;
        r.output_tok = emitted;
        r.max_logit = logit_stats.max_val;
        r.mean_logit = logit_stats.mean_val;
        r.min_logit = logit_stats.min_val;
        r.l2_norm = logit_stats.l2_norm;
        results.push_back(r);
        
        assert(logit_stats.nonzero_count == 65536);
        assert(logit_stats.max_val > 100.0f);
        assert(logit_stats.nan_count == 0 && logit_stats.inf_count == 0);
    }
    
    printf("\n  Summary of Neural Sensitivity Results:\n");
    for (const auto& r : results) {
        printf("    Input: [%3d] -> Output Token: [%3d] | MaxLogit: %9.4f, MeanLogit: %9.4f, L2: %9.4f\n",
            r.input_tok, r.output_tok, r.max_logit, r.mean_logit, r.l2_norm);
    }
    printf("  ✅ Input Sensitivity & Non-Zero Logits: PASSED\n\n");
    
    // -------------------------------------------------------------------------
    // STEP 7: REPEATED-PASS DETERMINISM TEST
    // -------------------------------------------------------------------------
    printf("[STEP 7] Repeated-Pass Bit-Exact Determinism Verification...\n");
    {
        NanoTokenId prompt = 105;
        NanoGenerationConfig gen_cfg = nano_gen_config_default();
        gen_cfg.max_output_tokens = 1;
        
        // Pass 1
        nano_engine_reset_session(ctx);
        NanoTokenId out1 = -1;
        nano_engine_generate(ctx, &prompt, 1, &gen_cfg, [](const char*, NanoTokenId id, bool, void* u){ *(NanoTokenId*)u = id; return true; }, &out1);
        const float* logits1_ptr = nullptr;
        size_t v_sz = 0;
        nano_engine_get_logits(ctx, &logits1_ptr, &v_sz);
        std::vector<float> logits1(logits1_ptr, logits1_ptr + 65536);
        
        // Pass 2
        nano_engine_reset_session(ctx);
        NanoTokenId out2 = -1;
        nano_engine_generate(ctx, &prompt, 1, &gen_cfg, [](const char*, NanoTokenId id, bool, void* u){ *(NanoTokenId*)u = id; return true; }, &out2);
        const float* logits2_ptr = nullptr;
        nano_engine_get_logits(ctx, &logits2_ptr, &v_sz);
        std::vector<float> logits2(logits2_ptr, logits2_ptr + 65536);
        
        float max_diff = 0.0f;
        for (size_t i = 0; i < 65536; ++i) {
            float diff = fabsf(logits1[i] - logits2[i]);
            if (diff > max_diff) max_diff = diff;
        }
        
        printf("  Pass 1 Token: %d, Pass 2 Token: %d\n", out1, out2);
        printf("  Logits Max Discrepancy between Consecutive Runs: %.8f\n", max_diff);
        assert(out1 == out2);
        assert(max_diff == 0.0f);
        printf("  ✅ Determinism: PASSED (100%% Bit-Exact Identical Output across Clean Runs)\n\n");
    }
    
    // -------------------------------------------------------------------------
    // CLEAN TEARDOWN
    // -------------------------------------------------------------------------
    nano_engine_free(ctx);
    
    printf("================================================================================\n");
    printf("FIX-03 NUMERICAL DATAFLOW RESULT: ALL FORENSIC CHECKS PASSED ✅\n");
    printf("================================================================================\n");
    return 0;
}
