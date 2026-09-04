/**
 * @file test_real_model_lmhead_neon.cpp
 * @brief FIX-A: Verification & Benchmark of Real Model Tensor-218 on itel A662L.
 *
 * Fulfills Sections 16, 17, 20:
 * 16. Real Model Weight Test (Tensor ID 218 from model.nano)
 * 17. Full 65536 Logits Comparison (Max/Mean Abs Diff, RMSE, Cosine Sim, Top-10)
 * 20. On-Device Physical Benchmark (Scalar vs NEON ms, Speedup)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "../../include/nano_types.h"
#include "../../include/kernels/neon_gemv_int8.h"

struct LogitEntry {
    int id;
    float val;
};

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("FIX-A: REAL MODEL TENSOR-218 LM-HEAD NEON VERIFIER & BENCHMARK\n");
    printf("================================================================================\n");

    const char* model_path = (argc > 1) ? argv[1] : "/data/local/tmp/model.nano";
    printf("[INFO] Model Path: %s\n", model_path);

    // 1. Open and mmap model.nano
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("❌ FAIL: Cannot open model file: %s\n", model_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    size_t file_size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    printf("[INFO] Model File Size: %zu bytes (%.2f MB)\n", file_size, (double)file_size / (1024.0 * 1024.0));

    NanoBinaryHeader header;
    if (fread(&header, 1, sizeof(NanoBinaryHeader), fp) != sizeof(NanoBinaryHeader)) {
        printf("❌ FAIL: Failed to read binary header\n");
        fclose(fp);
        return 1;
    }

    if (memcmp(header.magic, "NANO", 4) != 0) {
        printf("❌ FAIL: Invalid magic header\n");
        fclose(fp);
        return 1;
    }

    printf("[INFO] Header: version=0x%04x, tensors=%u, d_model=%u, vocab=%u\n",
           header.version, header.tensor_count, header.d_model, header.vocab_size);

    std::vector<NanoTensorDescriptor> descs(header.tensor_count);
    if (fread(descs.data(), sizeof(NanoTensorDescriptor), header.tensor_count, fp) != header.tensor_count) {
        printf("❌ FAIL: Failed to read descriptors\n");
        fclose(fp);
        return 1;
    }

    // Find Tensor 218 (lm_head)
    const NanoTensorDescriptor* lm_desc = nullptr;
    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        if (descs[i].tensor_id == 218) {
            lm_desc = &descs[i];
            break;
        }
    }

    if (!lm_desc) {
        printf("❌ FAIL: Tensor ID 218 not found in manifest!\n");
        fclose(fp);
        return 1;
    }

    printf("[INFO] Tensor 218 (lm_head): offset=%llu, size=%llu bytes, quant_type=%u, scale=%.8f\n",
           (unsigned long long)lm_desc->offset, (unsigned long long)lm_desc->size_bytes,
           lm_desc->quant_type, lm_desc->scale);

    size_t expected_size = (size_t)header.vocab_size * (size_t)header.d_model; // 65536 * 2560 = 167772160
    if (lm_desc->size_bytes != expected_size) {
        printf("❌ FAIL: Size mismatch for Tensor 218: expected %zu, got %llu\n",
               expected_size, (unsigned long long)lm_desc->size_bytes);
        fclose(fp);
        return 1;
    }

    // mmap tensor data
    fclose(fp);

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        printf("❌ FAIL: Cannot open file for mmap\n");
        return 1;
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        printf("❌ FAIL: mmap failed\n");
        return 1;
    }

    const int8_t* lm_head_ptr = (const int8_t*)((const uint8_t*)mapped + lm_desc->offset);
    float lm_head_scale = lm_desc->scale;

    printf("✅ Successfully memory-mapped Tensor 218 from real model at %p\n", (void*)lm_head_ptr);

    // 2. Prepare realistic activation vector (simulated post-RMSNorm INT8 quantized activation)
    std::vector<int8_t> h_state_int8(header.d_model);
    // Deterministic pseudo-random values simulating real quantized activation
    uint32_t rng = 0x12345678;
    for (size_t i = 0; i < header.d_model; ++i) {
        rng = rng * 1664525u + 1013904223u;
        // Range [-127, 127]
        int val = (int)((rng >> 16) & 0xFF) - 128;
        if (val < -127) val = -127;
        if (val > 127) val = 127;
        h_state_int8[i] = (int8_t)val;
    }

    float norm_scale = 0.00392157f; // ~ 1.0 / 255
    float combined_scale = norm_scale * lm_head_scale;

    std::vector<float> logits_scalar(header.vocab_size);
    std::vector<float> logits_neon(header.vocab_size);

    // 3. Compute with Scalar Reference
    printf("\n[SECTION 16 & 17] Executing Full 65,536-Row Forward Pass on Real Tensor 218...\n");
    nano_scalar_gemv_dense_int8_reference(
        lm_head_ptr,
        h_state_int8.data(),
        logits_scalar.data(),
        header.vocab_size,
        header.d_model,
        combined_scale
    );

    // 4. Compute with NEON Kernel
    nano_neon_gemv_dense_int8(
        lm_head_ptr,
        h_state_int8.data(),
        logits_neon.data(),
        header.vocab_size,
        header.d_model,
        combined_scale
    );

    // 5. Differential Analysis (Section 17)
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    double dot_prod = 0.0;
    double norm_s = 0.0;
    double norm_n = 0.0;
    int int32_mismatches = 0;

    for (size_t v = 0; v < header.vocab_size; ++v) {
        float fs = logits_scalar[v];
        float fn = logits_neon[v];
        double diff = fabs((double)fs - (double)fn);

        if (diff > max_abs_diff) max_abs_diff = diff;
        sum_abs_diff += diff;
        sum_sq_diff += diff * diff;

        dot_prod += (double)fs * (double)fn;
        norm_s += (double)fs * (double)fs;
        norm_n += (double)fn * (double)fn;

        // Verify recovered integer dot
        int64_t ds = (int64_t)round(fs / combined_scale);
        int64_t dn = (int64_t)round(fn / combined_scale);
        if (ds != dn) {
            int32_mismatches++;
        }
    }

    double mean_abs_diff = sum_abs_diff / (double)header.vocab_size;
    double rmse = sqrt(sum_sq_diff / (double)header.vocab_size);
    double cosine_sim = dot_prod / (sqrt(norm_s) * sqrt(norm_n) + 1e-12);

    // Top-10 ranking comparison
    std::vector<LogitEntry> top_scalar(header.vocab_size);
    std::vector<LogitEntry> top_neon(header.vocab_size);
    for (size_t v = 0; v < header.vocab_size; ++v) {
        top_scalar[v] = { (int)v, logits_scalar[v] };
        top_neon[v]   = { (int)v, logits_neon[v] };
    }

    auto comp = [](const LogitEntry& a, const LogitEntry& b) { return a.val > b.val; };
    std::partial_sort(top_scalar.begin(), top_scalar.begin() + 10, top_scalar.end(), comp);
    std::partial_sort(top_neon.begin(), top_neon.begin() + 10, top_neon.end(), comp);

    printf("\n=== NUMERICAL EQUIVALENCE REPORT (65,536 VOCABULARY ROWS) ===\n");
    printf("  INT32 Dot Mismatches: %d / 65536 (%s)\n",
           int32_mismatches, int32_mismatches == 0 ? "EXACT EQUALITY" : "MISMATCH");
    printf("  Max Absolute Error:   %.8e\n", max_abs_diff);
    printf("  Mean Absolute Error:  %.8e\n", mean_abs_diff);
    printf("  RMSE:                 %.8e\n", rmse);
    printf("  Cosine Similarity:    %.10f\n", cosine_sim);
    bool top1_match = (top_scalar[0].id == top_neon[0].id);
    printf("  Top-1 Match:          %s (ID %d vs %d, logit %.5f vs %.5f)\n",
           top1_match ? "EXACT" : "DIFF",
           top_scalar[0].id, top_neon[0].id, top_scalar[0].val, top_neon[0].val);

    bool top5_match = true;
    for (int k = 0; k < 5; ++k) {
        if (top_scalar[k].id != top_neon[k].id) top5_match = false;
    }
    printf("  Top-5 Match:          %s\n", top5_match ? "EXACT" : "DIFF");

    bool top10_match = true;
    for (int k = 0; k < 10; ++k) {
        if (top_scalar[k].id != top_neon[k].id) top10_match = false;
    }
    printf("  Top-10 Match:         %s\n", top10_match ? "EXACT" : "DIFF");

    printf("\n--- Top-10 Tokens Breakdown ---\n");
    printf("Rank | Scalar Token (Logit)         | NEON Token (Logit)           | Status\n");
    printf("-----+------------------------------+------------------------------+-------\n");
    for (int k = 0; k < 10; ++k) {
        printf(" %2d  | Token %-5d (%12.5f)    | Token %-5d (%12.5f)    | %s\n",
               k + 1,
               top_scalar[k].id, top_scalar[k].val,
               top_neon[k].id, top_neon[k].val,
               (top_scalar[k].id == top_neon[k].id && fabs(top_scalar[k].val - top_neon[k].val) < 1e-5f) ? "MATCH" : "DIFF");
    }

    // 6. Benchmark Comparison (Section 20)
    printf("\n================================================================================\n");
    printf("SECTION 20: ON-DEVICE PHYSICAL BENCHMARK (itel A662L, Cortex-A7)\n");
    printf("================================================================================\n");

    const int WARMUP_ITERS = 2;
    const int BENCH_ITERS = 5;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        nano_scalar_gemv_dense_int8_reference(lm_head_ptr, h_state_int8.data(), logits_scalar.data(), header.vocab_size, header.d_model, combined_scale);
        nano_neon_gemv_dense_int8(lm_head_ptr, h_state_int8.data(), logits_neon.data(), header.vocab_size, header.d_model, combined_scale);
    }

    // Benchmark Scalar
    double total_scalar_ms = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        nano_scalar_gemv_dense_int8_reference(lm_head_ptr, h_state_int8.data(), logits_scalar.data(), header.vocab_size, header.d_model, combined_scale);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_scalar_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double avg_scalar_ms = total_scalar_ms / BENCH_ITERS;

    // Benchmark NEON
    double total_neon_ms = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        nano_neon_gemv_dense_int8(lm_head_ptr, h_state_int8.data(), logits_neon.data(), header.vocab_size, header.d_model, combined_scale);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_neon_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double avg_neon_ms = total_neon_ms / BENCH_ITERS;
    double speedup = avg_scalar_ms / (avg_neon_ms > 0 ? avg_neon_ms : 1.0);

    printf("  LM-Head Scalar GEMV Latency: %.2f ms (averaged over %d runs)\n", avg_scalar_ms, BENCH_ITERS);
    printf("  LM-Head NEON GEMV Latency:   %.2f ms (averaged over %d runs)\n", avg_neon_ms, BENCH_ITERS);
    printf("  LM-Head Kernel Speedup:      %.2fx\n", speedup);
    printf("  Latency Reduction:           %.2f ms / token\n", avg_scalar_ms - avg_neon_ms);

    munmap(mapped, file_size);

    if (int32_mismatches == 0 && top1_match && max_abs_diff == 0.0) {
        printf("\nRESULT: ALL REAL TENSOR-218 VERIFICATION CHECKS PASSED ✅\n");
        return 0;
    } else {
        printf("\nRESULT: REAL TENSOR-218 VERIFICATION FAILED ❌\n");
        return 1;
    }
}
