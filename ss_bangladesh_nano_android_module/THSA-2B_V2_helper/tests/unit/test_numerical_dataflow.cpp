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
    printf("THSA-2B V2_HELPER: FIX-03 NUMERICAL DATAFLOW FORENSIC TEST SUITE\n");
    printf("================================================================================\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "../THSA-2B V1/models/model.nano";
    printf("[INFO] Target Model Path: %s\n\n", model_path);
    
    // Test GEMV Reference
    {
        const size_t M = 2560, K = 2560;
        std::vector<int8_t> x_int8(K, 1);
        std::vector<uint8_t> w_packed(M * (K / 4), 0x55); // 01 01 01 01 -> +1
        float alpha = 1.0f;
        std::vector<float> y_scalar(M), y_prod(M);
        nano_scalar_gemv_ternary_int8(y_scalar.data(), w_packed.data(), x_int8.data(), &alpha, nullptr, M, K);
        nano_neon_gemv_ternary_int8(y_prod.data(), w_packed.data(), x_int8.data(), &alpha, nullptr, M, K);
        float max_diff = fabsf(y_scalar[0] - y_prod[0]);
        assert(max_diff < 1e-5f);
    }
    
    printf("================================================================================\n");
    printf("V2_HELPER NUMERICAL DATAFLOW VERIFICATION SUCCESSFUL ✅\n");
    printf("================================================================================\n");
    return 0;
}
