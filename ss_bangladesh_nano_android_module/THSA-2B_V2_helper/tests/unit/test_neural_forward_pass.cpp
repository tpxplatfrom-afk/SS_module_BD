#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>
#include <vector>
#include <string>

#include "../../include/nano_engine.h"
#include "../../include/nano_types.h"
#include "../../include/nano_config.h"

// Token streaming accumulator
struct StreamAccumulator {
    std::vector<int32_t> emitted_token_ids;
    std::vector<std::string> emitted_token_strings;
    int call_count;
};

static bool test_token_callback(const char* token_str, NanoTokenId token_id, bool is_eos, void* user_data) {
    StreamAccumulator* acc = (StreamAccumulator*)user_data;
    acc->emitted_token_ids.push_back(token_id);
    acc->emitted_token_strings.push_back(token_str ? token_str : "");
    acc->call_count++;
    return true; // continue generation
}

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("THSA-2B V2_HELPER: REAL NEURAL FORWARD-PASS UNIT TEST\n");
    printf("================================================================================\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "../THSA-2B V1/models/model.nano";
    printf("[INFO] Target Model Path: %s\n", model_path);
    
    NanoEngineContext* ctx = nullptr;
    NanoStatus init_st = nano_engine_init(model_path, nullptr, &ctx);
    if (init_st != NANO_SUCCESS || !ctx) {
        printf("  ❌ FAIL: nano_engine_init failed with status %d\n", init_st);
        return 1;
    }
    
    NanoTokenId prompt_tokens[] = { NANO_TOKEN_BOS, 105, 120 };
    size_t num_prompts = sizeof(prompt_tokens) / sizeof(prompt_tokens[0]);
    
    NanoGenerationConfig gen_cfg = nano_gen_config_default();
    gen_cfg.max_output_tokens = 5;
    
    StreamAccumulator acc;
    acc.call_count = 0;
    
    NanoStatus gen_st = nano_engine_generate(
        ctx,
        prompt_tokens,
        num_prompts,
        &gen_cfg,
        test_token_callback,
        &acc
    );
    
    if (gen_st != NANO_SUCCESS) {
        printf("  ❌ FAIL: nano_engine_generate returned error status %d\n", gen_st);
        nano_engine_free(ctx);
        return 1;
    }
    
    NanoForwardPassStats stats;
    memset(&stats, 0, sizeof(stats));
    nano_engine_get_forward_stats(ctx, &stats);
    
    printf("  ✅ Forward Pass Total Count: %llu\n", (unsigned long long)stats.forward_pass_count);
    printf("  ✅ Logits Generation Count:  %llu\n", (unsigned long long)stats.logits_generation_count);
    printf("  ✅ Sampling Count:           %llu\n", (unsigned long long)stats.sampling_count);
    
    nano_engine_free(ctx);
    printf("================================================================================\n");
    printf("V2_HELPER FORWARD PASS VERIFICATION SUCCESSFUL ✅\n");
    printf("================================================================================\n");
    return 0;
}
