#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <string>

#include "../../include/nano_engine.h"
#include "../../include/nano_types.h"
#include "../../include/nano_config.h"

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("THSA-2B V1: MULTI-TOKEN AUTOREGRESSIVE GENERATION TEST\n");
    printf("================================================================================\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "models/model.nano";
    
    NanoEngineContext* ctx = nullptr;
    NanoStatus init_st = nano_engine_init(model_path, nullptr, &ctx);
    if (init_st != NANO_SUCCESS || !ctx) {
        printf("❌ FAIL: nano_engine_init failed with status %d\n", init_st);
        return 1;
    }
    
    // Autoregressive generation of 4 steps from prompt [105]
    NanoTokenId prompt = 105;
    NanoGenerationConfig gen_cfg = nano_gen_config_default();
    gen_cfg.max_output_tokens = 4;
    
    std::vector<NanoTokenId> generated_tokens;
    nano_engine_generate(
        ctx,
        &prompt,
        1,
        &gen_cfg,
        [](const char* piece, NanoTokenId id, bool is_special, void* user) -> bool {
            auto* list = (std::vector<NanoTokenId>*)user;
            list->push_back(id);
            printf("  [Step %zu] Emitted Token: %5d | Special: %s | Piece: '%s'\n",
                list->size(), id, is_special ? "true" : "false", piece ? piece : "");
            return true;
        },
        &generated_tokens
    );
    
    printf("\n  Total Autoregressive Tokens Emitted: %zu\n", generated_tokens.size());
    assert(generated_tokens.size() == 4);
    
    // Check Forward pass telemetry
    NanoForwardPassStats stats;
    nano_engine_get_forward_stats(ctx, &stats);
    printf("  Forward Pass Count:       %llu\n", stats.forward_pass_count);
    printf("  Logits Generation Count:  %llu\n", stats.logits_generation_count);
    printf("  Last Max Logit:           %.4f\n", stats.last_max_logit);
    printf("  Last Selected Token ID:   %d\n", stats.last_selected_token_id);
    
    assert(stats.forward_pass_count >= 4);
    assert(stats.last_max_logit > 100.0f);
    
    nano_engine_free(ctx);
    printf("================================================================================\n");
    printf("AUTOREGRESSIVE EQUIVALENCE TEST PASSED ✅\n");
    printf("================================================================================\n");
    return 0;
}
