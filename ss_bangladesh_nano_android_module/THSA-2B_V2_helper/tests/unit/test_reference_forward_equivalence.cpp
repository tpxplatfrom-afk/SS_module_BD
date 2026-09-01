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
    printf("THSA-2B V1: REFERENCE FORWARD-PASS EQUIVALENCE CHECKPOINT DUMPER\n");
    printf("================================================================================\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "models/model.nano";
    const char* out_dir = (argc > 2) ? argv[2] : "tests/artifacts";
    
    NanoEngineContext* ctx = nullptr;
    NanoStatus init_st = nano_engine_init(model_path, nullptr, &ctx);
    if (init_st != NANO_SUCCESS || !ctx) {
        printf("❌ FAIL: nano_engine_init failed with status %d\n", init_st);
        return 1;
    }
    
    NanoTokenId test_tokens[] = { 1, 2, 105, 120 };
    
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
        
        char out_filename[512];
        snprintf(out_filename, sizeof(out_filename), "%s/native_logits_token_%d.bin", out_dir, tok);
        FILE* fp = fopen(out_filename, "wb");
        if (fp) {
            fwrite(logits_ptr, sizeof(float), vocab_sz, fp);
            fclose(fp);
            printf("  ✅ Dumped native C++ logits for Token %d -> %s (Emitted: %d)\n", tok, out_filename, emitted);
        } else {
            printf("  ⚠️ Warning: Could not write %s\n", out_filename);
        }
    }
    
    nano_engine_free(ctx);
    printf("================================================================================\n");
    printf("NATIVE CHECKPOINT GENERATION COMPLETE ✅\n");
    printf("================================================================================\n");
    return 0;
}
