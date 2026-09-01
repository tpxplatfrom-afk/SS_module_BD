#include <stdio.h>
#include <assert.h>
#include "../../include/nano_engine.h"

int main() {
    printf("Running negative and adversarial tests...\n");
    NanoEngineContext* ctx = nullptr;
    assert(nano_engine_init(nullptr, nullptr, &ctx) == NANO_ERR_INVALID_PARAM);
    assert(nano_engine_generate(nullptr, nullptr, 0, nullptr, nullptr, nullptr) == NANO_ERR_INVALID_PARAM);
    assert(nano_engine_reset_session(nullptr) == NANO_ERR_INVALID_PARAM);
    assert(nano_engine_cancel(nullptr) == NANO_ERR_INVALID_PARAM);
    
    assert(nano_engine_init("models/model.nano", nullptr, &ctx) == NANO_SUCCESS);
    assert(ctx != nullptr);
    
    // Test invalid token
    NanoTokenId bad_tokens[] = { -1, 65536, 100000 };
    for (auto tok : bad_tokens) {
        NanoTokenId out = -1;
        NanoStatus st = nano_engine_generate(ctx, &tok, 1, nullptr, [](const char*, NanoTokenId, bool, void*){ return true; }, &out);
        assert(st == NANO_ERR_INVALID_TOKEN);
    }
    
    // Test 100 reset cycles
    for (int i = 0; i < 100; ++i) {
        assert(nano_engine_reset_session(ctx) == NANO_SUCCESS);
    }
    
    nano_engine_free(ctx);
    printf("ADVERSARIAL AND NEGATIVE TESTS PASSED!\n");
    return 0;
}
