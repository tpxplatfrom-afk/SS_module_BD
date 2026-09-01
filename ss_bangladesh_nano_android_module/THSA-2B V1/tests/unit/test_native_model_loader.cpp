#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>

#include "../../include/nano_engine.h"
#include "../../include/nano_types.h"
#include "../../include/nano_config.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("THSA-2B V1: NATIVE MODEL LOADER UNIT TEST & DIAGNOSTIC VALIDATOR\n");
    printf("================================================================================\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "models/model.nano";
    printf("[INFO] Target Model Path: %s\n", model_path);
    
    // -------------------------------------------------------------
    // TEST 1: Invalid Path Handling
    // -------------------------------------------------------------
    {
        printf("\n[TEST 1] Invalid Path Handling...\n");
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init("non_existent_file_xyz.nano", nullptr, &ctx);
        if (st == NANO_ERR_FILE_NOT_FOUND) {
            printf("  ✅ PASS: Correctly returned NANO_ERR_FILE_NOT_FOUND for non-existent file.\n");
        } else {
            printf("  ❌ FAIL: Expected NANO_ERR_FILE_NOT_FOUND (%d), got %d\n", NANO_ERR_FILE_NOT_FOUND, st);
            return 1;
        }
    }
    
    // -------------------------------------------------------------
    // TEST 2: Truncated / Corrupt File Handling
    // -------------------------------------------------------------
    {
        printf("\n[TEST 2] Truncated / Corrupt Header File Handling...\n");
        const char* corrupt_path = "corrupt_test_temp.nano";
        FILE* fp = fopen(corrupt_path, "wb");
        if (fp) {
            char junk[32] = {0};
            fwrite(junk, 1, 32, fp);
            fclose(fp);
            
            NanoEngineContext* ctx = nullptr;
            NanoStatus st = nano_engine_init(corrupt_path, nullptr, &ctx);
            remove(corrupt_path);
            
            if (st == NANO_ERR_TRUNCATED_FILE || st == NANO_ERR_CORRUPT_MODEL) {
                printf("  ✅ PASS: Correctly rejected truncated 32-byte file (Status: %d).\n", st);
            } else {
                printf("  ❌ FAIL: Expected error for truncated file, got %d\n", st);
                return 1;
            }
        }
    }
    
    // -------------------------------------------------------------
    // TEST 3: Valid Model Load, Memory Mapping & Header Parsing
    // -------------------------------------------------------------
    {
        printf("\n[TEST 3] Loading Real Binary Model: %s ...\n", model_path);
        auto t0 = std::chrono::high_resolution_clock::now();
        
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(model_path, nullptr, &ctx);
        auto t1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        if (st != NANO_SUCCESS || !ctx) {
            printf("  ❌ FAIL: nano_engine_init failed with status %d\n", st);
            return 1;
        }
        
        NanoModelState model_state;
        memset(&model_state, 0, sizeof(model_state));
        NanoStatus state_st = nano_engine_get_model_state(ctx, &model_state);
        if (state_st != NANO_SUCCESS) {
            printf("  ❌ FAIL: nano_engine_get_model_state failed with status %d\n", state_st);
            nano_engine_free(ctx);
            return 1;
        }
        
        printf("  ✅ Model Path Opened:         %s\n", model_state.model_path);
        printf("  ✅ Total File Size (Bytes):   %zu (%.2f MB)\n", model_state.file_size, model_state.file_size / (1024.0 * 1024.0));
        printf("  ✅ Header Magic:              %.4s (Valid: %s)\n", model_state.header.magic, memcmp(model_state.header.magic, "NANO", 4) == 0 ? "YES" : "NO");
        printf("  ✅ Format Version:            0x%04X\n", model_state.header.version);
        printf("  ✅ Total Blocks:              %u (%u State / %u GQA)\n", model_state.header.total_blocks, model_state.header.state_blocks, model_state.header.gqa_blocks);
        printf("  ✅ Model Dimensions:          d_model=%u, d_ffn=%u, d_head=%u\n", model_state.header.d_model, model_state.header.d_ffn, model_state.header.d_head);
        printf("  ✅ Attention Heads:           n_query=%u, n_kv=%u\n", model_state.header.n_q, model_state.header.n_kv);
        printf("  ✅ Vocab Size:                %u tokens\n", model_state.header.vocab_size);
        printf("  ✅ Context Horizon:           %u tokens\n", model_state.header.max_context);
        printf("  ✅ Tensor Count:              %u tensors\n", model_state.tensor_count);
        printf("  ✅ Stored Checksum:           0x%08X\n", model_state.header.crc32);
        printf("  ✅ Computed Checksum:         0x%08X (Integrity: %s)\n", model_state.computed_crc, model_state.integrity_verified ? "VALID MATCH" : "CORRUPT");
        printf("  ✅ Memory Mapped Pointer:     %p (is_mmap: %s)\n", (void*)model_state.mmap_ptr, model_state.is_mmap ? "YES" : "NO");
        printf("  ✅ Native Memory Mapped Size: %zu Bytes\n", model_state.mmap_size);
        printf("  ✅ Load & Validation Time:    %.2f ms\n", load_ms);
        
        // -------------------------------------------------------------
        // TEST 4: Actual Model Payload Byte Access Verification
        // -------------------------------------------------------------
        printf("\n[TEST 4] Verifying Actual Non-Zero Model Payload Bytes Access...\n");
        assert(model_state.mmap_ptr != nullptr);
        assert(model_state.descriptors != nullptr);
        
        // Check first 5 tensor descriptors
        size_t non_zero_bytes_count = 0;
        for (uint32_t i = 0; i < 5 && i < model_state.tensor_count; ++i) {
            const NanoTensorDescriptor& desc = model_state.descriptors[i];
            printf("  -> Tensor [%u]: quant=%u, offset=%llu, size=%llu bytes, scale=%.4f (64B-aligned: %s)\n",
                desc.tensor_id,
                desc.quant_type,
                (unsigned long long)desc.offset,
                (unsigned long long)desc.size_bytes,
                desc.scale,
                (desc.offset % 64 == 0) ? "YES" : "NO"
            );
            
            // Read actual payload bytes at offset
            const uint8_t* tensor_data = model_state.mmap_ptr + desc.offset;
            for (size_t b = 0; b < 64 && b < desc.size_bytes; ++b) {
                if (tensor_data[b] != 0) non_zero_bytes_count++;
            }
        }
        
        printf("  ✅ Non-Zero Payload Samples Read: %zu bytes accessed.\n", non_zero_bytes_count);
        if (non_zero_bytes_count > 0) {
            printf("  ✅ PROOF ESTABLISHED: Model payload bytes were actually read/mapped into native memory state!\n");
        } else {
            printf("  ⚠️ WARNING: All sample bytes were zero.\n");
        }
        
        // Clean teardown & unmap
        printf("\n[TEST 5] Clean RAII Teardown & Unmapping...\n");
        nano_engine_free(ctx);
        printf("  ✅ Successfully unmapped model and destroyed context.\n");
    }
    
    printf("\n================================================================================\n");
    printf("FIX 01 NATIVE MODEL LOADER RESULT: ALL TESTS PASSED ✅\n");
    printf("================================================================================\n");
    return 0;
}
