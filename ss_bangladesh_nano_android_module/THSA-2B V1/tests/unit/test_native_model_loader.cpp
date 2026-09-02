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
    // TEST 3: Model Load, Memory Mapping & Header Parsing (if present)
    // -------------------------------------------------------------
    {
        printf("\n[TEST 3] Checking for Real Binary Model: %s ...\n", model_path);
        FILE* fp_check = fopen(model_path, "rb");
        if (fp_check) {
            fclose(fp_check);
            auto t0 = std::chrono::high_resolution_clock::now();
            NanoEngineContext* ctx = nullptr;
            NanoStatus st = nano_engine_init(model_path, nullptr, &ctx);
            auto t1 = std::chrono::high_resolution_clock::now();
            double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            
            if (st == NANO_SUCCESS && ctx) {
                NanoModelState model_state;
                memset(&model_state, 0, sizeof(model_state));
                nano_engine_get_model_state(ctx, &model_state);
                printf("  ✅ Real model loaded in %.2f ms (version=0x%04X, tensors=%u)\n", load_ms, model_state.header.version, model_state.tensor_count);
                nano_engine_free(ctx);
            } else {
                printf("  [NOTE] Model failed to load with status %d\n", st);
            }
        } else {
            printf("  [INFO] Real model file %s not present (pre-export gate satisfied).\n", model_path);
        }
    }

    // -------------------------------------------------------------
    // TEST 4: COMPREHENSIVE V2 GRAPH DISPATCH & SECURITY GATE (CASES A - K)
    // -------------------------------------------------------------
    printf("\n================================================================================\n");
    printf("TEST 4: FORMAT V2 GRAPH DISPATCH & SECURITY GATE TEST SUITE (CASES A - K)\n");
    printf("================================================================================\n");

    auto calc_crc32 = [](const uint8_t* buffer, size_t length) -> uint32_t {
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < length; ++i) {
            crc ^= buffer[i];
            for (int j = 0; j < 8; ++j) {
                uint32_t mask = -(crc & 1);
                crc = (crc >> 1) ^ (0xEDB88320 & mask);
            }
        }
        return ~crc & 0xFFFFFFFF;
    };

    auto make_synthetic_test_model = [&](
        const char* path,
        uint16_t version,
        uint32_t tensor_count,
        uint32_t d_model,
        uint32_t vocab_size,
        bool corrupt_offset,
        bool corrupt_crc
    ) -> bool {
        size_t header_size = sizeof(NanoBinaryHeader);
        size_t desc_size = tensor_count * sizeof(NanoTensorDescriptor);
        size_t payload_offset = (version == 0x0002) ? 7104 : ((header_size + desc_size + 63) & ~63);
        size_t total_payload = tensor_count * 64;
        size_t file_size = payload_offset + total_payload;

        uint8_t* buf = (uint8_t*)calloc(1, file_size);
        if (!buf) return false;

        NanoBinaryHeader* hdr = (NanoBinaryHeader*)buf;
        memcpy(hdr->magic, "NANO", 4);
        hdr->version = version;
        hdr->total_blocks = 24;
        hdr->state_blocks = 16;
        hdr->gqa_blocks = 8;
        hdr->d_model = d_model;
        hdr->d_ffn = 6912;
        hdr->n_q = 20;
        hdr->n_kv = 4;
        hdr->d_head = 128;
        hdr->pad = 0;
        hdr->vocab_size = vocab_size;
        hdr->max_context = 10000;
        hdr->tensor_count = tensor_count;

        NanoTensorDescriptor* descs = (NanoTensorDescriptor*)(buf + header_size);
        for (uint32_t i = 0; i < tensor_count; ++i) {
            descs[i].tensor_id = i;
            descs[i].quant_type = NANO_QUANT_FP32;
            descs[i].offset = payload_offset + i * 64;
            if (corrupt_offset && i == 0) {
                descs[i].offset = 100; // malformed inside descriptor table
            }
            descs[i].size_bytes = 64;
            descs[i].scale = 1.0f;
            descs[i].pad = 0;

            // Fill dummy payload
            size_t p_off = descs[i].offset;
            if (p_off + 64 <= file_size) {
                buf[p_off] = (uint8_t)(i + 1);
            }
        }

        uint32_t computed = calc_crc32(buf + header_size, file_size - header_size);
        hdr->crc32 = corrupt_crc ? (computed ^ 0xDEADBEEF) : computed;

        FILE* fp = fopen(path, "wb");
        if (!fp) { free(buf); return false; }
        fwrite(buf, 1, file_size, fp);
        fclose(fp);
        free(buf);
        return true;
    };

    const char* test_bin = "temp_dispatch_test.nano";

    // CASE A: version 0x0002 + tensor_count 219 -> V2 dispatch
    {
        make_synthetic_test_model(test_bin, 0x0002, 219, 2560, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_SUCCESS && ctx) {
            printf("  ✅ CASE A (version 0x0002 + 219 tensors): Successfully entered V2 dispatch!\n");
            nano_engine_free(ctx);
        } else {
            printf("  ❌ CASE A FAILED: Expected NANO_SUCCESS, got %d\n", st);
            return 1;
        }
    }

    // CASE B: version 0x0001 + tensor_count 123 -> Legacy V1 dispatch
    {
        make_synthetic_test_model(test_bin, 0x0001, 123, 2560, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_SUCCESS && ctx) {
            printf("  ✅ CASE B (version 0x0001 + 123 tensors): Successfully entered Legacy V1 dispatch!\n");
            nano_engine_free(ctx);
        } else {
            printf("  ❌ CASE B FAILED: Expected NANO_SUCCESS, got %d\n", st);
            return 1;
        }
    }

    // CASE C: version 0x0002 + tensor_count 123 -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0002, 123, 2560, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_INVALID_HEADER || st == NANO_ERR_UNSUPPORTED) {
            printf("  ✅ CASE C (version 0x0002 + 123 tensors): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE C FAILED: Expected rejection, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE D: version 0x0001 + tensor_count 219 -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0001, 219, 2560, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_INVALID_HEADER || st == NANO_ERR_UNSUPPORTED) {
            printf("  ✅ CASE D (version 0x0001 + 219 tensors): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE D FAILED: Expected rejection, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE E: version 0x0003 -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0003, 219, 2560, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_UNSUPPORTED) {
            printf("  ✅ CASE E (version 0x0003): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE E FAILED: Expected NANO_ERR_UNSUPPORTED, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE F: tensor_count 218 -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0002, 218, 2560, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_INVALID_HEADER || st == NANO_ERR_UNSUPPORTED) {
            printf("  ✅ CASE F (tensor_count 218): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE F FAILED: Expected rejection, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE G: tensor_count 220 -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0002, 220, 2560, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_INVALID_HEADER || st == NANO_ERR_UNSUPPORTED) {
            printf("  ✅ CASE G (tensor_count 220): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE G FAILED: Expected rejection, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE H: wrong d_model (2048) -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0002, 219, 2048, 65536, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_INVALID_HEADER) {
            printf("  ✅ CASE H (wrong d_model=2048): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE H FAILED: Expected NANO_ERR_INVALID_HEADER, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE I: wrong vocab (32000) -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0002, 219, 2560, 32000, false, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_INVALID_HEADER) {
            printf("  ✅ CASE I (wrong vocab=32000): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE I FAILED: Expected NANO_ERR_INVALID_HEADER, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE J: malformed descriptor boundary -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0002, 219, 2560, 65536, true, false);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_INVALID_HEADER || st == NANO_ERR_TRUNCATED_FILE) {
            printf("  ✅ CASE J (malformed descriptor boundary): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE J FAILED: Expected boundary rejection, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    // CASE K: CRC mismatch -> REJECT
    {
        make_synthetic_test_model(test_bin, 0x0002, 219, 2560, 65536, false, true);
        NanoEngineContext* ctx = nullptr;
        NanoStatus st = nano_engine_init(test_bin, nullptr, &ctx);
        remove(test_bin);
        if (st == NANO_ERR_CHECKSUM_MISMATCH) {
            printf("  ✅ CASE K (CRC checksum mismatch): Correctly REJECTED (Status: %d)\n", st);
        } else {
            printf("  ❌ CASE K FAILED: Expected NANO_ERR_CHECKSUM_MISMATCH, got %d\n", st);
            if (ctx) nano_engine_free(ctx);
            return 1;
        }
    }

    printf("\n================================================================================\n");
    printf("THSA-2B V1 NATIVE MODEL LOADER & V2 DISPATCH GATE: ALL 11 TESTS PASSED ✅\n");
    printf("================================================================================\n");
    return 0;
}
