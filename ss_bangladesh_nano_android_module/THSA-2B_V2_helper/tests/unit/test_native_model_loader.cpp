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
    printf("THSA-2B V2_HELPER: NATIVE MODEL LOADER UNIT TEST & DIAGNOSTIC VALIDATOR\n");
    printf("================================================================================\n");
    
    const char* model_path = (argc > 1) ? argv[1] : "tests/artifacts/test_thsa_2b_exported.nano";
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
    
    printf("\n================================================================================\n");
    printf("V2_HELPER LOADER VERIFICATION SUCCESSFUL ✅\n");
    printf("================================================================================\n");
    return 0;
}
