cmake_minimum_required(VERSION 3.22.1)
project(nano_ai_engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization & Compiler Flags
if(MSVC)
    add_compile_options(/W3 /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NANO_BUILD_DLL)
else()
    add_compile_options(-O3 -Wall -Wextra -Wpedantic)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR ANDROID_ABI MATCHES "arm64-v8a")
        add_compile_options(-march=armv8-a+simd)
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
)

# Engine Core Source Files
set(NANO_ENGINE_SRCS
    src/kernels/neon_gemv_ternary.cpp
    src/kernels/neon_kv_cache.cpp
    src/kernels/neon_state_update.cpp
    src/kernels/neon_norm_act.cpp
    src/kernels/neon_gemv_int8.cpp        # FIX-A: ARMv7 NEON dense INT8 LM-head GEMV
    src/engine/memory_arena.cpp
    src/engine/nano_engine.cpp
    src/tokenizer/bpe_trie_runtime.cpp
    src/tokenizer/unicode_nfc.cpp
    src/tokenizer/utf8_ring_buffer.cpp
)

# Static Library Target
add_library(nano_engine_static STATIC ${NANO_ENGINE_SRCS})

# Shared Library Target (Android JNI or Host C API)
if(ANDROID)
    add_library(nano_engine SHARED ${NANO_ENGINE_SRCS} jni/nano_engine_jni.cpp)
    target_link_libraries(nano_engine log)
else()
    add_library(nano_engine SHARED ${NANO_ENGINE_SRCS})
endif()

# FIX-A: ARMv7 NEON compiler flags.
# -mfpu=neon   : Enable NEON FPU, defines __ARM_NEON so intrinsics are available.
# -mfloat-abi=softfp: Required on armeabi-v7a to use hardware FP with soft-float ABI linkage.
# Applied to shared library (production .so) and static library.
# arm64-v8a already has NEON unconditionally; this guard avoids double-flagging.
if(ANDROID_ABI MATCHES "armeabi-v7a")
    target_compile_options(nano_engine        PRIVATE -mfpu=neon -mfloat-abi=softfp)
    target_compile_options(nano_engine_static PRIVATE -mfpu=neon -mfloat-abi=softfp)
endif()

# Unit Test Executables
add_executable(test_neon_kernels
    tests/unit/test_neon_kernels.cpp
    ${NANO_ENGINE_SRCS}
)

add_executable(test_native_model_loader
    tests/unit/test_native_model_loader.cpp
    ${NANO_ENGINE_SRCS}
)

add_executable(test_neural_forward_pass
    tests/unit/test_neural_forward_pass.cpp
    ${NANO_ENGINE_SRCS}
)

if(ANDROID)
    target_link_libraries(test_neon_kernels log)
    target_link_libraries(test_native_model_loader log)
    target_link_libraries(test_neural_forward_pass log)
endif()

# FIX-A: Dense INT8 GEMV deterministic test (scalar reference vs NEON)
add_executable(test_dense_int8_gemv
    tests/unit/test_dense_int8_gemv.cpp
    src/kernels/neon_gemv_int8.cpp        # Only the new kernel — no full engine dependency
)
if(ANDROID)
    target_link_libraries(test_dense_int8_gemv log)
    if(ANDROID_ABI MATCHES "armeabi-v7a")
        target_compile_options(test_dense_int8_gemv PRIVATE -mfpu=neon -mfloat-abi=softfp)
    endif()
endif()

# FIX-A: Real model tensor-218 verifier & physical benchmark
add_executable(test_real_model_lmhead_neon
    tests/unit/test_real_model_lmhead_neon.cpp
    src/kernels/neon_gemv_int8.cpp
)
if(ANDROID)
    target_link_libraries(test_real_model_lmhead_neon log)
    if(ANDROID_ABI MATCHES "armeabi-v7a")
        target_compile_options(test_real_model_lmhead_neon PRIVATE -mfpu=neon -mfloat-abi=softfp)
    endif()
endif()

# FIX-B: State Conv numerical correctness test
add_executable(test_state_conv_numerical
    tests/unit/test_state_conv_numerical.cpp
    src/kernels/neon_state_update.cpp
    src/kernels/neon_norm_act.cpp
    src/kernels/neon_gemv_ternary.cpp
)
target_include_directories(test_state_conv_numerical PRIVATE include)
if(ANDROID)
    target_link_libraries(test_state_conv_numerical log m)
    if(ANDROID_ABI MATCHES "armeabi-v7a")
        target_compile_options(test_state_conv_numerical PRIVATE -mfpu=neon -mfloat-abi=softfp)
    endif()
else()
    target_link_libraries(test_state_conv_numerical m)
endif()

# FIX-C: GQA Attention numerical correctness test
add_executable(test_gqa_numerical
    tests/unit/test_gqa_numerical.cpp
    src/kernels/neon_kv_cache.cpp
    src/kernels/neon_norm_act.cpp
    src/kernels/neon_gemv_ternary.cpp
)
target_include_directories(test_gqa_numerical PRIVATE include)
if(ANDROID)
    target_link_libraries(test_gqa_numerical log m)
    if(ANDROID_ABI MATCHES "armeabi-v7a")
        target_compile_options(test_gqa_numerical PRIVATE -mfpu=neon -mfloat-abi=softfp)
    endif()
else()
    target_link_libraries(test_gqa_numerical m)
endif()

# FIX-14: INT8 Activation Quantization Contract & Numerical Consistency Test
add_executable(test_quantize_int8_numerical
    tests/unit/test_quantize_int8_numerical.cpp
    src/kernels/neon_norm_act.cpp
    src/kernels/neon_gemv_ternary.cpp
)
target_include_directories(test_quantize_int8_numerical PRIVATE include)
if(ANDROID)
    target_link_libraries(test_quantize_int8_numerical log m)
    if(ANDROID_ABI MATCHES "armeabi-v7a")
        target_compile_options(test_quantize_int8_numerical PRIVATE -mfpu=neon -mfloat-abi=softfp)
    endif()
else()
    target_link_libraries(test_quantize_int8_numerical m)
endif()

enable_testing()
add_test(NAME TestNeonKernels COMMAND test_neon_kernels)
add_test(NAME TestNativeModelLoader COMMAND test_native_model_loader)
add_test(NAME TestNeuralForwardPass COMMAND test_neural_forward_pass)
add_test(NAME TestDenseInt8Gemv COMMAND test_dense_int8_gemv)   # FIX-A
add_test(NAME TestRealModelLmheadNeon COMMAND test_real_model_lmhead_neon) # FIX-A
add_test(NAME TestStateConvNumerical COMMAND test_state_conv_numerical)    # FIX-B
add_test(NAME TestGqaNumerical COMMAND test_gqa_numerical)                # FIX-C
add_test(NAME TestQuantizeInt8Numerical COMMAND test_quantize_int8_numerical) # FIX-14


