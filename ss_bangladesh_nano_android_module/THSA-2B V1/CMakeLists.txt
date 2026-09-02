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

enable_testing()
add_test(NAME TestNeonKernels COMMAND test_neon_kernels)
add_test(NAME TestNativeModelLoader COMMAND test_native_model_loader)
add_test(NAME TestNeuralForwardPass COMMAND test_neural_forward_pass)
