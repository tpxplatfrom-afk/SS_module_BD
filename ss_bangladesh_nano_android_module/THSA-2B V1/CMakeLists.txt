cmake_minimum_required(VERSION 3.22.1)
project(nano_ai_engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization & Compiler Flags
if(MSVC)
    add_compile_options(/O2 /W4 /utf-8)
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
)

# Static Library Target
add_library(nano_engine_static STATIC ${NANO_ENGINE_SRCS})

# Unit Test Executable
add_executable(test_neon_kernels
    tests/unit/test_neon_kernels.cpp
    ${NANO_ENGINE_SRCS}
)

enable_testing()
add_test(NAME TestNeonKernels COMMAND test_neon_kernels)
