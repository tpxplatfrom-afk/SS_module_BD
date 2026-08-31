/**
 * @file nano_engine.h
 * @brief Primary Public C Engine Interface for THSA-2B V1 On-Device AI Runtime.
 * Standard Compliance: C99 / C++17 compatible header.
 */

#ifndef NANO_ENGINE_H
#define NANO_ENGINE_H

#include "nano_types.h"
#include "nano_config.h"
#include "nano_telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Engine Context Handle */
typedef struct NanoEngineContext NanoEngineContext;

/**
 * @brief Streaming Token Callback Function Pointer
 * @param token_str UTF-8 formatted decoded token text string
 * @param token_id Numerical token ID
 * @param is_eos True if this token marks end-of-sequence
 * @param user_data Opaque pointer passed during generation call
 * @return Return true to continue generation; false to request early halt
 */
typedef bool (*NanoTokenCallback)(const char* token_str, NanoTokenId token_id, bool is_eos, void* user_data);

/**
 * @brief Initialize the THSA-2B Native Engine and allocate static memory arena.
 * @param model_path Path to the serialized .nano binary model package on disk
 * @param config Optional model configuration overrides (pass NULL for default)
 * @param out_ctx Pointer to receive the allocated engine context handle
 * @return NANO_SUCCESS on success, or appropriate error code
 */
NANO_API NanoStatus nano_engine_init(
    const char* model_path,
    const NanoModelConfig* config,
    NanoEngineContext** out_ctx
);

/**
 * @brief Feed a prompt into the engine and stream output tokens via callback.
 * @param ctx Valid engine context handle
 * @param prompt_tokens Array of input token IDs
 * @param num_prompt_tokens Number of token IDs in prompt
 * @param gen_config Sampling parameters
 * @param callback Callback function invoked per generated token
 * @param user_data User context forwarded to callback
 * @return NANO_SUCCESS on completion, NANO_ERR_CANCELLED on abort, or error code
 */
NANO_API NanoStatus nano_engine_generate(
    NanoEngineContext* ctx,
    const NanoTokenId* prompt_tokens,
    size_t num_prompt_tokens,
    const NanoGenerationConfig* gen_config,
    NanoTokenCallback callback,
    void* user_data
);

/**
 * @brief Request non-blocking asynchronous cancellation of active generation.
 * Halts decode loop in <= 5.0 ms without corrupting state arenas (Section 9.6).
 * @param ctx Valid engine context handle
 * @return NANO_SUCCESS on flag set
 */
NANO_API NanoStatus nano_engine_cancel(NanoEngineContext* ctx);

/**
 * @brief Reset conversational session and reclaim KV-cache in O(1) time (Section 9.4).
 * @param ctx Valid engine context handle
 * @return NANO_SUCCESS on reset
 */
NANO_API NanoStatus nano_engine_reset_session(NanoEngineContext* ctx);

/**
 * @brief Query real-time operational telemetry and memory health (Section 9.8).
 * Execution latency <= 0.1 ms with zero lock contention.
 * @param ctx Valid engine context handle
 * @param out_telemetry Pointer to receive telemetry metrics
 * @return NANO_SUCCESS on success
 */
NANO_API NanoStatus nano_engine_get_telemetry(
    const NanoEngineContext* ctx,
    NanoEngineTelemetry* out_telemetry
);

/**
 * @brief Get current engine lifecycle state.
 * @param ctx Valid engine context handle
 * @return Current NanoEngineState enum value
 */
NANO_API NanoEngineState nano_engine_get_state(const NanoEngineContext* ctx);

/**
 * @brief Free all engine static arenas, unmap model file, and destroy context (Section 9.3).
 * Guarantees zero memory leaks (100% RAII teardown).
 * @param ctx Engine context handle to free
 */
NANO_API void nano_engine_free(NanoEngineContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* NANO_ENGINE_H */
