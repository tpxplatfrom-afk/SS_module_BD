/**
 * @file nano_engine_jni.cpp
 * @brief Android JNI Native Bridge for THSA-2B On-Device AI Engine.
 * Features RAII local frame guards (PushLocalFrame), async cancellation, and typed error handling.
 */

#include <jni.h>
#include <string.h>
#include <stdint.h>
#include "../include/nano_engine.h"
#include "../include/nano_types.h"
#include "../include/nano_config.h"
#include "../include/nano_telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Helper: Throw Kotlin NanoEngineException with typed error code
static void throw_nano_exception(JNIEnv* env, const char* message, int error_code) {
    jclass exc_class = env->FindClass("ai/nano/engine/NanoEngineException");
    if (exc_class) {
        jmethodID constructor = env->GetMethodID(exc_class, "<init>", "(Ljava/lang/String;I)V");
        if (constructor) {
            jstring msg_str = env->NewStringUTF(message);
            jthrowable exc_obj = (jthrowable)env->NewObject(exc_class, constructor, msg_str, (jint)error_code);
            env->Throw(exc_obj);
        }
    }
}

JNIEXPORT jlong JNICALL
Java_ai_nano_engine_NanoNative_nativeInit(
    JNIEnv* env,
    jclass clazz,
    jstring model_path_jstr
) {
    (void)clazz;
    if (!model_path_jstr) {
        throw_nano_exception(env, "Model path cannot be null", NANO_ERR_INVALID_PARAM);
        return 0;
    }
    
    const char* path_cstr = env->GetStringUTFChars(model_path_jstr, NULL);
    NanoEngineContext* ctx = NULL;
    NanoModelConfig config = nano_config_default_2b();
    
    NanoStatus status = nano_engine_init(path_cstr, &config, &ctx);
    env->ReleaseStringUTFChars(model_path_jstr, path_cstr);
    
    if (status != NANO_SUCCESS || !ctx) {
        throw_nano_exception(env, "Failed to initialize THSA-2B engine arena", status);
        return 0;
    }
    
    return (jlong)ctx;
}

JNIEXPORT jint JNICALL
Java_ai_nano_engine_NanoNative_nativeGenerate(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jintArray prompt_tokens_array,
    jfloat temperature,
    jfloat top_p,
    jint max_output_tokens,
    jobject token_callback_obj
) {
    (void)clazz;
    NanoEngineContext* ctx = (NanoEngineContext*)handle;
    if (!ctx) {
        throw_nano_exception(env, "Invalid engine handle", NANO_ERR_INVALID_PARAM);
        return NANO_ERR_INVALID_PARAM;
    }
    
    jsize num_prompt_tokens = env->GetArrayLength(prompt_tokens_array);
    jint* token_elems = env->GetIntArrayElements(prompt_tokens_array, NULL);
    
    NanoGenerationConfig gen_cfg = nano_gen_config_default();
    gen_cfg.temperature = temperature;
    gen_cfg.top_p = top_p;
    gen_cfg.max_output_tokens = max_output_tokens;
    
    // Lookup callback method: onToken(String token, int tokenId, boolean isEos) -> boolean
    jclass callback_class = env->GetObjectClass(token_callback_obj);
    jmethodID on_token_method = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;IZ)Z");
    
    // Callback struct wrapper
    struct CallbackState {
        JNIEnv*   env;
        jobject   callback_obj;
        jmethodID method_id;
    } cb_state = { env, token_callback_obj, on_token_method };
    
    auto native_callback = [](const char* token_str, NanoTokenId token_id, bool is_eos, void* user_data) -> bool {
        CallbackState* state = (CallbackState*)user_data;
        JNIEnv* local_env = state->env;
        
        // Push local frame (capacity 16) to strictly prevent Android 512 ART table overflow
        if (local_env->PushLocalFrame(16) < 0) {
            return false; // Out of memory in JNI local table
        }
        
        jstring j_str = local_env->NewStringUTF(token_str);
        jboolean keep_generating = local_env->CallBooleanMethod(
            state->callback_obj,
            state->method_id,
            j_str,
            (jint)token_id,
            (jboolean)is_eos
        );
        
        local_env->PopLocalFrame(NULL); // Free local references instantly
        return (bool)keep_generating;
    };
    
    NanoStatus status = nano_engine_generate(
        ctx,
        (const NanoTokenId*)token_elems,
        (size_t)num_prompt_tokens,
        &gen_cfg,
        native_callback,
        &cb_state
    );
    
    env->ReleaseIntArrayElements(prompt_tokens_array, token_elems, JNI_ABORT);
    return (jint)status;
}

JNIEXPORT jint JNICALL
Java_ai_nano_engine_NanoNative_nativeCancel(
    JNIEnv* env,
    jclass clazz,
    jlong handle
) {
    (void)env;
    (void)clazz;
    NanoEngineContext* ctx = (NanoEngineContext*)handle;
    if (!ctx) return NANO_ERR_INVALID_PARAM;
    return (jint)nano_engine_cancel(ctx);
}

JNIEXPORT jint JNICALL
Java_ai_nano_engine_NanoNative_nativeResetSession(
    JNIEnv* env,
    jclass clazz,
    jlong handle
) {
    (void)env;
    (void)clazz;
    NanoEngineContext* ctx = (NanoEngineContext*)handle;
    if (!ctx) return NANO_ERR_INVALID_PARAM;
    return (jint)nano_engine_reset_session(ctx);
}

JNIEXPORT jobject JNICALL
Java_ai_nano_engine_NanoNative_nativeGetTelemetry(
    JNIEnv* env,
    jclass clazz,
    jlong handle
) {
    (void)clazz;
    NanoEngineContext* ctx = (NanoEngineContext*)handle;
    if (!ctx) return NULL;
    
    NanoEngineTelemetry telem;
    nano_engine_get_telemetry(ctx, &telem);
    
    jclass telem_class = env->FindClass("ai/nano/engine/NanoTelemetry");
    if (!telem_class) return NULL;
    
    jmethodID constructor = env->GetMethodID(telem_class, "<init>", "(JIFFII)V");
    if (!constructor) return NULL;
    
    return env->NewObject(
        telem_class,
        constructor,
        (jlong)telem.resident_ram_bytes,
        (jint)telem.active_kv_tokens,
        (jfloat)telem.instantaneous_tok_per_s,
        (jfloat)telem.estimated_temp_c,
        (jint)telem.total_tokens_generated,
        (jint)telem.degraded_flags
    );
}

JNIEXPORT void JNICALL
Java_ai_nano_engine_NanoNative_nativeFree(
    JNIEnv* env,
    jclass clazz,
    jlong handle
) {
    (void)env;
    (void)clazz;
    NanoEngineContext* ctx = (NanoEngineContext*)handle;
    if (ctx) {
        nano_engine_free(ctx);
    }
}

#ifdef __cplusplus
}
#endif
