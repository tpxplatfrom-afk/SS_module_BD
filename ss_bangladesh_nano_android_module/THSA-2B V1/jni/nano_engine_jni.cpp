/**
 * @file nano_engine_jni.cpp
 * @brief Android JNI Native Bridge for THSA-2B On-Device AI Engine.
 * Features RAII local frame guards (PushLocalFrame), async cancellation, and typed error handling.
 */

#include <jni.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#define JNI_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NanoEngineJNI", __VA_ARGS__)
#define JNI_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NanoEngineJNI", __VA_ARGS__)
#else
#define JNI_LOGI(...)
#define JNI_LOGE(...)
#endif

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
    JNI_LOGI("NANO_NATIVE_LIBRARY_LOADED");
    if (!model_path_jstr) {
        JNI_LOGE("NANO_NATIVE_INIT failed: null model path");
        throw_nano_exception(env, "Model path cannot be null", NANO_ERR_INVALID_PARAM);
        return 0;
    }
    
    const char* path_cstr = env->GetStringUTFChars(model_path_jstr, NULL);
    JNI_LOGI("NANO_NATIVE_INIT: path=%s", path_cstr);
    
    NanoEngineContext* ctx = NULL;
    NanoModelConfig config = nano_config_default_2b();
    
    NanoStatus status = nano_engine_init(path_cstr, &config, &ctx);
    env->ReleaseStringUTFChars(model_path_jstr, path_cstr);
    
    if (status != NANO_SUCCESS || !ctx) {
        JNI_LOGE("NANO_NATIVE_INIT failed: status=%d", status);
        throw_nano_exception(env, "Failed to initialize THSA-2B engine arena", status);
        return 0;
    }
    
    JNI_LOGI("NANO_NATIVE_INIT_SUCCESS: handle=%p", (void*)ctx);
    return (jlong)ctx;
}

JNIEXPORT jintArray JNICALL
Java_ai_nano_engine_NanoNative_nativeEncode(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jstring text_jstr
) {
    (void)clazz;
    NanoEngineContext* ctx = (NanoEngineContext*)handle;
    if (!ctx || !text_jstr) {
        return NULL;
    }
    
    const char* text_cstr = env->GetStringUTFChars(text_jstr, NULL);
    size_t text_len = strlen(text_cstr);
    
    JNI_LOGI("NANO_CAUSAL_TOKENIZE_BEGIN: prompt_chars=%zu", text_len);

    NanoTokenId tokens[4096];
    size_t num_tokens = 0;
    
    NanoStatus status = nano_engine_encode(
        ctx,
        text_cstr,
        text_len,
        tokens,
        4096,
        &num_tokens
    );
    
    env->ReleaseStringUTFChars(text_jstr, text_cstr);
    
    JNI_LOGI("NANO_CAUSAL_TOKENIZE_RESULT: prompt_chars=%zu, token_count=%zu", text_len, num_tokens);

    if (status != NANO_SUCCESS || num_tokens == 0) {
        jintArray fallback = env->NewIntArray(1);
        jint unk = NANO_TOKEN_UNK;
        env->SetIntArrayRegion(fallback, 0, 1, &unk);
        JNI_LOGI("NANO_CAUSAL_INPUT_TOKENS: [%d]", unk);
        return fallback;
    }
    
    std::string token_list_str = "[";
    for (size_t i = 0; i < num_tokens; ++i) {
        token_list_str += std::to_string(tokens[i]);
        if (i + 1 < num_tokens) token_list_str += ", ";
    }
    token_list_str += "]";
    JNI_LOGI("NANO_CAUSAL_INPUT_TOKENS: %s", token_list_str.c_str());

    jintArray result = env->NewIntArray((jsize)num_tokens);
    env->SetIntArrayRegion(result, 0, (jsize)num_tokens, (const jint*)tokens);
    return result;
}

JNIEXPORT jstring JNICALL
Java_ai_nano_engine_NanoNative_nativeDecodeToken(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint token_id
) {
    (void)clazz;
    NanoEngineContext* ctx = (NanoEngineContext*)handle;
    if (!ctx) return NULL;
    
    char buf[128] = {0};
    size_t bytes_written = 0;
    nano_engine_decode_token(ctx, (NanoTokenId)token_id, buf, sizeof(buf), &bytes_written);
    if (bytes_written == 0) {
        buf[0] = '\0';
    }
    return env->NewStringUTF(buf);
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
    
    JNI_LOGI("NANO_GENERATE_BEGIN: prompt_tokens=%d, temp=%.2f, top_p=%.2f, max_tokens=%d",
             (int)num_prompt_tokens, temperature, top_p, max_output_tokens);
    
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
    JNI_LOGI("NANO_GENERATE_END: status=%d", status);
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

// FIX-12: Set diagnostic path from Kotlin before nativeInit
// Writes global env var so fix12_init() picks it up in nano_engine_init()
JNIEXPORT void JNICALL
Java_ai_nano_engine_NanoNative_nativeSetDiagPath(
    JNIEnv* env,
    jclass clazz,
    jstring diagPath
) {
    (void)clazz;
    if (!diagPath) {
        unsetenv("NANO_FIX12_DIAG_PATH");
        return;
    }
    const char* path = env->GetStringUTFChars(diagPath, nullptr);
    if (path && path[0] != '\0') {
        setenv("NANO_FIX12_DIAG_PATH", path, 1);
        __android_log_print(ANDROID_LOG_INFO, "NanoJNI",
            "FIX12_DIAG_PATH_SET=%s", path);
    } else {
        unsetenv("NANO_FIX12_DIAG_PATH");
    }
    if (path) env->ReleaseStringUTFChars(diagPath, path);
}

#ifdef __cplusplus
}
#endif
