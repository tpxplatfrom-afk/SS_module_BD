package com.example.thsa

import ai.nano.engine.NanoEngine as NativeNanoEngine
import android.util.Log
import java.io.File

/**
 * THSA-2B Native On-Device AI Engine.
 * Delegates 100% of inference directly to native C++ THSA-2B engine via JNI.
 */
class NanoEngine private constructor(
    private val nativeEngine: NativeNanoEngine,
    private val modelFile: File
) {
    companion object {
        private const val TAG = "NanoEngine"
        const val MODEL_NAME = "THSA-2B V1"
        const val MODEL_VERSION = "v1.0.0"
        const val MODEL_PARAMS = "2.41 Billion"

        @JvmStatic
        fun load(modelFile: File): NanoEngine {
            Log.i(TAG, "Loading native THSA-2B model from: ${modelFile.absolutePath}")
            val nativeEng = NativeNanoEngine.load(modelFile)
            return NanoEngine(nativeEng, modelFile)
        }
    }

    /**
     * Process user input directly through native THSA-2B model forward pass.
     */
    suspend fun ask(userInput: String): NanoResponse {
        val nativeResp = nativeEngine.ask(userInput)
        return NanoResponse(
            text = nativeResp.text,
            copyText = nativeResp.copyText
        )
    }

    val modelPath: String get() = modelFile.absolutePath
    val isModelLoaded: Boolean get() = nativeEngine.isModelLoaded
    val modelSizeFormatted: String
        get() {
            val bytes = if (modelFile.exists()) modelFile.length() else 0L
            val mb = bytes / (1024.0 * 1024.0)
            return if (mb >= 1.0) String.format("%.1f MB", mb) else "$bytes B"
        }
}

