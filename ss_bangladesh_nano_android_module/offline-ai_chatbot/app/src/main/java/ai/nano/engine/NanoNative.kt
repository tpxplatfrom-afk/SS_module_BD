package ai.nano.engine

/**
 * Low-level JNI bindings to libnano_engine.so.
 */
internal object NanoNative {
    init {
        System.loadLibrary("nano_engine")
    }

    @JvmStatic
    external fun nativeInit(modelPath: String): Long

    @JvmStatic
    external fun nativeEncode(handle: Long, text: String): IntArray

    @JvmStatic
    external fun nativeDecodeToken(handle: Long, tokenId: Int): String

    @JvmStatic
    external fun nativeGenerate(
        handle: Long,
        promptTokens: IntArray,
        temperature: Float,
        topP: Float,
        maxOutputTokens: Int,
        callback: NativeTokenCallback
    ): Int

    @JvmStatic
    external fun nativeCancel(handle: Long): Int

    @JvmStatic
    external fun nativeResetSession(handle: Long): Int

    @JvmStatic
    external fun nativeGetTelemetry(handle: Long): NanoTelemetry?

    @JvmStatic
    external fun nativeFree(handle: Long)
}

internal fun interface NativeTokenCallback {
    fun onToken(tokenStr: String, tokenId: Int, isEos: Boolean): Boolean
}
