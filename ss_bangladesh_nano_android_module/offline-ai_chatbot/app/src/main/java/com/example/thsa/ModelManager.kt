package com.example.thsa

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.security.MessageDigest

sealed class ModelDownloadState {
    object Idle : ModelDownloadState()
    data class Downloading(val progressPercent: Int, val downloadedBytes: Long, val totalBytes: Long) : ModelDownloadState()
    data class Ready(val modelFile: File, val sizeFormatted: String) : ModelDownloadState()
    data class Error(val message: String) : ModelDownloadState()
}

class ModelManager(private val context: Context) {
    companion object {
        const val MODEL_FILENAME = "model.nano"
        const val VOCAB_FILENAME = "thsa_tokenizer.vocab"
        // FIX-11: Production V2 model — 219 tensors, 2,050,296,320 parameters
        const val EXPECTED_MODEL_SIZE = 765477824L
        const val EXPECTED_MODEL_SHA256 = "0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64"
        private const val TAG = "ModelManager"

        fun computeSha256(file: File): String {
            if (!file.exists()) return "FILE_NOT_FOUND"
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(65536)
            FileInputStream(file).use { input ->
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }

    private val _downloadState = MutableStateFlow<ModelDownloadState>(ModelDownloadState.Idle)
    val downloadState: StateFlow<ModelDownloadState> = _downloadState.asStateFlow()

    val modelFile: File
        get() = File(context.filesDir, MODEL_FILENAME)

    fun getOrInitEngine(): NanoEngine {
        val file = modelFile
        ensureModelExtractedFromAssets(file)
        ensureVocabExtractedFromAssets()

        Log.i(TAG, "NANO_MODEL_PATH=${file.absolutePath}")
        Log.i(TAG, "NANO_MODEL_SIZE=${file.length()}")

        if (!file.exists() || file.length() != EXPECTED_MODEL_SIZE) {
            val errMsg = "Model file missing or invalid size: ${file.length()} (expected $EXPECTED_MODEL_SIZE)"
            Log.e(TAG, errMsg)
            Log.i(TAG, "NANO_MODEL_HASH_MATCH=false")
            _downloadState.value = ModelDownloadState.Error(errMsg)
            throw IllegalStateException(errMsg)
        }

        val hash = computeSha256(file)
        Log.i(TAG, "NANO_MODEL_SHA256=$hash")

        val match = hash.equals(EXPECTED_MODEL_SHA256, ignoreCase = true)
        Log.i(TAG, "NANO_MODEL_HASH_MATCH=$match")

        if (!match) {
            val errMsg = "Model SHA-256 mismatch! Got $hash, expected $EXPECTED_MODEL_SHA256"
            Log.e(TAG, errMsg)
            _downloadState.value = ModelDownloadState.Error(errMsg)
            throw IllegalStateException(errMsg)
        }

        Log.i(TAG, "NANO_ASSET_INTEGRITY=PASS")
        Log.i(TAG, "NANO_V2_MODEL_CONFIRMED: size=$EXPECTED_MODEL_SIZE sha256=$EXPECTED_MODEL_SHA256")
        Log.i(TAG, "NANO_NATIVE_INIT=START")
        val engine = NanoEngine.load(file)
        Log.i(TAG, "NANO_NATIVE_INIT=SUCCESS")

        _downloadState.value = ModelDownloadState.Ready(file, engine.modelSizeFormatted)
        return engine
    }

    private fun ensureModelExtractedFromAssets(targetFile: File) {
        try {
            if (targetFile.exists() && targetFile.length() == EXPECTED_MODEL_SIZE) {
                Log.i(TAG, "NANO_ASSET_OPEN: validated model present at ${targetFile.absolutePath}")
                return
            }

            // Wrong-size legacy model: delete it and re-extract
            if (targetFile.exists() && targetFile.length() != EXPECTED_MODEL_SIZE) {
                Log.w(TAG, "NANO_LEGACY_MODEL_PURGE: found ${targetFile.length()} bytes (expected $EXPECTED_MODEL_SIZE) — deleting")
                targetFile.delete()
            }

            targetFile.parentFile?.mkdirs()
            Log.i(TAG, "Checking assets for $MODEL_FILENAME...")

            var assetStream: InputStream? = null
            try {
                assetStream = context.assets.open(MODEL_FILENAME)
            } catch (e: Exception) {
                Log.d(TAG, "Asset not bundled directly in APK: ${e.message}")
            }

            if (assetStream != null) {
                assetStream.use { input ->
                    FileOutputStream(targetFile).use { output ->
                        val buffer = ByteArray(65536)
                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            output.write(buffer, 0, read)
                        }
                    }
                }
                Log.i(TAG, "Successfully unpacked model (${targetFile.length()} bytes) to local storage!")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting model from assets: ${e.message}", e)
        }
    }

    /**
     * Extract thsa_tokenizer.vocab from APK assets to filesDir so native C++ engine can load
     * the full 65,536-token SentencePiece vocabulary and decode real generated tokens.
     */
    private fun ensureVocabExtractedFromAssets() {
        val vocabFile = File(context.filesDir, VOCAB_FILENAME)
        if (vocabFile.exists() && vocabFile.length() > 100_000L) {
            Log.i(TAG, "TOKENIZER_VOCAB_PRESENT: path=${vocabFile.absolutePath} size=${vocabFile.length()}")
            return
        }
        try {
            Log.i(TAG, "Extracting $VOCAB_FILENAME from APK assets...")
            context.assets.open(VOCAB_FILENAME).use { input ->
                FileOutputStream(vocabFile).use { output ->
                    val buffer = ByteArray(65536)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                    }
                }
            }
            Log.i(TAG, "TOKENIZER_VOCAB_EXTRACTED: path=${vocabFile.absolutePath} size=${vocabFile.length()}")
        } catch (e: Exception) {
            Log.e(TAG, "TOKENIZER_VOCAB_EXTRACT_FAILED: ${e.message}", e)
        }
    }

    suspend fun downloadModel(
        onProgress: (Int) -> Unit = {}
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            _downloadState.value = ModelDownloadState.Downloading(0, 0, EXPECTED_MODEL_SIZE)
            val file = modelFile
            ensureModelExtractedFromAssets(file)
            _downloadState.value = ModelDownloadState.Downloading(100, file.length(), file.length())
            onProgress(100)

            val engine = getOrInitEngine()
            _downloadState.value = ModelDownloadState.Ready(file, engine.modelSizeFormatted)
            Result.success(file)
        } catch (e: Exception) {
            Log.e(TAG, "Local model initialization error: ${e.message}")
            _downloadState.value = ModelDownloadState.Error(e.message ?: "Model error")
            Result.failure(e)
        }
    }
}
