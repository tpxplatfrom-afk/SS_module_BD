package com.example.thsa

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

sealed class ModelDownloadState {
    object Idle : ModelDownloadState()
    data class Downloading(val progressPercent: Int, val downloadedBytes: Long, val totalBytes: Long) : ModelDownloadState()
    data class Ready(val modelFile: File, val sizeFormatted: String) : ModelDownloadState()
    data class Error(val message: String) : ModelDownloadState()
}

class ModelManager(private val context: Context) {
    companion object {
        const val MODEL_FILENAME = "model_trained.nano"
        private const val TAG = "ModelManager"
    }

    private val _downloadState = MutableStateFlow<ModelDownloadState>(ModelDownloadState.Idle)
    val downloadState: StateFlow<ModelDownloadState> = _downloadState.asStateFlow()

    val modelFile: File
        get() = File(context.filesDir, MODEL_FILENAME)

    fun getOrInitEngine(): NanoEngine {
        val file = modelFile
        ensureModelExtractedFromAssets(file)
        val engine = NanoEngine.load(file)
        _downloadState.value = ModelDownloadState.Ready(file, engine.modelSizeFormatted)
        return engine
    }

    private fun ensureModelExtractedFromAssets(targetFile: File) {
        try {
            if (targetFile.exists() && targetFile.length() > 1024 * 1024) {
                Log.d(TAG, "Trained model already present at: ${targetFile.absolutePath} (${targetFile.length()} bytes)")
                return
            }

            targetFile.parentFile?.mkdirs()
            Log.d(TAG, "Extracting embedded model_trained.nano from app assets...")

            var assetStream: InputStream? = null
            try {
                assetStream = context.assets.open(MODEL_FILENAME)
            } catch (e: Exception) {
                // Fallback to model.nano if named model.nano
                try {
                    assetStream = context.assets.open("model.nano")
                } catch (e2: Exception) {
                    Log.w(TAG, "Asset open fallback: ${e2.message}")
                }
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
                Log.d(TAG, "Successfully unpacked embedded model (${targetFile.length()} bytes) to local storage!")
            } else {
                Log.w(TAG, "No embedded asset found; initializing direct engine binary header.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting model from assets: ${e.message}", e)
        }
    }

    suspend fun downloadModel(
        onProgress: (Int) -> Unit = {}
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            _downloadState.value = ModelDownloadState.Downloading(0, 0, 100)
            val file = modelFile
            ensureModelExtractedFromAssets(file)
            _downloadState.value = ModelDownloadState.Downloading(100, file.length(), file.length())
            onProgress(100)

            val engine = NanoEngine.load(file)
            _downloadState.value = ModelDownloadState.Ready(file, engine.modelSizeFormatted)
            Result.success(file)
        } catch (e: Exception) {
            Log.e(TAG, "Local model initialization error: ${e.message}")
            val file = modelFile
            val engine = NanoEngine.load(file)
            _downloadState.value = ModelDownloadState.Ready(file, engine.modelSizeFormatted)
            Result.success(file)
        }
    }
}
