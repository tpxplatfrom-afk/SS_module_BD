package ai.nano.downloader

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * High-Speed, Zero-Cost Background Downloader for THSA-2.41B model.nano
 * Uses GitHub Releases Global Fastly/Azure CDN (100% Free, Zero-Card).
 */
object NanoModelManager {

    // Your 100% Live, Real GitHub Releases CDN Download Link
    const val MODEL_URL = "https://github.com/tpxplatfrom-afk/SS_module_BD/releases/download/v1.0.0/model.nano"
    const val MODEL_FILE_NAME = "model.nano"

    /**
     * Checks if the 654 MB model.nano file has already been downloaded.
     */
    fun isModelDownloaded(context: Context): Boolean {
        val modelFile = File(context.filesDir, MODEL_FILE_NAME)
        return modelFile.exists() && modelFile.length() > 600 * 1024 * 1024 // Verified > 600 MB
    }

    /**
     * Retrieves the local File pointer to model.nano.
     */
    fun getModelFile(context: Context): File {
        return File(context.filesDir, MODEL_FILE_NAME)
    }

    /**
     * Downloads model.nano with real-time percentage callbacks for splash screen progress bar.
     */
    suspend fun downloadModel(
        context: Context,
        onProgress: (percent: Int) -> Unit
    ): File = withContext(Dispatchers.IO) {
        val targetFile = File(context.filesDir, MODEL_FILE_NAME)
        if (isModelDownloaded(context)) {
            return@withContext targetFile
        }

        val tempFile = File(context.filesDir, "$MODEL_FILE_NAME.tmp")
        val url = URL(MODEL_URL)
        val connection = url.openConnection() as HttpURLConnection
        connection.instanceFollowRedirects = true
        connection.connectTimeout = 15000
        connection.readTimeout = 30000
        connection.connect()

        val fileLength = connection.contentLength
        val input = connection.inputStream
        val output = FileOutputStream(tempFile)

        val buffer = ByteArray(64 * 1024)
        var totalBytesRead = 0L
        var count: Int

        while (input.read(buffer).also { count = it } != -1) {
            output.write(buffer, 0, count)
            totalBytesRead += count
            if (fileLength > 0) {
                val progress = ((totalBytesRead * 100) / fileLength).toInt()
                withContext(Dispatchers.Main) {
                    onProgress(progress)
                }
            }
        }

        output.flush()
        output.close()
        input.close()

        // Rename temp file to final model.nano
        tempFile.renameTo(targetFile)
        targetFile
    }
}
