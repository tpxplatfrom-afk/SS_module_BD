package com.example.thsa

import java.io.File
import java.io.FileWriter
import java.io.IOException

/**
 * THSA-2.41B On-Device AI Engine Response Model.
 * Supports UI text, 1-Click clean clipboard text, and direct .txt / .md file exports.
 */
data class NanoResponse(
    val text: String,
    val copyText: String = text
) {
    /**
     * Direct .txt file export without markdown formatting artifacts.
     */
    fun saveAsTxt(file: File): Boolean {
        return try {
            file.parentFile?.mkdirs()
            FileWriter(file).use { writer ->
                writer.write(copyText)
            }
            true
        } catch (e: IOException) {
            e.printStackTrace()
            false
        }
    }

    /**
     * Direct .md file export with rich markdown and header metadata.
     */
    fun saveAsMd(file: File): Boolean {
        return try {
            file.parentFile?.mkdirs()
            FileWriter(file).use { writer ->
                val mdContent = buildString {
                    appendLine("# THSA-2.41B Offline AI Output")
                    appendLine("> Generated on-device with zero internet dependency.")
                    appendLine()
                    appendLine(text)
                }
                writer.write(mdContent)
            }
            true
        } catch (e: IOException) {
            e.printStackTrace()
            false
        }
    }
}
