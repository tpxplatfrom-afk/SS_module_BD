package ai.nano.engine

import java.io.File

/**
 * Universal, Copy-Paste Friendly Response Plugin for Android Developers.
 * Supports instant export to .txt, .md, and clean clipboard copying in 1 line.
 */
data class NanoResponse(
    val prompt: String,
    val text: String,
    val markdown: String,
    val copyText: String,
    val isScreenSafe: Boolean = true
) {
    /**
     * Get clean plain text ready for 1-click clipboard copy (no raw LaTeX/markdown symbols).
     */
    fun toPlainText(): String = copyText

    /**
     * Get standard Markdown format for Markwon or Jetpack Compose Markdown renderers.
     */
    fun toMarkdown(): String = markdown

    /**
     * Save response directly into a .txt file.
     */
    fun saveAsTxt(targetFile: File) {
        targetFile.writeText(copyText, Charsets.UTF_8)
    }

    /**
     * Save response directly into a .md (Markdown) file.
     */
    fun saveAsMd(targetFile: File) {
        targetFile.writeText(markdown, Charsets.UTF_8)
    }
}
