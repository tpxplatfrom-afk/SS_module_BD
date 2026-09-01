package ai.nano.engine

/**
 * Structured, screen-safe response wrapper for educational math & science queries.
 * Ensures numbers, LaTeX formulas, and Bengali text render safely without breaking UI layouts.
 */
data class NanoMathResponse(
    val query: String,
    val formattedMarkdown: String,
    val calculationBlock: String,
    val explanationBlock: String,
    val socraticHint: String,
    val isScreenSafe: Boolean = true
) {
    /**
     * Get clean, standard Markdown for rendering in Markwon, Jetpack Compose, or Noties Markdown.
     */
    fun toMarkdown(): String = formattedMarkdown

    /**
     * Get HTML formatted string for direct rendering in Android WebView or Html.fromHtml().
     */
    fun toHtml(): String {
        return """
            <div style="font-family: sans-serif; line-height: 1.6; color: #212529;">
                <div style="background: #f8f9fa; border-left: 4px solid #007bff; padding: 12px; margin-bottom: 12px;">
                    <h3>🔢 ১. গাণিতিক হিসাব</h3>
                    <pre style="white-space: pre-wrap; font-family: monospace;">$calculationBlock</pre>
                </div>
                <div style="background: #e8f5e9; border-left: 4px solid #28a745; padding: 12px; margin-bottom: 12px;">
                    <h3>💡 ২. সহজ ব্যাখ্যা</h3>
                    <p>$explanationBlock</p>
                </div>
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px;">
                    <h3>🎯 ৩. সহনশীল ইঙ্গিত</h3>
                    <p>$socraticHint</p>
                </div>
            </div>
        """.trimIndent()
    }
}
