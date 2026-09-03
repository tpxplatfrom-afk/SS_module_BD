package ai.nano.engine

/**
 * Generation parameters for on-device autoregressive decoding.
 */
data class NanoGenerationConfig(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val repetitionPenalty: Float = 1.1f,
    val maxOutputTokens: Int = 32
) {
    companion object {
        val DEFAULT = NanoGenerationConfig()
        val DETERMINISTIC = NanoGenerationConfig(temperature = 0.0f, topP = 1.0f, topK = 1)
        val CREATIVE = NanoGenerationConfig(temperature = 0.9f, topP = 0.95f, topK = 50)
    }
}
