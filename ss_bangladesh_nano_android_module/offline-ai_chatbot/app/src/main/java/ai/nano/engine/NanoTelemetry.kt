package ai.nano.engine

/**
 * Real-time operational telemetry and memory health packet (Section 9.8).
 */
data class NanoTelemetry(
    val residentRamBytes: Long,
    val activeKvTokens: Int,
    val instantaneousTokPerSec: Float,
    val estimatedTempCelsius: Float,
    val totalTokensGenerated: Int,
    val degradedFlags: Int
) {
    val residentRamMb: Double get() = residentRamBytes / (1024.0 * 1024.0)
    val isKiviEngaged: Boolean get() = (degradedFlags and 0x01) != 0
    val isThermalClamped: Boolean get() = (degradedFlags and 0x02) != 0
    val isContextEvicted: Boolean get() = (degradedFlags and 0x04) != 0
}
