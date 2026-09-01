package bd.sstutor.runtime

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process

enum class MemoryState {
    NORMAL,      // < 150 MB PSS
    WARNING,     // 150 - 200 MB PSS
    CRITICAL,    // 200 - 250 MB PSS
    EMERGENCY    // >= 250 MB PSS
}

data class MemorySnapshot(
    val totalPssMb: Double,
    val javaHeapMb: Double,
    val nativeHeapMb: Double,
    val availableSystemRamMb: Double,
    val memoryState: MemoryState,
    val isLowMemory: Boolean
)

object AndroidMemoryMonitor {
    const val PREFERRED_CEILING_MB = 150.0
    const val HARD_CEILING_MB = 200.0
    const val EMERGENCY_CEILING_MB = 250.0

    private var peakPssObservedMb: Double = 0.0

    fun getMemorySnapshot(context: Context? = null): MemorySnapshot {
        val runtime = Runtime.getRuntime()
        val javaHeap = (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0)
        val nativeHeap = Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)

        var pssMb = javaHeap + nativeHeap
        var availMb = 1024.0
        var isLow = false

        if (context != null) {
            val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            if (actManager != null) {
                val memInfo = ActivityManager.MemoryInfo()
                actManager.getMemoryInfo(memInfo)
                availMb = memInfo.availMem / (1024.0 * 1024.0)
                isLow = memInfo.lowMemory

                val pids = intArrayOf(Process.myPid())
                val pssInfo = actManager.getProcessMemoryInfo(pids)
                if (pssInfo.isNotEmpty()) {
                    pssMb = pssInfo[0].totalPss / 1024.0
                }
            }
        }

        if (pssMb > peakPssObservedMb) {
            peakPssObservedMb = pssMb
        }

        val state = when {
            pssMb >= EMERGENCY_CEILING_MB -> MemoryState.EMERGENCY
            pssMb >= HARD_CEILING_MB -> MemoryState.CRITICAL
            pssMb >= PREFERRED_CEILING_MB -> MemoryState.WARNING
            else -> MemoryState.NORMAL
        }

        return MemorySnapshot(
            totalPssMb = roundTwoDecimals(pssMb),
            javaHeapMb = roundTwoDecimals(javaHeap),
            nativeHeapMb = roundTwoDecimals(nativeHeap),
            availableSystemRamMb = roundTwoDecimals(availMb),
            memoryState = state,
            isLowMemory = isLow
        )
    }

    fun getPeakPss(): Double = roundTwoDecimals(peakPssObservedMb)

    fun resetPeak() {
        peakPssObservedMb = 0.0
    }

    private fun roundTwoDecimals(d: Double): Double = Math.round(d * 100.0) / 100.0
}
