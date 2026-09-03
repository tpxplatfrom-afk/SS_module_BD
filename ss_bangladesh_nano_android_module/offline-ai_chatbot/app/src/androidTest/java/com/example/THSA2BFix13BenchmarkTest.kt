package com.example

import ai.nano.engine.NanoEngine as NativeNanoEngine
import ai.nano.engine.NanoGenerationConfig
import ai.nano.engine.NanoNative
import android.content.Context
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.thsa.ModelManager
import org.junit.Assert.*
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File

/**
 * FIX-13 Physical Device Benchmark Test Suite
 * ===========================================
 * Executes 10-token generation across the 5 canonical prompts on itel A662L.
 * Measures:
 *   - Prefill latency
 *   - First-token latency
 *   - Per-token decode latency (TOKEN_01 .. TOKEN_10)
 *   - Total latency & tokens/sec
 *   - Memory telemetry (Java heap, Native heap, VmRSS, VmPeak)
 *   - Determinism across repeated runs
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class THSA2BFix13BenchmarkTest {

    companion object {
        private const val TAG = "FIX13_BENCHMARK"

        private val PROMPTS = listOf(
            Triple("TEST-A", "2+2=?",                             intArrayOf(360, 43226, 64782, 64792)),
            Triple("TEST-B", "বাংলাদেশের রাজধানী কী?",          intArrayOf(1620, 3715, 3101, 64792)),
            Triple("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?",intArrayOf(4874, 6494, 4186, 4289, 1357, 263, 5821, 19591, 64792)),
            Triple("TEST-D", "১২ × ৮ = ?",                       intArrayOf(2232, 15325, 1656, 1718, 2667)),
            Triple("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।",        intArrayOf(2829, 1620, 3715, 64705)),
        )

        private const val GENERATION_TOKENS = 10
    }

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun procStatus(): Map<String, Long> {
        return try {
            File("/proc/self/status").readLines().mapNotNull { line ->
                val p = line.split(Regex("\\s+"))
                if (p.size >= 2) p[0].trimEnd(':') to (p.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null) else null
            }.toMap()
        } catch (e: Exception) { emptyMap() }
    }

    private fun loadNativeEngine(): NativeNanoEngine {
        val mm = ModelManager(ctx)
        val modelFile = mm.modelFile
        if (!modelFile.exists()) {
            try { mm.getOrInitEngine() } catch (_: Exception) {}
        }
        require(modelFile.exists()) { "model.nano not found at ${modelFile.absolutePath}" }
        Log.i(TAG, "FIX13_MODEL_PATH=${modelFile.absolutePath} size=${modelFile.length()}")
        return NativeNanoEngine.load(modelFile)
    }

    @Test
    fun test01_runPhysicalBenchmark() {
        Log.i(TAG, "============================================================")
        Log.i(TAG, "FIX-13 PHYSICAL BENCHMARK SUITE: itel A662L")
        Log.i(TAG, "============================================================")

        val memInitial = procStatus()
        Log.i(TAG, "MEM_INITIAL: VmRSS=${memInitial["VmRSS"]}kB VmPeak=${memInitial["VmPeak"]}kB")

        val engine = loadNativeEngine()
        val handle = engine.javaClass.getDeclaredField("nativeHandle")
            .also { it.isAccessible = true }.getLong(engine)

        val memAfterLoad = procStatus()
        Log.i(TAG, "MEM_AFTER_LOAD: VmRSS=${memAfterLoad["VmRSS"]}kB VmPeak=${memAfterLoad["VmPeak"]}kB")

        PROMPTS.forEachIndexed { pIdx, (label, promptText, ids) ->
            Log.i(TAG, "------------------------------------------------------------")
            Log.i(TAG, "BENCHMARK_PROMPT: label=$label text='$promptText' tokens=${ids.size}")
            Log.i(TAG, "------------------------------------------------------------")

            val runOutputs = mutableListOf<List<Int>>()
            val runDurations = mutableListOf<Long>()

            // 4 Runs: Run 0 = Warm-up, Run 1..3 = Measured
            for (run in 0..3) {
                val isWarmup = (run == 0)
                val runTag = if (isWarmup) "WARMUP" else "RUN_$run"

                // Reset session state
                NanoNative.nativeResetSession(handle)

                val tokenTimestamps = mutableListOf<Long>()
                val generatedTokenIds = mutableListOf<Int>()
                val generatedTokenStrs = mutableListOf<String>()

                val tStart = System.nanoTime()
                var tFirstToken: Long = -1L

                NanoNative.nativeGenerate(
                    handle,
                    ids,
                    0.0f, 1.0f, GENERATION_TOKENS,
                    ai.nano.engine.NativeTokenCallback { tokenStr, tokenId, isEos ->
                        val now = System.nanoTime()
                        if (generatedTokenIds.isEmpty()) {
                            tFirstToken = now
                        }
                        tokenTimestamps.add(now)
                        generatedTokenIds.add(tokenId)
                        generatedTokenStrs.add(tokenStr)
                        generatedTokenIds.size < GENERATION_TOKENS
                    }
                )

                val tEnd = System.nanoTime()
                val totalMs = (tEnd - tStart) / 1_000_000L
                val ttftMs = if (tFirstToken > 0) (tFirstToken - tStart) / 1_000_000L else totalMs

                Log.i(TAG, "PROMPT=$label $runTag: total_ms=$totalMs ttft_ms=$ttftMs tokens=${generatedTokenIds.size}")
                Log.i(TAG, "PROMPT=$label $runTag OUTPUT_TOKENS=$generatedTokenIds")

                // Per-token latencies
                for (i in generatedTokenIds.indices) {
                    val latMs = if (i == 0) {
                        ttftMs
                    } else {
                        (tokenTimestamps[i] - tokenTimestamps[i - 1]) / 1_000_000L
                    }
                    Log.i(TAG, "PROMPT=$label $runTag TOKEN_%02d: id=%d lat_ms=%d text='%s'".format(i + 1, generatedTokenIds[i], latMs, generatedTokenStrs[i]))
                }

                if (!isWarmup) {
                    runOutputs.add(generatedTokenIds)
                    runDurations.add(totalMs)
                }
            }

            // Verify determinism across measured runs
            val run1Tokens = runOutputs.getOrNull(0) ?: emptyList()
            val run2Tokens = runOutputs.getOrNull(1) ?: emptyList()
            val run3Tokens = runOutputs.getOrNull(2) ?: emptyList()

            val isDeterministic = (run1Tokens == run2Tokens && run2Tokens == run3Tokens)
            Log.i(TAG, "PROMPT=$label DETERMINISTIC=$isDeterministic")
            assertTrue("Determinism failure for prompt $label", isDeterministic)

            // Median duration of measured runs
            val sortedDur = runDurations.sorted()
            val medianMs = sortedDur[sortedDur.size / 2]
            val tokPerSec = if (medianMs > 0) (GENERATION_TOKENS * 1000.0) / medianMs else 0.0
            Log.i(TAG, "PROMPT=$label SUMMARY: min_ms=${sortedDur.first()} median_ms=$medianMs max_ms=${sortedDur.last()} tok_per_sec=%.3f".format(tokPerSec))
        }

        val memFinal = procStatus()
        Log.i(TAG, "MEM_FINAL: VmRSS=${memFinal["VmRSS"]}kB VmPeak=${memFinal["VmPeak"]}kB")

        engine.close()
        Log.i(TAG, "FIX-13 PHYSICAL BENCHMARK COMPLETE: PASS")
    }
}
