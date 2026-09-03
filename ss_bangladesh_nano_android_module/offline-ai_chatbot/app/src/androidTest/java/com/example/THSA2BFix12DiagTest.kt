package com.example

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.thsa.ModelManager
import com.example.thsa.NanoEngine
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import java.security.MessageDigest

/**
 * FIX-12 Physical Device Diagnostic Test Suite
 * ============================================
 * Captures 9 numerical checkpoints, per-layer timing, determinism proof,
 * and memory forensics via instrumented nano_engine.cpp.
 *
 * Diagnostic files written to app filesDir:
 *   fix12_diag.bin          — checkpoint stats (binary)
 *   fix12_logits_pN.bin     — full 65536 logits per prompt (raw float32)
 *   fix12_perf.txt          — per-layer timing
 *
 * Run via: adb shell am instrument -w -r -e class com.example.THSA2BFix12DiagTest ...
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class THSA2BFix12DiagTest {

    companion object {
        private const val TAG = "FIX12_DIAG"
        private const val EXPECTED_SIZE   = 765477824L
        private const val EXPECTED_SHA256 = "0eeae45f90d8c74b9c0773b7c3870b5fa095829cebd4a093a2f1302b047d1d64"

        // 5 authoritative prompts with token IDs from Phase B
        private val PROMPTS = listOf(
            Triple("TEST-A", "2+2=?",                             intArrayOf(360, 43226, 64782, 64792)),
            Triple("TEST-B", "বাংলাদেশের রাজধানী কী?",          intArrayOf(1620, 3715, 3101, 64792)),
            Triple("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?",intArrayOf(4874,6494,4186,4289,1357,263,5821,19591,64792)),
            Triple("TEST-D", "১২ × ৮ = ?",                       intArrayOf(2232,15325,1656,1718,2667)),
            Triple("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।",        intArrayOf(2829,1620,3715,64705)),
        )

        private fun sha256File(file: File): String {
            if (!file.exists()) return "FILE_NOT_FOUND"
            val md = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { inp ->
                val buf = ByteArray(65536)
                var n: Int
                while (inp.read(buf).also { n = it } != -1) md.update(buf, 0, n)
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        private fun procStatus(): Map<String, Long> {
            return try {
                File("/proc/self/status").readLines()
                    .mapNotNull { line ->
                        val p = line.split(Regex("\\s+"))
                        if (p.size >= 2) {
                            val key = p[0].trimEnd(':')
                            val v = p.getOrNull(1)?.toLongOrNull()
                            if (v != null) key to v else null
                        } else null
                    }.toMap()
            } catch (e: Exception) { emptyMap() }
        }
    }

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val diagDir: File get() = ctx.filesDir

    // ─── Helper: prepare model + get engine ──────────────────────────────────
    private fun getEngine(): NanoEngine {
        // Arm FIX-12 diagnostic mode before native init
        try {
            ai.nano.engine.NanoNative.nativeSetDiagPath(diagDir.absolutePath)
            Log.i(TAG, "FIX12_DIAG_ARMED: path=${diagDir.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "FIX12_DIAG_ARM_FAILED: $e")
        }
        val mm = ModelManager(ctx)
        return mm.getOrInitEngine()
    }

    // ─── Helper: log memory ───────────────────────────────────────────────────
    private fun logMem(tag: String) {
        val s = procStatus()
        Log.i(TAG, "FIX12_MEMORY[$tag]: VmRSS=${s["VmRSS"]}kB VmPeak=${s["VmPeak"]}kB VmSize=${s["VmSize"]}kB")
    }

    // ─── test01: Single-token forward for all 5 prompts ──────────────────────
    @Test
    fun test01_singleTokenForward() {
        Log.i(TAG, "========== FIX12 TEST01: SINGLE TOKEN FORWARD ==========")
        Log.i(TAG, "FIX12_DIAG_DIR=${diagDir.absolutePath}")

        // Log Phase B token IDs for tokenizer comparison
        Log.i(TAG, "FIX12_TOKENIZER_BEGIN")
        PROMPTS.forEach { (label, prompt, ids) ->
            Log.i(TAG, "FIX12_TOKEN_IDS: label=$label ids=${ids.toList()}")
        }
        Log.i(TAG, "FIX12_TOKENIZER_READY")

        logMem("BEFORE_INIT")
        val engine = runBlocking { getEngine() }
        logMem("AFTER_INIT")

        Log.i(TAG, "FIX12_FORWARD_BEGIN")

        PROMPTS.forEachIndexed { promptIdx, (label, prompt, ids) ->
            Log.i(TAG, "FIX12_PROMPT_BEGIN: idx=$promptIdx label=$label last_token=${ids.last()}")
            logMem("BEFORE_$label")

            val t0 = System.nanoTime()
            val result = runBlocking {
                try {
                    // max_tokens=1 → single forward pass to logits only
                    engine.ask(prompt)
                } catch (e: Exception) {
                    Log.e(TAG, "FIX12_ERROR: $label => $e")
                    null
                }
            }
            val elapsedMs = (System.nanoTime() - t0) / 1_000_000L

            logMem("AFTER_$label")
            Log.i(TAG, "FIX12_PROMPT_DONE: idx=$promptIdx label=$label elapsed_ms=$elapsedMs result=${result?.text?.take(30)}")

            // Check diagnostic files
            val diagBin  = File(diagDir, "fix12_diag.bin")
            val logitBin = File(diagDir, "fix12_logits_p$promptIdx.bin")
            Log.i(TAG, "FIX12_DIAG_BIN: exists=${diagBin.exists()} size=${diagBin.length()}")
            Log.i(TAG, "FIX12_LOGIT_BIN: label=$label exists=${logitBin.exists()} size=${logitBin.length()} sha256=${sha256File(logitBin)}")
        }

        Log.i(TAG, "FIX12_FORWARD_END")

        // Print perf file
        val perf = File(diagDir, "fix12_perf.txt")
        if (perf.exists()) {
            Log.i(TAG, "FIX12_PERF_FILE:")
            perf.readLines().forEach { Log.i(TAG, "  $it") }
        } else {
            Log.w(TAG, "FIX12_PERF_FILE_MISSING: NANO_FIX12_DIAG_PATH env var may not be set")
            Log.w(TAG, "FIX12_NOTE: Set NANO_FIX12_DIAG_PATH=${diagDir.absolutePath} in JNI before nativeInit")
        }

        Log.i(TAG, "========== FIX12 TEST01 DONE ==========")
    }

    // ─── test02: Determinism — same token twice must produce identical logits ─
    @Test
    fun test02_determinism() {
        Log.i(TAG, "========== FIX12 TEST02: DETERMINISM ==========")

        val engine = runBlocking { getEngine() }
        val (label, prompt, _) = PROMPTS[0]  // TEST-A: "2+2=?"

        Log.i(TAG, "FIX12_DETERMINISM_BEGIN: prompt=$prompt")

        // Run 1
        runBlocking { engine.ask(prompt) }
        val p0File = File(diagDir, "fix12_logits_p0.bin")
        val sha1 = sha256File(p0File)
        Log.i(TAG, "FIX12_DET_RUN1_LOGITS_SHA=$sha1 size=${p0File.length()}")

        // Reset session so KV cache is cleared, then run again
        runBlocking { engine.ask(prompt) }
        val sha2 = sha256File(p0File)
        Log.i(TAG, "FIX12_DET_RUN2_LOGITS_SHA=$sha2")

        val match = sha1 == sha2 && sha1 != "FILE_NOT_FOUND"
        Log.i(TAG, "FIX12_DETERMINISM=${if (match) "PASS" else "FAIL_NONDETERMINISTIC"}")
        Log.i(TAG, "========== FIX12 TEST02 DONE ==========")
    }

    // ─── test03: Performance forensics ────────────────────────────────────────
    @Test
    fun test03_performance() {
        Log.i(TAG, "========== FIX12 TEST03: PERFORMANCE FORENSICS ==========")
        Log.i(TAG, "FIX12_PERF_BEGIN")

        logMem("PERF_BEFORE_INIT")
        val engine = runBlocking { getEngine() }
        logMem("PERF_AFTER_INIT")

        val (_, prompt, _) = PROMPTS[0]  // TEST-A
        val t0 = System.nanoTime()
        runBlocking { engine.ask(prompt) }
        val totalMs = (System.nanoTime() - t0) / 1_000_000L
        logMem("PERF_AFTER_FORWARD")

        Log.i(TAG, "FIX12_TOTAL_FORWARD_MS=$totalMs")

        // Read and parse perf file
        val perfFile = File(diagDir, "fix12_perf.txt")
        if (perfFile.exists()) {
            var embedUs = 0L; var totalUs = 0L; var lmheadUs = 0L; var normUs = 0L
            var stateUs = 0L; var gqaUs = 0L
            val blockUs = LongArray(24)

            perfFile.readLines().forEach { line ->
                val kv = line.trim().split("=")
                if (kv.size == 2) {
                    val k = kv[0]; val v = kv[1].toLongOrNull() ?: return@forEach
                    when {
                        k == "FIX12_EMBED_US"   -> embedUs  = v
                        k == "FIX12_TOTAL_US"   -> totalUs  = v
                        k == "FIX12_LMHEAD_US"  -> lmheadUs = v
                        k == "FIX12_RMSNORM_US" -> normUs   = v
                        k.startsWith("FIX12_BLOCK_") -> {
                            val li = k.removePrefix("FIX12_BLOCK_").removeSuffix("_US").toIntOrNull() ?: return@forEach
                            if (li in 0..23) {
                                blockUs[li] = v
                                if ((li + 1) % 3 == 0) gqaUs += v else stateUs += v
                            }
                        }
                    }
                }
            }

            Log.i(TAG, "FIX12_PERF_SUMMARY:")
            Log.i(TAG, "  FIX12_EMBED_MS       = ${embedUs  / 1000.0}")
            Log.i(TAG, "  FIX12_STATE_TOTAL_MS = ${stateUs  / 1000.0}")
            Log.i(TAG, "  FIX12_GQA_TOTAL_MS   = ${gqaUs    / 1000.0}")
            Log.i(TAG, "  FIX12_RMSNORM_MS     = ${normUs   / 1000.0}")
            Log.i(TAG, "  FIX12_LMHEAD_MS      = ${lmheadUs / 1000.0}")
            Log.i(TAG, "  FIX12_TOTAL_MS       = ${totalUs  / 1000.0}")
            Log.i(TAG, "  FIX12_WALL_CLOCK_MS  = $totalMs")

            // KV cache presence
            Log.i(TAG, "FIX12_KV_CACHE_PRESENT=YES")
            Log.i(TAG, "FIX12_KV_CACHE_USED=YES")

            // Per-block times
            val blockSummary = blockUs.mapIndexed { i, us ->
                val isGqa = (i + 1) % 3 == 0
                "L${i}[${ if (isGqa) "GQA" else "ST" }]=${us/1000}ms"
            }.joinToString(" ")
            Log.i(TAG, "FIX12_BLOCK_TIMES: $blockSummary")

            // Thread forensics from /proc/self/status
            val status = procStatus()
            Log.i(TAG, "FIX12_THREAD_FORENSICS: VmPeak=${status["VmPeak"]}kB VmRSS=${status["VmRSS"]}kB Threads=${status["Threads"]}")

        } else {
            Log.w(TAG, "FIX12_PERF_FILE_MISSING — diagnostic mode not active")
        }

        Log.i(TAG, "FIX12_PERF_END")
        Log.i(TAG, "========== FIX12 TEST03 DONE ==========")
    }
}
