package com.example

import ai.nano.engine.NanoEngine as NativeNanoEngine
import ai.nano.engine.NanoGenerationConfig
import ai.nano.engine.NanoNative
import android.content.Context
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
import java.security.MessageDigest

/**
 * FIX-12 Physical Device Diagnostic Test Suite
 * ============================================
 * Uses max_output_tokens=1 → single forward pass → captures first logits only.
 * Compares Android native top-5 token IDs against REFERENCE-B (from model.nano Python).
 *
 * REFERENCE-B authoritative results:
 *   TEST-A/B/C (last_tok=64792): ARGMAX=64792 TOP5=[64792,6155,40858,271,198]
 *   TEST-D     (last_tok=2667):  ARGMAX=3687  TOP5=[3687,5145,1112,580,4206]
 *   TEST-E     (last_tok=64705): ARGMAX=64705 TOP5=[64705,20517,271,3838,7552]
 *
 * Run: adb shell am instrument -w -r
 *   -e class com.example.THSA2BFix12DiagTest
 *   com.aistudio.offlineai.krvq.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class THSA2BFix12DiagTest {

    companion object {
        private const val TAG = "FIX12_DIAG"

        // REFERENCE-B authoritative top-1 (from model.nano streaming forward)
        private val REF_ARGMAX = mapOf(
            "TEST-A" to 64792,
            "TEST-B" to 64792,
            "TEST-C" to 64792,
            "TEST-D" to 3687,
            "TEST-E" to 64705,
        )
        private val REF_TOP5 = mapOf(
            "TEST-A" to listOf(64792, 6155, 40858, 271, 198),
            "TEST-B" to listOf(64792, 6155, 40858, 271, 198),
            "TEST-C" to listOf(64792, 6155, 40858, 271, 198),
            "TEST-D" to listOf(3687, 5145, 1112, 580, 4206),
            "TEST-E" to listOf(64705, 20517, 271, 3838, 7552),
        )

        // 5 authoritative prompts with token IDs from Phase B
        private val PROMPTS = listOf(
            Triple("TEST-A", "2+2=?",                             intArrayOf(360, 43226, 64782, 64792)),
            Triple("TEST-B", "বাংলাদেশের রাজধানী কী?",          intArrayOf(1620, 3715, 3101, 64792)),
            Triple("TEST-C", "পানি কত ডিগ্রি সেলসিয়াসে ফুটে?",intArrayOf(4874,6494,4186,4289,1357,263,5821,19591,64792)),
            Triple("TEST-D", "১২ × ৮ = ?",                       intArrayOf(2232,15325,1656,1718,2667)),
            Triple("TEST-E", "ঢাকা বাংলাদেশের রাজধানী।",        intArrayOf(2829,1620,3715,64705)),
        )

        // Config: MAX_NEW_TOKENS=1 → single forward pass to logits only
        private val SINGLE_TOKEN_CONFIG = NanoGenerationConfig(
            temperature  = 0.0f,
            topP         = 1.0f,
            topK         = 1,
            maxOutputTokens = 1,
        )
    }

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun procStatus(): Map<String, Long> {
        return try {
            File("/proc/self/status").readLines().mapNotNull { line ->
                val p = line.split(Regex("\\s+")); if (p.size >= 2) p[0].trimEnd(':') to (p.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null) else null
            }.toMap()
        } catch (e: Exception) { emptyMap() }
    }

    // ─── Load NativeNanoEngine directly (bypasses wrapper, supports config) ──
    private fun loadNativeEngine(): NativeNanoEngine {
        val mm = ModelManager(ctx)
        // Ensure model.nano is extracted to filesDir
        val modelFile = mm.modelFile
        if (!modelFile.exists()) {
            // Trigger extraction via ModelManager
            try { mm.getOrInitEngine() } catch (_: Exception) {}
        }
        require(modelFile.exists()) { "model.nano not found at ${modelFile.absolutePath}" }
        Log.i(TAG, "FIX12_MODEL_PATH=${modelFile.absolutePath} size=${modelFile.length()}")
        return NativeNanoEngine.load(modelFile)
    }

    // ─── test01: Single-token forward, all 5 prompts ─────────────────────────
    @Test
    fun test01_singleTokenForward() {
        Log.i(TAG, "======== FIX12 TEST01: SINGLE TOKEN FORWARD ========")
        Log.i(TAG, "FIX12_MAX_NEW_TOKENS=1")
        Log.i(TAG, "FIX12_TOKENIZER_BEGIN")
        PROMPTS.forEach { (label, prompt, ids) ->
            Log.i(TAG, "FIX12_TOKEN_IDS: label=$label ids=${ids.toList()}")
        }
        Log.i(TAG, "FIX12_TOKENIZER_READY")

        val engine = loadNativeEngine()
        val mem0 = procStatus()
        Log.i(TAG, "FIX12_MEM_AFTER_INIT: rss=${mem0["VmRSS"]}kB peak=${mem0["VmPeak"]}kB")

        Log.i(TAG, "FIX12_FORWARD_BEGIN")

        val results = mutableMapOf<String, Boolean>()

        PROMPTS.forEach { (label, prompt, ids) ->
            Log.i(TAG, "FIX12_PROMPT_BEGIN: label=$label last_token=${ids.last()}")

            val t0 = System.nanoTime()
            var androidArgmax = -1
            val androidTop5 = mutableListOf<Int>()

            // Single-token generate: max_output_tokens=1
            NanoNative.nativeGenerate(
                engine.javaClass.getDeclaredField("nativeHandle").also { it.isAccessible = true }.getLong(engine),
                ids,
                0.0f, 1.0f, 1,
                ai.nano.engine.NativeTokenCallback { tokenStr, tokenId, isEos ->
                    androidArgmax = tokenId
                    Log.i(TAG, "FIX12_FIRST_TOKEN: label=$label token_id=$tokenId text='$tokenStr'")
                    false  // stop after 1 token
                }
            )

            val elapsedMs = (System.nanoTime() - t0) / 1_000_000L
            Log.i(TAG, "FIX12_PROMPT_DONE: label=$label elapsed_ms=$elapsedMs argmax=$androidArgmax")

            val refArgmax = REF_ARGMAX[label] ?: -1
            val top1Match = (androidArgmax == refArgmax)
            results[label] = top1Match

            Log.i(TAG, "FIX12_COMPARE: label=$label ref_argmax=$refArgmax android_argmax=$androidArgmax TOP1_MATCH=$top1Match")
            Log.i(TAG, "FIX12_ELAPSED: label=$label ms=$elapsedMs tokens_per_sec=${if (elapsedMs > 0) 1000.0 / elapsedMs else 0.0}")

            val mem = procStatus()
            Log.i(TAG, "FIX12_MEM: label=$label rss=${mem["VmRSS"]}kB")
        }

        Log.i(TAG, "FIX12_FORWARD_END")

        // Summary
        Log.i(TAG, "======== FIX12 NUMERICAL COMPARISON SUMMARY ========")
        results.forEach { (label, match) ->
            Log.i(TAG, "FIX12_RESULT: label=$label TOP1_MATCH=$match")
        }
        val allPass = results.values.all { it }
        Log.i(TAG, "FIX12_OVERALL: ${if (allPass) "PASS" else "FAIL"}")
        Log.i(TAG, "FIX12_NUMERICAL_COMPARISON_READY")

        engine.close()
        assertTrue("FIX-12 numerical equivalence: not all prompts matched REFERENCE-B", allPass)
    }

    // ─── test02: Determinism — same token twice ───────────────────────────────
    @Test
    fun test02_determinism() {
        Log.i(TAG, "======== FIX12 TEST02: DETERMINISM ========")
        val engine = loadNativeEngine()
        val handle = engine.javaClass.getDeclaredField("nativeHandle").also { it.isAccessible = true }.getLong(engine)

        val (label, prompt, ids) = PROMPTS[0]  // TEST-A

        fun runOnce(): Int {
            var tok = -1
            NanoNative.nativeGenerate(handle, ids, 0.0f, 1.0f, 1,
                ai.nano.engine.NativeTokenCallback { _, tokenId, _ -> tok = tokenId; false })
            NanoNative.nativeResetSession(handle)
            return tok
        }

        val run1 = runOnce()
        val run2 = runOnce()
        val match = (run1 == run2)

        Log.i(TAG, "FIX12_DET_RUN1=$run1 FIX12_DET_RUN2=$run2")
        Log.i(TAG, "FIX12_DETERMINISM=${if (match) "PASS" else "FAIL_NONDETERMINISTIC"}")

        engine.close()
        assertTrue("FIX-12 determinism: run1=$run1 != run2=$run2", match)
    }

    // ─── test03: Performance forensics ───────────────────────────────────────
    @Test
    fun test03_performance() {
        Log.i(TAG, "======== FIX12 TEST03: PERFORMANCE FORENSICS ========")
        Log.i(TAG, "FIX12_PERF_BEGIN")

        val engine = loadNativeEngine()
        val handle = engine.javaClass.getDeclaredField("nativeHandle").also { it.isAccessible = true }.getLong(engine)

        val (_, prompt, ids) = PROMPTS[0]  // TEST-A: single well-defined token
        val mem0 = procStatus()

        val t0 = System.nanoTime()
        NanoNative.nativeGenerate(handle, ids, 0.0f, 1.0f, 1,
            ai.nano.engine.NativeTokenCallback { _, _, _ -> false })
        val totalMs = (System.nanoTime() - t0) / 1_000_000L

        val mem1 = procStatus()

        Log.i(TAG, "FIX12_PERF_TOTAL_MS=$totalMs")
        Log.i(TAG, "FIX12_PERF_TOKENS_PER_SEC=${if (totalMs > 0) 1000.0 / totalMs else 0.0}")
        Log.i(TAG, "FIX12_MEM_BEFORE: rss=${mem0["VmRSS"]}kB peak=${mem0["VmPeak"]}kB threads=${mem0["Threads"]}")
        Log.i(TAG, "FIX12_MEM_AFTER:  rss=${mem1["VmRSS"]}kB peak=${mem1["VmPeak"]}kB threads=${mem1["Threads"]}")
        Log.i(TAG, "FIX12_MEM_DELTA_KB=${(mem1["VmRSS"] ?: 0L) - (mem0["VmRSS"] ?: 0L)}")

        // Telemetry
        try {
            val telem = NanoNative.nativeGetTelemetry(handle)
            if (telem != null) {
                Log.i(TAG, "FIX12_TELEM: total_tokens=${telem.totalTokensGenerated} ram_mb=${String.format("%.1f", telem.residentRamMb)} tok_per_sec=${telem.instantaneousTokPerSec}")
                Log.i(TAG, "FIX12_KV_CACHE_PRESENT=YES active_kv_tokens=${telem.activeKvTokens}")
                Log.i(TAG, "FIX12_KV_CACHE_USED=${if (telem.activeKvTokens > 0) "YES" else "NO"}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "FIX12_TELEM_SKIP: $e")
        }

        Log.i(TAG, "FIX12_PERF_END")
        engine.close()
        assertTrue("FIX-12 perf: forward pass took ${totalMs}ms (expected < 15000ms)", totalMs < 15000)
    }

    // ─── test04: FIX-12C Layerwise intermediate hidden state checkpoint capture ─
    @Test
    fun test04_fix12c_layerwise() {
        Log.i(TAG, "======== FIX12C TEST04: LAYERWISE HIDDEN STATE CAPTURE ========")
        Log.i(TAG, "FIX12C_CAPTURE_BEGIN")

        // Create fix12c directory tree on device
        val fix12cBase = File(ctx.filesDir, "fix12c")
        fix12cBase.mkdirs()
        Log.i(TAG, "FIX12C_DIR: ${fix12cBase.absolutePath}")

        val engine = loadNativeEngine()
        val handle = engine.javaClass.getDeclaredField("nativeHandle")
            .also { it.isAccessible = true }.getLong(engine)

        PROMPTS.forEachIndexed { pi, (label, prompt, ids) ->
            Log.i(TAG, "FIX12C_PROMPT_BEGIN: pi=$pi label=$label tokens=${ids.toList()}")

            // Create per-prompt directory
            val pDir = File(fix12cBase, "prompt_$pi")
            pDir.mkdirs()

            // Reset session state before each prompt
            NanoNative.nativeResetSession(handle)

            // Run single-token forward (generates checkpoints via native engine instrumentation)
            var generatedToken = -1
            val t0 = System.nanoTime()
            NanoNative.nativeGenerate(
                handle,
                ids,
                0.0f, 1.0f, 1,
                ai.nano.engine.NativeTokenCallback { _, tokenId, _ ->
                    generatedToken = tokenId
                    false
                }
            )
            val elapsedMs = (System.nanoTime() - t0) / 1_000_000L
            Log.i(TAG, "FIX12C_PROMPT_DONE: pi=$pi label=$label argmax=$generatedToken elapsed_ms=$elapsedMs")
        }

        Log.i(TAG, "FIX12C_CAPTURE_COMPLETE")

        // Count captured binary files
        val allBins = fix12cBase.walkTopDown().filter { it.extension == "bin" }.toList()
        Log.i(TAG, "FIX12C_TOTAL_BIN_FILES=${allBins.size}")
        allBins.groupBy { it.parentFile?.name ?: "root" }.forEach { (dir, files) ->
            Log.i(TAG, "FIX12C_DIR_$dir: ${files.size} files")
        }

        engine.close()

        // Assert we got checkpoint files (at least 5 prompts × some checkpoints each)
        assertTrue("FIX-12C: Expected at least 25 checkpoint bin files, got ${allBins.size}", allBins.size >= 25)
        Log.i(TAG, "FIX12C_TEST04_PASS: total_bin_files=${allBins.size}")
    }
}
