package bd.sstutor.router

import bd.sstutor.math.MathEngine
import bd.sstutor.rag.KnowledgePackManager
import bd.sstutor.runtime.MicroRuntime
import bd.sstutor.session.SessionState
import bd.sstutor.validation.TutorValidator

data class TutorResponse(
    val text: String,
    val isMath: Boolean,
    val isGrounded: Boolean,
    val latencyMs: Long,
    val memoryUsageMb: Double,
    val mode: String
)

class TutorDecisionEngine(
    private val mathEngine: MathEngine = MathEngine,
    private val ragManager: KnowledgePackManager = KnowledgePackManager(),
    private val runtime: MicroRuntime,
    private val session: SessionState = SessionState()
) {
    fun processQuery(
        query: String,
        mode: String = "EXPLAIN",
        onToken: ((String) -> Unit)? = null
    ): TutorResponse {
        val t0 = System.currentTimeMillis()

        // 1. Deterministic Math Intent Detection
        val mathIntent = mathEngine.parseMathIntent(query)

        // 2. Offline RAG Retrieval
        val facts = ragManager.retrieve(query, topK = 2)
        val compressedFacts = ragManager.compressFacts(facts, maxWords = 40)

        // 3. Response Generation
        val rawResponse = if (mathIntent.isMath) {
            if (mode == "HINT") {
                "ইঙ্গিত: সমস্যাটির মূল সূত্র ও প্রয়োজনীয় ধাপগুলো চিন্তা করো। সরাসরি উত্তর খোঁজার আগে সূত্রের প্রতিটি চলক আলাদা করো।"
            } else {
                "গণনার ধাপসমূহ:\n${mathIntent.explanation}\nঅতএব সঠিক উত্তর: ${mathIntent.rawResult}।"
            }
        } else if (compressedFacts.isNotEmpty()) {
            "পাঠ্যপুস্তকের তথ্য অনুযায়ী:\n$compressedFacts"
        } else {
            // Neural MicroRuntime Generation
            val prompt = "[T] $mode\n[F] $compressedFacts\nপ্রশ্ন: $query\n[G] সহজ বাংলায় উত্তর দাও।\nউত্তর: "
            val genResult = runtime.generate(
                systemPrompt = "তুমি একজন বিশেষজ্ঞ বাংলা NCTB শিক্ষক।",
                userPrompt = prompt,
                onToken = onToken
            )
            genResult.text
        }

        // 4. Multi-Guard Validation Layer
        var cleaned = TutorValidator.cleanFormat(rawResponse)

        if (mathIntent.isMath) {
            cleaned = TutorValidator.validateAndCorrectMath(cleaned, mathIntent.rawResult)
        }

        if (mode == "HINT" && mathIntent.isMath) {
            cleaned = TutorValidator.validateHintCompliance(cleaned, mathIntent.rawResult)
        }

        onToken?.invoke(cleaned)
        session.updateTurn(query, cleaned)

        val latency = System.currentTimeMillis() - t0
        val memUsage = runtime.memoryUsageMb()

        return TutorResponse(
            text = cleaned,
            isMath = mathIntent.isMath,
            isGrounded = compressedFacts.isNotEmpty(),
            latencyMs = latency,
            memoryUsageMb = memUsage,
            mode = mode
        )
    }
}
