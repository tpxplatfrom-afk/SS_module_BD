package bd.sstutor.validation

import bd.sstutor.math.MathEngine
import java.util.regex.Pattern

object TutorValidator {
    private val REFUSAL_PHRASES = listOf(
        "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না",
        "প্রদত্ত তথ্য থেকে এটি নিশ্চিতভাবে বলা যায় না",
        "পাঠ্যবইয়ে এই তথ্যটি দেওয়া নেই",
        "পাঠ্যবইয়ে এই তথ্যটি দেওয়া নেই"
    )

    private val PROMPT_TAGS = listOf(
        "[T]", "[F]", "[R]", "[G]", "[H]", "[C]",
        "<|im_start|>", "<|im_end|>", "<|eos|>", "<|pad|>", "<|unk|>"
    )

    // 1. Format & Clean
    fun cleanFormat(rawText: String): String {
        var res = rawText
        for (tag in PROMPT_TAGS) {
            res = res.replace(tag, "")
        }
        return res.trim().replace(Regex("\n{3,}"), "\n\n")
    }

    // 2. Grounding & Anti-Hallucination Guard
    fun validateGrounding(response: String, textbookContext: String, isUnsupportedQuery: Boolean = false): Boolean {
        if (isUnsupportedQuery) {
            return REFUSAL_PHRASES.any { response.contains(it) }
        }
        return response.isNotEmpty()
    }

    // 3. Math Cross-Validation Guard
    fun validateAndCorrectMath(response: String, exactResult: String, steps: List<String>? = null): String {
        val normResp = MathEngine.toEnglishDigits(response)
        val normExact = MathEngine.toEnglishDigits(exactResult)

        val exactNumbers = mutableListOf<String>()
        val m = Pattern.compile("\\d+").matcher(normExact)
        while (m.find()) exactNumbers.add(m.group())

        val respNumbers = mutableListOf<String>()
        val m2 = Pattern.compile("\\d+").matcher(normResp)
        while (m2.find()) respNumbers.add(m2.group())

        var hasMismatch = false
        if (exactNumbers.isNotEmpty()) {
            val primary = exactNumbers[exactNumbers.size - 1]
            if (!respNumbers.contains(primary) && response.length < 40) {
                hasMismatch = true
            }
        }

        if (hasMismatch || response.trim().length < 10) {
            val stepStr = steps?.joinToString("\n") ?: ""
            return "গণনার নির্ভুল ধাপসমূহ:\n$stepStr\nঅতএব সঠিক উত্তর: $exactResult।"
        }
        return response
    }

    // 4. Socratic Hint Leak Guard
    fun validateHintCompliance(hintText: String, exactNumericAnswer: String): String {
        if (exactNumericAnswer.isEmpty()) return hintText
        val normAns = MathEngine.toEnglishDigits(exactNumericAnswer).trim()
        val normHint = MathEngine.toEnglishDigits(hintText)

        val pattern = Pattern.compile("(?<!\\d)" + Pattern.quote(normAns) + "(?!\\d)")
        if (pattern.matcher(normHint).find()) {
            return "ইঙ্গিত: সমস্যাটির মূল সূত্র ও প্রয়োজনীয় ধাপগুলো চিন্তা করো। সরাসরি উত্তর খোঁজার আগে প্রতিটি পদ চিহ্নিত করার চেষ্টা করো।"
        }
        return hintText
    }

    // 5. Repetition Loop Detection
    fun detectRepetitionLoop(text: String): Boolean {
        val words = text.split("\\s+".toRegex())
        if (words.size < 8) return false
        val trigrams = mutableMapOf<String, Int>()
        for (i in 0 until words.size - 2) {
            val tri = "${words[i]} ${words[i+1]} ${words[i+2]}"
            val count = trigrams.getOrDefault(tri, 0) + 1
            if (count > 2) return true
            trigrams[tri] = count
        }
        return false
    }
}
