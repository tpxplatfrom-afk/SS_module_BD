package bd.sstutor.tokenizer

import java.io.File
import org.json.JSONObject

class BengaliTokenizer {
    private val vocab = mutableMapOf<String, Int>()
    private val inverseVocab = mutableMapOf<Int, String>()

    init {
        // Core initial vocabulary seeds
        val coreSymbols = listOf(
            "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
            "[T]", "[F]", "[R]", "[G]", "[H]", "[C]",
            "x²", "√", "π", "≤", "≥", "=", "+", "-", "×", "÷", "%",
            "সরল", "মুনাফা", "আসল", "হার", "সময়", "ভগ্নাংশ", "লব", "হর", "যোগ",
            "পিথাগোরাস", "অতিভুজ", "ভূমি", "লম্ব", "বৃত্ত", "ক্ষেত্রফল", "পরিধি",
            "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "০"
        )
        for ((idx, sym) in coreSymbols.withIndex()) {
            vocab[sym] = idx
            inverseVocab[idx] = sym
        }
    }

    fun loadFromFile(tokenizerJsonFile: File) {
        if (!tokenizerJsonFile.exists()) return
        try {
            val content = tokenizerJsonFile.readText(Charsets.UTF_8)
            val json = JSONObject(content)
            val modelObj = json.optJSONObject("model")
            if (modelObj != null) {
                val vocabObj = modelObj.optJSONObject("vocab")
                if (vocabObj != null) {
                    val keys = vocabObj.keys()
                    while (keys.hasNext()) {
                        val k = keys.next()
                        val v = vocabObj.getInt(k)
                        vocab[k] = v
                        inverseVocab[v] = k
                    }
                }
            }
        } catch (e: Exception) {
            // Retain pre-seeded vocabulary
        }
    }

    fun encode(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        val words = text.split("\\s+".toRegex())
        for (word in words) {
            if (vocab.containsKey(word)) {
                tokens.add(vocab[word]!!)
            } else {
                // Character-level fallback
                for (ch in word) {
                    val s = ch.toString()
                    tokens.add(vocab.getOrDefault(s, 1))
                }
            }
        }
        return tokens
    }

    fun decode(tokens: List<Int>): String {
        val sb = StringBuilder()
        for (tok in tokens) {
            val word = inverseVocab.getOrDefault(tok, "")
            if (word.isNotEmpty()) {
                sb.append(word).append(" ")
            }
        }
        return sb.toString().trim()
    }

    fun getVocabSize(): Int = vocab.size
}
