package bd.sstutor.rag

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import java.io.File
import java.io.FileOutputStream

data class RetrievedFact(
    val chapterTitle: String,
    val text: String,
    val score: Double
)

class KnowledgePackManager(private val context: Context? = null) {
    private var db: SQLiteDatabase? = null
    private val inMemoryFacts = mutableListOf<RetrievedFact>()

    init {
        // Pre-seed core NCTB Class 8 curriculum facts as fallback
        inMemoryFacts.add(
            RetrievedFact(
                "মুনাফা",
                "সরল মুনাফার ক্ষেত্রে মুনাফা = আসল × মুনাফার হার × সময় অর্থাৎ I = Prn। চক্রবৃদ্ধি মূলধন C = P(1 + r)^n এবং চক্রবৃদ্ধি মুনাফা = C - P।",
                1.0
            )
        )
        inMemoryFacts.add(
            RetrievedFact(
                "পীথাগোরাসের উপপাদ্য",
                "সমকোণী ত্রিভুজের অতিভুজের উপর অঙ্কিত বর্গক্ষেত্রের ক্ষেত্রফল অপর দুই বাহুর উপর অঙ্কিত বর্গক্ষেত্রদ্বয়ের ক্ষেত্রফলের সমষ্টির সমান (c² = a² + b²)।",
                1.0
            )
        )
        inMemoryFacts.add(
            RetrievedFact(
                "বৃত্ত",
                "বৃত্তের পরিধি এবং ব্যাসের অনুপাত সর্বদা একটি ধ্রুবক সংখ্যা, যাকে পাই (π) বলে। পরিধি = 2πr এবং ক্ষেত্রফল = πr² যেখানে π ≈ ২২/৭।",
                1.0
            )
        )
        inMemoryFacts.add(
            RetrievedFact(
                "ভগ্নাংশ",
                "ভগ্নাংশের যোগ ও বিয়োগ করতে প্রথমে হরগুলোর ল.সা.গু নির্ণয় করে সমহর বিশিষ্ট ভগ্নাংশে রূপান্তর করতে হয়।",
                1.0
            )
        )
    }

    fun openDatabase(dbFile: File) {
        if (dbFile.exists()) {
            db = SQLiteDatabase.openDatabase(dbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY)
        }
    }

    fun retrieve(query: String, topK: Int = 2): List<RetrievedFact> {
        val results = mutableListOf<RetrievedFact>()

        // 1. Try SQLite FTS5 if open
        if (db != null && db!!.isOpen) {
            try {
                val cursor = db!!.rawQuery(
                    "SELECT c.content_text, ch.title FROM chunks c JOIN chapters ch ON c.chapter_id = ch.chapter_id WHERE c.content_text LIKE ? LIMIT ?",
                    arrayOf("%$query%", topK.toString())
                )
                while (cursor.moveToNext()) {
                    val text = cursor.getString(0)
                    val title = cursor.getString(1)
                    results.add(RetrievedFact(title, text, 1.0))
                }
                cursor.close()
            } catch (e: Exception) {
                // Fallback to in-memory search
            }
        }

        // 2. Fallback to in-memory curriculum matching
        if (results.isEmpty()) {
            val queryWords = query.split(" ").filter { it.length > 1 }
            for (fact in inMemoryFacts) {
                val matchCount = queryWords.count { fact.text.contains(it) || fact.chapterTitle.contains(it) }
                if (matchCount > 0) {
                    results.add(fact)
                }
            }
        }

        return results.take(topK)
    }

    fun compressFacts(facts: List<RetrievedFact>, maxWords: Int = 40): String {
        if (facts.isEmpty()) return ""
        val combined = facts.joinToString(" ") { it.text }
        val words = combined.split("\\s+".toRegex())
        return if (words.size > maxWords) {
            words.take(maxWords).joinToString(" ") + "..."
        } else {
            combined
        }
    }
}
