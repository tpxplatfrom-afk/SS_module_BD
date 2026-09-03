package com.example.data

import android.content.Context
import com.example.thsa.ModelManager
import com.example.thsa.NanoEngine
import com.example.thsa.NanoResponse
import kotlinx.coroutines.flow.Flow
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class ChatRepository(
    private val context: Context,
    private val chatDao: ChatDao
) {
    private val modelManager = ModelManager(context)
    val engine: NanoEngine by lazy {
        modelManager.getOrInitEngine()
    }

    val allSessions: Flow<List<ChatSessionEntity>> = chatDao.getAllSessions()

    fun getMessages(sessionId: String): Flow<List<ChatMessageEntity>> {
        return chatDao.getMessagesForSession(sessionId)
    }

    suspend fun createNewSession(title: String = "New Conversation", category: String = "General"): ChatSessionEntity {
        val session = ChatSessionEntity(
            id = UUID.randomUUID().toString(),
            title = title,
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
            category = category
        )
        chatDao.insertSession(session)
        return session
    }

    suspend fun processUserMessage(
        sessionId: String,
        userText: String,
        category: String = "General"
    ): NanoResponse = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
        val userMsg = ChatMessageEntity(
            sessionId = sessionId,
            role = "user",
            content = userText,
            copyText = userText,
            timestamp = System.currentTimeMillis(),
            category = category
        )
        chatDao.insertMessage(userMsg)

        // Process with Shanto On-Device AI Engine
        val response = engine.ask(userText)

        val assistantMsg = ChatMessageEntity(
            sessionId = sessionId,
            role = "assistant",
            content = response.text,
            copyText = response.copyText,
            timestamp = System.currentTimeMillis(),
            category = category
        )
        chatDao.insertMessage(assistantMsg)

        // Update session title if first message
        val currentSession = chatDao.getSessionById(sessionId)
        if (currentSession != null) {
            val updatedTitle = if (currentSession.title == "New Conversation" || currentSession.title.isBlank()) {
                val clean = userText.take(30).trim()
                if (userText.length > 30) "$clean..." else clean
            } else {
                currentSession.title
            }
            chatDao.updateSession(
                currentSession.copy(
                    title = updatedTitle,
                    updatedAt = System.currentTimeMillis(),
                    category = category
                )
            )
        }

        response
    }

    suspend fun clearSessionMessages(sessionId: String) {
        chatDao.deleteMessagesForSession(sessionId)
        val currentSession = chatDao.getSessionById(sessionId)
        if (currentSession != null) {
            chatDao.updateSession(
                currentSession.copy(
                    title = "New Conversation",
                    updatedAt = System.currentTimeMillis()
                )
            )
        }
    }

    suspend fun deleteSession(sessionId: String) {
        chatDao.deleteSession(sessionId)
    }

    suspend fun clearAllSessions() {
        chatDao.deleteAllSessions()
    }

    /**
     * Exports response as a TXT file in cache directory and returns the File.
     */
    fun exportResponseAsTxt(response: NanoResponse, prefix: String = "shanto_export"): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(context.cacheDir, "${prefix}_${timeStamp}.txt")
        response.saveAsTxt(file)
        return file
    }

    /**
     * Exports response as a MD file in cache directory and returns the File.
     */
    fun exportResponseAsMd(response: NanoResponse, prefix: String = "shanto_export"): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(context.cacheDir, "${prefix}_${timeStamp}.md")
        response.saveAsMd(file)
        return file
    }

    /**
     * Exports full conversation as TXT.
     */
    fun exportConversationAsTxt(messages: List<ChatMessageEntity>, title: String): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(context.cacheDir, "chat_${timeStamp}.txt")
        val content = buildString {
            appendLine("=== Shanto Offline Chat Export ===")
            appendLine("Title: $title")
            appendLine("Export Date: ${Date()}")
            appendLine("========================================")
            appendLine()
            for (msg in messages) {
                val roleName = if (msg.role == "user") "YOU" else "Shanto AI"
                val time = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date(msg.timestamp))
                appendLine("[$time] $roleName:")
                appendLine(msg.copyText)
                appendLine()
            }
        }
        file.parentFile?.mkdirs()
        file.writeText(content)
        return file
    }

    /**
     * Exports full conversation as MD.
     */
    fun exportConversationAsMd(messages: List<ChatMessageEntity>, title: String): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(context.cacheDir, "chat_${timeStamp}.md")
        val content = buildString {
            appendLine("# Shanto Offline Chat: $title")
            appendLine("> Exported on-device at ${Date()} with zero network dependency.")
            appendLine()
            for (msg in messages) {
                val roleName = if (msg.role == "user") "👤 **User**" else "🤖 **Shanto Offline AI**"
                appendLine("### $roleName")
                appendLine(msg.content)
                appendLine()
                appendLine("---")
                appendLine()
            }
        }
        file.parentFile?.mkdirs()
        file.writeText(content)
        return file
    }
}
