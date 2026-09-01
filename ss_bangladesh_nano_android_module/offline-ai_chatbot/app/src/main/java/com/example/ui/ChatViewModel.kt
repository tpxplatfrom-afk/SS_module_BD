package com.example.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.AppDatabase
import com.example.data.ChatMessageEntity
import com.example.data.ChatRepository
import com.example.data.ChatSessionEntity
import com.example.thsa.ModelDownloadState
import com.example.thsa.ModelManager
import com.example.thsa.NanoEngine
import com.example.thsa.NanoResponse
import com.example.ui.components.QuickPromptCategory
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

data class ChatUiState(
    val currentSessionId: String? = null,
    val currentSessionTitle: String = "New Conversation",
    val messages: List<ChatMessageEntity> = emptyList(),
    val isGenerating: Boolean = false,
    val inputText: String = "",
    val selectedCategory: String? = null,
    val downloadState: ModelDownloadState = ModelDownloadState.Idle,
    val showModelStatusSheet: Boolean = false,
    val toastMessage: String? = null
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AppDatabase.getInstance(application)
    private val repository = ChatRepository(application, db.chatDao())
    private val modelManager = ModelManager(application)

    val allSessions: StateFlow<List<ChatSessionEntity>> = repository.allSessions
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    val engine: NanoEngine get() = repository.engine

    private var messageCollectionJob: Job? = null

    init {
        viewModelScope.launch {
            modelManager.downloadState.collect { dState ->
                _uiState.value = _uiState.value.copy(downloadState = dState)
            }
        }

        // Initialize first chat session if needed
        viewModelScope.launch {
            allSessions.collect { sessions ->
                if (_uiState.value.currentSessionId == null) {
                    if (sessions.isNotEmpty()) {
                        switchSession(sessions.first().id, sessions.first().title)
                    } else {
                        val newSession = repository.createNewSession()
                        switchSession(newSession.id, newSession.title)
                    }
                }
            }
        }
    }

    fun onInputTextChanged(newText: String) {
        _uiState.value = _uiState.value.copy(inputText = newText)
    }

    fun selectCategory(category: QuickPromptCategory) {
        val current = _uiState.value.selectedCategory
        val newCategory = if (current == category.title) null else category.title
        _uiState.value = _uiState.value.copy(selectedCategory = newCategory)
    }

    fun switchSession(sessionId: String, title: String = "Conversation") {
        _uiState.value = _uiState.value.copy(
            currentSessionId = sessionId,
            currentSessionTitle = title,
            inputText = ""
        )

        messageCollectionJob?.cancel()
        messageCollectionJob = viewModelScope.launch {
            repository.getMessages(sessionId).collect { msgs ->
                _uiState.value = _uiState.value.copy(messages = msgs)
            }
        }
    }

    fun createNewSession(category: String = _uiState.value.selectedCategory ?: "General") {
        viewModelScope.launch {
            val newSession = repository.createNewSession(category = category)
            switchSession(newSession.id, newSession.title)
        }
    }

    fun deleteSession(sessionId: String) {
        viewModelScope.launch {
            repository.deleteSession(sessionId)
            if (_uiState.value.currentSessionId == sessionId) {
                val remaining = allSessions.value.filter { it.id != sessionId }
                if (remaining.isNotEmpty()) {
                    switchSession(remaining.first().id, remaining.first().title)
                } else {
                    val fresh = repository.createNewSession()
                    switchSession(fresh.id, fresh.title)
                }
            }
        }
    }

    fun clearCurrentChat() {
        val sessionId = _uiState.value.currentSessionId ?: return
        viewModelScope.launch {
            repository.clearSessionMessages(sessionId)
            _uiState.value = _uiState.value.copy(
                currentSessionTitle = "New Conversation",
                toastMessage = "Chat history cleared"
            )
        }
    }

    fun sendMessage(textOverride: String? = null) {
        val query = textOverride ?: _uiState.value.inputText
        val cleanQuery = query.trim()
        if (cleanQuery.isBlank() || _uiState.value.isGenerating) return

        val sessionId = _uiState.value.currentSessionId ?: return
        val category = _uiState.value.selectedCategory ?: "General"

        _uiState.value = _uiState.value.copy(
            inputText = "",
            isGenerating = true
        )

        viewModelScope.launch {
            try {
                repository.processUserMessage(
                    sessionId = sessionId,
                    userText = cleanQuery,
                    category = category
                )
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                _uiState.value = _uiState.value.copy(isGenerating = false)
            }
        }
    }

    fun exportMessageAsTxt(message: ChatMessageEntity): File {
        val dummyResponse = NanoResponse(message.content, message.copyText)
        val file = repository.exportResponseAsTxt(dummyResponse, "thsa_msg")
        _uiState.value = _uiState.value.copy(toastMessage = "Exported message to ${file.name}")
        return file
    }

    fun exportMessageAsMd(message: ChatMessageEntity): File {
        val dummyResponse = NanoResponse(message.content, message.copyText)
        val file = repository.exportResponseAsMd(dummyResponse, "thsa_msg")
        _uiState.value = _uiState.value.copy(toastMessage = "Exported Markdown to ${file.name}")
        return file
    }

    fun exportCurrentChatAsTxt(): File? {
        val msgs = _uiState.value.messages
        val title = _uiState.value.currentSessionTitle
        if (msgs.isEmpty()) {
            _uiState.value = _uiState.value.copy(toastMessage = "No messages to export.")
            return null
        }
        val file = repository.exportConversationAsTxt(msgs, title)
        _uiState.value = _uiState.value.copy(toastMessage = "Exported conversation to ${file.name}")
        return file
    }

    fun exportCurrentChatAsMd(): File? {
        val msgs = _uiState.value.messages
        val title = _uiState.value.currentSessionTitle
        if (msgs.isEmpty()) {
            _uiState.value = _uiState.value.copy(toastMessage = "No messages to export.")
            return null
        }
        val file = repository.exportConversationAsMd(msgs, title)
        _uiState.value = _uiState.value.copy(toastMessage = "Exported Markdown to ${file.name}")
        return file
    }

    fun downloadOrSyncModel() {
        viewModelScope.launch {
            modelManager.downloadModel()
        }
    }

    fun setModelStatusSheetVisible(visible: Boolean) {
        _uiState.value = _uiState.value.copy(showModelStatusSheet = visible)
    }

    fun clearToast() {
        _uiState.value = _uiState.value.copy(toastMessage = null)
    }
}
