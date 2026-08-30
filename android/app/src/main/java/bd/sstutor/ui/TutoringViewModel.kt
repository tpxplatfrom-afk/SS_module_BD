package bd.sstutor.ui

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import bd.sstutor.router.TutorDecisionEngine
import bd.sstutor.runtime.DeterministicFallbackRuntime
import bd.sstutor.runtime.MicroRuntime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TutoringViewModel : ViewModel() {
    private val runtime: MicroRuntime = DeterministicFallbackRuntime()
    private val decisionEngine = TutorDecisionEngine(runtime = runtime)

    private val _responseText = MutableLiveData<String>()
    val responseText: LiveData<String> = _responseText

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    init {
        runtime.load("")
        _responseText.value = "স্বাগতম! SS Tutor BD-তে তোমার গণিত ও বিজ্ঞানের যেকোনো প্রশ্ন জিজ্ঞেস করতে পারো।"
    }

    fun processQuery(query: String, mode: String = "EXPLAIN") {
        _isLoading.value = true
        viewModelScope.launch {
            val res = withContext(Dispatchers.Default) {
                decisionEngine.processQuery(query, mode)
            }
            _responseText.value = "প্রশ্ন: $query\n\n${res.text}"
            _isLoading.value = false
        }
    }

    fun handleMemoryPressure(level: Int) {
        if (level >= android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
            runtime.unload()
        }
    }

    override fun onCleared() {
        super.onCleared()
        runtime.unload()
    }
}
