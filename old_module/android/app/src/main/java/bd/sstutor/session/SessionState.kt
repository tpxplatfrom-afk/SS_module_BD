package bd.sstutor.session

data class SessionState(
    val sessionId: String = "default_session",
    var activeClass: Int = 8,
    var activeSubject: String = "Mathematics",
    var activeChapter: String = "মুনাফা",
    var turnCount: Int = 0,
    var lastQuestionSummary: String = "",
    var lastAnswerSummary: String = ""
) {
    fun updateTurn(question: String, answer: String) {
        turnCount++
        // Enforce strict character limits to maintain O(1) constant memory
        lastQuestionSummary = if (question.length > 60) question.take(60) else question
        lastAnswerSummary = if (answer.length > 60) answer.take(60) else answer
    }

    fun reset() {
        turnCount = 0
        lastQuestionSummary = ""
        lastAnswerSummary = ""
    }
}
