package bd.sstutor.ui

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import bd.sstutor.R
import bd.sstutor.runtime.AndroidMemoryMonitor
import bd.sstutor.runtime.MemoryState

class MainActivity : AppCompatActivity() {
    private lateinit var viewModel: TutoringViewModel
    private lateinit var tvChatHistory: TextView
    private lateinit var etQueryInput: EditText
    private lateinit var tvMemoryStatus: TextView
    private lateinit var spinnerChapter: Spinner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewModel = ViewModelProvider(this)[TutoringViewModel::class.java]

        tvChatHistory = findViewById(R.id.tvChatHistory)
        etQueryInput = findViewById(R.id.etQueryInput)
        tvMemoryStatus = findViewById(R.id.tvMemoryStatus)
        spinnerChapter = findViewById(R.id.spinnerChapter)

        val btnSend = findViewById<Button>(R.id.btnSend)
        val btnHint = findViewById<Button>(R.id.btnHint)
        val btnExplain = findViewById<Button>(R.id.btnExplain)
        val btnSolve = findViewById<Button>(R.id.btnSolve)

        btnSend.setOnClickListener { sendQuery("EXPLAIN") }
        btnHint.setOnClickListener { sendQuery("HINT") }
        btnExplain.setOnClickListener { sendQuery("EXPLAIN") }
        btnSolve.setOnClickListener { sendQuery("SOLVE") }

        viewModel.responseText.observe(this) { text ->
            tvChatHistory.text = text
            updateMemoryDisplay()
        }

        updateMemoryDisplay()
    }

    private fun sendQuery(mode: String) {
        val q = etQueryInput.text.toString().trim()
        if (q.isNotEmpty()) {
            viewModel.processQuery(q, mode)
            etQueryInput.setText("")
        }
    }

    private fun updateMemoryDisplay() {
        val snap = AndroidMemoryMonitor.getMemorySnapshot(this)
        tvMemoryStatus.text = "RAM PSS: ${snap.totalPssMb} MB | State: ${snap.memoryState}"
        if (snap.memoryState == MemoryState.CRITICAL || snap.memoryState == MemoryState.EMERGENCY) {
            tvMemoryStatus.setTextColor(resources.getColor(android.R.color.holo_red_dark, null))
        } else {
            tvMemoryStatus.setTextColor(resources.getColor(android.R.color.holo_green_dark, null))
        }
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        viewModel.handleMemoryPressure(level)
        updateMemoryDisplay()
    }
}
