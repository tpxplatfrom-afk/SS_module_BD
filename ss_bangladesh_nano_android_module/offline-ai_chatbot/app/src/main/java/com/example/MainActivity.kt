package com.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import android.content.Intent
import androidx.activity.viewModels
import com.example.ui.ChatScreen
import com.example.ui.ChatViewModel
import com.example.ui.theme.MyApplicationTheme

class MainActivity : ComponentActivity() {
    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                ChatScreen(viewModel = viewModel)
            }
        }
        handlePromptIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handlePromptIntent(intent)
    }

    private fun handlePromptIntent(intent: Intent?) {
        val prompt = intent?.getStringExtra("prompt")
        if (!prompt.isNullOrBlank()) {
            viewModel.sendMessage(prompt)
        }
    }
}

