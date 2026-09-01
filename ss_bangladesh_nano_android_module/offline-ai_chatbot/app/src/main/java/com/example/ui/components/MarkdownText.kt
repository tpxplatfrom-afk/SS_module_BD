package com.example.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun MarkdownText(
    content: String,
    modifier: Modifier = Modifier,
    isDarkText: Boolean = false
) {
    val lines = remember(content) { content.split("\n") }
    val clipboardManager = LocalClipboardManager.current

    SelectionContainer {
        Column(modifier = modifier) {
            var inCodeBlock = false
            var codeLanguage = ""
            val codeLines = mutableListOf<String>()

            for (i in lines.indices) {
                val line = lines[i]

                if (line.trimStart().startsWith("```")) {
                    if (inCodeBlock) {
                        // End of code block
                        val codeContent = codeLines.joinToString("\n")
                        CodeBlockCard(
                            code = codeContent,
                            language = codeLanguage,
                            onCopy = {
                                clipboardManager.setText(AnnotatedString(codeContent))
                            }
                        )
                        codeLines.clear()
                        inCodeBlock = false
                    } else {
                        // Start of code block
                        inCodeBlock = true
                        codeLanguage = line.trimStart().removePrefix("```").trim()
                    }
                    continue
                }

                if (inCodeBlock) {
                    codeLines.add(line)
                    continue
                }

                when {
                    line.startsWith("### ") -> {
                        Text(
                            text = line.removePrefix("### "),
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontWeight = FontWeight.Bold,
                                color = if (isDarkText) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.primary
                            ),
                            modifier = Modifier.padding(vertical = 4.dp)
                        )
                    }
                    line.startsWith("#### ") -> {
                        Text(
                            text = line.removePrefix("#### "),
                            style = MaterialTheme.typography.titleSmall.copy(
                                fontWeight = FontWeight.SemiBold,
                                color = if (isDarkText) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.secondary
                            ),
                            modifier = Modifier.padding(vertical = 3.dp)
                        )
                    }
                    line.startsWith("# ") -> {
                        Text(
                            text = line.removePrefix("# "),
                            style = MaterialTheme.typography.headlineSmall.copy(
                                fontWeight = FontWeight.ExtraBold,
                                color = if (isDarkText) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.primary
                            ),
                            modifier = Modifier.padding(vertical = 6.dp)
                        )
                    }
                    line.startsWith("---") -> {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 6.dp)
                                .height(1.dp)
                                .background(MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                        )
                    }
                    line.startsWith("> ") -> {
                        // Blockquote
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                                .clip(RoundedCornerShape(6.dp)),
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Box(
                                    modifier = Modifier
                                        .width(3.dp)
                                        .height(20.dp)
                                        .background(MaterialTheme.colorScheme.primary)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = line.removePrefix("> "),
                                    style = MaterialTheme.typography.bodyMedium.copy(
                                        fontStyle = FontStyle.Italic,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                )
                            }
                        }
                    }
                    line.startsWith("- ") || line.startsWith("* ") -> {
                        // Bullet point
                        val text = line.substring(2)
                        Row(
                            modifier = Modifier.padding(vertical = 2.dp),
                            verticalAlignment = Alignment.Top
                        ) {
                            Text(
                                text = "• ",
                                style = MaterialTheme.typography.bodyMedium.copy(
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            )
                            Text(
                                text = parseInlineMarkdown(text),
                                style = MaterialTheme.typography.bodyMedium.copy(
                                    lineHeight = 22.sp,
                                    color = if (isDarkText) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
                                )
                            )
                        }
                    }
                    line.matches(Regex("^\\d+\\.\\s+.*")) -> {
                        // Numbered list
                        val dotIndex = line.indexOf(". ")
                        val number = line.substring(0, dotIndex + 2)
                        val text = line.substring(dotIndex + 2)
                        Row(
                            modifier = Modifier.padding(vertical = 2.dp),
                            verticalAlignment = Alignment.Top
                        ) {
                            Text(
                                text = number,
                                style = MaterialTheme.typography.bodyMedium.copy(
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            )
                            Text(
                                text = parseInlineMarkdown(text),
                                style = MaterialTheme.typography.bodyMedium.copy(
                                    lineHeight = 22.sp,
                                    color = if (isDarkText) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
                                )
                            )
                        }
                    }
                    line.isBlank() -> {
                        Spacer(modifier = Modifier.height(4.dp))
                    }
                    else -> {
                        Text(
                            text = parseInlineMarkdown(line),
                            style = MaterialTheme.typography.bodyMedium.copy(
                                lineHeight = 22.sp,
                                color = if (isDarkText) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
                            ),
                            modifier = Modifier.padding(vertical = 2.dp)
                        )
                    }
                }
            }

            // Flush remaining code block if unclosed
            if (inCodeBlock && codeLines.isNotEmpty()) {
                val codeContent = codeLines.joinToString("\n")
                CodeBlockCard(
                    code = codeContent,
                    language = codeLanguage,
                    onCopy = {
                        clipboardManager.setText(AnnotatedString(codeContent))
                    }
                )
            }
        }
    }
}

@Composable
fun CodeBlockCard(
    code: String,
    language: String,
    onCopy: () -> Unit
) {
    var copied by remember { mutableStateOf(false) }

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .border(1.dp, Color(0xFF334155), RoundedCornerShape(8.dp))
            .clip(RoundedCornerShape(8.dp)),
        color = Color(0xFF0F172A)
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF1E293B))
                    .padding(horizontal = 12.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = if (language.isNotBlank()) language.uppercase() else "CODE",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = Color(0xFF38BDF8),
                        fontWeight = FontWeight.Bold
                    )
                )
                Spacer(modifier = Modifier.weight(1f))
                IconButton(
                    onClick = {
                        onCopy()
                        copied = true
                    },
                    modifier = Modifier.padding(0.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ContentCopy,
                        contentDescription = "Copy code block",
                        tint = if (copied) Color(0xFF34D399) else Color(0xFF94A3B8)
                    )
                }
            }

            Text(
                text = code,
                style = MaterialTheme.typography.bodySmall.copy(
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFFE2E8F0),
                    fontSize = 12.5.sp,
                    lineHeight = 18.sp
                ),
                modifier = Modifier.padding(12.dp)
            )
        }
    }
}

/**
 * Parses bold (**text** or __text__), bold-italic (***text***), italics (*text* or _text_),
 * strikethrough (~~text~~), and inline code (`code`).
 */
fun parseInlineMarkdown(text: String): AnnotatedString {
    return buildAnnotatedString {
        var i = 0
        val len = text.length

        while (i < len) {
            when {
                // Triple asterisk/underscore: Bold + Italic (***text*** or ___text___)
                (text.startsWith("***", i) || text.startsWith("___", i)) -> {
                    val delim = text.substring(i, i + 3)
                    val endIndex = text.indexOf(delim, i + 3)
                    if (endIndex != -1) {
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontStyle = FontStyle.Italic)) {
                            append(text.substring(i + 3, endIndex))
                        }
                        i = endIndex + 3
                    } else {
                        append(delim)
                        i += 3
                    }
                }

                // Double asterisk/underscore: Bold (**text** or __text__)
                (text.startsWith("**", i) || text.startsWith("__", i)) -> {
                    val delim = text.substring(i, i + 2)
                    val endIndex = text.indexOf(delim, i + 2)
                    if (endIndex != -1) {
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                            append(text.substring(i + 2, endIndex))
                        }
                        i = endIndex + 2
                    } else {
                        append(delim)
                        i += 2
                    }
                }

                // Double tilde: Strikethrough (~~text~~)
                text.startsWith("~~", i) -> {
                    val endIndex = text.indexOf("~~", i + 2)
                    if (endIndex != -1) {
                        withStyle(SpanStyle(textDecoration = androidx.compose.ui.text.style.TextDecoration.LineThrough)) {
                            append(text.substring(i + 2, endIndex))
                        }
                        i = endIndex + 2
                    } else {
                        append("~~")
                        i += 2
                    }
                }

                // Single backtick: Inline Code (`code`)
                text[i] == '`' -> {
                    val endIndex = text.indexOf('`', i + 1)
                    if (endIndex != -1) {
                        withStyle(
                            SpanStyle(
                                fontFamily = FontFamily.Monospace,
                                background = Color(0xFF64748B).copy(alpha = 0.22f),
                                fontWeight = FontWeight.Medium,
                                fontSize = 13.sp,
                                color = Color(0xFF38BDF8)
                            )
                        ) {
                            append(" " + text.substring(i + 1, endIndex) + " ")
                        }
                        i = endIndex + 1
                    } else {
                        append('`')
                        i++
                    }
                }

                // Single asterisk/underscore: Italic (*text* or _text_)
                (text[i] == '*' || text[i] == '_') -> {
                    val delim = text[i]
                    val endIndex = text.indexOf(delim, i + 1)
                    // Ensure it's not preceded/followed by whitespace if desired, or standard markdown matching
                    if (endIndex != -1 && endIndex > i + 1) {
                        withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                            append(text.substring(i + 1, endIndex))
                        }
                        i = endIndex + 1
                    } else {
                        append(delim)
                        i++
                    }
                }

                else -> {
                    append(text[i])
                    i++
                }
            }
        }
    }
}
