package com.example.ui.components

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Calculate
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.MenuBook
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.Spellcheck
import androidx.compose.material.icons.filled.Work
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

data class QuickPromptCategory(
    val title: String,
    val icon: ImageVector,
    val samplePrompts: List<String>
)

val CATEGORIES = listOf(
    QuickPromptCategory(
        title = "Math Solver",
        icon = Icons.Default.Calculate,
        samplePrompts = listOf(
            "Solve quadratic formula x^2 - 5x + 6 = 0 step by step",
            "What is 25% of 850?",
            "Calculate 145 * 38",
            "How to find the area and perimeter of a trapezoid?"
        )
    ),
    QuickPromptCategory(
        title = "Essay Writer",
        icon = Icons.Default.MenuBook,
        samplePrompts = listOf(
            "Write an essay on Artificial Intelligence and Society",
            "Write a structured composition about Renewable Energy",
            "Write an essay on the importance of mental health"
        )
    ),
    QuickPromptCategory(
        title = "CV & Letters",
        icon = Icons.Default.Work,
        samplePrompts = listOf(
            "Generate a professional CV for a Senior Android Engineer",
            "Write a cover letter for a Software Engineer position",
            "Write a formal resignation letter with 2 weeks notice",
            "Write a formal leave application for personal reasons"
        )
    ),
    QuickPromptCategory(
        title = "Science",
        icon = Icons.Default.Science,
        samplePrompts = listOf(
            "Explain photosynthesis and its chemical equation",
            "Explain Newton's Three Laws of Motion with formulas",
            "What is the structure of DNA and how does it replicate?",
            "Explain the laws of thermodynamics"
        )
    ),
    QuickPromptCategory(
        title = "Grammar",
        icon = Icons.Default.Spellcheck,
        samplePrompts = listOf(
            "Correct this: i definately didnt recieve teh email yesterday",
            "Improve this sentence for an executive email: we need to do this fast",
            "Check grammar and polish this paragraph"
        )
    ),
    QuickPromptCategory(
        title = "Code Helper",
        icon = Icons.Default.Code,
        samplePrompts = listOf(
            "Write a Kotlin function to group and count list items",
            "How to use Room Database with Flow in Jetpack Compose?",
            "Explain binary search algorithm with time complexity"
        )
    )
)

@Composable
fun QuickPromptSelector(
    selectedCategory: String?,
    onSelectCategory: (QuickPromptCategory) -> Unit,
    onSelectSamplePrompt: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val scrollState = rememberScrollState()

    Row(
        modifier = modifier
            .fillMaxWidth()
            .horizontalScroll(scrollState)
            .padding(horizontal = 12.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        for (category in CATEGORIES) {
            val isSelected = selectedCategory == category.title
            FilterChip(
                selected = isSelected,
                onClick = { onSelectCategory(category) },
                label = {
                    Text(
                        text = category.title,
                        fontSize = 12.sp
                    )
                },
                leadingIcon = {
                    Icon(
                        imageVector = category.icon,
                        contentDescription = category.title,
                        modifier = Modifier.padding(end = 2.dp)
                    )
                },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                    selectedLabelColor = MaterialTheme.colorScheme.onPrimaryContainer
                ),
                modifier = Modifier.height(32.dp)
            )
        }
    }
}
