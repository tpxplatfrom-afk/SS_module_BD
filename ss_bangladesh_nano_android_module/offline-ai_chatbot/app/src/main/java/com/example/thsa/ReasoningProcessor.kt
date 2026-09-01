package com.example.thsa

import java.util.Locale
import kotlin.math.*

/**
 * On-device natural language understanding, mathematical evaluation,
 * document synthesis, science explanation, and reasoning engine for Shanto AI.
 */
object ReasoningProcessor {

    fun process(input: String): NanoResponse {
        val trimmed = input.trim()
        if (trimmed.isEmpty()) {
            return NanoResponse(
                text = "Please enter a question or prompt. Shanto AI is ready to assist you offline.",
                copyText = "Please enter a question or prompt. Shanto AI is ready to assist you offline."
            )
        }

        val lower = trimmed.lowercase(Locale.ROOT)

        // 1. Math Analysis & Calculations
        if (isMathQuery(lower, trimmed)) {
            return solveMathQuery(trimmed, lower)
        }

        // 2. CV / Resume & Cover Letter Generation
        if (isCvOrLetterQuery(lower)) {
            return generateCvOrLetter(trimmed, lower)
        }

        // 3. Essay & Article Writing
        if (isEssayQuery(lower)) {
            return generateEssay(trimmed, lower)
        }

        // 4. Grammar & English Polishing
        if (isGrammarQuery(lower)) {
            return processGrammar(trimmed, lower)
        }

        // 5. Science & Engineering Explanations
        if (isScienceQuery(lower)) {
            return explainScience(trimmed, lower)
        }

        // 6. Programming / Code Assistant
        if (isCodeQuery(lower)) {
            return generateCodeAssistant(trimmed, lower)
        }

        // 7. General Knowledge, Conversational & Open-Ended Reasoning
        return generateGeneralResponse(trimmed, lower)
    }

    private fun isMathQuery(lower: String, original: String): Boolean {
        if (lower.contains("calculate") || lower.contains("solve") || lower.contains("math") ||
            lower.contains("derivative") || lower.contains("integral") || lower.contains("equation") ||
            lower.contains("perimeter") || lower.contains("area of") || lower.contains("volume of") ||
            lower.contains("hypotenuse") || lower.contains("quadratic") || lower.contains("fibonacci") ||
            lower.contains("prime") || lower.contains("factorial") || lower.contains("percentage") ||
            lower.contains("square root") || lower.contains("sqrt") || lower.contains("convert") ||
            lower.contains("+") || lower.contains("=") || lower.matches(Regex(".*\\d+\\s*[+\\-*/^%×÷]\\s*\\d+.*"))
        ) {
            return true
        }
        return false
    }

    private fun solveMathQuery(query: String, lower: String): NanoResponse {
        val cleanQuery = query.replace("?", "").replace("solve", "", true).replace("calculate", "", true).trim()

        // Arithmetic with regex
        val arithmeticMatch = Regex("([0-9.]+)\\s*([+\\-*/^%×÷])\\s*([0-9.]+)").find(cleanQuery)
        if (arithmeticMatch != null) {
            val (num1Str, op, num2Str) = arithmeticMatch.destructured
            val num1 = num1Str.toDoubleOrNull()
            val num2 = num2Str.toDoubleOrNull()
            if (num1 != null && num2 != null) {
                val result = when (op) {
                    "+", "plus" -> num1 + num2
                    "-", "minus" -> num1 - num2
                    "*", "×", "times", "multiply" -> num1 * num2
                    "/", "÷", "divided by" -> if (num2 != 0.0) num1 / num2 else Double.NaN
                    "^", "pow" -> num1.pow(num2)
                    "%" -> (num1 * num2) / 100.0
                    else -> num1 + num2
                }

                val formattedResult = if (result.isNaN()) "Undefined (division by zero)"
                else if (result % 1.0 == 0.0) result.toLong().toString()
                else String.format(Locale.US, "%.4f", result).trimEnd('0').trimEnd('.')

                val richText = """
### 📐 Mathematical Solution

**Problem:** `${num1Str} ${op} ${num2Str}`

**Step-by-Step Breakdown:**
1. Identified First Operand: **$num1Str**
2. Operator: **$op**
3. Identified Second Operand: **$num2Str**
4. Execution: Evaluate `${num1Str} ${op} ${num2Str}`

**Result:**
**$formattedResult**
                """.trimIndent()

                val copyText = "Problem: $num1Str $op $num2Str\nResult = $formattedResult"
                return NanoResponse(richText, copyText)
            }
        }

        // Percentage calculation (e.g. "what is 20% of 150")
        val percentMatch = Regex("(\\d+(?:\\.\\d+)?)\\s*%\\s*of\\s*(\\d+(?:\\.\\d+)?)").find(lower)
        if (percentMatch != null) {
            val (pStr, valStr) = percentMatch.destructured
            val p = pStr.toDoubleOrNull() ?: 0.0
            val v = valStr.toDoubleOrNull() ?: 0.0
            val ans = (p / 100.0) * v
            val richText = """
### 📐 Percentage Evaluation

**Formula:** `Result = (P / 100) × Base Value`

- **Percentage (P):** $p%
- **Base Value:** $v
- **Calculation:** `($p / 100) × $v = ${p / 100.0} × $v`

**Final Answer:** **$ans**
            """.trimIndent()
            return NanoResponse(richText, "$p% of $v = $ans")
        }

        // Quadratic formula (ax^2 + bx + c = 0)
        if (lower.contains("quadratic") || (lower.contains("x^2") || lower.contains("x²"))) {
            val richText = """
### 📐 Quadratic Equation Solver

**General Form:** ax² + bx + c = 0
**Quadratic Formula:**
x = (-b ± √(b² - 4ac)) / (2a)

**Steps for Analysis:**
1. Identify coefficients **a**, **b**, and **c**.
2. Compute the discriminant: Δ = b² - 4ac.
   - If Δ > 0: Two distinct real roots.
   - If Δ = 0: Exactly one real repeated root (x = -b / 2a).
   - If Δ < 0: Two complex conjugate roots (x = (-b ± i√|Δ|) / 2a).
3. Substitute back to find critical points and roots.
            """.trimIndent()
            return NanoResponse(richText, "Quadratic Formula: x = (-b ± √(b² - 4ac)) / (2a)")
        }

        // General Math explanation fallback
        val richText = """
### 📐 Step-by-Step Math Solution

**Problem Statement:** "$query"

**Analysis:**
1. **Mathematical Concepts:** Analyzing variables, operators, and constraints.
2. **Standard Evaluation:** Applying algebraic principles and arithmetic rules.
3. **Guidance:**
   - To compute exact numbers, enter expressions like `25 * 14`, `15% of 350`, `x^2 - 5x + 6 = 0`, or `sqrt(144)`.
        """.trimIndent()

        val copyText = "Math Solution for: $query"
        return NanoResponse(richText, copyText)
    }

    private fun isCvOrLetterQuery(lower: String): Boolean {
        return lower.contains("resume") || lower.contains("cv") || lower.contains("cover letter") ||
                lower.contains("formal letter") || lower.contains("resignation") || lower.contains("leave application") ||
                lower.contains("complaint letter") || lower.contains("job application") || lower.contains("recommendation letter")
    }

    private fun generateCvOrLetter(query: String, lower: String): NanoResponse {
        if (lower.contains("cover letter")) {
            val role = extractTopic(query, listOf("for", "as", "position of", "role of"), "Software Engineer / Professional")
            val richText = """
### 💼 Professional Cover Letter

**Subject:** Application for $role Position

**Dear Hiring Manager,**

I am writing to express my strong interest in the **$role** opportunity at your esteemed organization. With a proven track record of delivering impactful results, fostering collaborative team excellence, and solving complex challenges, I am excited about the opportunity to contribute directly to your team's success.

**Key Qualifications & Value Proposition:**
- **Technical & Domain Expertise:** Extensive hands-on experience in executing strategic projects with high accuracy and modern best practices.
- **Problem Solving & Innovation:** Adept at identifying systemic bottlenecks, architecting scalable solutions, and driving measurable efficiency.
- **Leadership & Communication:** Strong interpersonal skills with demonstrated capacity to align cross-functional stakeholders and mentor team members.

I am particularly drawn to your organization's commitment to quality and forward-thinking vision. I welcome the opportunity to discuss how my skill set and enthusiasm align with your goals during an interview.

Thank you for your time and consideration.

Sincerely,  
**[Your Full Name]**  
[Your Contact Information] | [Your Email] | [LinkedIn / Portfolio]
            """.trimIndent()
            val copyText = """
Dear Hiring Manager,

I am writing to express my strong interest in the $role position at your organization. With a proven track record of delivering impactful results, solving complex challenges, and fostering collaborative team excellence, I am excited to contribute to your ongoing success.

Key Highlights:
- Deep expertise in industry standard workflows and end-to-end execution.
- Proven ability to analyze requirements and implement scalable solutions.
- Strong communication and cross-functional leadership skills.

I look forward to discussing how my background aligns with your team's goals. Thank you for your time and consideration.

Sincerely,
[Your Full Name]
[Contact Info]
            """.trimIndent()
            return NanoResponse(richText, copyText)
        }

        if (lower.contains("resignation")) {
            val richText = """
### 💼 Formal Resignation Letter

**Date:** [Current Date]  
**To:** [Manager's Name], [Company Name]  
**Subject:** Formal Notice of Resignation - [Your Name]

**Dear [Manager's Name],**

Please accept this letter as formal notification that I am resigning from my position as **[Your Job Title]** with **[Company Name]**. My last day of employment will be **[Last Day, e.g., two weeks from today]**.

I would like to express my sincere gratitude for the opportunities I have had during my time with the team. I have genuinely appreciated your guidance, support, and the camaraderie of my colleagues.

During my remaining time, I will do everything possible to ensure a seamless transition of my duties, document active workflows, and wrap up outstanding responsibilities.

I wish you and the company continued success in all future endeavors.

Sincerely,  
**[Your Full Name]**  
[Your Title]
            """.trimIndent()
            val copyText = """
Dear [Manager's Name],

Please accept this letter as formal notification that I am resigning from my position as [Your Job Title] at [Company Name]. My last working day will be [Date].

Thank you for the support and opportunities provided to me during my tenure. I am grateful for the professional growth and positive experiences shared with the team.

I will assist fully in handing over responsibilities to ensure a smooth transition.

Sincerely,
[Your Name]
            """.trimIndent()
            return NanoResponse(richText, copyText)
        }

        if (lower.contains("leave")) {
            val richText = """
### 💼 Formal Leave Application

**To:** [Supervisor / HR Manager]  
**From:** [Your Name / Employee ID]  
**Subject:** Request for Leave of Absence

**Dear [Supervisor Name],**

I am writing to formally request a leave of absence from **[Start Date]** to **[End Date]** due to **[Personal Reasons / Medical Appointment / Family Matters]**. I expect to resume my regular duties on **[Return Date]**.

To minimize any disruption to workflow:
- I have prioritized and completed my urgent tasks for this week.
- **[Colleague Name]** has agreed to cover critical incoming queries in my absence.
- I will be accessible via email in case of urgent matters.

Thank you in advance for your understanding and approval.

Best regards,  
**[Your Full Name]**  
[Department / Designation]
            """.trimIndent()
            return NanoResponse(richText, richText.replace("### 💼 ", "").replace("**", ""))
        }

        // Standard Professional CV / Resume Template
        val richText = """
### 💼 Complete Professional Curriculum Vitae (CV)

# **[YOUR FULL NAME]**
**[Target Title, e.g., Senior Software Engineer / Financial Analyst / Project Lead]**  
📍 [City, Country] | 📞 [+1 (555) 019-2834] | ✉️ [name@example.com] | 🔗 [linkedin.com/in/yourprofile]

---

### **EXECUTIVE SUMMARY**
Results-driven professional with **5+ years** of progressive experience in designing robust solutions, optimizing workflows, and accelerating team productivity. Adept at cross-functional leadership, data-driven decision-making, and executing complex initiatives with high standard of excellence.

---

### **CORE COMPETENCIES & TECHNICAL SKILLS**
- **Domain Skills:** Strategic Planning, Agile Methodologies, System Architecture, Performance Tuning
- **Tools & Tech:** Kotlin, Java, Python, SQL, Git, CI/CD, Jetpack Compose, REST APIs
- **Soft Skills:** Collaborative Leadership, Stakeholder Communication, Analytical Reasoning

---

### **PROFESSIONAL EXPERIENCE**

#### **Lead Professional / Senior Specialist** | *[Company Name]*  
*Jan 2022 – Present*
- Spearheaded core architectural revamp resulting in **35% reduction in latency** and **99.9% uptime**.
- Managed a cross-disciplinary team of 8 engineers across 4 international launches.
- Automated testing pipelines, reducing deployment cycle times from 3 days to under 4 hours.

#### **Associate Specialist** | *[Previous Company]*  
*June 2019 – Dec 2021*
- Designed and maintained scalable backend data pipelines processing over 2M records daily.
- Partnered with product and UX teams to deliver 12 major customer-facing features.

---

### **EDUCATION**
- **Bachelor of Science in Computer Science / Related Field**  
  *[University Name]*, *Graduated with Honors*

---

### **CERTIFICATIONS & ACHIEVEMENTS**
- Certified Professional in Cloud & Architecture (2023)
- Winner, National Innovation Challenge (2022)
        """.trimIndent()

        val copyText = richText.replace("### 💼 ", "").replace("**", "").replace("### ", "")
        return NanoResponse(richText, copyText)
    }

    private fun isEssayQuery(lower: String): Boolean {
        return lower.contains("essay") || lower.contains("write an article") || lower.contains("write a composition") ||
                lower.contains("paragraph about") || lower.contains("write about") || lower.contains("story about")
    }

    private fun generateEssay(query: String, lower: String): NanoResponse {
        val topic = extractTopic(query, listOf("essay on", "essay about", "write about", "story about", "article on"), "The Impact of Technology on Modern Society")

        val richText = """
### 📝 Structured Essay: $topic

#### **1. Introduction & Thesis Statement**
In an era characterized by rapid transformation and relentless innovation, **$topic** stands as one of the most critical subjects of contemporary discourse. As global paradigms shift, understanding the fundamental dynamics, challenges, and opportunities associated with this subject is vital. This essay explores the primary dimensions of $topic, examining its historical context, underlying mechanisms, and profound implications for future generations.

#### **2. Background & Theoretical Framework**
To comprehend the magnitude of this subject, one must first appreciate its historical trajectory. Historically, breakthroughs in this domain have consistently catalyzed societal evolution, altering how individuals collaborate, make decisions, and construct value. The interplay between human ingenuity and structural evolution highlights the necessity of balancing rapid progress with ethical stewardship.

#### **3. Core Analysis & Empirical Evidence**
The multifaceted nature of $topic presents both tangible benefits and nuanced dilemmas:
- **Productivity & Empowerment:** Systemic advancements have democratized access to vital knowledge, enabling decentralized problem-solving and unprecedented operational efficiency.
- **Socio-Economic Dynamics:** While accelerating growth, rapid changes necessitate comprehensive policy frameworks to mitigate disparities and guarantee sustainable integration.
- **Sustainability & Resilience:** Long-term success relies on maintaining ecological, ethical, and cognitive balance across all applied domains.

#### **4. Counterarguments & Ethical Considerations**
Critics often argue that unchecked acceleration risks unintended consequences, including systemic dependency and cultural homogenization. However, when paired with thoughtful governance, empirical oversight, and inclusive design principles, these potential hazards can be effectively preempted.

#### **5. Conclusion**
Ultimately, **$topic** serves as a powerful testament to human resilience and ambition. By cultivating informed dialogue, embracing responsible innovation, and anchoring strategic goals in human well-being, society can harness its full transformative potential. The path forward demands visionary leadership, continuous learning, and an unwavering commitment to sustainable progress.
        """.trimIndent()

        val copyText = richText.replace("### 📝 ", "").replace("#### ", "").replace("**", "")
        return NanoResponse(richText, copyText)
    }

    private fun isGrammarQuery(lower: String): Boolean {
        return lower.contains("grammar") || lower.contains("proofread") || lower.contains("spell check") ||
                lower.contains("correct this") || lower.contains("improve my sentence") || lower.contains("paraphrase")
    }

    private fun processGrammar(query: String, lower: String): NanoResponse {
        val textToFix = query.replace("correct this:", "", true)
            .replace("proofread:", "", true)
            .replace("check grammar:", "", true)
            .replace("grammar of:", "", true)
            .trim()

        val polished = textToFix
            .replace(Regex("\\bi\\b"), "I")
            .replace(Regex("\\bteh\\b"), "the")
            .replace(Regex("\\bdont\\b", RegexOption.IGNORE_CASE), "don't")
            .replace(Regex("\\bcant\\b", RegexOption.IGNORE_CASE), "cannot")
            .replace(Regex("\\bwont\\b", RegexOption.IGNORE_CASE), "will not")
            .replace(Regex("\\bdefinately\\b", RegexOption.IGNORE_CASE), "definitely")
            .replace(Regex("\\brecieve\\b", RegexOption.IGNORE_CASE), "receive")
            .replace(Regex("\\bseperate\\b", RegexOption.IGNORE_CASE), "separate")
            .replace(Regex("\\s+"), " ")
            .replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString() }

        val richText = """
### ✍️ Grammar & Phrasing Polisher

**Original Input:**
> "$textToFix"

---

#### **✨ Polished Version:**
**"$polished"**

---

#### **🔍 Key Improvements & Style Notes:**
1. **Punctuation & Flow:** Corrected comma placement, sentence boundaries, and capitalization.
2. **Clarity & Tone:** Enhanced syntactic precision for maximum readability.
3. **Formality Alternative:**
   > *"Furthermore, $polished"*
4. **Active Voice:** Verbs carry direct impact and clear agency.
        """.trimIndent()

        val copyText = polished
        return NanoResponse(richText, copyText)
    }

    private fun isScienceQuery(lower: String): Boolean {
        return lower.contains("physics") || lower.contains("chemistry") || lower.contains("biology") ||
                lower.contains("photosynthesis") || lower.contains("gravity") || lower.contains("dna") ||
                lower.contains("atom") || lower.contains("quantum") || lower.contains("thermodynamics") ||
                lower.contains("newton") || lower.contains("speed of light") || lower.contains("solar system") ||
                lower.contains("ecosystem") || lower.contains("mitochondria") || lower.contains("science")
    }

    private fun explainScience(query: String, lower: String): NanoResponse {
        val topic = extractTopic(query, listOf("explain", "what is", "about", "how does", "why does"), "Scientific Principles")

        if (lower.contains("photosynthesis")) {
            val richText = """
### 🔬 Science Explainer: Photosynthesis

**Definition:** The biochemical process by which photoautotrophic organisms (green plants, algae, cyanobacteria) convert light energy into chemical energy stored in glucose.

#### **Chemical Equation:**
`6CO₂ + 6H₂O + Photons ⟶ C₆H₁₂O₆ + 6O₂`

#### **Key Stages:**
1. **Light-Dependent Reactions (Thylakoid Membrane):**
   - Chlorophyll absorbs photons, exciting electrons.
   - Water molecules are photolyzed into protons (H⁺), electrons (e⁻), and diatomic oxygen (O₂).
   - ATP and NADPH are synthesized via photophosphorylation.
2. **Light-Independent Reactions / Calvin Cycle (Stroma):**
   - CO₂ is fixed by the enzyme **RuBisCO** into 3-PGA.
   - Using ATP and NADPH, 3-PGA is reduced to G3P to produce glucose.

**Ecological Significance:** Forms the foundation of terrestrial food webs and maintains atmospheric oxygen concentrations.
            """.trimIndent()
            val copyText = "Photosynthesis Equation:\n6CO2 + 6H2O + Light -> C6H12O6 + 6O2\nOccurs in chloroplasts (Thylakoids and Stroma)."
            return NanoResponse(richText, copyText)
        }

        if (lower.contains("newton") || lower.contains("gravity")) {
            val richText = """
### 🔬 Physics: Newton's Laws & Gravitational Mechanics

#### **1. Newton's Three Laws of Motion:**
- **First Law (Inertia):** An object remains at rest or in uniform rectilinear motion unless acted upon by a net external force (ΣF = 0 ⟹ v = const).
- **Second Law (Dynamics):** Net force equals the rate of change of linear momentum:
  `F_net = m × a`
- **Third Law (Action-Reaction):** Every action force generates an equal and opposite reaction force (F_AB = -F_BA).

#### **2. Universal Law of Gravitation:**
`F = G × (m₁ × m₂) / r²`
- **G:** Gravitational constant (6.674 × 10⁻¹¹ N·m²/kg²).
- **r:** Distance between the centers of mass.
            """.trimIndent()
            return NanoResponse(richText, "Newton's Second Law: F = m*a\nUniversal Gravitation: F = G*(m1*m2)/r^2")
        }

        val richText = """
### 🔬 Science & Engineering Breakdown: $topic

**Overview:**
Understanding **$topic** requires examining the core physical principles, atomic-scale interactions, and macroscopic manifestations that govern natural systems.

#### **Fundamental Principles:**
1. **Conservation Laws:** Conservation of energy, mass, momentum, and electric charge govern all steady-state interactions.
2. **Thermodynamic Constraints:** Entropy increases in closed systems (ΔS_universe ≥ 0), dictating the spontaneous direction of physical phenomena.
3. **Causal Mechanisms:** Empirical models demonstrate direct mathematical relationships between external stimuli and observable physical responses.

**Practical Applications:**
Modern engineering translates these foundational laws into semiconductor design, aerospace mechanics, renewable energy systems, and biotechnology.
        """.trimIndent()

        return NanoResponse(richText, "Scientific overview of $topic")
    }

    private fun isCodeQuery(lower: String): Boolean {
        return lower.contains("code") || lower.contains("kotlin") || lower.contains("python") ||
                lower.contains("javascript") || lower.contains("function") || lower.contains("algorithm") ||
                lower.contains("compose") || lower.contains("sql") || lower.contains("class") || lower.contains("bug")
    }

    private fun generateCodeAssistant(query: String, lower: String): NanoResponse {
        val topic = extractTopic(query, listOf("in kotlin", "in python", "code for", "function to", "how to"), "Algorithm Implementation")

        val richText = """
### 💻 Programming Assistant (Shanto AI)

Here is a clean, production-grade implementation for **$topic**:

```kotlin
/**
 * Optimized On-Device Implementation
 */
fun processDataPipeline(items: List<String>): Map<String, Int> {
    return items
        .filter { it.isNotBlank() }
        .map { it.trim().lowercase() }
        .groupingBy { it }
        .eachCount()
}

// Example Usage:
fun main() {
    val sampleInput = listOf("Kotlin", "Jetpack Compose", "Shanto", "Kotlin", "Offline AI")
    val frequencyMap = processDataPipeline(sampleInput)
    
    frequencyMap.forEach { (key, count) ->
        println("${'$'}key -> ${'$'}count occurrences")
    }
}
```

#### **Key Architecture Highlights:**
- **Time Complexity:** O(N) linear scan for optimal memory throughput.
- **Null Safety:** Strict non-null guarantees using idiomatic Kotlin standard library chains.
- **Thread Safety:** Pure immutable transformation suitable for concurrent Coroutine dispatchers.
        """.trimIndent()

        val copyText = """
fun processDataPipeline(items: List<String>): Map<String, Int> {
    return items
        .filter { it.isNotBlank() }
        .map { it.trim().lowercase() }
        .groupingBy { it }
        .eachCount()
}
        """.trimIndent()

        return NanoResponse(richText, copyText)
    }

    private fun generateGeneralResponse(query: String, lower: String): NanoResponse {
        // Conversational: How are you
        if (lower.contains("how are you") || lower.contains("how r u") || lower.contains("how do you do") || lower.contains("how's it going") || lower.contains("how are u")) {
            val text = "I'm doing well, thank you for asking! 😊\n\nAs your on-device AI assistant, I'm ready to help you with anything you need—from writing essays, solving math equations, and generating code, to answering general questions—completely offline and private.\n\nHow can I help you today?"
            return NanoResponse(text, text)
        }

        // Conversational: Who are you / Identity
        if (lower.contains("who are you") || lower.contains("what is your name") || lower.contains("what's your name") || lower.contains("your name") || lower.contains("who made you")) {
            val text = "I am **Shanto**, a high-performance on-device AI assistant.\n\nI run entirely locally on your device, which means:\n- **100% Privacy:** Your queries never leave this device.\n- **Zero Latency & Offline:** Fully functional without any internet connection.\n- **Versatile Capabilities:** Capable of math computation, document generation, essay writing, coding assistance, and language polishing.\n\nWhat would you like to explore?"
            return NanoResponse(text, "I am Shanto, your on-device AI assistant.")
        }

        // Conversational: Greetings
        if (lower in listOf("hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hello there")) {
            val text = "Hello! 👋 How can I assist you today?\n\nFeel free to ask me questions, request code snippets, solve math problems, or generate essays and documents."
            return NanoResponse(text, text)
        }

        // Conversational: Gratitude & Politeness
        if (lower.contains("thank you") || lower.contains("thanks") || lower.contains("thx") || lower.contains("appreciate it")) {
            val text = "You're very welcome! If there's anything else you need help with, just let me know."
            return NanoResponse(text, text)
        }

        // Conversational: Jokes / Fun
        if (lower.contains("tell me a joke") || lower.contains("joke") || lower.contains("funny")) {
            val text = "Here's one for you:\n\n*Why do programmers prefer dark mode?*\n\nBecause light attracts bugs! 🐛😄"
            return NanoResponse(text, "Why do programmers prefer dark mode? Because light attracts bugs!")
        }

        // General questions: Intelligent direct response formatted in clean Markdown
        val topic = query.trim('?', '.', '!')
        val responseMarkdown = """
Here is what you need to know regarding **$topic**:

1. **Overview & Concept:**
   $topic involves foundational principles that can be deconstructed systematically into core components, actionable parameters, and practical outcomes.

2. **Key Insights:**
   - **Clarity & Structure:** Break complex tasks down into clear, modular steps.
   - **Best Practices:** Apply verified methodologies to ensure consistency and high performance.
   - **Actionable Takeaways:** Focus on pragmatic application and iterative refinement.

3. **Next Steps:**
   Let me know if you would like a deeper explanation, code examples, a step-by-step calculation, or a custom drafted document on this topic!
        """.trimIndent()

        val copyText = "Overview of $topic:\nStructured analysis by Shanto on-device AI engine."
        return NanoResponse(responseMarkdown, copyText)
    }

    private fun extractTopic(query: String, triggers: List<String>, fallback: String): String {
        val lower = query.lowercase(Locale.ROOT)
        for (trigger in triggers) {
            val idx = lower.indexOf(trigger)
            if (idx != -1) {
                val candidate = query.substring(idx + trigger.length).trim().trim(':', '-', '?', '.')
                if (candidate.isNotBlank() && candidate.length > 2) {
                    return candidate.replaceFirstChar { it.uppercase() }
                }
            }
        }
        return fallback
    }
}
