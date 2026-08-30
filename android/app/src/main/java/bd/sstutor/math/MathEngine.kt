package bd.sstutor.math

import java.util.regex.Pattern
import kotlin.math.sqrt

object MathEngine {
    private val BENGALI_DIGITS = arrayOf("০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯")

    fun toBengaliNumber(num: Long): String {
        val s = num.toString()
        val sb = StringBuilder()
        for (ch in s) {
            if (ch.isDigit()) {
                sb.append(BENGALI_DIGITS[ch - '0'])
            } else {
                sb.append(ch)
            }
        }
        return sb.toString()
    }

    fun toBengaliNumber(d: Double): String {
        if (d % 1.0 == 0.0) return toBengaliNumber(d.toLong())
        val s = String.format("%.2f", d)
        val sb = StringBuilder()
        for (ch in s) {
            if (ch.isDigit()) {
                sb.append(BENGALI_DIGITS[ch - '0'])
            } else if (ch == '.') {
                sb.append(".")
            } else {
                sb.append(ch)
            }
        }
        return sb.toString()
    }

    fun toEnglishDigits(text: String): String {
        var res = text
        for (i in 0..9) {
            res = res.replace(BENGALI_DIGITS[i], i.toString())
        }
        return res
    }

    fun gcd(a: Long, b: Long): Long {
        var n1 = Math.abs(a)
        var n2 = Math.abs(b)
        while (n2 != 0L) {
            val temp = n2
            n2 = n1 % n2
            n1 = temp
        }
        return if (n1 == 0L) 1L else n1
    }

    // --- 1. Exact Fraction Arithmetic ---
    data class FractionResult(
        val finalAnswerBengali: String,
        val stepsBengali: List<String>,
        val num: Long,
        val den: Long
    )

    fun addFractions(n1: Long, d1: Long, n2: Long, d2: Long): FractionResult {
        val lcm = (d1 * d2) / gcd(d1, d2)
        val m1 = lcm / d1
        val m2 = lcm / d2
        val stepN1 = n1 * m1
        val stepN2 = n2 * m2
        val totalNum = stepN1 + stepN2
        val g = gcd(totalNum, lcm)
        val redNum = totalNum / g
        val redDen = lcm / g

        val finalAns = if (redDen == 1L) {
            toBengaliNumber(redNum)
        } else if (redNum > redDen) {
            val whole = redNum / redDen
            val rem = redNum % redDen
            "${toBengaliNumber(whole)} সমস্ত ${toBengaliNumber(rem)}/${toBengaliNumber(redDen)}"
        } else {
            "${toBengaliNumber(redNum)}/${toBengaliNumber(redDen)}"
        }

        val steps = listOf(
            "ধাপ ১: হরদ্বয় ${toBengaliNumber(d1)} ও ${toBengaliNumber(d2)} এর ল.সা.গু = ${toBengaliNumber(lcm)}।",
            "ধাপ ২: সমহর বিশিষ্ট ভগ্নাংশ: ${toBengaliNumber(stepN1)}/${toBengaliNumber(lcm)} + ${toBengaliNumber(stepN2)}/${toBengaliNumber(lcm)}।",
            "ধাপ ৩: লব যোগ করে পাই: ${toBengaliNumber(totalNum)}/${toBengaliNumber(lcm)} = $finalAns।"
        )

        return FractionResult(finalAns, steps, redNum, redDen)
    }

    // --- 2. Financial & Geometric Calculations ---
    fun simpleInterest(p: Double, r: Double, n: Double): Map<String, Any> {
        val interest = (p * r * n) / 100.0
        val total = p + interest
        return mapOf(
            "interest" to interest,
            "total" to total,
            "formula" to "I = Prn",
            "bengali_text" to "আসল P = ${toBengaliNumber(p)} টাকা, হার r = ${toBengaliNumber(r)}%, সময় n = ${toBengaliNumber(n)} বছর। মোট সরল মুনাফা I = ${toBengaliNumber(interest)} টাকা, সবৃদ্ধিমূল = ${toBengaliNumber(total)} টাকা।"
        )
    }

    fun compoundInterest(p: Double, r: Double, n: Double): Map<String, Any> {
        val amount = p * Math.pow(1.0 + (r / 100.0), n)
        val interest = amount - p
        return mapOf(
            "amount" to amount,
            "interest" to interest,
            "formula" to "C = P(1+r)^n",
            "bengali_text" to "সবৃদ্ধিমূল C = ${toBengaliNumber(amount)} টাকা, চক্রবৃদ্ধি মুনাফা = ${toBengaliNumber(interest)} টাকা।"
        )
    }

    fun pythagoras(a: Double, b: Double): Map<String, Any> {
        val c = sqrt(a * a + b * b)
        return mapOf(
            "hypotenuse" to c,
            "bengali_text" to "পিথাগোরাসের উপপাদ্য অনুসারে অতিভুজ c = √(a² + b²) = √(${toBengaliNumber(a)}² + ${toBengaliNumber(b)}²) = ${toBengaliNumber(c)} সেমি।"
        )
    }

    fun seriesSum(n: Long): Map<String, Any> {
        val sum = (n * (n + 1)) / 2
        return mapOf(
            "sum" to sum,
            "bengali_text" to "১ থেকে ${toBengaliNumber(n)} পর্যন্ত স্বাভাবিক সংখ্যার সমষ্টি Sₙ = n(n+1)/2 = ${toBengaliNumber(sum)}।"
        )
    }

    fun circleMetrics(r: Double): Map<String, Any> {
        val pi = 22.0 / 7.0
        val area = pi * r * r
        val perimeter = 2.0 * pi * r
        return mapOf(
            "area" to area,
            "perimeter" to perimeter,
            "bengali_text" to "ব্যাসার্ধ r = ${toBengaliNumber(r)} সেমি হলে পরিধি = ${toBengaliNumber(perimeter)} সেমি এবং ক্ষেত্রফল = ${toBengaliNumber(area)} বর্গসেমি।"
        )
    }

    // --- 3. Mathematical Intent Detection ---
    data class MathIntent(
        val type: String,
        val isMath: Boolean,
        val params: Map<String, Double> = emptyMap(),
        val rawResult: String = "",
        val explanation: String = ""
    )

    fun parseMathIntent(query: String): MathIntent {
        val qEng = toEnglishDigits(query)

        // Fractions: e.g. "3/4 + 5/6" or "৩/৪ এবং ৫/৬ যোগ"
        val fracPattern = Pattern.compile("(\\d+)\\s*/\\s*(\\d+)")
        val matcher = fracPattern.matcher(qEng)
        val fractions = mutableListOf<Pair<Long, Long>>()
        while (matcher.find()) {
            fractions.add(Pair(matcher.group(1).toLong(), matcher.group(2).toLong()))
        }

        if (fractions.size >= 2 && (query.contains("যোগ") || query.contains("+") || query.contains("যোগফল"))) {
            val res = addFractions(fractions[0].first, fractions[0].second, fractions[1].first, fractions[1].second)
            return MathIntent("fraction_addition", true, emptyMap(), res.finalAnswerBengali, res.stepsBengali.joinToString("\n"))
        }

        // Simple Interest: P, r, n
        if ((query.contains("মুনাফা") || query.contains("সুদ")) && !query.contains("চক্রবৃদ্ধি")) {
            val numMatcher = Pattern.compile("\\d+(?:\\.\\d+)?").matcher(qEng)
            val nums = mutableListOf<Double>()
            while (numMatcher.find()) nums.add(numMatcher.group().toDouble())
            if (nums.size >= 3) {
                nums.sort()
                val n = nums[0]
                val r = if (nums[1] <= 35.0) nums[1] else nums[0]
                val p = nums[nums.size - 1]
                val res = simpleInterest(p, r, n)
                return MathIntent("simple_interest", true, mapOf("p" to p, "r" to r, "n" to n), toBengaliNumber(res["interest"] as Double), res["bengali_text"] as String)
            }
        }

        // Compound Interest
        if (query.contains("চক্রবৃদ্ধি")) {
            val numMatcher = Pattern.compile("\\d+(?:\\.\\d+)?").matcher(qEng)
            val nums = mutableListOf<Double>()
            while (numMatcher.find()) nums.add(numMatcher.group().toDouble())
            if (nums.size >= 3) {
                nums.sort()
                val n = nums[0]
                val r = if (nums[1] <= 35.0) nums[1] else nums[0]
                val p = nums[nums.size - 1]
                val res = compoundInterest(p, r, n)
                return MathIntent("compound_interest", true, mapOf("p" to p, "r" to r, "n" to n), toBengaliNumber(res["amount"] as Double), res["bengali_text"] as String)
            }
        }

        // Pythagoras
        if (query.contains("অতিভুজ") || query.contains("পিথাগোরাস")) {
            val numMatcher = Pattern.compile("\\d+(?:\\.\\d+)?").matcher(qEng)
            val nums = mutableListOf<Double>()
            while (numMatcher.find()) nums.add(numMatcher.group().toDouble())
            if (nums.size >= 2) {
                val a = nums[0]
                val b = nums[1]
                val res = pythagoras(a, b)
                return MathIntent("pythagoras", true, mapOf("a" to a, "b" to b), toBengaliNumber(res["hypotenuse"] as Double), res["bengali_text"] as String)
            }
        }

        // Series Sum
        if (query.contains("স্বাভাবিক সংখ্যার যোগফল") || query.contains("ক্রমিক সংখ্যার যোগফল") || query.contains("পর্যন্ত সংখ্যার যোগফল")) {
            val numMatcher = Pattern.compile("\\d+").matcher(qEng)
            val nums = mutableListOf<Long>()
            while (numMatcher.find()) nums.add(numMatcher.group().toLong())
            if (nums.isNotEmpty()) {
                val n = nums[nums.size - 1]
                val res = seriesSum(n)
                return MathIntent("series_sum", true, mapOf("n" to n.toDouble()), toBengaliNumber(res["sum"] as Long), res["bengali_text"] as String)
            }
        }

        // Circle metrics
        if (query.contains("বৃত্তের ক্ষেত্রফল") || query.contains("বৃত্তের পরিধি")) {
            val numMatcher = Pattern.compile("\\d+(?:\\.\\d+)?").matcher(qEng)
            if (numMatcher.find()) {
                val r = numMatcher.group().toDouble()
                val res = circleMetrics(r)
                return MathIntent("circle_metrics", true, mapOf("r" to r), toBengaliNumber(res["area"] as Double), res["bengali_text"] as String)
            }
        }

        return MathIntent("general_concept", false)
    }
}
