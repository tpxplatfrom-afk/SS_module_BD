# 📱 THSA-2.41B Android Developer Integration & Cloudflare CDN Distribution Guide

Welcome to the **THSA-2.41B On-Device AI Module for Bangladesh (Classes 1–12)**.  
This guide explains:
1. How Android developers can integrate the module in just **3–4 lines of Kotlin**.
2. How the **1-Click Copy-Paste (`.txt` / `.md`)** system works.
3. How to host the 654 MB `model.nano` file on **Cloudflare R2 / CDN** for high-speed, zero-cost downloads.

---

## 🚀 Part 1: Android Developer Quickstart (3–4 Lines of Code)

### 📦 Step 1: Add Gradle Dependency (`build.gradle.kts` / `build.gradle`)

In your project `settings.gradle.kts`:
```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://cdn.yourdomain.com/maven") } // Your Cloudflare CDN
    }
}
```

In your app-level `build.gradle.kts`:
```kotlin
dependencies {
    implementation("ai.nano:ss-bangladesh-nano:1.0.0")
}
```

---

### 💻 Step 2: Universal 1-Method Kotlin Code (MainActivity.kt)

Android developers only need to write **3 lines of code**:

```kotlin
package com.example.myeducationapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import ai.nano.engine.NanoEngine
import java.io.File

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 1. Initialize Engine (Zero-RAM Leak, Monolithic Arena)
        val engine = NanoEngine.load(File(filesDir, "model.nano"))

        // 2. Universal Ask (Math, Science, English CV, Bangladesh Laws, Socratic Hints)
        val response = engine.ask("৯ম শ্রেণি অনুশীলনী ৩.১ এর ২ এর খ")

        // 3. Display on Screen & 1-Click Copy-Paste
        textView.text = response.text            // Screen-Safe Display
        val copyReady = response.copyText        // 1-Click Clipboard Ready (clean text)
        response.saveAsTxt(File(cacheDir, "solution.txt")) // Instant .txt file export
        response.saveAsMd(File(cacheDir, "solution.md"))   // Instant .md file export
    }
}
```

---

## 📋 Part 2: 1-Click Copy-Paste & `.txt` / `.md` Plugin Explained

Mobile students often struggle to copy math equations, English CVs, or paragraphs into WhatsApp or Word documents because raw markdown contains `#`, `*`, `$$`, and backticks.

### 🛡️ How `NanoResponse` Solves This:
```kotlin
data class NanoResponse(
    val prompt: String,
    val text: String,        // Formatted for UI display (LaTeX equations, markdown bold)
    val markdown: String,    // Standard Markdown (.md)
    val copyText: String,    // 100% clean plaintext (all markdown hashes, math tags stripped)
    val isScreenSafe: Boolean = true
) {
    fun toPlainText(): String = copyText
    fun toMarkdown(): String = markdown
    fun saveAsTxt(targetFile: File)
    fun saveAsMd(targetFile: File)
}
```

### 💬 Automatic Chat Detection:
When a student types in chat:
> *"আমাকে এই সিভিটা কপি-পেস্ট করার মতো করে দাও"* বা *"Make it copy-paste friendly"*

The engine automatically encapsulates the response inside a dedicated **`[১-ক্লিক কপি-পেস্ট ফ্রেন্ডলি টেক্সট ব্লক]`**:
```text
📋 [১-ক্লিক কপি-পেস্ট ফ্রেন্ডলি টেক্সট ব্লক]:
===================================================================
Curriculum Vitae (CV) for Assistant English Teacher
Name: Tanvir Ahmed
Education: M.A in English (CGPA 3.75, DU)
...
===================================================================
```

---

## ☁️ Part 3: Cloudflare R2 CDN Hosting & Model Downloader Guide

To keep your Android APK size lightweight (< 15 MB), **do not bundle the 654 MB `model.nano` inside the APK**. Instead, host it on Cloudflare CDN and download it on the first launch of the app.

---

### 🌐 Step A: Setup Cloudflare R2 (100% Free Egress Bandwidth)
1. Log in to your **[Cloudflare Dashboard](https://dash.cloudflare.com/)**.
2. In the left sidebar, navigate to **R2 -> Create Bucket**.
   - Bucket Name: `ss-bangladesh-assets`
   - Location: `Automatic` (or Asia Pacific)
3. Upload `model.nano` (654.39 MB) to your R2 bucket.
4. Click on **Settings -> Custom Domains** -> Connect your domain (e.g. `cdn.yourdomain.com`).
5. Your public download URL will be:
   ```
   https://cdn.yourdomain.com/models/model.nano
   ```
   *(Cloudflare R2 charges **$0 for egress bandwidth**, meaning 1 million students downloading the model costs $0 in bandwidth fees).*

---

### 📥 Step B: In-App Automatic Model Downloader (Kotlin)

Add this background downloader in your Android app to download `model.nano` on first launch with a progress bar:

```kotlin
package ai.nano.downloader

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

object NanoModelManager {

    private const val MODEL_URL = "https://cdn.yourdomain.com/models/model.nano"
    private const val MODEL_FILE_NAME = "model.nano"

    fun isModelDownloaded(context: Context): Boolean {
        val modelFile = File(context.filesDir, MODEL_FILE_NAME)
        return modelFile.exists() && modelFile.length() > 600 * 1024 * 1024 // > 600 MB
    }

    suspend fun downloadModel(
        context: Context,
        onProgress: (percent: Int) -> Unit
    ): File = withContext(Dispatchers.IO) {
        val targetFile = File(context.filesDir, MODEL_FILE_NAME)
        val tempFile = File(context.filesDir, "$MODEL_FILE_NAME.tmp")

        val connection = URL(MODEL_URL).openConnection() as HttpURLConnection
        connection.connectTimeout = 15000
        connection.readTimeout = 30000
        connection.connect()

        val fileLength = connection.contentLength
        val input = connection.inputStream
        val output = FileOutputStream(tempFile)

        val buffer = ByteArray(64 * 1024)
        var totalBytesRead = 0L
        var count: Int

        while (input.read(buffer).also { count = it } != -1) {
            output.write(buffer, 0, count)
            totalBytesRead += count
            if (fileLength > 0) {
                val progress = ((totalBytesRead * 100) / fileLength).toInt()
                withContext(Dispatchers.Main) {
                    onProgress(progress)
                }
            }
        }

        output.flush()
        output.close()
        input.close()

        // Rename temp file to final model.nano
        tempFile.renameTo(targetFile)
        targetFile
    }
}
```

---

### 📱 Step C: Complete Android Splash/Launch Flow

```kotlin
class SplashActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        if (NanoModelManager.isModelDownloaded(this)) {
            // Model already exists -> Open Main Chat Activity
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        } else {
            // Download model on first launch with progress bar
            progressBar.visibility = View.VISIBLE
            statusText.text = "শিক্ষার্থী এআই মডেল ডাউনলোড হচ্ছে (একবারই প্রয়োজন)..."

            lifecycleScope.launch {
                NanoModelManager.downloadModel(this@SplashActivity) { progress ->
                    progressBar.progress = progress
                    statusText.text = "মডেল ডাউনলোড হচ্ছে: $progress%"
                }
                startActivity(Intent(this@SplashActivity, MainActivity::class.java))
                finish()
            }
        }
    }
}
```

---

## 🎯 Summary of Capabilities

| Feature | Android Implementation | Description |
|---|---|---|
| **Single Universal API** | `engine.ask(prompt)` | Handles Math, Science, English CV, Bangladesh Laws, Socratic hints |
| **1-Click Copy-Paste** | `response.copyText` | Clean, stripped plaintext for notepad, assignments & WhatsApp |
| **.txt & .md Export** | `response.saveAsTxt()` / `.saveAsMd()` | 1-line file export to storage |
| **Sibling Profile Memory** | Automatic State Tracking | Remembers active class & switches when brother/sister takes phone |
| **Zero Bandwidth Cost** | Cloudflare R2 CDN | Free egress hosting for `model.nano` (654 MB) |
