# AUDIT-ANDROID-NANO-LIVENESS-01
## PRE-FIX-05 Android App Real Nano Inference Forensic Audit

**Audit Date:** 2026-09-02  
**Auditor:** Antigravity Forensic Engine  
**Status:** COMPLETE — NO MODIFICATIONS MADE

---

## 1. EXECUTIVE VERDICT

> [!CAUTION]
> **The Android application does NOT use model.nano or the native THSA-2B inference engine to generate responses.**
>
> All responses are produced by `ReasoningProcessor.kt`, a 679-line Kotlin `object` containing hardcoded decision trees, static string templates, pattern-matched responses, and scripted fallback text. The native JNI path (`ai.nano.engine.NanoNative`) exists in source code but is **architecturally isolated** from the actual app execution path and is **never called** by any running code. `libnano_engine.so` is **not packaged** in the built APK.

---

## 2. EXACT APK AUDITED

| Property | Value |
|---|---|
| APK path | `ss_bangladesh_nano_android_module/offline-ai_chatbot/app/build/outputs/apk/debug/app-debug.apk` |
| APK size | 171,119,436 bytes |
| APK SHA-256 | `A18CDA788DE550FA9D9E08A14513EDAC6046DE56659E6E3F06BCE1AE25133E35` |
| Installed APK path on device | `/data/app/~~uwFmWf6Xz_MUf8xD4ihMyg==/com.aistudio.offlineai.krvq-qbaBBoFVYbVPDmu94BNYvA==/base.apk` |

---

## 3. EXACT PACKAGE / applicationId

| Property | Value |
|---|---|
| `applicationId` | `com.aistudio.offlineai.krvq` |
| Namespace | `com.example` |
| App Label | "Shanto" / "Offline On-Device AI" |
| Main Activity | `com.example.MainActivity` |
| Package installed on device | **YES** (confirmed via `pm list packages`) |

---

## 4. DEVICE INFORMATION

| Property | Value |
|---|---|
| Device Model | itel A662L |
| Android Version | 12 |
| ABI | armeabi-v7a (32-bit ARM) |
| App PID | 13777 |
| Physical Device Connected | **YES** |
| App Running | **YES** (confirmed via `ps -A`) |

---

## 5. ANDROID APPLICATION CALL GRAPH

### 5.1 Actual Runtime Path (as executed)

```
USER INPUT (typed text)
    ↓
ChatScreen.kt (Jetpack Compose UI) — Line 430: onSend = { viewModel.sendMessage() }
    ↓
ChatViewModel.kt — Line 147: repository.processUserMessage(sessionId, cleanQuery, category)
    ↓
ChatRepository.kt — Line 57: val response = engine.ask(userText)
    ↓
com.example.thsa.NanoEngine.kt — Line 58-60:
    fun ask(userInput: String): NanoResponse {
        return ReasoningProcessor.process(userInput)   // ← ALL INFERENCE HAPPENS HERE
    }
    ↓
ReasoningProcessor.kt — Line 19: fun process(input: String): NanoResponse
    [679 lines of Kotlin decision-tree / hardcoded responses]
    ↓
NanoResponse (com.example.thsa.NanoResponse) — static text object
    ↓
ChatRepository.kt — saves to Room DB
    ↓
ChatScreen UI displays response
```

### 5.2 Critical Arrow Analysis — Each Arrow Evaluated

| Arrow | File | Function | Line | Caller | Callee | Status |
|---|---|---|---|---|---|---|
| UI → ViewModel | ChatScreen.kt | onSend | 430 | ChatScreen | ChatViewModel.sendMessage | **EXISTS** |
| ViewModel → Repository | ChatViewModel.kt | sendMessage | 147 | ChatViewModel | ChatRepository.processUserMessage | **EXISTS** |
| Repository → NanoEngine | ChatRepository.kt | processUserMessage | 57 | ChatRepository | com.example.thsa.NanoEngine.ask | **EXISTS** |
| NanoEngine → ReasoningProcessor | NanoEngine.kt | ask | 59 | NanoEngine | ReasoningProcessor.process | **EXISTS** |
| ReasoningProcessor → JNI | (none) | — | — | — | — | **BROKEN / ABSENT** |
| JNI → native C++ | (none) | — | — | — | — | **BROKEN / ABSENT** |
| native C++ → nano_engine | (none) | — | — | — | — | **BROKEN / ABSENT** |
| nano_engine → model.nano | (none) | — | — | — | — | **BROKEN / ABSENT** |
| model.nano → logits | (none) | — | — | — | — | **BROKEN / ABSENT** |
| logits → token selection | (none) | — | — | — | — | **BROKEN / ABSENT** |
| token selection → decoded text | (none) | — | — | — | — | **BROKEN / ABSENT** |

**Every arrow below `ReasoningProcessor` is BROKEN / ABSENT.**

---

## 6. REASONINGPROCESSOR.KT FORENSIC AUDIT

**File:** [`ReasoningProcessor.kt`](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/offline-ai_chatbot/app/src/main/java/com/example/thsa/ReasoningProcessor.kt)  
**Lines:** 679  
**Package:** `com.example.thsa`  
**Type:** Kotlin `object` (singleton)

### 6.1 Structural Findings

| Mechanism | Lines | Evidence |
|---|---|---|
| `object ReasoningProcessor` | Line 17 | Singleton, no AI state |
| Multi-tier `if()` dispatch | Lines 31–76 | 10 explicit if-branches on input |
| `lower.contains(it)` pattern matching | Lines 90–98, 181–186, etc. | String contains checks |
| Hardcoded greeting strings | Lines 84–98 | `listOf("হাই", "হ্যালো", "hello", "hi", "hey", ...)` |
| Hardcoded identity response | Lines 108–118 | Static multi-line Bangla string literal |
| Hardcoded Operation Jackpot answer | Lines 309–327 | Word-for-word prewritten historical essay |
| Hardcoded 1971 liberation war answer | Lines 354–372 | Pre-written essay in Kotlin string |
| Hardcoded 10 kg / 5 meter physics answer | Lines 447–496 | Specific numeric answer hard-coded |
| Hardcoded Newton's Laws | Lines 500–507 | Static F=ma text |
| Template fallback (generateGeneralResponse) | Lines 644–679 | `"Here is what you need to know regarding **$query**:"` template |
| Calculus/Optimization scripted response | Lines 518–553 | Full static math essay |
| Code assistant canned response | Lines 631–641 | `println("Hello from Shanto On-Device AI!")` static stub |
| CV/Cover letter scripted template | Lines 608–624 | Boilerplate letter |

### 6.2 Hardcoded Response Catalog

| File | Lines | Input Condition | Static Output |
|---|---|---|---|
| ReasoningProcessor.kt | 108–118 | `original.contains("তুমি কে")` OR `"who are you"` | Fixed Bangla identity essay |
| ReasoningProcessor.kt | 124–129 | `original.contains("কেমন আছো")` OR `lower.contains("kemon")` | Fixed wellness greeting |
| ReasoningProcessor.kt | 133–135 | `original.contains("ধন্যবাদ")` | Fixed 1-sentence Bangla thanks |
| ReasoningProcessor.kt | 161–163 | `lower.contains("how are you")` | Fixed English greeting |
| ReasoningProcessor.kt | 309–327 | `lower.contains("operation jackpot")` | 6-paragraph prewritten essay |
| ReasoningProcessor.kt | 354–372 | `lower.contains("1971")` OR `"মুক্তিযুদ্ধ"` | Full Bangla Liberation War essay |
| ReasoningProcessor.kt | 447–496 | `lower.contains("10 kg")` + `contains("5")` | Physics answer: **490 Joules** hardcoded |
| ReasoningProcessor.kt | 500–507 | Any `physics` / `newton` hit | Newton's Laws static text |
| ReasoningProcessor.kt | 663–679 | All unmatched inputs | Template with `$query` interpolation |

### 6.3 Token-Level Evidence

- **No logits**: Zero evidence of floating-point logit computation.
- **No argmax**: No sampling or top-p/temperature computation.
- **No token IDs**: No integer token arrays exist in `ReasoningProcessor.kt`.
- **No autoregressive loop**: No iterative generation.
- **No tokenizer call**: No tokenizer is invoked in the response path.

---

## 7. STATIC RESPONSE FINDINGS

### 7.1 DEX Inspection (APK)

The following strings were confirmed **present in `classes5.dex`** of the built APK:

| String | DEX File | Interpretation |
|---|---|---|
| `ReasoningProcessor` | classes5.dex | Response engine is static Kotlin object |
| `SHANTO_NANO_WEIGHTS` | classes5.dex | Fake "model" header written by NanoEngine.load() |
| `Operation Jackpot` | classes5.dex | Hardcoded answer preloaded in APK |
| `490 Joules` | classes5.dex | **Physics answer pre-embedded in APK binary** |
| `model_trained.nano` | classes5.dex | Asset filename referenced in Kotlin |
| `model.nano` | classes5.dex | Fallback asset name in Kotlin |
| `Wakon Yosai` | classes5.dex | Preloaded history response in APK |
| `Harpedonaptai` | classes5.dex | Preloaded history response in APK |

> [!CAUTION]
> **`490 Joules` is present in classes5.dex.** This confirms that the physics answer is embedded in the APK binary and can be produced **without loading or querying any model file.**

### 7.2 String Comparison: ReasoningProcessor vs. Actual DB Response

The device database confirmed that the query `"Tell me about 1971 Operation Jackpot"` produced the **exact verbatim text from `ReasoningProcessor.kt` lines 329–347**, character-for-character.

The query `"If a 10 kg object falls from 5 meters, calculate energy"` in one session returned:
```
Math evaluation for: If a 10 kg object falls from 5 meters, calculate energy
```
This is the **exact verbatim fallback from `solveMathQuery()` line 599–600** (the regex didn't match the operator pattern). In another session with a slightly different phrasing, it returned the **full hardcoded physics essay from lines 474–495**.

---

## 8. MODEL.NANO APK PACKAGING EVIDENCE

### 8.1 Asset Found in APK

| Property | Value |
|---|---|
| Asset path in APK | `assets/model_trained.nano` |
| Uncompressed size in APK | **166,728,000 bytes (159 MB)** |
| Compressed size in APK | 148,010,842 bytes |
| SHA-256 (APK asset) | `966a2ecf3ca4d2c9100b9ab6d688379f92b35f8d0c6e821e40a249524f039c16` |

### 8.2 Model SHA-256 Comparison

| Model | Size | SHA-256 |
|---|---|---|
| `android/app/src/main/model.nano` (reference) | 686,176,192 bytes | `638d51bd6813893145a2c64a46e33581c78b2a8134df0b580f4de1645e164791` ✅ matches expected |
| `app/src/main/assets/model_trained.nano` (bundled) | **166,728,000 bytes** | `966a2ecf3ca4d2c9100b9ab6d688379f92b35f8d0c6e821e40a249524f039c16` ❌ DIFFERENT |

> [!WARNING]
> The model packaged in the APK (`model_trained.nano`, 159 MB) does **NOT match** the validated THSA-2B model (`model.nano`, 686 MB). The hashes are completely different. The APK is bundling an **entirely different, unverified file** under the name `model_trained.nano`.

### 8.3 Device File Evidence

On the connected device, the file `/data/data/com.aistudio.offlineai.krvq/files/model_trained.nano` exists:
```
-rw-rw-rw- 1 u0_a96 u0_a96 166728000 2026-09-02 02:31 model_trained.nano
```
This file was extracted from APK assets by `ModelManager.ensureModelExtractedFromAssets()`. It is the **159 MB file, not the 686 MB validated model**.

### 8.4 Model Not Mapped in Process Memory

A search for `model_trained.nano` and `/data/data/*offlineai*` in `/proc/13777/maps` returned **zero results**. The 159 MB file is on disk but is **NOT memory-mapped or file-descriptor-opened by the running process**.

---

## 9. LIBNANO_ENGINE.SO APK EVIDENCE

### 9.1 Presence in Source / jniLibs

| File | Architecture | Size | JNI Symbols |
|---|---|---|---|
| `jniLibs/arm64-v8a/libnano_engine.so` | AArch64 (64-bit) | 37,912 bytes | ALL 10 PRESENT (static) |
| `jniLibs/armeabi-v7a/libnano_engine.so` | ARM 32-bit | 32,212 bytes | ALL 10 PRESENT (static) |

All 10 JNI symbols are present in symbol tables of both `.so` files:
- `Java_ai_nano_engine_NanoNative_nativeInit`
- `Java_ai_nano_engine_NanoNative_nativeGenerate`
- `Java_ai_nano_engine_NanoNative_nativeCancel`
- `Java_ai_nano_engine_NanoNative_nativeFree`
- `Java_ai_nano_engine_NanoNative_nativeResetSession`
- `Java_ai_nano_engine_NanoNative_nativeGetTelemetry`
- `nano_engine_init`, `nano_engine_generate`, `nano_engine_reset_session`, `nano_engine_get_logits`

### 9.2 Critical Finding: libnano_engine.so NOT in Built APK

> [!CAUTION]
> **`libnano_engine.so` is NOT packaged in the built APK.**
>
> The APK's `lib/` directory contains only:
> - `lib/arm64-v8a/libandroidx.graphics.path.so`
> - `lib/arm64-v8a/libdatastore_shared_counter.so`
> - `lib/armeabi-v7a/libandroidx.graphics.path.so`
> - `lib/armeabi-v7a/libdatastore_shared_counter.so`
> - (x86 and x86_64 equivalents)
>
> `libnano_engine.so` is absent from the APK entirely.

### 9.3 libnano_engine.so NOT Mapped in Process Memory

A search for `libnano` in `/proc/13777/maps` returned **zero results**. The library is not loaded at runtime.

---

## 10. JNI CALL-CHAIN EVIDENCE

### 10.1 Architecture Analysis

Two separate `NanoEngine` classes exist in the repository — a critical architectural split:

| Class | Package | Role in App |
|---|---|---|
| `com.example.thsa.NanoEngine` | thsa package | **ACTUALLY USED** by ChatRepository |
| `ai.nano.engine.NanoEngine` | nano engine package | **NOT USED** — completely isolated |

The **actually-used** class (`com.example.thsa.NanoEngine`) at [`NanoEngine.kt`](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/offline-ai_chatbot/app/src/main/java/com/example/thsa/NanoEngine.kt):

```kotlin
fun ask(userInput: String): NanoResponse {
    return ReasoningProcessor.process(userInput)  // Line 59 — NO JNI, NO NATIVE
}
```

The `ai.nano.engine.NanoNative` JNI bridge class is **never imported** anywhere in the actual application flow. A grep of the entire `app/src/main/` directory for `import ai.nano.engine` returns **zero results** in any non-engine file.

### 10.2 NanoNative.kt — Dead Code Status

[`NanoNative.kt`](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/offline-ai_chatbot/app/src/main/java/ai/nano/engine/NanoNative.kt) at line 8:
```kotlin
init {
    System.loadLibrary("nano_engine")
}
```
This would load `libnano_engine.so` — but:
1. `NanoNative` is only referenced by `ai.nano.engine.NanoEngine`
2. `ai.nano.engine.NanoEngine` is never called by any UI/ViewModel/Repository path
3. `libnano_engine.so` is not in the APK anyway
4. The device `/proc/maps` confirms no `libnano_engine.so` is loaded

**Result: NanoNative.kt is dead code. The JNI bridge is architecturally disconnected.**

---

## 11. NATIVE RUNTIME EVIDENCE

### 11.1 Logcat Analysis

`adb logcat` was searched for:
- `NanoEngine`, `ModelManager`, `nano_engine`, `nativeInit`, `nativeGenerate`
- `System.load`, `loadLibrary`, `libnano`, `Shanto`, `THSA`
- Tag filters, PID filters

**Result: Zero matches.** No native engine initialization messages, no JNI load messages, no model loading logs from the running app.

The `ModelManager.kt` log message `"Loading Shanto AI model from..."` is produced by `com.example.thsa.NanoEngine.load()`, but this log was also not visible in logcat (app logging may have been suppressed or log buffer rolled over).

### 11.2 Process Memory Forensics

From `/proc/13777/maps`:
- **DEX files present**: `classes.dex`, `classes2.dex` through `classes9.dex` from the APK — confirmed loaded
- **`libnano_engine.so`**: **ABSENT** from maps
- **`model_trained.nano`**: **ABSENT** from maps (not mmap'd)
- **App installed APK file**: Present as `base.apk` via read-only memory mapping

### 11.3 NanoEngine.load() Behavior (com.example.thsa)

At startup, `ModelManager.getOrInitEngine()` calls `NanoEngine.load(modelFile)`.  
[`NanoEngine.kt` lines 38–48](file:///c:/Users/User/Desktop/SS_module_BD/ss_bangladesh_nano_android_module/offline-ai_chatbot/app/src/main/java/com/example/thsa/NanoEngine.kt#L38-L48):

```kotlin
if (!modelFile.exists() || modelFile.length() == 0L) {
    RandomAccessFile(modelFile, "rw").use { raf ->
        raf.setLength(1024 * 1024 * 2) // 2MB binary header structure
        raf.writeUTF("SHANTO_NANO_WEIGHTS_V1_0_0")
    }
}
val size = if (modelFile.exists()) modelFile.length() else 0L
return NanoEngine(modelFile, size)
```

This shows the "engine loading" is purely cosmetic — it creates a 2MB placeholder file if the model doesn't exist, then returns a Kotlin object wrapping a `File` reference. **No C++, no JNI, no parsing.**

---

## 12. PROCESS MEMORY EVIDENCE

| Evidence Type | Result |
|---|---|
| `libnano_engine.so` in `/proc/maps` | **ABSENT** |
| `model_trained.nano` memory-mapped | **ABSENT** |
| Any app-specific `.so` beyond system libs | **ABSENT** |
| Classes loaded from APK dex | CONFIRMED (arm32 dalvik JIT) |
| RAM usage of app | ~94 MB RSS — consistent with Compose + Room DB only |

The device is armeabi-v7a (32-bit ARM). The `arm64-v8a` `.so` in `jniLibs/` would not be used on this device anyway, but the `armeabi-v7a` `.so` is also absent from the APK.

---

## 13. UNKNOWN-INPUT TESTS

These prompts were **not submitted at runtime** because the audit cannot modify the production environment. However, the following static analysis determines what would happen:

| Novel Prompt | Would Reach Native Engine? | Evidence |
|---|---|---|
| `"ZXQ-7391-NANO-LIVENESS-ORANGE"` | **NO** | Falls to `generateGeneralResponse()` template in ReasoningProcessor |
| `"Calculate 8173 × 9419"` | **NO** | Hits `isMathQuery()` → `solveMathQuery()` → Kotlin regex arithmetic |
| `"Invent a sentence with Q7-MANGO-481"` | **NO** | Falls to `generateGeneralResponse()` template |
| `"What is the next prime after 999983?"` | **NO** | No prime checker; falls to generic template |
| `"Repeat exactly: NANO_RUNTIME_NONCE_7F3A91C2"` | **NO** | Falls to `generateGeneralResponse()` template |

**Device DB confirms this pattern:** A query `"আমেরিকায় কি আছে ভাই?"` (not in any knowledge base) produced verbatim template output:
```
Here is what you need to know regarding **আমেরিকায় কি আছে ভাই**:
1. Overview & Concept: আমেরিকায় কি আছে ভাই involves foundational principles...
```
This is **line-for-line** from `generateGeneralResponse()` (lines 663–679 of ReasoningProcessor.kt).

---

## 14. MODEL DEPENDENCY EXPERIMENT (Static Analysis)

> [!IMPORTANT]
> **No modifications to production files were made.** The following is a static analysis conclusion based on the code path.

**Hypothesis:** Does the app require `model_trained.nano` to produce responses?

**Answer: NO.**

The call chain is:
```
ChatRepository.engine.ask(userText)
  → com.example.thsa.NanoEngine.ask()
  → ReasoningProcessor.process()
  → (static string logic) → return NanoResponse(text)
```

`ReasoningProcessor.process()` makes **zero file I/O calls**. It never opens, reads, or references the model file. All string computation is purely in-memory Kotlin.

**Conclusion:**
```
MODEL_DEPENDENCY_TEST = FAILED
```
The application produces the same responses whether or not `model_trained.nano` exists on disk.

---

## 15. NATIVE LIBRARY DEPENDENCY EXPERIMENT (Static Analysis)

**Hypothesis:** Does the app require `libnano_engine.so` to produce responses?

**Answer: NO.**

`libnano_engine.so` is not in the APK. The app runs normally on device (PID 13777 confirmed active). The absence of `libnano_engine.so` causes zero errors because `NanoNative.kt` is dead code that is never executed.

**Conclusion:**
```
NATIVE_ENGINE_DEPENDENCY = NOT_PROVEN
```

---

## 16. STATIC RESPONSE BYPASS TEST

```
ReasoningProcessor.kt
  → hardcoded/template response (679 lines)
  → NanoResponse text field (com.example.thsa.NanoResponse)
  → ChatRepository saves to Room DB
  → ChatScreen.kt displays via MessageBubble
```

This chain exists **without any call to**:
- `nano_engine_generate()`
- `nano_engine_get_logits()`
- `model_trained.nano`
- Any native code

**Conclusion:**
```
ANDROID_AI_INFERENCE = BYPASSED
STATIC_RESPONSE_BYPASS = CONFIRMED
```

---

## 17. TOKEN-LEVEL PROOF

| Token-level Mechanism | Evidence | Present? |
|---|---|---|
| Logit array produced | None | **NO** |
| argmax / sampling | None | **NO** |
| Token IDs (IntArray) | `NanoNative.nativeGenerate()` signature exists but never called | **NO (dead code)** |
| Autoregressive loop | None in response path | **NO** |
| Tokenizer decode | None in response path | **NO** |
| BPE trie used at runtime | Source exists in C++ but .so absent from APK | **NO** |

There is no token-level generation. The output is directly constructed Kotlin string objects.

---

## 18. UNRESOLVED ANOMALIES

### Anomaly 1: Two Competing NanoEngine Architectures
The repository contains two classes both named `NanoEngine` in different packages:
- `com.example.thsa.NanoEngine` — actually wired into the app, delegates to ReasoningProcessor
- `ai.nano.engine.NanoEngine` — calls JNI (dead code path)

This dual-architecture creates the appearance of real AI infrastructure without it functioning.

### Anomaly 2: model_trained.nano ≠ model.nano
The APK bundles `model_trained.nano` (166 MB, SHA-256: `966a2ecf...`) which does not match the validated THSA-2B model `model.nano` (686 MB, SHA-256: `638d51bd...`). The nature and content of the 166 MB file are unknown.

### Anomaly 3: libnano_engine.so in jniLibs but not in APK
Both `arm64-v8a` and `armeabi-v7a` variants of `libnano_engine.so` exist in the source `jniLibs/` directory but are **absent from the built APK**. This suggests they were either excluded by Gradle configuration or a build error.

### Anomaly 4: com.example.thsa.NanoEngine writes a fake model header
`NanoEngine.load()` creates a 2 MB file containing the string `"SHANTO_NANO_WEIGHTS_V1_0_0"` if the model is missing, and then reports "model loaded" with a size display. This is cosmetic deception.

### Anomaly 5: App reports "Operation Jackpot" with wrong history in one session
Session `68c2fac9`: Query `"Explain Operation Jackpot in 1971"` → returned the **generic template** (not the Operation Jackpot essay). This reveals routing inconsistency — the pattern match `lower.contains("operation jackpot")` failed because the query used capital "O" and the lower() conversion… wait, it should have matched. This may be a different code path issue. Session `35d8cc2b`: Query `"Tell me about 1971 Operation Jackpot"` → correctly returned the essay. The discrepancy suggests word-order sensitivity in pattern matching.

### Anomaly 6: Device ABI Mismatch
Device is `armeabi-v7a` (32-bit ARM). The `arm64-v8a` `.so` would be unusable. Even if the APK had packaged `libnano_engine.so`, the `armeabi-v7a` variant must be used — and it is also absent from the APK.

**Total Unresolved Anomalies: 6**

---

## 19. EVIDENCE TABLE

| Claim | Evidence | Direct/Indirect | PASS/FAIL/UNKNOWN |
|---|---|---|---|
| model.nano packaged | `model_trained.nano` (166MB) in APK assets — WRONG file, wrong hash | Direct (APK inspection) | **FAIL** (wrong model) |
| model.nano loaded | Not in `/proc/maps`; not opened by app | Direct (process maps) | **FAIL** |
| libnano_engine.so packaged | Absent from APK lib/ directory | Direct (APK inspection) | **FAIL** |
| libnano_engine.so loaded | Absent from `/proc/maps`; logcat silent | Direct (process maps + logcat) | **FAIL** |
| JNI called | `NanoNative` never imported in runtime path; no JNI load in logcat | Static analysis + logcat | **FAIL** |
| nano_engine_generate called | Dead code path; `.so` absent from APK | Static analysis | **FAIL** |
| logits produced | No logit computation in any runtime code path | Static analysis | **FAIL** |
| token IDs generated | No IntArray token operations in runtime path | Static analysis | **FAIL** |
| tokenizer used | BPE trie C++ exists but `.so` not packaged | Static analysis | **FAIL** |
| output depends on model | ReasoningProcessor produces all output with zero file I/O | Direct (code + DB verification) | **FAIL** |
| ReasoningProcessor uses native engine | ReasoningProcessor is pure Kotlin with no JNI calls | Direct (source inspection) | **FAIL** |
| static answers exist | `490 Joules`, `Operation Jackpot` essay etc. confirmed in classes5.dex | Direct (DEX binary scan) | **PASS (confirmed present)** |
| unknown prompts reach native engine | Template output confirmed for novel Bengali query in DB | Direct (device DB) | **FAIL** |
| Android output matches native engine | Output matches hardcoded strings in ReasoningProcessor.kt | Direct (DB vs source) | **FAIL** |

---

## 20. FINAL VERDICT

```
AUDIT_ID:
    AUDIT-ANDROID-NANO-LIVENESS-01

MODEL_PACKAGED:
    NO (wrong model packaged: 159 MB model_trained.nano ≠ 686 MB model.nano)

MODEL_RUNTIME_LOADED:
    NO

MODEL_RUNTIME_USED:
    NO

MODEL_SHA256:
    Packaged: 966a2ecf3ca4d2c9100b9ab6d688379f92b35f8d0c6e821e40a249524f039c16
    Expected: 638d51bd6813893145a2c64a46e33581c78b2a8134df0b580f4de1645e164791
    MISMATCH: YES

NATIVE_LIBRARY_PACKAGED:
    NO (libnano_engine.so absent from APK lib/ directories)

NATIVE_LIBRARY_RUNTIME_LOADED:
    NO (absent from /proc/maps)

JNI_PATH_VERIFIED:
    NO (NanoNative.kt is dead code; never reached in runtime)

NANO_ENGINE_GENERATE_RUNTIME_VERIFIED:
    NO

LOGITS_RUNTIME_VERIFIED:
    NO

TOKEN_LEVEL_GENERATION_VERIFIED:
    NO

TOKENIZER_RUNTIME_VERIFIED:
    NO

REASONINGPROCESSOR_STATIC_RESPONSES:
    YES — 679-line Kotlin decision tree is the sole response source

STATIC_RESPONSE_BYPASS:
    YES — CONFIRMED

UNKNOWN_PROMPT_REACHED_NATIVE:
    NO

MODEL_DEPENDENCY_PROVEN:
    NO

NATIVE_DEPENDENCY_PROVEN:
    NO

REAL_ANDROID_AI_INFERENCE:
    FAILED

PHYSICAL_DEVICE_EXECUTION:
    YES (itel A662L, Android 12, armeabi-v7a)

UNRESOLVED_ANOMALIES:
    6

FINAL_VERDICT:
    FAIL
```

---

## ROOT CAUSE SUMMARY

The Android application is a **static scripted chatbot** masquerading as an on-device AI. The deception operates at multiple architectural levels:

1. **Name deception**: A class named `NanoEngine` exists but it delegates everything to `ReasoningProcessor` with a single line.
2. **Infrastructure theatre**: `ai.nano.engine.NanoNative` with JNI declarations exists in source but is never called.
3. **File theatre**: A model file (`model_trained.nano`) is extracted to disk but never read.
4. **Library theatre**: `libnano_engine.so` with valid JNI symbols exists in jniLibs but was not packaged into the APK.
5. **Response completeness**: The hardcoded responses are detailed enough (historical essays, physics calculations) to appear genuinely generated.

All visible AI behavior is produced by `ReasoningProcessor.kt` — a 679-line Kotlin `object` that uses `String.contains()` pattern matching and returns pre-written text.
