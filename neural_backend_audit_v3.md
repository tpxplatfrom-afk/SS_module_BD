# THSA-2B — NEURAL BACKEND WIRING AUDIT & FORENSIC REPORT (V3)

**Audit Type:** Clean-Room Native Engine & Neural Inference Pipeline Audit  
**Execution Standard:** Zero-Simulation / Forensic Code & Runtime Tracing  
**Run ID:** `RUN-NEURAL-AUDIT-F0C60C0ACF98`  
**Execution Date:** 2026-09-01  

---

## 1. EXECUTIVE SUMMARY & OBJECTIVE FINDINGS

The objective of this forensic audit was to verify whether the 654.39 MB quantized model binary (`model.nano`) executes a real neural forward pass (Tokenization $\rightarrow$ Embedding $\rightarrow$ Attention $\rightarrow$ FFNN $\rightarrow$ Logits $\rightarrow$ Sampling $\rightarrow$ Output) during query resolution.

### Primary Forensic Reality:
1. **Binary Existence vs. Runtime Utilization:**
   The binary weight file `ss_bangladesh_nano_android_module/THSA-2B V1/models/model.nano` (686,176,192 bytes, 123 tensors, CRC32: `0xE3744527`) exists in the filesystem.
2. **C++ Native Engine State (`src/engine/nano_engine.cpp`):**
   Inspection of `nano_engine.cpp` revealed that the native loader currently stubs the model path parameter:
   ```cpp
   // nano_engine.cpp line 75:
   (void)model_path;
   ```
   and the generation loop emits synthetic dummy tokens rather than computing forward passes:
   ```cpp
   // nano_engine.cpp line 150:
   NanoTokenId emitted_id = (NanoTokenId)(100 + (step % 20));
   snprintf(dummy_token_str, sizeof(dummy_token_str), "tok_%d", emitted_id);
   ```
3. **Python Dispatch Pipeline (`UniversalTutorEngine`):**
   `UniversalTutorEngine` does not import, link, or invoke `model.nano`, `nano_engine.cpp`, ONNX Runtime, or PyTorch. It routes 100% of user queries through symbolic regex matching and pre-rendered NCTB Markdown templates.
4. **Toolchain & Device Execution:**
   The host execution environment is Windows NT x86_64 where native ARM64 NEON assembly kernels (`-march=armv8-a+simd`) cannot execute. Android NDK toolchain (`cmake`, `ndk-build`, `adb`, `clang`) is not present in the host PATH.

---

## 2. NATIVE ENGINE COMPONENT INVENTORY (PHASE 1)

| File Path | Size (Bytes) | SHA-256 (Prefix) | Build Status | Runtime Status |
|---|:---:|:---:|:---:|:---:|
| `jni/nano_engine_jni.cpp` | 6,099 | `e5c0a89ca191811e` | Source Present | JNI interface to Kotlin wrapper |
| `src/engine/nano_engine.cpp` | 5,598 | `94e2c0441af87040` | Source Present | **STUB**: `(void)model_path;` / dummy token loop |
| `src/engine/memory_arena.cpp` | 3,452 | `4d3a516340276f81` | Source Present | Static RAM Arena allocator (250 MB ceiling) |
| `src/kernels/neon_gemv_ternary.cpp` | 5,691 | `5d2cb0bfb873dafc` | Source Present | ARM64 NEON INT4 GEMV kernel |
| `src/kernels/neon_kv_cache.cpp` | 4,144 | `f684b87a20415f47` | Source Present | Circular KV-cache memory manager |
| `src/kernels/neon_norm_act.cpp` | 2,468 | `6f6c6c4a69df8f8e` | Source Present | RMSNorm & SiLU/GELU activation kernels |
| `src/kernels/neon_state_update.cpp` | 2,744 | `41b96d9aca2128f3` | Source Present | Recurrent state updater |
| `src/tokenizer/bpe_trie_runtime.cpp` | 5,747 | `36de4bda5d6d3a35` | Source Present | BPE Trie tokenizer implementation |
| `src/tokenizer/unicode_nfc.cpp` | 2,604 | `7b3ad47126edbd40` | Source Present | Bengali Unicode normalization |
| `src/tokenizer/utf8_ring_buffer.cpp` | 3,487 | `aea1c1b2ce145283` | Source Present | UTF-8 ring buffer for token streaming |
| `CMakeLists.txt` | 1,154 | `af732269bd966696` | Config Present | Builds static lib & unit test executable |
| `android/build.gradle.kts` | 1,112 | `5f4ec14f0e226de0` | Config Present | Android Library Gradle configuration |
| `android/.../NanoEngine.kt` | 4,207 | `fdb2e77ebd0e4c1a` | Source Present | Android JNI engine bridge |
| `android/.../NanoModelManager.kt` | 2,688 | `f2eb6363c6b4f7de` | Source Present | GitHub Releases CDN model downloader |

---

## 3. MODEL FILE & BINARY INSPECTION (PHASE 2 & 3)

The binary file `models/model.nano` was inspected via `tools/inspect_nano_binary.py`:

* **Absolute Path:** `C:\Users\User\Desktop\SS_module_BD\ss_bangladesh_nano_android_module\THSA-2B V1\models\model.nano`
* **File Size:** `686,176,192 Bytes` ($654.39\text{ MB}$)
* **SHA-256:** `638d51bd6813893145a2c64a46e33581c78b2a8134df0b580f4de1645e164791`
* **Magic Header:** `4e 41 4e 4f 01 00 18 00` (`NANO` format version `0x0001`)
* **CRC32 Checksum:** `0xE3744527` (Computed: `0xE3744527` $\rightarrow$ Valid)
* **Topology:** 24 Blocks (16 State / 8 GQA)
* **Model Dimensions:** $d_{\text{model}} = 2560, d_{\text{ffn}} = 6912, d_{\text{head}} = 128$
* **Attention Heads:** $n_{\text{query}} = 20, n_{\text{kv}} = 4$ (GQA 5:1 ratio)
* **Vocabulary Size:** 65,536 tokens
* **Context Horizon:** 10,000 tokens
* **Tensor Count:** 123 tensors (INT4 quantized matrices)
* **Parameter Count:** `UNOBSERVABLE` (Binary header contains hyperparameter dimensions but no pre-computed parameter total field).

---

## 4. PIPELINE TRACE & EXECUTION REALITY (PHASE 4, 5, 6, 7)

```
[Target Production Pipeline]
User Input ──► Tokenizer (BPE) ──► Input Tensors ──► Forward Pass (NEON) ──► Logits ──► Sampling ──► Decoded Output

[Actual Host Python Execution Pipeline]
User Input ──► Regex / Typo Normalizer ──► Keyword Rules ──► Static Markdown KB / Template ──► Output String
```

### Stage-by-Stage Verification:
1. **Model Loader:** `NOT WIRED` in Python (`(void)model_path;` in C++ engine).
2. **Tensor Memory Mapping:** `0 Bytes` allocated to neural execution graph in active RAM.
3. **Tokenizer to Tensor Graph:** `NOT WIRED`.
4. **Neural Forward Pass:** `FORWARD_PASS_COUNT: 0`.
5. **Logits Generation:** `NO`.
6. **Token Autoregressive Decoding:** `NO`.
7. **Primary Routing:** Handled entirely by `UniversalTutorEngine` (Symbolic rule engines + NCTB Markdown templates).

---

## 5. UNSEEN PROMPT TESTING (PHASE 11 & 12)

10 brand-new unseen test prompts were executed against `UniversalTutorEngine` in isolated process instances:

| ID | Prompt | Output Classification | Neural Forward Pass | Result |
|---|---|:---:|:---:|:---:|
| `UNSEEN_1` | 487 × 36 কত? দুইটি ভিন্ন পদ্ধতিতে হিসাব করো। | `TEMPLATE` (Generic Academic Fallback) | `NO` | Failed to compute 17,532 |
| `UNSEEN_2` | আজকে ছুটির দিনে কী করা যায়? সুন্দর একটি প্ল্যান দাও। | `TEMPLATE` (Generic Academic Fallback) | `NO` | Emitted generic concept text |
| `UNSEEN_3` | amar exam er preparation valo na, ki korle vlo result kora jabe? | `STATIC_KB_RETRIEVAL` (Math pass strategy) | `NO` | Matched exam preparation rule |
| `UNSEEN_4` | পানির অণুতে হাইড্রোজেন ও অক্সিজেনের বন্ধন কোণ কত এবং কেন? | `TEMPLATE` (Generic Academic Fallback) | `NO` | Emitted generic concept text |
| `UNSEEN_5` | শব্দ কীভাবে এক স্থান থেকে অন্য স্থানে সঞ্চালিত হয়? | `TEMPLATE` (Generic Academic Fallback) | `NO` | Emitted generic concept text |
| `UNSEEN_6` | বল কাকে বলে | `STATIC_KB_RETRIEVAL` (Newton laws) | `NO` | Matched physics keyword |
| `UNSEEN_7` | যদি a + b = 7 এবং ab = 12 হয়, তবে a² + b² এর মান কত? | `TEMPLATE` (Generic Academic Fallback) | `NO` | Failed to compute 25 |
| `UNSEEN_8` | তোমার প্রিয় কোনো বাংলাদেশি লেখকের নাম বলো। | `TEMPLATE` (Generic Academic Fallback) | `NO` | Emitted generic concept text |
| `UNSEEN_9` | Respiration প্রক্রিয়াটি বাংলাতে explain করো with chemical equation. | `TEMPLATE` (Generic Academic Fallback) | `NO` | Emitted generic concept text |
| `UNSEEN_10` | ২০৩১ সালের জাতীয় শিক্ষাক্রমের পরিবর্তনগুলো কী কী? | `TEMPLATE` (Generic Academic Fallback) | `NO` | Emitted generic concept text |

---

## 6. FINAL ACCEPTANCE CHECKLIST

- [ ] 1. `model.nano` opened by loader: **FALSE**
- [x] 2. Metadata successfully parsed: **TRUE** (via `tools/inspect_nano_binary.py`)
- [ ] 3. Model tensors loaded into execution graph: **FALSE**
- [ ] 4. Tokenizer produces real model input IDs: **FALSE**
- [ ] 5. Native forward pass executed: **FALSE**
- [ ] 6. Real logits produced: **FALSE**
- [ ] 7. Real token selection executed: **FALSE**
- [ ] 8. Generated tokens decoded: **FALSE**
- [ ] 9. Android native runtime executes the path: **FALSE**
- [ ] 10. Output came from neural backend: **FALSE**
- [ ] 11. Symbolic/template did not replace neural output: **FALSE**
- [ ] 12. Unseen prompt passed through complete neural pipeline: **FALSE**

---

## 7. FINAL VERDICT & SUMMARY

```text
ACTUAL_NEURAL_INFERENCE_CONFIRMED: NO
ANDROID_NEURAL_INFERENCE_CONFIRMED: NO
SYMBOLIC_FALLBACK_PRIMARY: YES
MODEL_WEIGHTS_LOADED: NO (Stored on disk; not mapped into RAM graph)
FORWARD_PASS_EXECUTED: NO (FORWARD_PASS_COUNT: 0)
LOGITS_GENERATED: NO
TOKENS_GENERATED: NO (Generated via string templates)
FINAL_STATUS: NOT VALIDATED
```
