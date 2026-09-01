# THSA-2B V1: Master Implementation Roadmap & Engineering Execution Plan
## End-to-End Delivery Architecture for Android Native On-Device Inference Engine

**Document Identifier:** `PLAN-THSA2B-IMPL-001`  
**Revision:** `1.0.0` (Production Baseline Roadmap)  
**Target Subsystem:** `ss_bangladesh_nano_android_module/THSA-2B V1`  
**Standard Compliance:** RFC 2119 (MUST, MUST NOT, SHOULD, SHOULD NOT, MAY)  
**Governing Architecture:** `NANO_MODULE_FINAL_ARCHITECTURE.md` (Revision `3.4.0`)  
**Governing Goal:** `NANO_MODULE_GOAL.md` (Revision `2.0.0`)  

---

## 1. Executive Summary & Architectural Compliance Mandates

This document establishes the binding, step-by-step engineering roadmap for implementing the **THSA-2B (Ternary Hybrid State-Attention 2B)** on-device inference engine from micro-kernel foundation to production Android deployment.

### 1.1 Non-Negotiable Resource Quadrilateral Invariants
Every line of code implemented across all phases MUST strictly adhere to the following hard system constraints:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THSA-2B HARD SYSTEM CONSTRAINTS (INVARIANTS)                 │
├────────────────────────────────┬────────────────────────────────────────────────┤
│ **Physical Working RAM Ceiling**│ **<= 250.0 MB** (Preferred <= 200.0 MB)        │
│ **Flash Storage / ROM Footprint**│ **<= 1.0 GB** (Serialized model: 400 - 500 MB) │
│ **Context Horizon Capacity**   │ **10,000 tokens** (Zero silent truncation)     │
│ **Thermal & Energy Envelope**   │ **<= 45.0°C sustained**, **2.0 - 3.5 mJ/token**│
│ **Compute Hardware Tier**       │ **ARM64 CPU + NEON SIMD** (No NPU/GPU reliance)│
│ **Model Scale Target**         │ **~2.0 Billion parameters** (1.95B - 2.00B)    │
│ **Multilingual Tokenizer**     │ **V = 65,536** (English + Bengali native)       │
└────────────────────────────────┴────────────────────────────────────────────────┘
```

---

## 2. Target Directory & Module Structure Layout

All native code, training pipelines, exporter tools, and Android bridge bindings will reside in the following structured directory tree:

```
ss_bangladesh_nano_android_module/THSA-2B V1/
├── CMakeLists.txt                         # Top-level NDK CMake build configuration
├── THSA_2B_MASTER_IMPLEMENTATION_PLAN.md  # This master implementation roadmap
│
├── include/                               # Public C/C++ Header Interfaces
│   ├── nano_engine.h                      # Primary native C engine API
│   ├── nano_config.h                      # ModelConfig struct & scale descriptors
│   ├── nano_types.h                       # Data types, error codes, and enums
│   ├── nano_telemetry.h                   # Telemetry and observability packet struct
│   └── nano_tokenizer.h                   # Tokenizer C runtime interface
│
├── src/                                   # Native C++ Implementation Core
│   ├── engine/
│   │   ├── nano_engine.cpp                # Engine lifecycle, state machine, decode loop
│   │   ├── memory_arena.cpp               # Static memory arena allocator & zero-leak RAII
│   │   ├── dma_ring_loader.cpp            # 16 MB double-buffered mmap streaming loader
│   │   └── session_manager.cpp            # Context reset, state continuity, rolling window
│   │
│   ├── kernels/                           # High-Performance ARM64 NEON Kernels
│   │   ├── neon_gemv_ternary.cpp          # INT8 activation x Ternary {-1,0,+1} dot products
│   │   ├── neon_kv_cache.cpp              # INT4 grouped KV quant/dequant & attention logits
│   │   ├── neon_state_update.cpp          # 1D causal short-conv & linear state update
│   │   ├── neon_norm_act.cpp              # Vectorized RMSNorm and fast SwiGLU activations
│   │   └── neon_sampling.cpp              # Top-K, Top-P, Temperature & Argmax samplers
│   │
│   └── tokenizer/
│       ├── bpe_trie_runtime.cpp           # Compact C++ trie tokenizer runtime (<= 8 MB)
│       ├── unicode_nfc.cpp                # Deterministic Unicode NFC normalizer
│       └── utf8_ring_buffer.cpp           # 16-byte multi-byte accumulation stream buffer
│
├── jni/                                   # Android JNI Native Bridge
│   ├── nano_engine_jni.cpp                # JNI bindings, error translation, local frame guards
│   └── nano_telemetry_jni.cpp             # JNI telemetry getter bindings
│
├── android/                               # Kotlin Android Library Module
│   ├── src/main/java/ai/nano/engine/
│   │   ├── NanoEngine.kt                  # Kotlin main API wrapper (StateFlow & Coroutines)
│   │   ├── NanoConfig.kt                  # Model configuration builder & presets
│   │   ├── NanoTelemetry.kt               # Kotlin data class for runtime observability
│   │   └── NanoEngineException.kt         # Typed exception hierarchy for native error codes
│   └── build.gradle.kts                   # Android library build script
│
├── tools/                                 # Conversion, Exporter & Packaging Tools
│   ├── export_to_nano.py                  # PyTorch checkpoint -> 64-byte aligned .nano binary
│   ├── inspect_nano_binary.py             # Binary header, CRC32, and tensor metadata validator
│   └── calibrate_quantization.py          # Post-training QAT calibration & scale generator
│
├── training/                              # Pre-Training & Distillation Pipelines
│   ├── config/
│   │   ├── proxy_350m_config.json         # 350M micro-THSA proxy pilot configuration
│   │   └── thsa_2b_config.json            # Full 2B THSA production model configuration
│   ├── models/
│   │   ├── thsa_hybrid_model.py           # PyTorch THSA-2B architecture definition
│   │   ├── ternary_layers.py              # BitNet-style ternary linear layers with STE
│   │   └── state_conv_block.py            # Short-conv / SSD state block implementations
│   ├── distillation/
│   │   ├── teacher_ensemble.py            # Sarvam-1 (Indic) + Qwen-2.5 (Reasoning) wrapper
│   │   └── distillation_loss.py           # Combined CE + KL soft-label distillation loss
│   └── train_qat.py                       # Temperature-annealed QAT training loop
│
└── tests/                                 # Multi-Tier Verification & Benchmark Suite
    ├── unit/
    │   ├── test_neon_kernels.cpp          # Bit-exact NEON vs scalar differential tests
    │   ├── test_memory_arena.cpp          # Static arena bounds & zero-leak alloc/free tests
    │   ├── test_tokenizer.cpp             # Unicode NFC, Bengali fertility & UTF-8 buffer tests
    │   └── test_binary_loader.cpp         # .nano format header parsing & CRC32 tests
    ├── integration/
    │   ├── test_multi_turn_dialogue.cpp   # 500+ turn attention-sink rolling window test
    │   ├── test_async_cancellation.cpp    # <= 5 ms thread-safe cancellation test
    │   └── test_load_unload_leak.cpp      # 1,000 cycle load/unload RAII leak test
    └── benchmarks/
        ├── benchmark_gemv_neon.cpp        # GFLOPS, memory bandwidth & latency benchmarks
        └── benchmark_android_device.py    # Automated physical Android device test runner
```

---
## 3. PHASE 1: Mathematical & Physical Validation Baseline (STATUS: COMPLETED ✅)

### 3.1 Objectives & Deliverables
Phase 1 formulated, simulated, and proved the mathematical and physical feasibility of all THSA-2B core assumptions prior to writing micro-kernel code.

### 3.2 Executed Validation Tests & Results
All 3 simulation engines located in the workspace root were executed and achieved 100% PASS against Revision 3.4.0:

1. **`01_memory_model_validator.py` (Memory Model):**
   - **Primary 16/8 Topology:** Working RAM = **$229.06	ext{ MB}$** (Safety Margin = $+20.94	ext{ MB}$ under $250.0	ext{ MB}$ ceiling).
   - **Elastic Fallback 12/12 Topology:** Working RAM = **$248.59	ext{ MB}$** (Safety Margin = $+1.41	ext{ MB}$).
   - **Context Scalability Tiers:** 4K ($206.0	ext{ MB}$), 8K ($222.0	ext{ MB}$), 10K ($229.1	ext{ MB}$), 20K KIVI mode ($240.0	ext{ MB}$).

2. **`02_quantization_error_simulator.py` (Quantization Stability):**
   - **Sensitive Layer Shielding (Bridge 1):** INT8/INT4 protection on Embeddings and LM Head keeps bulk ternary error to $\le 1.85\%$.
   - **Cascading 24-Layer Error:** $2.57\%$, yielding an end-to-end **Perplexity Loss of $1.49\%$** ($\le 5.0\%$ target SLA).

3. **`03_energy_thermal_simulator.py` (Energy & Thermals):**
   - **Human-Paced DVFS (10-12 tok/s @ 1.8 GHz):** Total system power = **$1.47	ext{ W}$** (vs $3.5	ext{ W}$ peak limit).
   - **Steady-State Chassis Temp:** **$36.0^\circ	ext{C}$** ($\le 45.0^\circ	ext{C}$ threshold, $+9.0^\circ	ext{C}$ margin).
   - **Kernel Compute Energy:** **$2.52	ext{ mJ / token}$** ($2.0 - 3.5	ext{ mJ/token}$ target).
   - **Battery Drain:** **$4.78\% / 	ext{hour}$** during conversational dialogue ($\le 5.0\%/	ext{hr}$ SLA).

4. **Master Orchestrator (`run_phase1_validation.py`):** Unified pass decision authorizing Phase 2.

---

## 4. PHASE 2: Core Native Engine & ARM64 NEON Vector Micro-Kernels

### 4.1 Phase Overview & SLA Targets
Phase 2 builds the high-performance C++17 / ARM64 NEON micro-kernel foundation and the static memory arena allocator. All compute paths run entirely in native code with zero third-party framework dependencies.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2 PERFORMANCE TARGETS                           │
├────────────────────────────────┬────────────────────────────────────────────────┤
│ **Ternary GEMV SIMD Speed**    │ **>= 45.0 GFLOPS equivalent** (ARM Cortex-A78) │
│ **INT4 Grouped KV Dequant**    │ **<= 0.8 ms** per 10,000-token attention layer │
│ **Short-Conv State Update**    │ **<= 0.15 ms** per 16 state blocks             │
│ **Chunked Prefill Throughput** │ **>= 40.0 tokens / sec** (256-token chunk)     │
│ **Instantaneous Decode Latency**│ **<= 90.0 ms / token** (11.0 tokens/sec sustained)│
│ **Working Memory Allocation**  │ **Zero `malloc`/`free` during decode loop**    │
└────────────────────────────────┴────────────────────────────────────────────────┘
```

### 4.2 Phase 2 Sub-Module Breakdown

#### Phase 2A: High-Performance Ternary GEMV/GEMM NEON Kernel
* **Source Files:**
  - `include/kernels/neon_gemv_ternary.h`
  - `src/kernels/neon_gemv_ternary.cpp`
  - `tests/unit/test_neon_kernels.cpp`
* **Mathematical Formulation:**
  $$\mathbf{y} = lpha \cdot (\mathbf{W}_{	ext{ternary}} \cdot \mathbf{x}_{	ext{int8}}) + 	ext{bias}$$
  Where $\mathbf{W} \in \{-1, 0, +1\}^{M 	imes N}$, packed at 2 bits per weight or unrolled in-register.
* **NEON Implementation Details:**
  - Uses `sdot` / `saddw` / `tbl` NEON instructions to replace hardware multipliers with vector add/sub pipelines.
  - Processes 16 weights per NEON vector register (`int8x16_t`).
  - Implements 4-way loop unrolling across output channels to saturate dual-issue SIMD pipelines.
* **Verification Gate (`GATE-NEON-001`):** Bit-exact match against scalar double-precision reference ($\Delta \le 1	ext{ ULP}$).

#### Phase 2B: INT4 Grouped KV-Cache SIMD & Attention-Sink Rolling Arena
* **Source Files:**
  - `include/kernels/neon_kv_cache.h`
  - `src/kernels/neon_kv_cache.cpp`
  - `tests/unit/test_kv_cache.cpp`
* **Mathematical Formulation:**
  $$	ext{Logit}_i = rac{\mathbf{q} \cdot 	ext{Dequant}_{	ext{int4}}(\mathbf{k}_i)}{\sqrt{d_{	ext{head}}}}, \quad \mathbf{y} = \sum_i 	ext{Softmax}(	ext{Logit}_i) \cdot 	ext{Dequant}_{	ext{int4}}(\mathbf{v}_i)$$
* **Implementation Details:**
  - KV-Cache packed at 4 bits per element (2 elements per byte).
  - Grouped Query Attention (GQA): 20 Query heads share 4 KV heads ($5:1$ ratio).
  - Vectorized INT4 unpack + Fused FP16 dot-product kernel using `vld1_u8`, `vshl_n_u8`, `vmla_f16`.
  - Attention-Sink circular FIFO: 4 sink tokens permanently pinned at slots `0..3`.
* **Verification Gate (`GATE-KV-001`):** Attention score reconstruction KL divergence $\le 0.0120$ over 10K tokens.

#### Phase 2C: 16-Block Linear State/Short-Conv Kernel & DMA Ring Loader
* **Source Files:**
  - `include/kernels/neon_state_update.h`
  - `src/kernels/neon_state_update.cpp`
  - `src/engine/dma_ring_loader.cpp`
  - `tests/unit/test_state_kernel.cpp`
* **Implementation Details:**
  - 1D causal depthwise convolution across temporal window $K=4$ with SiLU gating.
  - Memory consumption: Strictly $O(1)$ state memory ($\le 128	ext{ KB}$ total across all 16 state blocks).
  - **16 MB Double-Buffered DMA Ring Loader (Bridge 2):** Two 16 MB ping-pong RAM buffers. While compute kernel processes Block $N$ in Buffer A, asynchronous kernel prefetch thread reads Block $N+1$ into Buffer B via `mmap` / `O_DIRECT`.
* **Verification Gate (`GATE-DMA-001`):** Zero page-fault stalls ($< 0.5	ext{ ms}$ overhead per block transition).

#### Phase 2D: Chunked Streaming Prefill & Vectorized Normalization/Activation
* **Source Files:**
  - `include/kernels/neon_norm_act.h`
  - `src/kernels/neon_norm_act.cpp`
  - `src/engine/memory_arena.cpp`
  - `tests/unit/test_memory_arena.cpp`
* **Implementation Details:**
  - Vectorized RMSNorm: $\mathbf{y} = rac{\mathbf{x}}{\sqrt{rac{1}{d} \sum x_i^2 + \epsilon}} \odot \gamma$ using NEON `vrsqrteq_f32`.
  - Vectorized SwiGLU: $	ext{SwiGLU}(x) = (x \cdot 	ext{sigmoid}(eta x)) \odot 	ext{gate}$.
  - **Chunked Prefill Pipeline (Bridge 5):** Evaluates prompts in 256-token micro-chunks, keeping working activation RAM capped at $\le 25.0	ext{ MB}$ regardless of prompt length.
  - **Static Memory Arena Allocator:** Single monolithic `mmap` allocation at engine startup. Zero dynamic heap allocations (`malloc`/`free`) during generation.
* **Verification Gate (`GATE-ARENA-001`):** Zero memory leaks and total working RSS $\le 229.1	ext{ MB}$.

---
## 5. PHASE 3: Multilingual Tokenizer Runtime & 350M Proxy Pilot Training

### 5.1 Phase Overview & SLA Targets
Phase 3 delivers the compact C++ multilingual tokenizer runtime and trains the 350M micro-THSA proxy model to empirically prove QAT training stability on 50B tokens before committing full 2B pretraining compute.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 3 PERFORMANCE TARGETS                           │
├────────────────────────────────┬────────────────────────────────────────────────┤
│ **Tokenizer Resident Memory**  │ **<= 8.0 MB** (Compact Trie table in RAM)      │
│ **Encoding Throughput**        │ **<= 5.0 ms** per 1,000 characters             │
│ **Bengali Subword Fertility**  │ **<= 1.8 tokens / word** (Target: 1.4 - 1.8)   │
│ **English Subword Fertility**  │ **<= 1.2 tokens / word**                       │
│ **350M Proxy Training Tokens** │ **50 Billion high-quality multilingual tokens**│
│ **Proxy QAT Perplexity Gap**   │ **<= 4.0% loss** vs unquantized FP16 baseline  │
└────────────────────────────────┴────────────────────────────────────────────────┘
```

### 5.2 Phase 3 Sub-Module Breakdown

#### Phase 3A: C++ Embedded Trie Tokenizer Runtime
* **Source Files:**
  - `include/nano_tokenizer.h`
  - `src/tokenizer/bpe_trie_runtime.cpp`
  - `src/tokenizer/unicode_nfc.cpp`
  - `src/tokenizer/utf8_ring_buffer.cpp`
  - `tests/unit/test_tokenizer.cpp`
* **Implementation Details:**
  - Fixed vocabulary size $V = 65{,}536$ tokens (36.6% English, 30.5% Bengali, 18.3% Code, 14.2% Shared roots, 0.4% Special/Byte fallback).
  - **Unicode NFC Normalizer:** Deterministically combines decomposed Bengali vowel signs (কার / ফলা) and conjuncts before BPE lookup.
  - **Single-Token Complex Conjuncts:** Direct token IDs for high-frequency Bengali conjuncts (`ক্ষ`, `জ্ঞ`, `ঞ্চ`, `ম্ভ`, `ন্ত্র`), ensuring subword fertility $\le 1.8	ext{ tok/word}$.
  - **16-Byte UTF-8 Streaming Accumulation Ring Buffer:** Reassembles multi-byte UTF-8 character boundaries during streaming emission, preventing character corruption on client displays.
* **Verification Gate (`GATE-TOK-001`):** 100% tokenization coverage with zero `<|unk|>` tokens across a 100,000-character mixed English/Bengali test corpus.

#### Phase 3B: 350M Micro-THSA QAT Pre-Flight Training (Bridge 4)
* **Source Files:**
  - `training/config/proxy_350m_config.json`
  - `training/models/thsa_hybrid_model.py`
  - `training/models/ternary_layers.py`
  - `training/distillation/teacher_ensemble.py`
  - `training/distillation/distillation_loss.py`
  - `training/train_qat.py`
* **Architecture Specifications (350M Proxy):**
  - 14 Backbone Blocks (10 State / 4 GQA), $d_{	ext{model}} = 1024$, $d_{	ext{ffn}} = 2764$, $N_q = 8$, $N_{kv} = 2$.
* **Training Protocol:**
  - **Dataset:** 50 Billion tokens (60% English/Code, 30% Bengali, 10% Math/Socratic reasoning).
  - **Teacher-Student Distillation:** Soft-label loss distillation from `Sarvam-1 (2B)` for Bengali syntax and `Qwen-2.5-7B` for math logic:
    $$\mathcal{L}_{	ext{total}} = (1-lpha)\mathcal{L}_{	ext{CE}} + lpha \cdot 	au^2 \mathcal{D}_{	ext{KL}}(p_{	ext{student}} \,||\, p_{	ext{teacher}}), \quad lpha=0.65, \, 	au=2.0$$
  - **Temperature-Annealed QAT:** Smooth ternary relaxation $	ilde{W} = 	anh(eta \cdot W)$ where $eta: 1 ightarrow 100$ over 10K steps, backed by Straight-Through Estimator (STE).
* **Verification Gate (`GATE-PROXY-001`):** Proxy QAT perplexity degradation $\le 4.0\%$ and Needle-In-A-Haystack retrieval score $\ge 95.0\%$ over 8K context.

---

## 6. PHASE 4: Full-Scale 2B Pre-Training, QAT Annealing & .nano Exporter

### 6.1 Phase Overview & SLA Targets
Phase 4 scales the validated QAT training pipeline to the full ~2B parameter THSA architecture across 2.0 Trillion tokens, exports the trained weights, and serializes them into the SIMD-aligned `.nano` binary format.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 4 PERFORMANCE TARGETS                           │
├────────────────────────────────┬────────────────────────────────────────────────┤
│ **Total Parameter Count**      │ **1.95B - 2.00B parameters**                   │
│ **Pre-Training Token Horizon** │ **2.0 Trillion high-quality tokens**           │
│ **QAT Annealing Schedule**     │ **Final 20% of tokens (20k - 50k steps)**      │
│ **Serialized Model Size**      │ **400 - 500 MB** (.nano binary package)        │
│ **File Header & CRC Overhead** │ **<= 64 KB total metadata**                    │
│ **Cache-Line SIMD Alignment**  │ **100% of weight tensors on 64-byte boundary** │
└────────────────────────────────┴────────────────────────────────────────────────┘
```

### 6.2 Phase 4 Sub-Module Breakdown

#### Phase 4A: 2B Pre-Training & QAT Annealing
* **Source Files:**
  - `training/config/thsa_2b_config.json`
  - `training/train_qat.py`
  - `tools/calibrate_quantization.py`
* **Architecture Configuration:**
  - 24 Backbone Blocks (16 State / 8 GQA), $d_{	ext{model}} = 2560$, $d_{	ext{ffn}} = 6912$, $N_q = 20$, $N_{kv} = 4$, $d_{	ext{head}} = 128$.
* **Curriculum Strategy:**
  - First 80% (1.6T tokens): Standard FP16 / INT8 mixed-precision base pre-training.
  - Final 20% (400B tokens): Continuous QAT fine-tuning with Sensitive Layer Shielding (Embeddings & LM Head in INT8/INT4, FFN in ternary).
* **Verification Gate (`GATE-TRAIN-001`):** 2B Quantized model perplexity $\le 10.50$ (relative degradation $\le 2.0\%$ vs FP16 baseline).

#### Phase 4B: Weight Exporter & Cache-Aligned `.nano` Serializer
* **Source Files:**
  - `tools/export_to_nano.py`
  - `tools/inspect_nano_binary.py`
  - `src/engine/nano_engine.cpp`
  - `tests/unit/test_binary_loader.cpp`
* **Binary File Structure (`model.nano`):**
  - **64-Byte Header:** Magic `0x4E414E4F` ("NANO"), format version `0x0001`, CRC32 checksum, dimension descriptors (`total_blocks=24`, `d_model=2560`, `vocab_size=65536`).
  - **Tensor Descriptor Table:** Array of `{uint32_t tensor_id, uint64_t offset, uint64_t size_bytes, uint8_t quant_type}` entries.
  - **Tensor Payload Area:** All weight offsets strictly aligned to **64 bytes** (ARM Cortex-A cache line boundary) to enable direct zero-copy NEON vector load instructions without alignment faults.
* **Verification Gate (`GATE-BIN-001`):** Binary passes CRC32 validation, all tensor offsets $\equiv 0 \pmod{64}$, file size $\le 500	ext{ MB}$.

---
## 7. PHASE 5: Android JNI Bridge, Engine State Machine & Kotlin API

### 7.1 Phase Overview & SLA Targets
Phase 5 develops the production JNI bridge, native lifecycle state machine, asynchronous token cancellation, and clean Kotlin Coroutine / Flow API for Android client applications.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 5 PERFORMANCE TARGETS                           │
├────────────────────────────────┬────────────────────────────────────────────────┤
│ **Native Library Binary Size** │ **<= 6.0 MB** (Stripped `libnano_engine.so`)   │
│ **Cold-Start Init Latency**    │ **<= 150 ms** (`T_init` to `READY` state)      │
│ **Async Cancellation Response**│ **<= 5.0 ms** from UI button tap to halt       │
│ **Telemetry Query Latency**    │ **<= 0.1 ms** non-blocking atomic getter       │
│ **JNI Local Reference Floor**  │ **<= 16 active local frames** (ART limit: 512) │
│ **Memory Leaks (1k cycles)**   │ **0.0 KB** resident RSS delta across reloads   │
└────────────────────────────────┴────────────────────────────────────────────────┘
```

### 7.2 Phase 5 Sub-Module Breakdown

#### Phase 5A: Thread-Safe C++ Engine State Machine & JNI Bridge
* **Source Files:**
  - `include/nano_engine.h`
  - `include/nano_telemetry.h`
  - `src/engine/nano_engine.cpp`
  - `src/engine/session_manager.cpp`
  - `jni/nano_engine_jni.cpp`
  - `jni/nano_telemetry_jni.cpp`
* **Implementation Details:**
  - **Engine State Machine:** `UNINITIALIZED` -> `INITIALIZING` -> `READY` -> `GENERATING` -> `PAUSED` -> `ERROR`.
  - **Thread-Safety Model:** Thread-affine decode loop pinned to dedicated high-priority background thread; atomic command queue for UI interaction.
  - **Async Cancellation Flag:** `std::atomic<bool> cancel_requested` polled at every token boundary; halts execution in $\le 5	ext{ ms}$ with zero state corruption.
  - **Android ZRAM Hinting:** Issues `madvise(MADV_DONTNEED)` on processed DMA pages.
  - **Native Telemetry Struct:** Populates `NanoEngineTelemetry` (RSS bytes, active KV slots, decode tok/s, chassis temperature).
  - **Scoped JNI Frames:** Wraps all streaming token emission callbacks inside `env->PushLocalFrame(16)` / `env->PopLocalFrame()`.
* **Verification Gate (`GATE-JNI-001`):** Zero JNI local reference leaks and thread-safe cancellation in $\le 5.0	ext{ ms}$.

#### Phase 5B: Kotlin Coroutine / Flow API Wrapper
* **Source Files:**
  - `android/src/main/java/ai/nano/engine/NanoEngine.kt`
  - `android/src/main/java/ai/nano/engine/NanoConfig.kt`
  - `android/src/main/java/ai/nano/engine/NanoTelemetry.kt`
  - `android/src/main/java/ai/nano/engine/NanoEngineException.kt`
  - `android/build.gradle.kts`
* **Kotlin Interface Definition:**
  ```kotlin
  class NanoEngine private constructor(private val nativeHandle: Long) : AutoCloseable {
      fun generateStream(prompt: String, config: GenerationConfig = GenerationConfig.DEFAULT): Flow<String>
      fun cancelGeneration()
      fun resetSession()
      fun getTelemetry(): NanoTelemetry
      override fun close()
  }
  ```
* **Verification Gate (`GATE-KOTLIN-001`):** Seamless streaming output in Android UI with zero ANR (Application Not Responding) events.

---

## 8. PHASE 6: Hardware-in-the-Loop Multi-SoC Test Farm Benchmarking & Certification

### 8.1 Phase Overview & SLA Targets
Phase 6 validates the compiled native engine across a physical multi-SoC Android device matrix (Snapdragon, Dimensity, Exynos, Tensor) under sustained worst-case operational loads.

### 8.2 Physical Test Matrix & Pass/Fail Criteria

| Test Suite ID | Test Description | Target SLA / Quality Gate | Verification Script / Tool |
| :--- | :--- | :--- | :--- |
| **`TEST-MEM-001`** | **Peak RAM Ceiling** | $\le 250.0	ext{ MB}$ Working RSS @ 10K context | `benchmarks/benchmark_android_device.py` |
| **`TEST-STB-001`** | **Long-Session Stability** | Memory delta $\le 1.0	ext{ MB}$ across 500+ turns | `tests/integration/test_multi_turn_dialogue.cpp` |
| **`TEST-LEAK-001`** | **Repeated Load/Unload** | $0.0	ext{ KB}$ RSS growth across 1,000 load/unloads | `tests/integration/test_load_unload_leak.cpp` |
| **`TEST-THM-001`** | **Thermal Equilibrium** | Chassis surface $\le 45.0^\circ	ext{C}$ after 60 min | Physical thermal camera & sysfs logging |
| **`TEST-LAT-001`** | **TTFT & Decode Speed** | TTFT $\le 165	ext{ ms}$ (100 tok), Decode $\ge 10.0	ext{ tok/s}$ | `benchmarks/benchmark_gemv_neon.cpp` |
| **`TEST-BNG-001`** | **Bengali Subword Fertility**| $\le 1.8	ext{ tokens / word}$ on Bengali news/QA | `tests/unit/test_tokenizer.cpp` |
| **`TEST-QAT-001`** | **Retrieval Accuracy** | $\ge 95.0\%$ Needle-In-A-Haystack retrieval @ 10K | 10K NIAH synthetic evaluation run |

---

## 9. Traceability Matrix & Master Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THSA-2B V1 MASTER IMPLEMENTATION TIMELINE                    │
├───────┬──────────────────────────────────────────┬───────────┬──────────────────┤
│ Phase │ Milestone / Subsystem                    │ Duration  │ Target Gate      │
├───────┼──────────────────────────────────────────┼───────────┼──────────────────┤
│ **1** │ **Mathematical & Physical Validation**   │ COMPLETE  │ `GATE-PHYS-001`  │
│ **2** │ **Native NEON Kernels & Memory Arena**   │ Weeks 1-3 │ `GATE-NEON-001`  │
│ **3** │ **Tokenizer & 350M Proxy Pilot QAT**     │ Weeks 4-7 │ `GATE-PROXY-001` │
│ **4** │ **Full 2B Training & .nano Serialization**│ Weeks 8-13│ `GATE-TRAIN-001` │
│ **5** │ **Android JNI Bridge & Kotlin Wrapper**  │ Weeks 14-16│ `GATE-JNI-001`   │
│ **6** │ **Multi-SoC Certification & Release**    │ Weeks 17-18│ `GATE-CERT-001`  │
└───────┴──────────────────────────────────────────┴───────────┴──────────────────┘
```

---

## 10. Immediate Next Action: Commencing Phase 2 Implementation

With this master implementation roadmap formally established, the project transitions directly into **Phase 2 Implementation**:
1. Create top-level `CMakeLists.txt` and native headers under `include/` (`nano_engine.h`, `nano_config.h`, `nano_types.h`).
2. Implement Phase 2A: The high-performance ARM64 NEON ternary GEMV kernel (`src/kernels/neon_gemv_ternary.cpp`).
3. Implement Phase 2B: The INT4 grouped KV-cache SIMD kernel (`src/kernels/neon_kv_cache.cpp`).
4. Execute unit test harness to prove bit-exactness and GFLOPS throughput against scalar references.

---
*Roadmap approved and authorized for direct execution under the `ss_bangladesh_nano_android_module` tree.*
