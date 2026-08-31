# THSA-2B: Final V1 Architecture Specification
## Ternary Hybrid State-Attention 2B Engine for Android (Hardened Baseline)

**Document Identifier:** `SPEC-NANO-ARCH-THSA2B-001`  
**Revision:** `2.0.0` (Hardened Architecture with Validation & Recovery Protocols)  
**Status:** FINAL V1 ARCHITECTURE SPECIFICATION — ARCHITECTURE TARGET  
**Target Subsystem:** `ss_bangladesh_nano_android_module`  
**Standard Compliance:** RFC 2119 (MUST, MUST NOT, SHOULD, SHOULD NOT, MAY)  

---

> ### **CRITICAL SPECIFICATION NOTICE & RESEARCH DISCIPLINE**
> 1. **Architecture Target Status:** This document establishes `THSA-2B` as the official, selected Version 1 (V1) architectural design target for the Nano-AI Android Module. All numerical memory, latency, compute, power, and thermal claims represent **formal engineering hypotheses and target contracts** that MUST be experimentally validated on physical Android hardware.
> 2. **Physical Feasibility Mandate:** The project MUST NOT claim or imply that the ~2B parameter class, 10,000-token context length, or <= 250 MB working RAM target have already been achieved prior to complete physical-device benchmarking.
> 3. **Research Discipline Principle:** *"External models provide evidence, not architecture."* Prior research systems (BitNet b1.58, Mamba-2/SSD, Liquid LFM2, Gemma 3n, KIVI, Gemini Nano MTP) are cited strictly as empirical precedent and design evidence. `THSA-2B` is an independent, clean-room systems architecture derived specifically from the Nano-AI device constraints.

---

## 1. Project Identity & Primary Constraints

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT IDENTITY & BOUNDARIES                         │
├───────────────────────────────┬─────────────────────────────────────────────────┤
│ **Project Name**              │ Nano-AI Engine (Android Native Inference Core) │
│ **V1 Milestone Goal**         │ Global Core Offline Nano-AI                     │
│ **Primary Target Platform**   │ Android Smartphones (Physical Edge Devices)     │
│ **Primary Compute Tier**      │ ARM64 CPU + NEON Vector SIMD                    │
│ **Core Parameter Target**     │ ~2 BILLION PARAMETERS (~2B Parameter Class)     │
│ **Context Target**            │ 10,000 TOKENS (10K Context Horizon)             │
│ **RAM Hard Ceiling**          │ 250 MB Total Working RAM Under Peak Load        │
│ **Preferred RAM Target**      │ <= 200 MB Total Working RAM                     │
│ **ROM Target**                │ <= 1.0 GB Total Persistent Package Footprint   │
│ **Execution Boundary**        │ 100% Offline / Zero Network Dependency          │
└───────────────────────────────┴─────────────────────────────────────────────────┘
```

### 1.1 The Device Resource Quadrilateral
The entire `THSA-2B` engine is constrained by four co-equal, first-class physical device resources:

```
                NANO-AI DEVICE
                      |
      +---------------+---------------+
      |               |               |
     RAM             ROM          PROCESSOR
      |               |               |
      +---------------+---------------+
                      |
                   BATTERY
```

* **RAM (Volatile Working Memory):** Working memory envelope strictly <= 250 MB (preferred <= 200 MB).
* **ROM (Non-Volatile Storage):** Flash footprint target <= 1.0 GB (model storage target: 400–500 MB for bit-packed ternary weights).
* **PROCESSOR (Compute & Memory Bandwidth):** Optimized for ARM Cortex-A CPU clusters via NEON vector SIMD; memory bandwidth pressure minimized.
* **BATTERY (Energy & Thermals):** Energy target of 2.0–3.5 mJ per token, <= 3.5 W peak power draw, and sustained thermal ceiling <= 45°C.

---

## 2. Core Architecture: THSA-2B Topology

The **Ternary Hybrid State-Attention 2B (THSA-2B)** architecture is a purpose-built hybrid topology engineered specifically to solve the fundamental memory-bandwidth and KV-cache scaling bottlenecks of mobile edge inference.

```
┌────────────────────────────────────────────────────────────────────────┐
│               THSA-2B HYBRID 24-BLOCK BACKBONE TOPOLOGY                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   24 TOTAL BACKBONE BLOCKS                                             │
│   ├── 16 State / Short-Conv Blocks  (~66.7% of backbone)               │
│   │   └── O(1) Memory Footprint, Fast Recurrent / Conv Context State   │
│   └── 8 Grouped Query Attention (GQA) Blocks (~33.3% of backbone)      │
│       └── High-Fidelity Causal Token Retrieval, INT4 KV-Cache          │
│                                                                        │
│   CORE ACCELERATION & COMPRESSION PILLARS                              │
│   ├── BitNet-Style Ternary Weight Matrices (W in {-1, 0, +1})          │
│   ├── Aggressively Constrained INT4 KV-Cache (8 layers only)           │
│   ├── Memory-Mapped (mmap) Paged Weight Residency (<= 130 MB working)  │
│   └── Optional Integrated Multi-Token Prediction (MTP) Head (<= 32M)   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Architectural Composition Rationale
* **Standard Dense Transformers Fail Mobile 10K:** A 24-layer standard Transformer with Multi-Head Attention (MHA) over 10,000 tokens consumes > 2.2 GB of RAM for KV-cache alone in FP16, and > 550 MB in INT4, making on-device 250 MB execution physically impossible.
* **Pure State-Space Models (SSM) Lack Needle Retrieval:** While pure SSMs achieve O(1) state memory, empirical research shows quality degradation on long-context associative recall, precise copy tasks, and complex multi-hop reasoning over 10K tokens.
* **The THSA-2B Solution (66.7% State / 33.3% GQA):** By interleaving 16 State/Short-Conv blocks with 8 GQA blocks, THSA-2B achieves O(1) memory complexity across two-thirds of the network while retaining exact token retrieval in the remaining one-third.

---

## 3. Core Dimensions & Structural Specifications

```
┌────────────────────────────────────────────────────────────────────────┐
│                      THSA-2B STRUCTURAL DIMENSIONS                     │
├────────────────────────────────────────┬───────────────────────────────┤
│ **Total Backbone Blocks ($N_{\text{blocks}}$)** │ **24**                        │
│ **State / Short-Conv Blocks**          │ **16** (Blocks 1-2, 4-5, 7-8, etc.)│
│ **GQA Attention Blocks**               │ **8**  (Every 3rd block: 3, 6, 9... )│
│ **Hidden Dimension ($d_{\text{model}}$)**    │ **2560**                      │
│ **FFN Intermediate Dimension ($d_{\text{ffn}}$)**│ **6912** ($2.7 \times d_{\text{model}}$)     │
│ **Attention Query Heads ($N_q$)**      │ **20**                        │
│ **Attention KV Heads ($N_{kv}$)**      │ **4** (GQA Group Ratio = 5:1) │
│ **Head Dimension ($d_{\text{head}}$)**       │ **128** ($20 \times 128 = 2560$)           │
│ **Target Context Horizon ($L$)**       │ **10,000 tokens**             │
│ **Vocabulary Size ($V$)**              │ **TBD — Tokenizer Phase (32k-64k)**│
│ **Total Parameter Class Target**       │ **1.95B – 2.0B Parameters**   │
└────────────────────────────────────────┴───────────────────────────────┘
```

---

## 4. Parameter Budget Breakdown

| Subsystem / Layer Type | Mathematical Formulation | Parameter Count | Precision Tier | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Token Embeddings** | $V \times d_{\text{model}} = V \times 2560$ | `TBD` (e.g. $\approx 81.9\text{M}$ for $V=32\text{k}$) | FP16 / INT8 / Ternary | `TBD — Tokenizer Phase` |
| **8 GQA Self-Attention Layers** | $8 \times (W_q + W_k + W_v + W_o)$ | $8 \times (2560^2 + 2 \cdot 2560 \cdot 512 + 2560^2) = \mathbf{125.83\text{M}}$ | Ternary $\{-1, 0, +1\}$ | **EXACT** |
| **16 State / Short-Conv Layers** | $16 \times (\text{Proj}_{\text{in}} + \text{State/Conv} + \text{Proj}_{\text{out}})$ | $16 \times (\sim 20.97\text{M}) = \mathbf{\sim 335.5\text{M}}$ | Ternary $\{-1, 0, +1\}$ | `TBD — State Kernel Phase` |
| **24 Gated SwiGLU FFN Layers** | $24 \times (W_{\text{gate}} + W_{\text{up}} + W_{\text{down}})$ | $24 \times (3 \times 2560 \times 6912) = \mathbf{1{,}274.02\text{M}}$ | Ternary $\{-1, 0, +1\}$ | **EXACT** |
| **Layer Normalizations (RMSNorm)** | $24 \times 2 \times 2560 + 2560$ | $\mathbf{\approx 0.25\text{M}}$ | FP16 / FP32 | **EXACT** |
| **Output LM Head** | $d_{\text{model}} \times V$ (Tied or Untied) | `TBD` ($0$ if tied to embedding, $\sim 81.9\text{M}$ if untied) | FP16 / Ternary | `TBD — Training Phase` |
| **Optional MTP Head** | Consumes $h_{\text{last}}$, reuses trunk | $\le \mathbf{32.00\text{M}}$ (Cap target) | Ternary / FP16 | `TBD — MTP Training Phase` |
| **TOTAL TARGET MODEL CLASS** | $\mathbf{\sum \text{All Subsystems}}$ | **$\mathbf{\sim 1.95\text{B} - 2.00\text{B} \text{ Parameters}}$** | **Hybrid / Ternary** | **ARCHITECTURE TARGET** |

---

## 5. Ternary Weight Design (BitNet b1.58 Principle)

The primary weight representation for all dense linear projections across the 24 backbone blocks MUST be ternary:

$$\mathbf{W} \in \{-1, \, 0, \, +1\}^{M \times N}$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                     TERNARY WEIGHT ENCODING & SCALING                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   WEIGHT QUANTIZATION (Off-line / Serialization):                      │
│   W_ternary = Clip( Round( W / gamma ), -1, +1 )                       │
│   gamma = (1 / (M * N)) * sum(|W_ij|)   (Per-tensor or per-group)      │
│                                                                        │
│   ACTIVATION QUANTIZATION (Hot Path):                                  │
│   X_quant = Clip( Round( X * (127 / max(|X|)) ), -128, +127 ) (INT8)   │
│                                                                        │
│   COMPUTE EXECUTION (ARM NEON):                                        │
│   Y = (X_quant * W_ternary) * (scale_X * scale_W)                      │
│   -> Replaces expensive FP32/FP16 multiply-accumulate with             │
│      integer additions, subtractions, and bit-manipulations.           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Precision Allocations Across Subsystems
* **Core Weight Tensors (Ternary Target):** GQA $W_q, W_k, W_v, W_o$; FFN $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$; State in/out projections.
* **Sensitive Tensors (Higher Precision Tier):**
  * Token Embeddings: FP16 or INT8 (to preserve semantic input density).
  * Layer Normalization Gains: FP16 / FP32 (to avoid activation drift).
  * State Recurrence Parameters ($A, B, C$ / $\Delta$): FP16 / FP32 (to maintain state stability over 10K steps).
  * Output LM Head / Logits: FP16 or high-accuracy INT8.
  * Softmax & Attention Logits: FP16 / FP32 accumulation.

---

## 6. Attention Design: 8-Block Grouped Query Attention

The 8 attention blocks utilize Grouped Query Attention (GQA) with a 5:1 query-to-KV head ratio:

```
    QUERY HEADS (N_q = 20)           KV HEADS (N_kv = 4)
    [Q0  Q1  Q2  Q3]  ─────────────► [KV0] (Head dim = 128)
    [Q4  Q5  Q6  Q7]  ─────────────► [KV1] (Head dim = 128)
    [Q8  Q9  Q10 Q11] ─────────────► [KV2] (Head dim = 128)
    [Q12 Q13 Q14 Q15] ─────────────► [KV3] (Head dim = 128)
    [Q16 Q17 Q18 Q19] ─────────────► [KV4 (mapped / grouped)]
```

### 6.1 GQA Specifications
* **Query Dimension:** $20 \times 128 = 2560$
* **Key/Value Dimension:** $4 \times 128 = 512$
* **Causal Masking:** Standard lower-triangular causal attention over sequence length $L \le 10{,}000$.
* **Sliding Window Status:** Full causal attention over 10K tokens is the V1 baseline. Sliding-window attention is designated strictly as an **optional future optimization branch**.

---

## 7. KV-Cache Design & 10K Context Memory Proof

Only the **8 GQA attention blocks** allocate and maintain Key-Value cache buffers. The 16 State/Short-Conv blocks maintain a fixed-size recurrent state independent of context length.

### 7.1 Formal KV-Cache Memory Equation
$$\mathbf{M_{\text{KV}}} = 2 \times L_{\text{context}} \times N_{\text{attention}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times B_{\text{KV}}$$

### 7.2 Numerical Calculation for THSA-2B Baseline (INT4 Precision)
* $L_{\text{context}} = 10{,}000\text{ tokens}$
* $N_{\text{attention}} = 8\text{ blocks}$
* $N_{\text{kv\_heads}} = 4\text{ heads}$
* $D_{\text{head}} = 128\text{ elements}$
* $B_{\text{KV}} = 0.5\text{ bytes (INT4 quantized K and V)}$

$$\mathbf{M_{\text{KV}}} = 2 \times 10{,}000 \times 8 \times 4 \times 128 \times 0.5\text{ bytes} = 40{,}960{,}000\text{ bytes} = \mathbf{39.0625\text{ MB}}$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                 KV-CACHE MEMORY COMPARISON AT 10K CONTEXT              │
├────────────────────────────────────────────────────────────────────────┤
│ Standard 24-Layer MHA (FP16, 20 heads):   2,457.6 MB  [IMPOSSIBLE]     │
│ Standard 24-Layer GQA (FP16, 4 heads):      491.5 MB  [EXCEEDS RAM]    │
│ Standard 24-Layer GQA (INT4, 4 heads):      122.8 MB  [HIGH PRESSURE]  │
│ **THSA-2B Hybrid (8 GQA Layers, INT4, 4 heads): 39.1 MB** [FEASIBLE]   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 8. State / Short-Conv Block Design

The 16 non-attention backbone blocks implement a linear-time sequence mixing transformation:

$$\mathbf{X} \in \mathbb{R}^{B \times S \times 2560} \longrightarrow \mathbf{Y} \in \mathbb{R}^{B \times S \times 2560}$$

```
                          ┌──────────────────────────┐
                          │     Input Tensor X       │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   RMSNorm(X) [FP16/32]   │
                          └─────────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌─────────────────────┐
              │ Proj_in (Branch A)  │       │ Proj_gate (Branch B)│
              │ Ternary Matrix      │       │ Ternary Matrix      │
              └──────────┬──────────┘       └──────────┬──────────┘
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │ Gated Short-Conv /  │                  │
              │ SSM State Recurrence│                  │
              └──────────┬──────────┘                  │
                         │                             │
                         ▼                             ▼
              ┌───────────────────────────────────────────────────┐
              │             Gating Multiplication (*)             │
              └─────────────────────────┬─────────────────────────┘
                                        │
                                        ▼
              ┌───────────────────────────────────────────────────┐
              │               Proj_out (Ternary W)                │
              └─────────────────────────┬─────────────────────────┘
                                        │
                                        ▼
              ┌───────────────────────────────────────────────────┐
              │               Residual Add (Y + X)                │
              └───────────────────────────────────────────────────┘
```

### 8.1 Architectural Contract
* **Contract:** Memory consumption per state block MUST be $O(1)$ with respect to sequence length $L$, consuming $\le 128\text{ KB}$ of state memory per block during decode.
* **Option A (Structured State-Space / SSD):** Continuous state evolution $\mathbf{h}_t = \mathbf{A} \mathbf{h}_{t-1} + \mathbf{B} \mathbf{x}_t$, $\mathbf{y}_t = \mathbf{C} \mathbf{h}_t + \mathbf{D} \mathbf{x}_t$.
* **Option B (Gated Depthwise Short-Convolution):** 1D causal convolution across temporal kernel window $K \in [3, 4, 7]$ with gating non-linearities.

### 8.2 Implementation Selection Phase Gate & Benchmark Protocol
To resolve the SSM vs. Short-Conv implementation choice empirically without premature lock-in:
1. **Phase 1 (V1 Architecture):** Freeze the abstract contract ($O(1)$ state, $\le 128\text{ KB}$ per block).
2. **Phase 2A (Kernel Micro-benchmarks):** Benchmark Mamba-2/SSD recurrence vs. Gated Depthwise Short-Conv on ARM64 NEON for throughput ($\text{tok/sec}$) and cache locality.
3. **Phase 2B (Long-Context Retrieval Gate):** Measure needle-in-a-haystack retrieval accuracy, associative recall, and perplexity across 10K tokens.
4. **Formal Selection Gate Criterion:** The selected variant MUST achieve **$\ge 95\%$ of pure-attention retrieval quality** on synthetic needle benchmarks while strictly preserving $O(1)$ memory.

---

## 9. Memory Topology: Paged Weight Residency & Arena Allocation

The complete ~2B parameter model ($400\text{ MB} - 500\text{ MB}$ on flash) **MUST NOT** be loaded entirely into RAM at startup.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MEMORY TOPOLOGY PIPELINE                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   1. NON-VOLATILE FLASH (ROM)                                          │
│      Compact Ternary Weight File (Serialized Model Package)            │
│                        │                                               │
│                        ▼ Zero-Copy Kernel Mapping                      │
│   2. MEMORY-MAPPED VIRTUAL ADDRESS SPACE (mmap)                        │
│      MAP_SHARED / PROT_READ (OS Page Cache Backed)                     │
│                        │                                               │
│                        ▼ Demand Paging / Sequential Prefetch           │
│   3. RESIDENT WORKING RAM SET (<= 130 MB)                              │
│      Active block weights resident in physical RAM pages               │
│                        │                                               │
│                        ▼ Static Pre-Allocated Buffers                  │
│   4. STATIC WORKING ARENA (<= 120 MB)                                  │
│      KV Cache (39 MB) + Activations (25 MB) + Workspace (20 MB) + Meta │
│                        │                                               │
│                        ▼ Vector SIMD Execution                         │
│   5. ARM64 CPU / NEON REGISTERS                                        │
│      Ternary GEMV / Integer Addition Hot Loops                         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 9.1 Zero Dynamic Heap Allocation Mandate
* The hot token generation loop MUST execute with **ZERO dynamic heap allocations** (`0` calls to `malloc`, `calloc`, `realloc`, or C++ `new`).
* All intermediate activations, scratchpads, and KV-cache blocks MUST reside in pre-allocated static arenas initialized once during model startup.

### 9.2 Memory-Mapped I/O Performance Guarantees & Page Fault Budgets
To prevent token generation jitter caused by flash read stalls:
* **Sustained Sequential Read Throughput:** $\ge 800\text{ MB/s}$ on modern Android UFS 2.2/3.1/4.0 flash storage.
* **P99 Page Fault Latency Budget:** $\le 2.0\text{ ms}$ under standard operating conditions; peak allowable $\le 10\text{ ms}$ under background system pressure.
* **Sequential Readahead & Prefetch Policy:** The runtime MUST issue asynchronous `madvise(MADV_WILLNEED)` hints for Block $N+1$ while Block $N$ is executing its compute kernel.
* **Memory Pressure Strategy:** If Android OS broadcasts `onTrimMemory(TRIM_MEMORY_RUNNING_CRITICAL)`, the runtime MUST release cached non-active weight pages and evict temporary scratchpads without dropping active KV context.

---

## 10. RAM Budget Allocation (Working Memory Envelope)

```
┌────────────────────────────────────────────────────────────────────────┐
│                  THSA-2B WORKING RAM BUDGET ALLOCATION                 │
├────────────────────────────────────────┬───────────────────────────────┤
│ **Subsystem / Memory Component**       │ **Target Allocation Ceiling** │
├────────────────────────────────────────┼───────────────────────────────┤
│ **Resident Weight Pages ($M_{\text{weights}}$)**│ **<= 130.0 MB**               │
│ **KV-Cache ($M_{\text{kv\_cache}}$ - 10K tokens)**│ **<=  45.0 MB** (Nominal 39.1)│
│ **Activation Tensors ($M_{\text{activations}}$)**│ **<=  25.0 MB**               │
│ **Temporary Workspace ($M_{\text{workspace}}$)** │ **<=  20.0 MB**               │
│ **Runtime / JNI / Metadata ($M_{\text{meta}}$)**│ **<=  15.0 MB**               │
│ **Safety Margin Buffer**               │ **~  15.0 MB**                │
├────────────────────────────────────────┼───────────────────────────────┤
│ **HARD WORKING RAM CEILING**           │ **250.0 MB** (Peak Maximum)   │
│ **PREFERRED WORKING TARGET**           │ **<= 200.0 MB**               │
└────────────────────────────────────────┴───────────────────────────────┘
```

---

## 11. ROM & Persistent Storage Architecture

$$\mathbf{TOTAL\_STORAGE} = S_{\text{model}} + S_{\text{tokenizer}} + S_{\text{runtime}} + S_{\text{metadata}} + S_{\text{required\_assets}}$$

* **Model File ($S_{\text{model}}$):** Bit-packed ternary weights with quantized scaling headers. Target: $\approx 400\text{ MB} - 500\text{ MB}$ on flash. Total package ceiling: $\le 1.0\text{ GB}$.
* **Sequential Locality:** Tensors within the model binary MUST be organized sequentially by execution order (Block 0 $\rightarrow$ Block 23) to maximize flash sequential read throughput and Linux readahead efficiency during `mmap`.
* **Zero Packaging Bloat:** Native libraries (`.so`) MUST be stripped of debug symbols. Test fixtures, golden datasets, and calibration tools MUST NOT be packaged into production APK/AAB builds.

---

## 12. Processor Execution Tier: ARM64 CPU + NEON

Version 1 is engineered strictly for **ARM64 CPU execution utilizing NEON vector SIMD**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     ARM64 NEON KERNEL TAXONOMY                         │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Ternary GEMV / GEMM:  INT8 activation x Ternary weight dot products │
│                          utilizing NEON add/sub/tbl instructions.      │
│ 2. GQA Projections:      Batched Q/K/V linear transformations.         │
│ 3. KV Quant / Dequant:   On-the-fly INT4 packing/unpacking kernels.    │
│ 4. State Update:         Linear recurrence / 1D conv SIMD pipelines.   │
│ 5. Normalization:        Vectorized RMSNorm with reciprocal sqrt.      │
│ 6. Activation Functions: Fast vectorized Swish / SiLU / GeLU.          │
│ 7. Residual Operations:  Vectorized FP16/FP32 in-place addition.       │
│ 8. Sampling / Logits:    Top-K, Top-P, Temperature, Argmax kernels.   │
└────────────────────────────────────────────────────────────────────────┘
```

### 12.1 Dynamic SIMD Dispatch & Multi-SoC Heterogeneity Strategy
Android hardware exhibits diverse CPU microarchitectures (Qualcomm Kryo, ARM Cortex-X/A7xx/A5xx, MediaTek Dimensity, Samsung Exynos). To ensure smooth execution across all tiers:
1. **Dynamic Capability Detection:** The runtime MUST query `getauxval(AT_HWCAP)` / `AT_HWCAP2` at startup to detect ARMv8.2-A+ features (`FEAT_DotProd`, `FEAT_FP16`, `FEAT_I8MM`).
2. **Multi-Tier Kernel Dispatch:**
   * *Tier 1 (High Performance):* NEON `I8MM` / `DotProd` vector instructions (Snapdragon 8 Gen 1/2/3, Dimensity 9000+).
   * *Tier 2 (Standard ARM64):* Baseline 128-bit NEON vector SIMD (universal across all 64-bit Android chips).
   * *Tier 3 (Scalar Fallback):* Pure ISO C++17 scalar implementation for verified cross-platform correctness testing.
3. **Thermal-Aware Core Affinity:** The thread pool MUST monitor core thermal status via `/sys/devices/virtual/thermal/`. Compute threads MUST bind to performance cores during prefill bursts and adaptively balance across efficiency cores during continuous decode if temperature exceeds $40^\circ\text{C}$.

---

## 13. Battery & Energy Dissipation Model

$$\mathbf{Energy\_per\_token} = \frac{\text{Total Joules Consumed}}{\text{Generated Tokens}}$$

### 13.1 Concrete Energy & Thermal Target Contracts
* **Energy Efficiency Target:** **$2.0 - 3.5\text{ mJ}$ per generated token** during the autoregressive decode phase on reference hardware.
* **Continuous Power Consumption Ceiling:** **$\le 3.5\text{ W}$ peak system power draw** during active generation.
* **Thermal Throttling Ceiling:** Device skin/package temperature MUST remain **$\le 45^\circ\text{C}$** during sustained 10-minute continuous inference sessions.
* **Battery Drain Profile:** **$\le 5\%$ total battery consumption** over 1 hour of sustained continuous generation at $10\text{ tokens/sec}$.
* **Rejection-Aware MTP Profiling:** MTP speculative speedup MUST be measured alongside candidate rejection rates; if verification rejection exceeds $40\%$, the MTP module MUST be throttled dynamically to conserve battery.

---

## 14. Multi-Token Prediction (MTP) Speculative Head

Multi-Token Prediction (MTP) is included in THSA-2B as an **optional, integrated speculative decoding acceleration module**:

```
                              ┌───────────────────────────┐
                              │ Backbone Hidden State h_N │
                              └─────────────┬─────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
        ┌─────────────────────────┐                   ┌─────────────────────────┐
        │  Standard LM Head (t+1) │                   │ Optional MTP Head (t+2) │
        │  (Direct Token Emission)│                   │ (Speculative Candidate) │
        └────────────┬────────────┘                   └────────────┬────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                              ┌───────────────────────────┐
                              │ Exact Verification Step   │
                              │ (Backbone State Reuse)    │
                              └───────────────────────────┘
```

* **Architectural Principles:** Parameter budget $\le 32\text{M}$, reuses backbone representations, up to 4 candidate tokens, exact backbone verification pass, zero correctness deviation.

---

## 15. Explicit Architectural Deferrals: MoE & MatFormer

### 15.1 Mixture-of-Experts (MoE) — DEFERRED FOR V1
* **Decision:** MoE is explicitly excluded from the THSA-2B V1 core architecture due to random flash page faults, mobile CPU branch misprediction overhead, and non-deterministic working RAM spikes.

### 15.2 MatFormer / Elastic Parameter Scaling — DEFERRED FOR V1
* **Decision:** MatFormer-style nested sub-network slicing is deferred to maintain a deterministic, fixed ~2B baseline for V1 physical-device benchmarking.

---

## 16. Numerical Precision Policy & Quantization Calibration

```
┌────────────────────────────────────────────────────────────────────────┐
│                     NUMERICAL PRECISION POLICY                         │
├────────────────────────────────┬───────────────────────────────────────┤
│ **Weights**                    │ Ternary {-1, 0, +1} Target            │
│ **KV-Cache**                   │ INT4 Baseline                         │
│ **Accumulation**               │ Higher Precision (INT32 / FP16 / FP32)│
│ **Layer Normalization**        │ FP16 / FP32                           │
│ **Softmax Computation**        │ Numerically Stable FP32               │
│ **Logits & Sampling**          │ FP16 / FP32                           │
└────────────────────────────────┴───────────────────────────────────────┘
```

### 16.1 Numerical Tolerance Policy & Error Budgets
To prevent cascading quantization error across the ternary $\rightarrow$ INT8 $\rightarrow$ INT4 chain:
1. **Ternary Weight Error Budget:** Relative tensor reconstruction error MUST be $\le 2.0\%$ compared to the unquantized FP32 reference.
2. **INT4 KV-Cache Error Budget:** Softmax output probability divergence MUST remain within a Kullback-Leibler (KL) divergence threshold of $D_{\text{KL}} \le 0.015$ ($\le 1.5\%$ relative softmax error).
3. **Accumulation Contract:** All ternary dot products MUST accumulate in **INT32 or FP32** before scaling and clamping to avoid intermediate overflow/underflow.
4. **Post-Training Quantization (PTQ) Calibration:** Min-max symmetric channel-wise quantization calibrated across a standardized 512-sample long-sequence dataset.
5. **Perplexity Degradation Limit:** End-to-end quantized model perplexity on WikiText-103 / C4 validation subsets MUST exhibit $\le 5.0\%$ degradation compared to the unquantized baseline.

---

## 17. End-to-End Execution Data Flow

```
User Prompt (Text)
      │
      ▼
Tokenizer (Vocab / Trie Encoding)
      │
      ▼
Input Token IDs [S]
      │
      ▼
Token Embedding Lookup (FP16 / INT8)
      │
      ▼
Initial Activation Tensor X_0 [B, S, 2560]
      │
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 24-BLOCK HYBRID THSA-2B BACKBONE PIPELINE                              │
│                                                                        │
│  Block 00: State / Short-Conv Block (O(1) Recurrent State Update)      │
│  Block 01: State / Short-Conv Block (O(1) Recurrent State Update)      │
│  Block 02: Grouped Query Attention Block (INT4 KV-Cache Update, 10K)   │
│  Block 03: State / Short-Conv Block (O(1) Recurrent State Update)      │
│  ...                                                                   │
│  Block 23: Grouped Query Attention Block (INT4 KV-Cache Update, 10K)   │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
Final RMSNorm Layer (FP16 / FP32)
      │
      ▼
Final Hidden State h_24 [B, 2560]
      │
      ├───► Standard LM Head ────► Logits (t+1) ──► Sampler ──► Next Token
      │
      └───► Optional MTP Head ───► Speculative Candidate Tokens (t+2..t+4)
                                         │
                                         ▼
                               Verification Pipeline (Backbone Pass)
```

---

## 18. Formal Architecture Trade-Off Matrix

| Feature / Technique | Primary Benefit | System Cost | Architectural Risk | V1 Baseline Status |
| :--- | :--- | :--- | :--- | :--- |
| **BitNet Ternary Weights** | 4x - 8x weight compression, low DRAM BW | Requires custom NEON kernels | Quantization noise on complex reasoning | **CORE V1 TARGET** |
| **16 State / Short-Conv Blocks** | O(1) memory, fast sequence mixing | Weaker associative recall than dense Attn | Implementation details need benchmark | **CORE V1 TARGET** |
| **8 GQA Attention Blocks** | Exact long-context needle retrieval | Allocates KV-cache | Memory grows with sequence length | **CORE V1 TARGET** |
| **INT4 KV-Cache** | Fits 10K context into 39.1 MB RAM | Quantization / dequant overhead | Potential accuracy loss on attention scores | **CORE V1 TARGET** |
| **mmap Paged Residency** | Allows 2B model on 250 MB device | I/O latency on page fault | Flash read latency under memory pressure | **CORE V1 TARGET** |
| **Multi-Token Prediction** | 1.5x - 2.5x decode speedup | <= 32M parameter budget | Rejection rate under low confidence | **OPTIONAL V1 MODULE** |
| **Mixture-of-Experts (MoE)** | Lower active FLOPs per token | Random flash I/O, branch overhead | RAM spikes, thermal unpredictability | **DEFERRED (V2/V3)** |
| **MatFormer Elastic Slicing** | Multiple model sizes in one file | Complex training and weight layout | Benchmark non-determinism | **DEFERRED (V2/V3)** |
| **INT2 KV-Cache** | Further halves KV RAM to 19.5 MB | Severe quantization noise | Severe perplexity degradation | **RESEARCH PATH** |

---

## 19. Mathematical Feasibility Formulations

$$\mathbf{Peak\_RAM} = M_{\text{weights}} + M_{\text{kv\_cache}} + M_{\text{activations}} + M_{\text{workspace}} + M_{\text{runtime}}$$

$$\mathbf{V1\_Feasibility\_Condition} = \begin{cases} \text{TRUE} & \text{if } \mathbf{Peak\_RAM} \le 250\text{ MB} \;\land\; L_{\text{context}} \ge 10{,}000 \;\land\; \theta \approx 2\text{B} \;\land\; \text{Offline} = 1 \\ \text{FALSE} & \text{otherwise} \end{cases}$$

---

## 20. Explicit Failure Conditions & Recovery Protocols

```
┌────────────────────────────────────────────────────────────────────────┐
│                       FORMAL FAILURE CRITERIA                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Peak Working RAM > 250 MB under full 10K context load.              │
│ 2. 10,000-Token Context is unavailable or artificially truncated.       │
│ 3. Model Correctness / Numerical Tolerance checks fail.                │
│ 4. Sustained Thermal Throttling renders generation unusable (< TBD).   │
│ 5. Packaged Storage Footprint exceeds distribution limits (> 1.0 GB).  │
│ 6. Energy Consumption per token causes excessive battery drain.        │
└────────────────────────────────────────────────────────────────────────┘
```

### 20.1 Explicit Recovery Strategies & Non-Silent Degraded Modes
If physical benchmarking encounters memory pressure exceeding the 250 MB ceiling during 10K context execution, the runtime MUST execute the following explicit recovery protocol:
1. **Immediate Fallback:** Truncate active KV-cache allocation to $8{,}192\text{ tokens}$ (or $4{,}096\text{ tokens}$) and retry generation.
2. **Mandatory Diagnostic Logging:** Emit structured log event `WARN_DEGRADED_CONTEXT_BUDGET` with exact heap and PSS telemetry (silent truncation is strictly prohibited).
3. **Application-Facing State Notification:** Propagate degraded status flag (`FLAG_CONTEXT_CONSTRAINED`) across the JNI bridge so host applications can inform the user transparently.
4. **Root-Cause Telemetry Trigger:** Record memory dump to local test artifact for architectural review (determining if quantization scales or scratchpad arenas caused the regression).

---

## 21. Research Basis & Prior Art Distinction

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH BASIS MAPPING                          │
├───────────────────────────────┬────────────────────────────────────────┤
│ **Research Precedent**        │ **Empirical Insight Extracted**        │
├───────────────────────────────┼────────────────────────────────────────┤
│ Microsoft BitNet b1.58        │ Ternary weights W in {-1, 0, +1}       │
│ Mamba / Mamba-2 / SSD         │ Linear-time recurrent state space      │
│ Liquid LFM2                   │ Hybrid SSM-Attention block ratios      │
│ Google Gemma 3n / MatFormer   │ Mobile weight locality & packaging     │
│ KIVI Research                 │ 2-bit/4-bit KV quantization methods    │
│ Google Gemini Nano            │ Speculative Multi-Token Prediction     │
└───────────────────────────────┴────────────────────────────────────────┘
```

---

## 22. Final V1 Architecture Specification Summary

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THSA-2B FINAL SPECIFICATION SUMMARY                  │
├───────────────────────────────────┬────────────────────────────────────┤
│ **Architecture Name**             │ **THSA-2B (Ternary Hybrid State-Attn)**│
│ **Target Parameter Class**        │ **~2 BILLION Parameters**          │
│ **Backbone Blocks ($N_{\text{blocks}}$)**  │ **24 Total Blocks**                │
│ **State / Short-Conv Blocks**     │ **16 Blocks (~66.7%)**             │
│ **GQA Attention Blocks**          │ **8 Blocks (~33.3%)**              │
│ **Hidden Dimension ($d_{\text{model}}$)**│ **2560**                           │
│ **FFN Dimension ($d_{\text{ffn}}$)**     │ **6912**                           │
│ **Attention Query Heads ($N_q$)** │ **20**                             │
│ **Attention KV Heads ($N_{kv}$)** │ **4**                              │
│ **Head Dimension ($d_{\text{head}}$)**   │ **128**                            │
│ **Context Target**                │ **10,000 Tokens (10K Context)**    │
│ **Weight Representation**         │ **BitNet-Style Ternary {-1, 0, +1}**│
│ **KV-Cache Representation**       │ **INT4 Baseline (39.1 MB @ 10K)**  │
│ **Primary Target Runtime**        │ **Android Native ARM64 CPU + NEON**│
│ **Weight Loading Mechanism**      │ **Zero-Copy mmap + Paged Residency**│
│ **Multi-Token Prediction (MTP)**  │ **Optional Integrated Drafter (<=32M)**│
│ **Working RAM Hard Ceiling**      │ **250 MB (Peak Maximum)**          │
│ **Preferred RAM Target**          │ **<= 200 MB**                      │
│ **ROM Package Target**            │ **<= 1.0 GB (400-500 MB Model)**   │
│ **Energy Target**                 │ **2.0 - 3.5 mJ / generated token** │
│ **Mixture-of-Experts (MoE)**      │ **DEFERRED to V2/V3**              │
│ **Hardware Accelerators (NPU)**   │ **DEFERRED to V2/V3**              │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 23. Training & Quantization-Aware Training (QAT) Strategy

While runtime construction precedes training execution, THSA-2B defines the necessary model training pipeline to guarantee seamless conversion to ternary weights and INT4 KV-caches:

### 23.1 Base Model Pre-Training Pipeline
* **Token Budget:** Pre-training horizon of $\approx 2.0\text{ Trillion}$ high-quality tokens across multilingual and code corpora.
* **Tokenizer Configuration:** Byte-Pair Encoding (BPE) or SentencePiece with vocabulary size $V \in [32{,}768, \, 65{,}536]$.
* **Target Objective:** Causal autoregressive language modeling targeting validation perplexity $\le 10.0$ on standard benchmark distributions.

### 23.2 Quantization-Aware Training (QAT) Protocol
Naive Post-Training Quantization (PTQ) is insufficient for sub-2-bit ternary weights. THSA-2B enforces Quantization-Aware Training:
1. **Fake Quantization Forward Pass:** Weights are simulated as ternary values during forward passes using $\text{Round}(\text{Clamp}(W/\gamma, -1, +1))$, with INT8 activation scaling.
2. **Straight-Through Estimator (STE):** Backward pass gradients flow directly through the rounding operator: $\frac{\partial \text{loss}}{\partial W} \approx \frac{\partial \text{loss}}{\partial W_{\text{quantized}}}$.
3. **QAT Fine-Tuning Stage:** $20{,}000 - 50{,}000\text{ steps}$ of continuous QAT fine-tuning on high-quality instruct/reasoning datasets.
4. **Validation Suite:** Continuous evaluation on standard benchmarks (MMLU 5-shot, GSM8K, ARC-Challenge, ANLI) ensuring quantized model score retention $\ge 95\%$ relative to FP32 baseline.

---
*Specification formulated and finalized for the `ss_bangladesh_nano_android_module` architecture tree.*
