# THSA-2B: Final V1 Architecture Specification
## Ternary Hybrid State-Attention 2B Engine for Android

**Document Identifier:** `SPEC-NANO-ARCH-THSA2B-001`  
**Revision:** `1.0.0` (V1 Final Architecture Baseline)  
**Status:** FINAL V1 ARCHITECTURE SPECIFICATION — ARCHITECTURE TARGET  
**Target Subsystem:** `ss_bangladesh_nano_android_module`  
**Standard Compliance:** RFC 2119 (MUST, MUST NOT, SHOULD, SHOULD NOT, MAY)  

---

> ### **CRITICAL SPECIFICATION NOTICE & RESEARCH DISCIPLINE**
> 1. **Architecture Target Status:** This document establishes `THSA-2B` as the official, selected Version 1 (V1) architectural design target for the Nano-AI Android Module. All numerical memory, latency, compute, and battery claims herein represent **formal engineering hypotheses and target contracts** that MUST be experimentally validated on physical Android hardware.
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
* **ROM (Non-Volatile Storage):** Flash footprint minimization across serialized model, tokenizer, and native `.so` libraries.
* **PROCESSOR (Compute & Memory Bandwidth):** Optimized for ARM Cortex-A CPU clusters via NEON vector SIMD; memory bandwidth pressure minimized.
* **BATTERY (Energy & Thermals):** Minimization of Joules per generated token and mitigation of thermal throttling during continuous decode.

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

> **Directive `REQ-ARCH-001` (No Pure Mamba Lock-In):** The State / Short-Conv block is an **architectural contract and mathematical abstraction**. It MAY be instantiated via Mamba-2/SSD structured state-space mathematics or a mobile-optimized gated short-convolution mechanism. The final kernel choice will be determined by ARM NEON micro-benchmarking.

---

## 3. Core Dimensions & Structural Specifications

The structural dimensions of THSA-2B are calibrated to hit the ~2B parameter class horizon while maintaining clean SIMD alignment for ARM64 NEON:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      THSA-2B STRUCTURAL DIMENSIONS                     │
├────────────────────────────────────────┬───────────────────────────────┤
│ **Total Backbone Blocks ($N_{\\text{blocks}}$)** │ **24**                        │
│ **State / Short-Conv Blocks**          │ **16** (Blocks 1-2, 4-5, 7-8, etc.)│
│ **GQA Attention Blocks**               │ **8**  (Every 3rd block: 3, 6, 9... )│
│ **Hidden Dimension ($d_{\\text{model}}$)**    │ **2560**                      │
│ **FFN Intermediate Dimension ($d_{\\text{ffn}}$)**│ **6912** ($2.7 \\times d_{\\text{model}}$)     │
│ **Attention Query Heads ($N_q$)**      │ **20**                        │
│ **Attention KV Heads ($N_{kv}$)**      │ **4** (GQA Group Ratio = 5:1) │
│ **Head Dimension ($d_{\\text{head}}$)**       │ **128** ($20 \\times 128 = 2560$)           │
│ **Target Context Horizon ($L$)**       │ **10,000 tokens**             │
│ **Vocabulary Size ($V$)**              │ **TBD — Tokenizer Phase**     │
│ **Total Parameter Class Target**       │ **1.95B – 2.0B Parameters**   │
└────────────────────────────────────────┴───────────────────────────────┘
```

---

## 4. Parameter Budget Breakdown

The parameter budget targets the **~2B total model class**. The table below establishes exact structural counts and identifies items subject to training/tokenizer discovery:

| Subsystem / Layer Type | Mathematical Formulation | Parameter Count | Precision Tier | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Token Embeddings** | $V \\times d_{\\text{model}} = V \\times 2560$ | `TBD` (e.g. $\\approx 81.9\\text{M}$ for $V=32\\text{k}$) | FP16 / INT8 / Ternary | `TBD — Tokenizer Phase` |
| **8 GQA Self-Attention Layers** | $8 \\times (W_q + W_k + W_v + W_o)$ | $8 \\times (2560^2 + 2 \\cdot 2560 \\cdot 512 + 2560^2) = \\mathbf{125.83\\text{M}}$ | Ternary $\\{-1, 0, +1\\}$ | **EXACT** |
| **16 State / Short-Conv Layers** | $16 \\times (\\text{Proj}_{\\text{in}} + \\text{State/Conv} + \\text{Proj}_{\\text{out}})$ | $16 \\times (\\sim 20.97\\text{M}) = \\mathbf{\\sim 335.5\\text{M}}$ | Ternary $\\{-1, 0, +1\\}$ | `TBD — State Kernel Phase` |
| **24 Gated SwiGLU FFN Layers** | $24 \\times (W_{\\text{gate}} + W_{\\text{up}} + W_{\\text{down}})$ | $24 \\times (3 \\times 2560 \\times 6912) = \\mathbf{1{,}274.02\\text{M}}$ | Ternary $\\{-1, 0, +1\\}$ | **EXACT** |
| **Layer Normalizations (RMSNorm)** | $24 \\times 2 \\times 2560 + 2560$ | $\\mathbf{\\approx 0.25\\text{M}}$ | FP16 / FP32 | **EXACT** |
| **Output LM Head** | $d_{\\text{model}} \\times V$ (Tied or Untied) | `TBD` ($0$ if tied to embedding, $\\sim 81.9\\text{M}$ if untied) | FP16 / Ternary | `TBD — Training Phase` |
| **Optional MTP Head** | Consumes $h_{\\text{last}}$, reuses trunk | $\\le \\mathbf{32.00\\text{M}}$ (Cap target) | Ternary / FP16 | `TBD — MTP Training Phase` |
| **TOTAL TARGET MODEL CLASS** | $\\mathbf{\\sum \\text{All Subsystems}}$ | **$\\mathbf{\\sim 1.95\\text{B} - 2.00\\text{B} \\text{ Parameters}}$** | **Hybrid / Ternary** | **ARCHITECTURE TARGET** |

---

## 5. Ternary Weight Design (BitNet b1.58 Principle)

The primary weight representation for all dense linear projections across the 24 backbone blocks MUST be ternary:

$$\\mathbf{W} \\in \\{-1, \\, 0, \\, +1\\}^{M \\times N}$$

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
Not all tensors tolerate aggressive 1.58-bit quantization. THSA-2B strictly differentiates precision tiers:

* **Core Weight Tensors (Ternary Target):** GQA $W_q, W_k, W_v, W_o$; FFN $W_{\\text{gate}}, W_{\\text{up}}, W_{\\text{down}}$; State in/out projections.
* **Sensitive Tensors (Higher Precision Tier):**
  * Token Embeddings: FP16 or INT8 (to preserve semantic input density).
  * Layer Normalization Gains: FP16 / FP32 (to avoid activation drift).
  * State Recurrence Parameters ($A, B, C$ / $\\Delta$): FP16 / FP32 (to maintain state stability over 10K steps).
  * Output LM Head / Logits: FP16 or high-accuracy INT8.
  * Softmax & Attention Logits: FP16 / FP32 accumulation.

> **Directive `REQ-TERN-001`:** Ternary weight representation drastically compresses persistent ROM storage and reduces DRAM memory bandwidth traffic. However, developers MUST NOT assume ternary weights automatically guarantee <= 250 MB RAM. RAM is governed by resident working set pages, activations, and KV-cache.

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
* **Query Dimension:** $20 \\times 128 = 2560$
* **Key/Value Dimension:** $4 \\times 128 = 512$
* **Causal Masking:** Standard lower-triangular causal attention over sequence length $L \\le 10{,}000$.
* **Sliding Window Status:** Full causal attention over 10K tokens is the V1 baseline. Sliding-window attention is designated strictly as an **optional future optimization branch**.

---

## 7. KV-Cache Design & 10K Context Memory Proof

Only the **8 GQA attention blocks** allocate and maintain Key-Value cache buffers. The 16 State/Short-Conv blocks maintain a fixed-size recurrent state independent of context length.

### 7.1 Formal KV-Cache Memory Equation
$$\\mathbf{M_{\\text{KV}}} = 2 \\times L_{\\text{context}} \\times N_{\\text{attention}} \\times N_{\\text{kv\\_heads}} \\times D_{\\text{head}} \\times B_{\\text{KV}}$$

### 7.2 Numerical Calculation for THSA-2B Baseline (INT4 Precision)
* $L_{\\text{context}} = 10{,}000\\text{ tokens}$
* $N_{\\text{attention}} = 8\\text{ blocks}$
* $N_{\\text{kv\\_heads}} = 4\\text{ heads}$
* $D_{\\text{head}} = 128\\text{ elements}$
* $B_{\\text{KV}} = 0.5\\text{ bytes (INT4 quantized K and V)}$

$$\\mathbf{M_{\\text{KV}}} = 2 \\times 10{,}000 \\times 8 \\times 4 \\times 128 \\times 0.5\\text{ bytes} = 40{,}960{,}000\\text{ bytes} = \\mathbf{39.0625\\text{ MB}}$$

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

### 7.3 Advanced KV Research Horizons (Post-V1)
* **INT2 KV Quantization:** Potential reduction from $39.1\\text{ MB} \\rightarrow 19.5\\text{ MB}$.
* **Asymmetric K/V Quantization:** e.g., INT4 Keys + INT2 Values.
* **Dynamic Token Eviction / Selective KV:** Pruning uninformative historical tokens based on attention scores.
* **Low-Rank KV Projections:** Factorizing KV state vectors.

---

## 8. State / Short-Conv Block Design

The 16 non-attention backbone blocks implement a linear-time sequence mixing transformation:

$$\\mathbf{X} \\in \\mathbb{R}^{B \\times S \\times 2560} \\longrightarrow \\mathbf{Y} \\in \\mathbb{R}^{B \\times S \\times 2560}$$

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

### 8.1 Architectural Contract vs. Implementation Options
* **Contract:** Memory consumption per state block MUST be O(1) with respect to sequence length $L$, consuming <= 128 KB of state memory per block during decode.
* **Option A (Structured State-Space / SSD):** Continuous state evolution $\\mathbf{h}_t = \\mathbf{A} \\mathbf{h}_{t-1} + \\mathbf{B} \\mathbf{x}_t$, $\\mathbf{y}_t = \\mathbf{C} \\mathbf{h}_t + \\mathbf{D} \\mathbf{x}_t$.
* **Option B (Gated Depthwise Short-Convolution):** 1D causal convolution across temporal kernel window $K \\in [3, 4, 7]$ with gating non-linearities.

---

## 9. Memory Topology: Paged Weight Residency & Arena Allocation

The complete ~2B parameter model ($400\\text{ MB} - 500\\text{ MB}$ on flash) **MUST NOT** be loaded entirely into RAM at startup.

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

---

## 10. RAM Budget Allocation (Working Memory Envelope)

The table below defines the formal engineering budget allocated across all physical memory subsystems. The sum of peak resident components MUST remain within the hard ceiling:

```
┌────────────────────────────────────────────────────────────────────────┐
│                  THSA-2B WORKING RAM BUDGET ALLOCATION                 │
├────────────────────────────────────────┬───────────────────────────────┤
│ **Subsystem / Memory Component**       │ **Target Allocation Ceiling** │
├────────────────────────────────────────┼───────────────────────────────┤
│ **Resident Weight Pages ($M_{\\text{weights}}$)**│ **<= 130.0 MB**               │
│ **KV-Cache ($M_{\\text{kv\\_cache}}$ - 10K tokens)**│ **<=  45.0 MB** (Nominal 39.1)│
│ **Activation Tensors ($M_{\\text{activations}}$)**│ **<=  25.0 MB**               │
│ **Temporary Workspace ($M_{\\text{workspace}}$)** │ **<=  20.0 MB**               │
│ **Runtime / JNI / Metadata ($M_{\\text{meta}}$)**│ **<=  15.0 MB**               │
│ **Safety Margin Buffer**               │ **~  15.0 MB**                │
├────────────────────────────────────────┼───────────────────────────────┤
│ **HARD WORKING RAM CEILING**           │ **250.0 MB** (Peak Maximum)   │
│ **PREFERRED WORKING TARGET**           │ **<= 200.0 MB**               │
└────────────────────────────────────────┴───────────────────────────────┘
```

> **Directive `REQ-BUDGET-001`:** These values represent rigorous engineering budget allocations, NOT completed physical benchmark measurements. Verification requires empirical validation under peak 10K context load on physical hardware.

---

## 11. ROM & Persistent Storage Architecture

$$\\mathbf{TOTAL\\_STORAGE} = S_{\\text{model}} + S_{\\text{tokenizer}} + S_{\\text{runtime}} + S_{\\text{metadata}} + S_{\\text{required\\_assets}}$$

* **Model File ($S_{\\text{model}}$):** Bit-packed ternary weights with quantized scaling headers. Target: $\\approx 400\\text{ MB} - 500\\text{ MB}$ on flash.
* **Sequential Locality:** Tensors within the model binary MUST be organized sequentially by execution order (Block 0 $\\rightarrow$ Block 23) to maximize flash sequential read throughput and Linux readahead efficiency during `mmap`.
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

* **Fallback Mandate:** Every NEON kernel MUST be accompanied by a portable, verified pure C/C++ scalar fallback implementation.
* **Secondary Accelerators (Deferred):** NPU, Qualcomm Hexagon DSP, MediaTek NeuroPilot, OpenCL, and Vulkan backends are deferred to V2/V3.

---

## 13. Battery & Energy Dissipation Model

Energy efficiency is treated as a first-class architectural metric:

$$\\mathbf{Energy\\_per\\_token} = \\frac{\\text{Total Joules Consumed}}{\\text{Generated Tokens}}$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                   BATTERY CONSUMPTION RISK DRIVERS                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Weight Fetch Bandwidth: Repeatedly streaming hundreds of MBs across │
│                            the LPDDR bus drains battery rapidly.       │
│ 2. KV-Cache Read Traffic:  Autoregressive attention reading large KV   │
│                            buffers on every token multiplies mJ/tok.   │
│ 3. Frequent CPU Wakeups:   Sub-optimal thread scheduling preventing    │
│                            CPU cores from entering low-power states.   │
└────────────────────────────────────────────────────────────────────────┘
```

* **Thermal Throttling Protection:** Sustained 10-minute generation benchmarks MUST record token generation throughput before and after thermal equilibrium ($\\Delta_{\\text{thermal}}$).

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

* **Architectural Principles:**
  * **Parameter Budget:** <= 32M parameters dedicated to the MTP module.
  * **No Separate Drafter Model:** Reuses backbone representations; avoids duplicating 10K KV-caches.
  * **Exact Verification:** The main backbone verifies speculative candidates in parallel during the next forward step.
  * **Zero Correctness Compromise:** If speculative tokens fail verification, the runtime falls back seamlessly to standard single-token autoregression.

---

## 15. Explicit Architectural Deferrals: MoE & MatFormer

### 15.1 Mixture-of-Experts (MoE) — DEFERRED FOR V1
* **Decision:** MoE is explicitly excluded from the THSA-2B V1 core architecture.
* **Technical Rationale:**
  1. *Paging Disruption:* Dynamic routing to sparse experts forces random, unpredictable flash `mmap` page reads, breaking sequential prefetch.
  2. *CPU Branch Overhead:* Dynamic dispatching on mobile CPUs creates severe branch misprediction penalties.
  3. *RAM Unpredictability:* Routing variance creates non-deterministic working RAM spikes exceeding 250 MB.
* **Roadmap Status:** Deferred to V2/V3 research branches.

### 15.2 MatFormer / Elastic Parameter Scaling — DEFERRED FOR V1
* **Decision:** MatFormer-style nested sub-network slicing is deferred.
* **Technical Rationale:** V1 requires a fixed, deterministic ~2B baseline to establish reproducible benchmarks across physical Android devices.

---

## 16. Numerical Precision Policy

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

$$\\mathbf{Peak\\_RAM} = M_{\\text{weights}} + M_{\\text{kv\\_cache}} + M_{\\text{activations}} + M_{\\text{workspace}} + M_{\\text{runtime}}$$

$$\\mathbf{V1\\_Feasibility\\_Condition} = \\begin{cases} \\text{TRUE} & \\text{if } \\mathbf{Peak\\_RAM} \\le 250\\text{ MB} \\;\\land\\; L_{\\text{context}} \\ge 10{,}000 \\;\\land\\; \\theta \\approx 2\\text{B} \\;\\land\\; \\text{Offline} = 1 \\\\ \\text{FALSE} & \\text{otherwise} \\end{cases}$$

$$\\mathbf{Memory\\_Bandwidth\\_Per\\_Token} = \\frac{\\text{Resident\\_Weights\\_Read} + \\text{KV\\_Cache\\_Read\\_Write}}{\\text{Token\\_Duration\\_Seconds}}$$

---

## 20. Explicit Failure Conditions

The THSA-2B project MUST be declared a **FAILURE** under any of the following conditions:

```
┌────────────────────────────────────────────────────────────────────────┐
│                       FORMAL FAILURE CRITERIA                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Peak Working RAM > 250 MB under full 10K context load.              │
│ 2. 10,000-Token Context is unavailable or artificially truncated.       │
│ 3. Model Correctness / Numerical Tolerance checks fail.                │
│ 4. Sustained Thermal Throttling renders generation unusable (< TBD).   │
│ 5. Packaged Storage Footprint exceeds distribution limits.             │
│ 6. Energy Consumption per token causes excessive battery drain.        │
└────────────────────────────────────────────────────────────────────────┘
```

> **Directive `REQ-FAIL-001`:** If physical benchmarking reveals a failure condition, engineers MUST report the limitation transparently. Silently reducing the context length (e.g. from 10K to 4K) to pass a RAM test is strictly prohibited.

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

* **Distinction Mandate:** The external works above provide scientific evidence and validation methodology. `THSA-2B` is an independent architecture tailored specifically to the Nano-AI 250 MB / 10K Android target.

---

## 22. Final V1 Architecture Specification Summary

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THSA-2B FINAL SPECIFICATION SUMMARY                  │
├───────────────────────────────────┬────────────────────────────────────┤
│ **Architecture Name**             │ **THSA-2B (Ternary Hybrid State-Attn)**│
│ **Target Parameter Class**        │ **~2 BILLION Parameters**          │
│ **Backbone Blocks ($N_{\\text{blocks}}$)**  │ **24 Total Blocks**                │
│ **State / Short-Conv Blocks**     │ **16 Blocks (~66.7%)**             │
│ **GQA Attention Blocks**          │ **8 Blocks (~33.3%)**              │
│ **Hidden Dimension ($d_{\\text{model}}$)**│ **2560**                           │
│ **FFN Dimension ($d_{\\text{ffn}}$)**     │ **6912**                           │
│ **Attention Query Heads ($N_q$)** │ **20**                             │
│ **Attention KV Heads ($N_{kv}$)** │ **4**                              │
│ **Head Dimension ($d_{\\text{head}}$)**   │ **128**                            │
│ **Context Target**                │ **10,000 Tokens (10K Context)**    │
│ **Weight Representation**         │ **BitNet-Style Ternary {-1, 0, +1}**│
│ **KV-Cache Representation**       │ **INT4 Baseline (39.1 MB @ 10K)**  │
│ **Primary Target Runtime**        │ **Android Native ARM64 CPU + NEON**│
│ **Weight Loading Mechanism**      │ **Zero-Copy mmap + Paged Residency**│
│ **Multi-Token Prediction (MTP)**  │ **Optional Integrated Drafter (<=32M)**│
│ **Working RAM Hard Ceiling**      │ **250 MB (Peak Maximum)**          │
│ **Preferred RAM Target**          │ **<= 200 MB**                      │
│ **Mixture-of-Experts (MoE)**      │ **DEFERRED to V2/V3**              │
│ **Hardware Accelerators (NPU)**   │ **DEFERRED to V2/V3**              │
└───────────────────────────────────┴────────────────────────────────────┘
```

---
*Specification formulated and finalized for the `ss_bangladesh_nano_android_module` architecture tree.*
