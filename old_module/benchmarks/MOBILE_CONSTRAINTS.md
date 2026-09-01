# Low-End Mobile Hardware Constraints & Memory Reality Check

**Document Version:** 1.0.0  
**Target Class:** Budget Android Devices (~2.0 GB Physical RAM / ~16 GB Internal Storage)  
**Reference SoCs:** MediaTek Helio A22 / P22, Qualcomm Snapdragon 425 / 439 / 450, Unisoc SC9863A (Cortex-A53 / Cortex-A55 @ 1.4–2.0 GHz)

---

## 1. The Myth of "2GB Usable RAM"

A common architectural error in mobile ML development is assuming that a device with 2GB physical RAM offers 2GB (or even 1GB) of free memory to a user application.

### Realistic Physical RAM Budget Allocation on Android (2GB Physical RAM Device)

```
Total Physical RAM: 2048 MB
├── Android OS Kernel, Drivers & Hardware Reserved: ~350 MB - 450 MB
├── Android System Server & Core Daemons:           ~300 MB - 400 MB
├── SurfaceFlinger & Display Graphics Buffers:      ~100 MB - 150 MB
├── Essential Background Services (RIL, System UI): ~150 MB - 250 MB
└── Total OS Baseline Overhead:                     ~900 MB - 1250 MB

AVAILABLE FOR FOREGROUND APPS:                      ~550 MB - 750 MB (Maximum Safe Margin)
CRITICAL LMK THRESHOLD:                             Typically fires when free RAM drops < 180-250 MB
```

> [!CAUTION]
> If a foreground app causes total available system memory to drop below the kernel's low-memory watermark ($\approx 180\text{–}250\text{ MB}$), the **Android Low Memory Killer (LMK)** will aggressively terminate the foreground process without warning.

---

## 2. Memory Terminology & Allocations Breakdown

To build a reliable offline engine, we must strictly distinguish between six distinct memory tiers:

```
+-----------------------------------------------------------------------------------------+
|                                    TOTAL PROCESS RSS                                    |
|                                                                                         |
|  +---------------------------+  +--------------------------+  +----------------------+  |
|  |     Clean Mapped Memory   |  |     Dirty Working RAM    |  |    Host App Heap     |  |
|  |     (Model Weights via    |  |  (C++ Scratchpad, KV     |  |  (Kotlin/Java UI,    |  |
|  |          mmap())          |  |   Cache, Tokenizer Buff) |  |   Bitmaps, SQLite)   |  |
|  |         ~350 MB           |  |         ~150 MB          |  |       ~80 MB         |  |
|  +---------------------------+  +--------------------------+  +----------------------+  |
+-----------------------------------------------------------------------------------------+
```

### 1. Model File Size (Storage on Disk)
* The binary size of the quantized `.gguf` file stored on internal eMMC flash (e.g. $350\text{ MB}$).
* Consumes non-volatile storage, not RAM, until accessed.

### 2. Mapped Model Memory (`mmap()`)
* Model weights mapped into the virtual address space via `mmap()`.
* **Behavior:** Paged into physical RAM on demand from flash. Because this memory is *read-only and clean*, the Linux kernel can discard pages under memory pressure without writing to swap/ZRAM and page them back in as needed.
* *Trap:* If too much of the model is accessed rapidly during generation, the active resident set size (RSS) still counts against the process memory budget.

### 3. Dirty Working RAM (Native Heap Allocations)
* Dynamic C++ allocations created during matrix multiplications, intermediate tensor activations, and compute graphs.
* **Behavior:** *Dirty memory* that cannot be paged out. Must remain in physical RAM or compressed into ZRAM.
* **Budget Target:** Must be strictly constrained to $\le 100\text{–}150\text{ MB}$.

### 4. Key-Value (KV) Cache Memory
* Memory holding attention keys and values for prior tokens in the conversation context.
* **Formula:** $\text{KV Cache Size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times n_{\text{ctx}} \times \text{bytes per element}$.
* For a 1B model at $n_{\text{ctx}} = 2048$ with 16-bit float cache: $\approx 64\text{–}128\text{ MB}$.
* *Mitigation:* We will evaluate **Quantized KV Cache (FP8 or Q4_0 KV Cache)** in `llama.cpp` to cut cache memory by $50\%\text{–}75\%$.

### 5. Application Memory (Java/Kotlin Android Runtime)
* The host app's ART/JVM heap, view hierarchies, Jetpack Compose layouts, SQLite connection pools, and bitmaps.
* **Budget Target:** Typically consumes $60\text{–}120\text{ MB}$.

### 6. ZRAM & Swap Behavior on Low-End Devices
* Most budget Android devices configure $512\text{ MB}\text{–}1024\text{ MB}$ of **ZRAM** (compressed in-RAM swap).
* When dirty memory is pushed to ZRAM, CPU cycles are consumed by the kernel decompressing memory pages, causing massive latency spikes (stuttering) and thermal heating.

---

## 3. CPU & Thermal Constraints on Low-End SoCs

* **Cores & Clock Speeds:** Most entry SoCs feature 4 or 8 ARM Cortex-A53 cores running at $1.4\text{–}2.0\text{ GHz}$.
* **Lack of FP16/DotProd Hardware:** Older Cortex-A53 cores lack ARMv8.2-A `dotprod` instructions, meaning INT8/INT4 matrix multiplications fall back to standard NEON vector instructions.
* **Thermal Throttling Dynamics:**
  * Running 4 cores at $100\%$ CPU utilization causes SoC temperatures to exceed $65^\circ\text{C}$ within 60–90 seconds.
  * The thermal governor then aggressively throttles CPU clock frequencies down to $800\text{–}1000\text{ MHz}$ (a $50\%$ drop in inference throughput).
* **Architectural Rule:** The inference engine must default to **2 worker threads** (or at most 3), leaving headroom for OS UI threads and preventing rapid thermal collapse.

---

## 4. Storage & eMMC Constraints (16GB Devices)

* On a 16GB device, Android OS and preinstalled OEM apps occupy 9–11 GB, leaving only **3–5 GB of actual user storage**.
* Apps downloading $> 1\text{ GB}$ of model assets suffer high user uninstallation rates and download failures.
* **Engine Target:** Core Engine + Base Model $\le 400\text{ MB}$; individual Knowledge Packs $\le 50\text{ MB}$.
