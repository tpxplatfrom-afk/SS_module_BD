# SS Bangladesh Core Model Master — Capability Matrix

**Document Version:** 1.0.0  
**Phase:** 8.3 — Core Model Master Capability Characterization  
**Date:** 2026-08-30  
**Status:** **EMPIRICALLY VERIFIED BASELINE** (Untrained, Domain-Neutral)  

---

## 1. Executive Capability Matrix

| Category | Dimension | Empirical Value / Limit | Classification |
| :--- | :--- | :--- | :--- |
| **Model** | Total Parameter Count | **71,528,256 (71.53M)** | Deterministic (Seed 42) |
| | Tensor Count | **93 Tensors** | Complete Safetensors |
| | Transformer Layers | **10 Layers** | Fixed Architecture |
| | Hidden Dimension ($d$) | **576** | Standard LLaMA |
| | Intermediate Dimension | **2,304** (SwiGLU) | Standard LLaMA |
| | Attention Heads | **8 Attention / 8 KV** | Multi-Head Attention |
| | Vocabulary Capacity | **16,000 Byte-level BPE** | Bengali-First Dedicated |
| | Configured Context ($L$) | **256 tokens** | Safe Target |
| | Extrapolated Forward Limit | **Up to 1,024 tokens** | Unsupported Extrapolation |
| **Text Capacity** | Bengali Tokens per Word | **5.61 tokens/word** (Avg) | Empirical (1–10k words) |
| | UTF-8 Bytes per Token | **3.48 bytes/token** (Avg) | Compression Ratio |
| | Safe Bengali Input Words | **~45 words** (256 tokens) | Safe Operating Window |
| | Safe Bengali Input Chars | **~325 characters** | Safe Operating Window |
| | Bengali Unicode Integrity | **100% Roundtrip Match** | Swaraborno, Byanjon, Juktakkhor |
| | Pathological Worst Case | **2.0 tokens/char** (ZWJ/ZWNJ repeats) | Known Stress Bound |
| **Generation** | Maximum Safe Output | **256 tokens** (within context) | Configured Ceiling |
| | Forward Pass Latency | **120.3 ms – 216.2 ms** | CPU Host Baseline |
| | Generation Throughput | **27.4 – 30.9 tokens/sec** | Host CPU Benchmark |
| | Time to First Token (TTFT) | **~157 ms** (Cold model load) | Initialization |
| **Memory** | Process Idle Memory | **~378 – 501 MB** | Full Python/PyTorch Dev |
| | Loaded Model Delta | **+3.6 MB – +40 MB** (Host RSS) | Clean Allocation |
| | Peak Inference Memory | **~759 MB** (Host PyTorch RSS) | Peak Allocator |
| | Turn Drift (500 turns) | **0.00 MB / Flat** | Strictly $O(1)$ Bounded |
| | Recovery on Unload | **239.1 MB recovered** | Clean Garbage Collection |
| | Cycling Drift (20 cycles) | **-6.46 MB net drift** | Zero Leakage Detected |
| **Android 2GB** | Physical Target Device | **itel A662L (Android 12 Go)** | Physical Tested |
| | Total Addressable RAM | **1.87 GB (1,911.4 MB)** | Real Device Hardware |
| | Available Free RAM | **923.0 – 988.1 MB** | System Active |
| | CPU Architecture / ABI | **ARMv7-a 32-bit (Unisoc SC9832E)** | Low-End Cortex-A55 |
| | Total Device Storage | **26 GB Total / 8.4 GB Free** | Ample for Module |
| | Operating Temperature | **32.5°C – 32.6°C** | Zero Thermal Throttling |
| **Offline** | Internet Dependency | **Zero / None Required** | 100% Local Execution |
| | Network Sockets Required | **0 Sockets** | Pure Offline Engine |
| | Cloud API Dependency | **$0 / Zero Remote Endpoints** | Standalone Embedded |
| **Storage** | Core Master Bundle Size | **272.99 MB** (FP32 weights) | Master Archive |
| | SS Tutor BD Specialization | **207.33 MB** (Class 8 Math) | Downstream Fork |
| | Exported INT4 Module | **34.12 MB** | Embedded Runtime |
