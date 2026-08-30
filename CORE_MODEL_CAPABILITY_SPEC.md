# SS Bangladesh Core Model Master — Authoritative Capability Specification

**Document:** `CORE_MODEL_CAPABILITY_SPEC.md`  
**Model Identity:** `ss_bangladesh`  
**Version:** 0.8.2  
**Role:** Root Reusable AI Foundation for Bengali-First Downstream Specializations  
**Status:** **UNTRAINED / DOMAIN-NEUTRAL MASTER BASELINE**  

---

## 1. What the Model IS

* The **SS Bangladesh Core Model Master** (`ss_bangladesh`) is the single root foundation model of the SS Tutor BD ecosystem.
* It consists of a 10-layer, 576-hidden-dimension `LlamaForCausalLM` neural network initialized deterministically with seed 42 truncated normal distribution ($\sigma=0.02$).
* Total parameter count is **71,528,256 (71.53M)** across **93 tensors**.
* It is packaged with a dedicated 16,000-vocabulary Byte-level BPE Bengali tokenizer, generation configuration, and canonical architecture metadata.
* It is **domain-neutral** and contains zero hardcoded textbook facts or curriculum bias.

---

## 2. What the Model CAN Handle

1. **Bengali Unicode Encoding & Decoding:** Flawlessly roundtrips Bengali Swaraborno, Byanjonborno, Kaar, Fola, Juktakkhor (all compound conjuncts), Bengali numerals, Arabic numerals, math symbols, and mixed Bengali/English sentences with 100% fidelity.
2. **Deterministic Context Processing:** Forward passes execute reliably from 1 to 256 tokens within a 120 ms – 216 ms latency window on single-thread CPU.
3. **Continuous Generation:** Generates output sequences up to the context budget at 27.4 – 30.9 tokens/sec on standard CPU.
4. **$O(1)$ Bounded Memory Lifecycle:** Sustains 500+ consecutive inference turns with zero memory growth and releases memory cleanly upon unloading.
5. **100% Offline Embedded Execution:** Operates completely disconnected from the internet, requiring 0 network sockets and zero cloud APIs.
6. **2 GB RAM Android Compatibility:** Fits comfortably within the resource constraints of low-end hardware (e.g., physical `itel A662L` with 923 MB available RAM).

---

## 3. What the Model CANNOT Handle (Current Boundaries)

1. **Domain Knowledge / Factual Tutoring:** Because the Core Master is **untrained**, it cannot answer NCTB curriculum questions, solve physics problems, or tutor students without downstream training/specialization.
2. **Context Beyond 256 Tokens in Production:** While the forward pass mechanically executes up to 1,024 tokens via RoPE extrapolation, sequences $>256$ tokens are outside the calibrated positional encoding and must be truncated in production.
3. **Ultra-Long Multi-Paragraph Input:** A single prompt containing $>45$ Bengali words will exceed the 256-token context window and must be chunked or summarized before passing to the model.

---

## 4. What Is Configured

* `num_hidden_layers`: 10
* `hidden_size`: 576
* `intermediate_size`: 2,304 (SwiGLU)
* `num_attention_heads`: 8
* `num_key_value_heads`: 8
* `max_position_embeddings`: 256
* `vocab_size`: 16,000
* `rms_norm_eps`: $1 \times 10^{-6}$
* `rope_theta`: 10,000.0

---

## 5. What Was Empirically Verified

* **Cryptographic Immutability:** Pre-benchmark and post-benchmark SHA-256 matches anchor `bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb`.
* **Tokenizer Scaling Table:** Measured exact character/byte/token ratios from 1 to 10,000 words. Average: 5.61 tokens/word, 3.48 bytes/token.
* **Context Scaling:** 64, 128, 192, 256 tokens classified `SAFE`. 320 to 1,024 tokens classified `UNSUPPORTED_EXTRAPOLATION`.
* **Physical Device Specs:** Physical `itel A662L` running Android 12 Go (API 31) on Unisoc SC9832E with 1.87 GB RAM verified via ADB.
* **Thermal Profile:** Idle battery temperature 32.5°C with zero throttling.

---

## 6. What Remains Unknown

* **Convergence Rate on NCTB Class 6–10:** Exact training loss trajectory and epoch requirements for high-school science/math have not yet been measured on the Core Master.
* **Post-Training Perplexity vs. Quantization Degradation:** Perplexity retention under INT4 quantization after domain fine-tuning will be measured in future specialization phases.

---

## 7. Android Deployment & Offline Capability

* **Target Runtime:** Runs embedded via ONNX Runtime / MicroRuntime adapter.
* **Physical Hardware:** Tested on `itel A662L` (ARMv7-a 32-bit Cortex-A55).
* **Storage Footprint:** Master FP32 is 272.99 MB; exported INT4 module is 34.12 MB.
* **Offline Independence:** 100% local execution. Zero network calls.

---

## 8. Safe Production Operating Limits

| Parameter | Safe Production Limit | Hard Failure Threshold |
| :--- | :--- | :--- |
| Max Input Length | 200 tokens (~35 Bengali words) | $>256$ tokens (Truncation Required) |
| Max Output Length | 56 tokens | $>256 - \text{input\_len}$ |
| Total Sequence Budget | 256 tokens | $>256$ tokens |
| Process RAM Allowance | $< 200$ MB (INT4 on Android) | $> 250$ MB (Emergency Ceiling) |
| Max Consecutive Turns | Unlimited ($O(1)$ Bounded) | N/A |

---

## 9. Future Specialization Implications

1. **Clean Forking Protocol:** To build `SS Tutor BD (Class 6–10)` or `SS Mechanics`, copy `models/core/ss_bangladesh/` as the initial checkpoint.
2. **Tokenizer Sharing:** The 16,000 Byte-level BPE tokenizer is domain-agnostic and should be reused across all Bengali educational specializations without re-training.
3. **Multi-Model Scalability:** Our company can train and version multiple downstream modules from this single master without architectural divergence.
