# Evaluation Metrics Specification: SS Tutor BD

**Document Version:** 1.0.0  
**Purpose:** Formally define the measurement protocols, mathematical definitions, and status classification for evaluating candidate models.  
**Rule:** Every metric value is strictly categorized as `Measured`, `Target`, or `Unknown`. No simulated or estimated values may be reported as measured.

---

## 1. Metric Classification Definitions

* **`Target`:** Engineering design threshold defined in the architecture specification.
* **`Measured`:** Value obtained through reproducible benchmark execution on specified hardware.
* **`Unknown`:** Pending benchmark execution in Phase 1 / Phase 2 experiments.

---

## 2. Core Metrics Registry

### A. Model Static Metrics

| Metric Identifier | Metric Name | Definition / Measurement Unit | Baseline Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| `MTR-MOD-01` | **Parameter Count** | Total number of trainable weights (Millions / Billions) | $0.4\text{B} \le N \le 1.8\text{B}$ | `Target` |
| `MTR-MOD-02` | **Quantized Binary Size** | Size of the `.gguf` / `.pte` file on disk (Megabytes) | $\le 450\text{ MB}$ (INT4/INT3) | `Target` |
| `MTR-MOD-03` | **Quantization Schemes Tested** | Formats evaluated (e.g., `Q4_K_M`, `Q3_K_M`, `IQ3_XXS`) | GGUF K-Quants | `Target` |

---

### B. Runtime & Latency Metrics

| Metric Identifier | Metric Name | Definition / Measurement Unit | Baseline Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| `MTR-RUN-01` | **Cold Model Load Time** | Time from `llama_load_model_from_file()` to ready state (Milliseconds) | $\le 2500\text{ ms}$ on mobile | `Target` |
| `MTR-RUN-02` | **Time to First Token (TTFT)** | Time from prompt submission to first generated token (Milliseconds) | $\le 1800\text{ ms}$ on mobile | `Target` |
| `MTR-RUN-03` | **Prompt Processing Speed** | Prefill evaluation throughput ($\text{tokens/second}$) | $\ge 20.0\text{ tok/s}$ | `Target` |
| `MTR-RUN-04` | **Generation Throughput** | Autoregressive token generation speed ($\text{tokens/second}$) | $\ge 4.0\text{ tok/s}$ on budget ARM | `Target` |
| `MTR-RUN-05` | **Total Generation Latency** | Time to complete a standard 128-token response (Seconds) | $\le 30.0\text{ s}$ | `Target` |

---

### C. Memory Footprint Metrics

| Metric Identifier | Metric Name | Definition / Measurement Unit | Baseline Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| `MTR-MEM-01` | **Peak Resident Set Size (RSS)** | Maximum physical RAM held by the process during inference (Megabytes) | $\le 650\text{ MB}$ | `Target` |
| `MTR-MEM-02` | **Clean Mapped Memory** | Model weights memory mapped via `mmap()` (Megabytes) | Varies with model size | `Target` |
| `MTR-MEM-03` | **Dirty Working RAM** | Active heap allocations that cannot be paged out (Megabytes) | $\le 150\text{ MB}$ | `Target` |
| `MTR-MEM-04` | **KV Cache Memory Footprint** | Context buffer allocation at 2048 token context (Megabytes) | $\le 80\text{ MB}$ | `Target` |

---

### D. Language & Tokenization Efficiency Metrics

| Metric Identifier | Metric Name | Definition / Measurement Unit | Baseline Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| `MTR-LNG-01` | **Bengali Token Expansion Ratio** | $\frac{\text{Total Tokens Produced by Tokenizer}}{\text{Total Bengali Words in Sample Text}}$ | $\le 1.80\text{ tokens/word}$ | `Target` |
| `MTR-LNG-02` | **Bengali Script Grammatical Coherence** | Human evaluation score (1–5 scale) on conjuncts (*যুক্তবর্ণ*), tense, and spelling | $\ge 4.0 / 5.0$ | `Target` |
| `MTR-LNG-03` | **Language Drift / English Leakage** | Percentage of unsolicited English words in pure Bengali prompt responses (%) | $\le 2.0\%$ | `Target` |

---

### E. Reasoning & Mathematical Accuracy Metrics

| Metric Identifier | Metric Name | Definition / Measurement Unit | Baseline Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| `MTR-RSN-01` | **Final Answer Correctness** | Percentage of Category B math problems solved with correct final answer (%) | $\ge 75.0\%$ (with RAG/prompt) | `Target` |
| `MTR-RSN-02` | **Derivation Step Validity** | Percentage of intermediate mathematical steps logically and algebraically valid (%) | $\ge 80.0\%$ | `Target` |
| `MTR-RSN-03` | **Formula / LaTeX Rendering Validity** | Percentage of formulas rendered with valid LaTeX syntax (%) | $\ge 90.0\%$ | `Target` |

---

### F. Pedagogical & Tutor Behavior Metrics

| Metric Identifier | Metric Name | Definition / Measurement Unit | Baseline Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| `MTR-TUT-01` | **Socratic Scaffolding Score** | Human score (1–5) measuring progressive guidance vs raw answer dump | $\ge 4.0 / 5.0$ | `Target` |
| `MTR-TUT-02` | **Hint Constraint Compliance** | Adherence rate on Test 4 (withholding final answer when hint requested) (%) | $100\%$ | `Target` |
| `MTR-TUT-03` | **Hallucination Rejection Rate** | Successful refusal/clarification on Category E (fake chapters/theorems) (%) | $\ge 85.0\%$ | `Target` |
| `MTR-TUT-04` | **Instruction Format Compliance** | Compliance with JSON/Markdown/bullet formatting constraints (Category D) (%) | $\ge 90.0\%$ | `Target` |

---

## 3. Status Summary of Current Measurements

* **Total Metrics Defined:** 19
* **Metrics Currently `Measured`:** 0 (Benchmarking harness not yet executed)
* **Metrics Classified as `Target`:** 19
* **Metrics Classified as `Unknown`:** All empirical candidate values pending Phase 1 test execution
