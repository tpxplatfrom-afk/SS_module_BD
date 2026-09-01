# SS Tutor BD

**A Developer-Friendly, Offline-First AI Education Platform for Bangladesh NCTB (Class 6–10)**

---

## Overview

SS Tutor BD is an open-source, modular AI education engine and embeddable SDK designed specifically for the Bangladesh National Curriculum and Textbook Board (NCTB) curriculum. It enables EdTech developers, offline kiosks, and educational hardware vendors to deploy a culturally attuned, Bengali-native AI tutor that operates **100% offline** on budget Android devices (~2GB RAM, ~16GB storage).

```
SS Tutor BD Architecture
├── SS Core (Single reusable small language model + offline C++ runtime)
└── Education Knowledge Packs (.ssp)
    ├── Class 6 Pack (Mathematics, Science, ...)
    ├── Class 7 Pack (Mathematics, Science, ...)
    ├── Class 8 Pack (Mathematics, Science, ...)
    └── Class 9–10 Pack (Mathematics, Science, English, ...)
```

---

## Phase 1: Research & Model Benchmarking Harness

This repository currently hosts the reproducible **Phase 1 Model Feasibility & Benchmarking Harness**.

### Quick Start Commands

```bash
# 1. View Candidate Models & License Gate Status
python benchmark_runner/cli.py list

# 2. Check Host Storage Capacity
python scripts/check_disk.py

# 3. Run Bengali Tokenizer Efficiency Benchmark
python benchmark_runner/cli.py tokenizer Qwen/Qwen2.5-0.5B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 4. Run Dry-Run Validation Benchmark (100 Items)
python benchmark_runner/cli.py benchmark-mock CAND-01

# 5. Download & Benchmark CAND-01 (Qwen2.5-0.5B-Instruct Q4_K_M)
python benchmark_runner/cli.py download CAND-01
python benchmark_runner/cli.py benchmark CAND-01

# 6. Purge Model Weights (Preserves disk space)
python benchmark_runner/cli.py purge
```

---

## Documentation Index

* [`ARCHITECTURE.md`](ARCHITECTURE.md): Complete platform architecture specification (v1.1.0).
* [`ARCHITECTURE_CHANGELOG.md`](ARCHITECTURE_CHANGELOG.md): Architecture revision tracking and ADR records.
* [`MODEL_SELECTION.md`](MODEL_SELECTION.md): 100-point scorecard, sequential gate protocol, and candidate tiers.
* [`MODEL_CANDIDATES.md`](MODEL_CANDIDATES.md): Master registry for candidate models (CAND-01 to CAND-10).
* [`MODEL_SELECTION_CHANGELOG.md`](MODEL_SELECTION_CHANGELOG.md): Rationale for research framework decisions.
* [`benchmarks/`](benchmarks/): 100-item standardized evaluation dataset and measurement specifications.

---

## License

SS Tutor BD engine and benchmark harness codebase is licensed under the [Apache 2.0 License](LICENSE).
Third-party models and curriculum knowledge packs maintain their own independent provenance and licenses.
