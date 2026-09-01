# SS Tutor BD — Core Model Master Manifest (Proposed)

**Document Version:** 1.0.0  
**Phase:** 8.1 — Forensic Discovery  

> [!NOTE]
> This manifest classifies existing repository files into architectural roles for future separation without physically moving or deleting any files.

---

## 1. File Classification Manifest

```text
========================================================================================
CATEGORY           FILE / DIRECTORY PATH                        DESCRIPTION
========================================================================================
CORE_REQUIRED      training/train_micro_model.py                Architecture builder (70M Transformer)
CORE_REQUIRED      models/tokenizer_bengali_16k/                16K Byte-level BPE Tokenizer
CORE_REQUIRED      configs/phase8_training.json                 Core Transformer configuration
CORE_REQUIRED      core/runtime/micro_runtime.py                Bounded native inference runtime
CORE_REQUIRED      core/runtime/memory_budget.py                Strict 200MB PSS contract engine
CORE_REQUIRED      core/runtime/session_manager.py              O(1) constant state session engine
CORE_REQUIRED      core/curriculum/schema.py                    Hierarchical ontology & deterministic IDs
CORE_REQUIRED      core/curriculum/boundaries.py                CurriculumScope abstraction
CORE_REQUIRED      core/tutor_module.py                         Developer module integration contract

CORE_OPTIONAL      models/manager.py                            Model lifecycle manager
CORE_OPTIONAL      models/registry.json                         Model candidates registry
CORE_OPTIONAL      core/sanitization/cleaner.py                 Output text sanitizer

SS_TUTOR_SPECIFIC  models/sstutor_bengali_70m_edu/model.safetensors  Trained Class 8 Math weights (207MB)
SS_TUTOR_SPECIFIC  models/export_int4/                          Exported 34MB INT4 binary
SS_TUTOR_SPECIFIC  packs/class8_math/                           Class 8 NCTB Math SQLite RAG pack
SS_TUTOR_SPECIFIC  data/phase4/                                 13,000 synthetic Class 8 training pairs
SS_TUTOR_SPECIFIC  core/math/                                   Exact NCTB math solvers (fraction, etc.)
SS_TUTOR_SPECIFIC  core/validation/                             Grounding & pedagogical validators

RUNTIME_ONLY       runtimes/base.py                             Abstract runtime interface
RUNTIME_ONLY       runtimes/llama_cpp_runtime.py                llama.cpp engine wrapper
RUNTIME_ONLY       runtimes/mock_runtime.py                     Fast unit test mock engine

ANDROID_ONLY       android/app/                                 Native Android UI & JNI bindings
ANDROID_ONLY       benchmarks/android/real_device/              Real-device ADB profilers (itel A662L)

TRAINING_ONLY      scripts/train_micro_model.py                 Training CLI entry point
TRAINING_ONLY      scripts/generate_phase7_queries.py           Benchmark query generator
TRAINING_ONLY      scripts/purge_training_artifacts.py          Disk cleanup utility

TEST_ONLY          tests/                                       23 automated regression & unit tests
TEST_ONLY          benchmarks/                                  Diagnostic & 13D evaluation suites

BUILD_ONLY         scripts/build_android_assets.py              Asset packaging script
BUILD_ONLY         scripts/audit_release.py                     Secret & license auditor

TEMPORARY          scratch/                                     Scratch data files
TEMPORARY          __pycache__/                                 Python bytecode cache
========================================================================================
```
