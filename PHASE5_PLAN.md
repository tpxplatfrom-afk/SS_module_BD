# SS Tutor BD — Phase 5 Master Implementation Plan

**Project:** SS Tutor BD — Offline-First Bangladesh NCTB AI Tutor  
**Phase:** 5 — Android / Native UI Binding, Offline APK Packaging & Real-Device Validation  
**Strict Memory Contract:** Preferred $\le 150\text{ MB}$, Hard Production Ceiling $\le 200\text{ MB}$  
**Target Hardware:** Low-End Android (2 GB RAM, 16 GB Storage)  
**Development Cost:** \$0 USD  

---

## 1. Subsystem Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    SS Tutor BD Android                    │
├───────────────────────────────────────────────────────────┤
│ UI Layer (Material 3 Native Android UI)                   │
│   Subject / Class / Chapter Selector                      │
│   Chat / Tutoring View with Token Streaming               │
│   Action Buttons: [ইঙ্গিত (Hint)] [ব্যাখ্যা (Explain)]    │
│                   [সমাধান (Solve)] [অনুশীলন (Practice)]   │
├───────────────────────────────────────────────────────────┤
│ TutorDecisionEngine & Router                              │
│   Math intent? → Deterministic Engine (100% precision)   │
│   Textbook concept? → SQLite FTS5 RAG                     │
│   Verbalization/Explanation? → MicroRuntime (70M INT4)    │
├───────────────────────────────────────────────────────────┤
│ Deterministic Math Subsystem                              │
│   Fraction Arithmetic | Calculator | Pythagoras | Linear  │
├───────────────────────────────────────────────────────────┤
│ Offline RAG Layer (.ssp Knowledge Pack)                   │
│   Class 8 Mathematics (164 KB SQLite FTS5, < 20 ms)       │
├───────────────────────────────────────────────────────────┤
│ Dedicated Bengali Tokenizer (16K BPE, 3.65 tok/word)      │
├───────────────────────────────────────────────────────────┤
│ Native MicroRuntime Bridge                                │
│   loadModel() | unloadModel() | generate() | cancel()     │
│   Lazy loading policies: AUTO / LOAD_PER_QUERY            │
├───────────────────────────────────────────────────────────┤
│ Multi-Guard Validation Layer                              │
│   GroundingGuard | MathGuard | HintGuard | LangGuard      │
├───────────────────────────────────────────────────────────┤
│ AndroidMemoryMonitor & Lifecycle Controller               │
│   Real-time PSS / Native Heap / Java Heap tracking        │
│   NORMAL (<150MB) | WARNING (150-200) | CRITICAL (>200)   │
│   Auto-unload on background & onTrimMemory(CRITICAL)      │
└───────────────────────────────────────────────────────────┘
```

---

## 2. 26-Step Sequential Execution Roadmap

1. **Architecture Audit & Decision Docs:** `PHASE5_PRECHECK.md`, `PHASE5_PLAN.md`, `docs/ANDROID_RUNTIME_DECISION.md`, `docs/ANDROID_PRODUCTION_MEMORY.md`, `docs/OFFLINE_ARCHITECTURE.md`, `docs/RELEASE_ARCHITECTURE.md`.
2. **Android Project Structure:** Create `android/` with standard Gradle build configuration (`app/build.gradle`, `settings.gradle`, `AndroidManifest.xml`).
3. **Native Engine & Decision Router (`android/app/src/main/java/bd/sstutor/`):**
   * `runtime/MicroRuntime.kt`: Model-agnostic runtime interface.
   * `runtime/AndroidMemoryMonitor.kt`: Real PSS, native heap, Java heap monitoring with NORMAL/WARNING/CRITICAL/EMERGENCY states.
   * `math/`: Complete deterministic math port (Fractions, Calculator, Pythagoras, Linear systems, Expression parser).
   * `rag/`: SQLite FTS5 database helper and `.ssp` Knowledge Pack loader.
   * `tokenizer/`: Bengali 16K BPE tokenizer engine.
   * `validation/`: Grounding, Math, Socratic Hint, Language, and Format validators.
   * `router/TutorDecisionEngine.kt`: Deterministic-first query dispatcher.
   * `session/SessionState.kt`: $O(1)$ constant-memory session manager.
4. **Android UI Layer (`ui/`):**
   * Modern Material Design 3 tutoring activity with Subject/Class/Chapter drawer, message recycler view, and Socratic hint mode.
5. **Testing & Golden Test Cases:**
   * `tests/android/golden/`: Golden test cases for fractions, algebra, circle area, Pythagoras, and interest.
   * `tests/android/test_android_engine.py`: Unit and integration test runner.
6. **Real Android Memory Benchmark:**
   * `benchmarks/android/android_memory_benchmark.py`: Cold launch, 10/25/50/100-turn multi-turn sessions, model load/unload cycles.
7. **Quality Benchmark:**
   * `benchmarks/android/run_android_quality_benchmark.py`: 100-question curriculum benchmark on Android engine.
8. **Release Packaging & Auditing:**
   * `scripts/audit_release.py`: Scans APK/release assets for unauthorized models, training artifacts, or credentials.
   * `THIRD_PARTY_LICENSES.md`: Complete license inventory.
   * `scripts/run_phase5_validation.py`: Master automated validation runner.
9. **Final Report:**
   * `PHASE5_REPORT.md`: Comprehensive evaluation report with all hard acceptance gates documented.
