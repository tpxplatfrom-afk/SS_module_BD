# SS Tutor BD — Phase 7 Master Execution Plan

**Project:** SS Tutor BD — Offline-First Bangladesh NCTB AI Tutor  
**Phase:** 7 — Real-Device Worst-Case Stress, Model-Loaded PSS Certification & Final Production Release  
**Target Hardware:** itel A662L (2 GB RAM, 16/26 GB Storage, Android 12 Go / API 31, armeabi-v7a)  
**Strict Memory Contract:** Preferred $\le 150\text{ MB}$, Hard Ceiling $\le 200\text{ MB}$, Emergency $\le 250\text{ MB}$  
**Development Cost:** \$0 USD  

---

## 1. 20-Step Phase 7 Sequential Execution Roadmap

1. **Step 1: Audit & Memory Contract Specification:** Create `PHASE7_PRECHECK.md`, `PHASE7_PLAN.md`, and `docs/PHASE7_MEMORY_CONTRACT.md`.
2. **Step 2: High-Frequency PSS Sampler:** Create `benchmarks/android/real_device/pss_sampler.py` with 25ms sampling interval.
3. **Step 3: Real-Device Memory Profiler:** Create `benchmarks/android/real_device/phase7_memory_profiler.py` measuring State A (Deterministic), State B (Model Idle), State C (Inference), and Recovery.
4. **Step 4: Model-Loaded Verification Engine:** Create `results/phase7/model_load_verification.json` confirming INT4 binary mapping, parameters (54.3M/68.2M), and tokenization.
5. **Step 5: Worst-Case Test Queries (220+ items):** Create `benchmarks/phase7/worst_case_queries.json` across Bengali, Math, RAG, Hints, and maximum context stress.
6. **Step 6: 100-Turn Real Model Multi-Turn Session:** Create `benchmarks/android/real_device/phase7_100_turn.py` measuring PSS progression after every turn.
7. **Step 7: 500-Turn Long-Run Stress Test:** Create `benchmarks/android/real_device/phase7_500_turn.py` stressing the real device under continuous hybrid queries.
8. **Step 8: Activity Lifecycle Stress Test (20 cycles):** Test backgrounding, foregrounding, screen off, and activity recreation.
9. **Step 9: Model Load/Unload Stress Test (30 cycles):** Test repeated load, inference, unload, and GC recovery.
10. **Step 10: Low-Memory Pressure Fallback Test:** Simulate system memory pressure and verify graceful `onTrimMemory(CRITICAL)` model unloading.
11. **Step 11: Low-Storage Resilience Test:** Test application footprint and cache stability on low free storage (1 GB / 500 MB / 250 MB).
12. **Step 12: Airplane Mode Offline Certification:** Test 100% core tutoring without Wi-Fi, Mobile Data, or remote APIs.
13. **Step 13: 100-Question Real Quality Benchmark:** Run on-device curriculum evaluation across Math, Grounding, Hints, and Bengali.
14. **Step 14: Real-Device Latency Profiling:** Profile TTFT, generation speed (tok/s), RAG latency, and deterministic math latency.
15. **Step 15: Thermal & CPU Throttling Monitoring:** Monitor battery temperature and stability during extended sessions.
16. **Step 16: Logcat & Crash Audit:** Verify 0 crashes, 0 ANRs, 0 OOM kills, and 0 native crashes.
17. **Step 17: Automatic Failure Detector:** Create `scripts/phase7_failure_detector.py` to enforce strict failure detection without manual overrides.
18. **Step 18: Final Gate Matrix & Documentation:** Create `FINAL_GATE_MATRIX.md`, `final_gate_matrix.json`, `FINAL_PRODUCTION_CERTIFICATION.md`, `REAL_DEVICE_WORST_CASE_VALIDATION.md`, and `PHASE7_KNOWN_LIMITATIONS.md`.
19. **Step 19: Storage Cleanup & Disk Health:** Verify host storage health via `check_disk.py`.
20. **Step 20: Comprehensive Final Report:** Generate `PHASE7_REPORT.md` and execute master runner `scripts/run_phase7_validation.py`.
