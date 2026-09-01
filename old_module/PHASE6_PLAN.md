# SS Tutor BD — Phase 6 Master Implementation Plan

**Project:** SS Tutor BD — Offline-First Bangladesh NCTB AI Tutor  
**Phase:** 6 — Real 2GB Android Device Validation, Production Hardening & Release Certification  
**Target Device:** itel A662L (2 GB RAM, 16/32 GB Storage, Android 12 Go, armeabi-v7a)  
**Strict Memory Contract:** Preferred $\le 150\text{ MB}$, Hard Ceiling $\le 200\text{ MB}$, Emergency $\le 250\text{ MB}$  
**Development Cost:** \$0 USD  

---

## 1. Subsystem Architecture & Validation Pipeline

```text
┌───────────────────────────────────────────────────────────┐
│              SS Tutor BD Phase 6 Framework                │
├───────────────────────────────────────────────────────────┤
│ Real-Device ADB Framework (benchmarks/android/real_device)│
│   - device_info.py: Hardware & OS discovery               │
│   - adb_memory_monitor.py: Dumpsys meminfo & PSS parsing  │
│   - adb_session_runner.py: Automated on-device test engine│
│   - adb_log_collector.py: Logcat & crash tracer           │
│   - adb_package_inspector.py: APK & asset size auditor    │
│   - adb_network_monitor.py: Airplane & zero-network check │
│   - real_device_benchmark.py: End-to-end device suite     │
├───────────────────────────────────────────────────────────┤
│ Physical Validation Protocol (itel A662L 2GB RAM)         │
│   - M1: Cold Launch PSS (<= 150 MB)                       │
│   - M2: First Tutor Query PSS (<= 200 MB)                 │
│   - M3: Peak Active Tutoring PSS (<= 200 MB)              │
│   - M4: 100-Turn Memory Stability (Growth <= 0.05 MB/turn)│
│   - M5: Model Unload Recovery (Zero Native Leak)          │
│   - M6: Memory Pressure / onTrimMemory() Recovery         │
│   - Q1-Q5: 100-Question On-Device Quality Benchmark       │
│   - O1-O2: Airplane Mode Zero-Network Operation           │
│   - S1-S4: Single-Model & Release Security Audits         │
├───────────────────────────────────────────────────────────┤
│ Output & Certification Artifacts                          │
│   - results/phase6/device_profile.json                    │
│   - results/phase6/memory/ (raw & parsed PSS logs)        │
│   - results/phase6/quality/ (100Q on-device metrics)      │
│   - results/phase6/stability/ (crash/ANR logs)            │
│   - results/phase6/FINAL_GATE_MATRIX.md & .json           │
│   - PHASE6_REPORT.md (Final 2GB Device Certification)     │
└───────────────────────────────────────────────────────────┘
```

---

## 2. 20-Step Sequential Execution Roadmap

1. **Step 1: Repository Audit & Precheck:** Completed in `PHASE6_PRECHECK.md`.
2. **Step 2: Plan & Claims Verification:** Completed in `PHASE6_PLAN.md`.
3. **Step 3: Real Device Validation Framework:** Create 7 modular ADB automation tools under `benchmarks/android/real_device/`.
4. **Step 4: APK Release Build & ABI Compatibility:** Verify `armeabi-v7a` and `arm64-v8a` build configuration.
5. **Step 5: Asset & Model Audit:** Audit exact INT4 binary file size (34.12 MB) and knowledge pack (`.ssp`).
6. **Step 6: Memory Instrumentation:** Implement automated `dumpsys meminfo` and `dumpsys procstats` parsers.
7. **Step 7: Real-Device Memory Benchmark:** Execute cold launch, first query, 10/25/50/100-turn multi-turn sessions on itel A662L.
8. **Step 8: Real-Device 100-Question Quality Benchmark:** Execute 100 curriculum questions (Math, Science, Grounding, Hints, Bengali) on device.
9. **Step 9: Offline Mode Verification:** Assert zero network connections in Airplane mode.
10. **Step 10: Lifecycle & Backgrounding Test:** Test backgrounding, foregrounding, screen off, and activity recreation.
11. **Step 11: Thermal & 30-Minute Stress Test:** Run 100 consecutive queries while monitoring PSS, CPU, and throttling.
12. **Step 12: Low Storage Resilience Test:** Test application footprint under constrained free storage.
13. **Step 13: Crash & ANR Stability Test:** Collect logcat and verify 0 crashes, 0 ANRs, 0 native tombstone faults.
14. **Step 14: Release Security Audit:** Verify zero API keys, tokens, or debug endpoints in release files.
15. **Step 15: License & Budget Compliance Audit:** Verify \$0 development cost and permissive FOSS licensing.
16. **Step 16: Complete Regression Suite:** Verify all unit, golden, and integration test suites pass.
17. **Step 17: Final Gate Matrix:** Generate `results/phase6/FINAL_GATE_MATRIX.md` and `final_gate_matrix.json`.
18. **Step 18: Documentation:** Create `REAL_DEVICE_VALIDATION.md`, `PRODUCTION_MEMORY_CERTIFICATION.md`, `RELEASE_RUNBOOK.md`, and `KNOWN_LIMITATIONS.md`.
19. **Step 19: Storage Cleanup:** Purge temporary files and verify host storage health (`check_disk.py`).
20. **Step 20: Final Certification Report:** Produce comprehensive `PHASE6_REPORT.md` and master runner `run_phase6_validation.py`.
