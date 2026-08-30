# Android Production Memory Architecture Specification

**Document Version:** 1.0.0  
**Phase:** 5 — Android / Native UI Binding  
**Target Hardware:** 2 GB Physical RAM / 16 GB Storage  

---

## 1. Process Memory Breakdown & Subsystem Allocations

```
========================================================================
SUBSYSTEM                              ALLOCATION (MB)   MAX CEILING (MB)
========================================================================
1. Dalvik / ART Java Heap (UI & App)   15 – 25 MB        35 MB
2. Native Heap (INT4 Model mmap)       34 – 40 MB        50 MB
3. MicroRuntime Execution & KV Cache   15 – 25 MB        35 MB
4. SQLite FTS5 RAG Index Buffer         2 – 5 MB         10 MB
5. Deterministic Math & Validators      1 – 3 MB          5 MB
6. Safety Headroom Buffer Margin       35 – 45 MB        65 MB
------------------------------------------------------------------------
TOTAL TARGET APPLICATION PSS          102 – 143 MB      <= 200 MB (Hard)
PREFERRED PROCESS WORKING SET         <= 150 MB
========================================================================
```

---

## 2. Hard Memory State Transitions & Action Triggers

```
        PSS < 150 MB                    150 MB <= PSS <= 200 MB
     ┌─────────────────┐                 ┌─────────────────┐
     │  NORMAL STATE   │ ──────────────> │  WARNING STATE  │
     │ Full operation  │ <────────────── │ Trim RAG cache  │
     └─────────────────┘                 └─────────────────┘
                                                  │
                                                  │ PSS > 200 MB
                                                  ▼
     ┌─────────────────┐                 ┌─────────────────┐
     │ EMERGENCY STATE │ <────────────── │ CRITICAL STATE  │
     │ Unload model    │   PSS >= 250 MB │ Reduce context  │
     │ Fallback only   │                 │ Stop background │
     └─────────────────┘                 └─────────────────┘
```

* **NORMAL ($< 150\text{ MB}$):** Standard operation; lazy model caching allowed for up to 60 seconds of inactivity.
* **WARNING ($150–200\text{ MB}$):** Flush inactive RAG chunk caches; restrict max generation tokens to 64.
* **CRITICAL ($200–250\text{ MB}$):** Immediately unload neural model; reject concurrent tasks; transition to deterministic-only mode.
* **EMERGENCY ($\ge 250\text{ MB}$):** Force garbage collection; release all non-essential buffers; retain only active `SessionState` string.
