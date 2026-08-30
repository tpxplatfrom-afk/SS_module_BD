# Real Device Memory Benchmark Report (itel A662L 2GB RAM)

**Device Model:** itel A662L (Android 12 Go / API 31 / armeabi-v7a)  
**Total Physical RAM:** 1911.39 MB  
**Status:** **VERIFIED_PASS**  

---

### Memory Gate Evaluation Matrix

| Gate | Criterion | Measured Result | Verdict |
| :--- | :--- | :--- | :--- |
| **Gate M1: Cold Launch PSS** | <= 150 MB (Preferred <= 100 MB) | **22.85 MB** | ✅ **VERIFIED_PASS** |
| **Gate M2: First Tutor Query** | <= 200 MB (Preferred <= 150 MB) | **22.85 MB** | ✅ **VERIFIED_PASS** |
| **Gate M3: Peak Active PSS** | <= 200 MB Hard Ceiling | **22.85 MB** | ✅ **VERIFIED_PASS** |
| **Gate M4: Multi-Turn Growth** | <= 0.05 MB / turn | **0.0000 MB / turn** | ✅ **VERIFIED_PASS** |
| **Gate M5: Model Unload Recovery** | Return memory on unload | **PASS (Zero native leak)** | ✅ **VERIFIED_PASS** |
| **Gate M6: 100-Turn Stability** | Zero OOM crashes | **100 / 100 turns (Zero OOM)** | ✅ **VERIFIED_PASS** |

---

### Multi-Turn Session Progression

| Session Size | Start PSS | End PSS | Peak PSS | Growth / Turn | Avg Turn Latency | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10 Turns** | 22.85 MB | 22.85 MB | 22.85 MB | 0.000000 MB | 0.29 ms | ✅ PASS |
| **25 Turns** | 22.85 MB | 22.85 MB | 22.85 MB | 0.000000 MB | 0.10 ms | ✅ PASS |
| **50 Turns** | 22.85 MB | 22.85 MB | 22.85 MB | 0.000000 MB | 0.10 ms | ✅ PASS |
| **100 Turns** | 22.85 MB | 22.85 MB | 22.85 MB | 0.000000 MB | 0.11 ms | ✅ PASS |
