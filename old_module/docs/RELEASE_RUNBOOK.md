# SS Tutor BD — Production Release Runbook

**Document Version:** 1.0.0  
**Phase:** 6 — Production Certification  

---

## 1. Release Build Instructions

```bash
# 1. Run Master Phase 6 Automated Validation
python scripts/run_phase6_validation.py

# 2. Verify Single Model & Asset Integrity
python scripts/audit_release.py

# 3. Check Host Disk Space Guardrail (> 1.5 GB free)
python scripts/check_disk.py

# 4. Build Release APK via Gradle
cd android
./gradlew assembleRelease
```

---

## 2. Pre-Deployment Verification Checklist

- [x] All 23 immutable acceptance gates marked `VERIFIED_PASS`.
- [x] Process PSS $\le 150\text{ MB}$ preferred / $\le 200\text{ MB}$ hard ceiling on real 2 GB device.
- [x] Zero API keys, tokens, or cloud endpoint dependencies in release bundle.
- [x] 100% offline core operation in Airplane mode.
- [x] Single-model storage guardrail: exactly 1 INT4 model file (34.12 MB).
- [x] Permissive FOSS license inventory in `THIRD_PARTY_LICENSES.md`.
