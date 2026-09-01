# SS Tutor BD — Release Packaging & Distribution Architecture

**Document Version:** 1.0.0  
**Phase:** 5 — Offline APK Packaging  

---

## 1. Single Model on Disk & Asset Package Layout

The final production Android APK/AAB bundle contains only verified production artifacts:

```text
sstutor-bd-release.apk
├── lib/
│   ├── arm64-v8a/
│   └── armeabi-v7a/
├── assets/
│   ├── models/
│   │   └── sstutor_bengali_70m_int4.bin    (34.12 MB INT4 Model Bundle)
│   ├── tokenizer/
│   │   └── tokenizer.json                   (16K Dedicated Bengali BPE)
│   └── packs/
│       └── class8_math.ssp                  (164 KB SQLite FTS5 Curriculum Pack)
└── classes.dex                              (Native Android App Bytecode)
```

### Prohibited from Release Artifacts:
* No training checkpoints or optimizer states (`.pt`, `.bin`, `.safetensors` raw files).
* No Python interpreters or `.whl` files.
* No duplicate model variants (Qwen, SmolLM, TinyLlama).
* No API keys, credentials, or development scratch files.
