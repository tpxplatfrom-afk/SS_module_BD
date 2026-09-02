# FIX-08 — BASELINE VERIFICATION REPORT

**Fix Identifier:** `FIX-08-BASELINE-VERIFICATION`
**Date / Timestamp:** `2026-09-02T21:50:00+06:00`
**Project Scope:** `ss_bangladesh_nano_android_module / THSA-2B V1`
**Absolute Prohibition:** `ss_bangladesh/` and all external modules untouched.

---

## 1. Repository & Working Tree State

- **Active Branch:** `main`
- **Authoritative Baseline Commit SHA:** `2fcc4d507df64a5b2abbcd243c59897f1627712f`
- **Origin Sync Status:** Synchronized with `origin/main`
- **Tracked Build Configuration:**
  - `CMakeLists.txt` verified for host (MSVC x64) and Android NDK (ARM64-v8a).
  - `src/tokenizer/bpe_trie_runtime.cpp` updated with required standard headers.

---

## 2. Authoritative Checkpoint Provenance

### Step-30 Continuation Target Checkpoint
- **Authoritative Google Drive Path:**
  `/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000030.pt`
- **Expected Byte Size:** `4,106,953,961 bytes`
- **Authoritative Cryptographic SHA-256:**
  `0d8d3f31830fd682324708795ab0ebd91b7213a0f28027290216323892f0e667`
- **Step-30 Manifest Path:**
  `/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000030.manifest.json`
- **Manifest Expected SHA-256:**
  `45f6c4c3478825ec6b7d8274ec9d861aa86d660ef3b13a3d67be9856e8fe1d75`
- **Parameter Count:** `2,050,296,320`
- **Trainable Tensors:** `219`
- **NaN / Inf:** `0 / 0 (Finite & Clean)`
- **Global Step:** `30` (20-step continuation from Step 10)

### Step-10 Immutable Baseline Checkpoint
- **Authoritative Google Drive Path:**
  `/content/drive/MyDrive/THSA-2B/checkpoints/checkpoint_step_000010.pt`
- **Expected Byte Size:** `4,106,949,417 bytes`
- **Authoritative Cryptographic SHA-256:**
  `5e83d361a657cb22177d9117b1e31794ec80681efd9f6c60656bf5956709ab99`
- **Immutability Status:** Byte-for-byte verified and frozen across all prior fixes.

---

## 3. Strict Operating Invariants for FIX-08

1. **Zero Retraining:** No GPU or CPU training cycles will be launched.
2. **Zero Checkpoint Mutation:** Checkpoint files remain strictly read-only; no modification, overwrite, or deletion.
3. **No Synthetic Weights:** No fallback, placeholder, or synthetic random weights will be substituted.
4. **No Premature Production Export:** `model.nano` production distribution package will NOT be exported until the native graph is 100% completed, structurally verified, and numerically proven.
5. **Architectural Integrity:** All 219 trainable tensors of `THSAHybridForCausalLM` must be accounted for without fabricated or hand-waved parameter folding.

**Baseline Verification Result:** `PASS`
