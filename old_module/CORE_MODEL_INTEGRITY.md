# SS Bangladesh Core Model Master — Integrity & Reproducibility Specification

**Document Version:** 1.0.0  
**Phase:** 8.2 — Core Model Master Assembly  

---

## 1. Cryptographic Checksum Registry (SHA-256)

All component files inside `models/core/ss_bangladesh/` have been hashed with SHA-256:

```text
====================================================================================================
FILE PATH                                   SHA-256 CHECKSUM
====================================================================================================
model/config.json                           40d6c41b80db296996d99df47ef715562725dd7c31d1ddce59ff8e75cfdcbb80
model/generation_config.json                b3bf27ecb8243beea63fa2898cf88dfb9f30b92e59e9a4f47ce0e0ee0c3886f7
model/model.safetensors                     bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb
tokenizer/tokenizer.json                    69da59ce5a7a28e88bbdbdddfb09b5ca8bc7ae1d533ecf5ca6524316d946e632
tokenizer/tokenizer_config.json             530e137c413b5e43a6d9eb4f3ec18a7c5b64263f112e75e927c9d724ea5ee4be
config/architecture.json                    a938634863372c0c7a5223f03b544da5a864700d33e9bb6cf896cfa7b878297f
====================================================================================================
```

---

## 2. Deterministic Reproducibility Verification

The baseline model can be deterministically reproduced in any Python/PyTorch environment using:

```python
import torch
from training.train_micro_model import build_70m_micro_model

torch.manual_seed(42)
model = build_70m_micro_model(vocab_size=16000)
# Model will instantiate with identical 71,528,256 parameters across 93 tensors
```

---

## 3. Immutability Verification Protocol

To verify that the Core Model Master has not been modified:
```bash
python -c "
import hashlib
from pathlib import Path
p = Path('models/core/ss_bangladesh/model/model.safetensors')
h = hashlib.sha256(p.read_bytes()).hexdigest()
assert h == 'bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb', 'Core Model Master corrupted!'
print('Core Model Master Checksum Verified!')
"
```
