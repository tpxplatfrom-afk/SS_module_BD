"""
SS Tutor BD — Phase 8.2 Core Model Master Assembly Script
Assembles the canonical reusable Core Model Master bundle: `models/core/ss_bangladesh/` and `ss_bangladesh/`.
Includes untrained baseline weights (deterministic seed 42), 16K Bengali Tokenizer,
architecture configs, manifest, lineage, and SHA-256 integrity checksums.
"""
import sys
import os
import json
import shutil
import hashlib
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from training.train_micro_model import build_70m_micro_model

MASTER_DIR = PROJECT_ROOT / "models" / "core" / "ss_bangladesh"
TOKENIZER_SRC = PROJECT_ROOT / "models" / "tokenizer_bengali_16k"


def calculate_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def assemble_core_master(seed: int = 42) -> dict:
    print("\n" + "=" * 75)
    print("  SS TUTOR BD — PHASE 8.2: CORE MODEL MASTER ASSEMBLY (ss_bangladesh)")
    print("=" * 75)

    # 1. Clean destination directory
    if MASTER_DIR.exists():
        shutil.rmtree(MASTER_DIR)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = MASTER_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = MASTER_DIR / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = MASTER_DIR / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # 2. Instantiate Deterministic Untrained Baseline Model (Seed 42)
    torch.manual_seed(seed)
    print(f"[Assembly] Instantiating baseline architecture with seed {seed}...")
    model = build_70m_micro_model(vocab_size=16000)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Assembly] Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 3. Save Baseline Model & Configs to model/
    print("[Assembly] Saving model weights and configs to model/...")
    model.save_pretrained(str(model_dir))
    
    # 4. Copy 16K Bengali Tokenizer to tokenizer/
    print("[Assembly] Copying canonical 16K Bengali Tokenizer to tokenizer/...")
    for f in TOKENIZER_SRC.glob("*"):
        if f.is_file():
            shutil.copy2(f, tok_dir / f.name)

    # 5. Save Architecture Config to config/
    arch_config = {
        "model_id": "ss_bangladesh_core_v0.8.2",
        "model_name": "SS Bangladesh Core Model Master",
        "model_family": "SS_Bangladesh_Transformer",
        "version": "0.8.2",
        "status": "UNTRAINED_REUSABLE_BASELINE",
        "architecture_class": "LlamaForCausalLM",
        "seed": seed,
        "config": {
            "vocab_size": 16000,
            "hidden_size": 576,
            "intermediate_size": 2304,
            "num_hidden_layers": 10,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 256,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False
        },
        "parameter_metrics": {
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "layers_count": 10,
            "attention_heads": 8,
            "context_length": 256
        }
    }
    with open(cfg_dir / "architecture.json", "w", encoding="utf-8") as f:
        json.dump(arch_config, f, indent=2, ensure_ascii=False)

    # 6. Generate Checksums
    checksums = {}
    for p in MASTER_DIR.rglob("*"):
        if p.is_file() and p.name != "checksums.sha256" and p.name != "manifest.json":
            rel = str(p.relative_to(MASTER_DIR)).replace('\\', '/')
            checksums[rel] = calculate_sha256(p)

    with open(MASTER_DIR / "checksums.sha256", "w", encoding="utf-8") as f:
        for fpath, h in sorted(checksums.items()):
            f.write(f"{h}  {fpath}\n")

    # 7. Generate Machine-Readable Master Manifest
    manifest = {
        "manifest_version": "1.0.0",
        "model_id": "ss_bangladesh",
        "canonical_name": "SS Bangladesh Core Model Master",
        "model_family": "SS_Bangladesh",
        "version": "0.8.2",
        "model_type": "Causal Autoregressive Transformer",
        "architecture": "LlamaForCausalLM (10L / 576H / 2304FFN / 8A / 256Ctx)",
        "parameter_count": total_params,
        "parameter_count_str": f"{round(total_params/1e6, 2)}M",
        "tensor_count": len(list(model.state_dict().keys())),
        "dtype": "float32",
        "vocabulary_size": 16000,
        "context_length": 256,
        "tokenizer_id": "tokenizer_bengali_16k",
        "tokenizer_version": "0.4.0",
        "tokenizer_tokens_per_word": "3.65 - 3.86",
        "base_checkpoint_type": "UNTRAINED_RECONSTRUCTABLE_BASELINE",
        "training_status": "UNTRAINED_REUSABLE_BASE",
        "domain_status": "DOMAIN_NEUTRAL (No Curriculum/Domain Bias)",
        "creation_timestamp": "2026-08-30T23:30:00+06:00",
        "initialization_seed": seed,
        "initialization_method": "Truncated Normal Distribution (sigma=0.02)",
        "license": "Apache-2.0",
        "files_checksums_sha256": checksums,
        "lineage": {
            "parent": None,
            "type": "ROOT_MASTER_CORE_AI",
            "downstream_specializations": [
                {
                    "specialization_id": "ss_tutor_bd",
                    "domain": "Bangladesh High School Education (NCTB Class 6-10)",
                    "active_checkpoint": "models/sstutor_bengali_70m_edu/model.safetensors"
                },
                {
                    "specialization_id": "mechanics_module",
                    "domain": "Applied Physics & Mechanics Education",
                    "active_checkpoint": "PLANNED_FUTURE_FORK"
                }
            ]
        }
    }
    with open(MASTER_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 8. Generate Lineage Specification
    lineage_doc = {
        "root_master": "ss_bangladesh",
        "lineage_graph": {
            "node_0": {
                "name": "SS Bangladesh Core Model Master",
                "role": "Root Reusable Foundation",
                "training_level": "Untrained / Domain-Neutral",
                "location": "models/core/ss_bangladesh/"
            },
            "branch_1": {
                "name": "SS Tutor BD Specialization",
                "role": "NCTB Class 6-10 High School Tutor",
                "training_level": "Domain-Trained (Class 8 Math baseline active)",
                "location": "models/sstutor_bengali_70m_edu/"
            },
            "branch_2": {
                "name": "Mechanics AI Module Specialization",
                "role": "Applied Physics & Mechanics Tutor",
                "training_level": "Planned Future Fork",
                "location": "models/specializations/mechanics/"
            }
        }
    }
    with open(MASTER_DIR / "lineage.json", "w", encoding="utf-8") as f:
        json.dump(lineage_doc, f, indent=2, ensure_ascii=False)

    # 9. Generate Master README.md
    readme_content = f"""# SS Bangladesh Core Model Master (`ss_bangladesh`)

**Canonical Identity:** SS Bangladesh Core Model Master  
**Version:** 0.8.2  
**Role:** Root Reusable AI Foundation for Bengali-First Educational Specializations  
**Training Status:** **UNTRAINED / DOMAIN-NEUTRAL BASELINE**  

---

## 1. Overview & Architectural Role

`ss_bangladesh` is the authoritative master core model from which all downstream specialized models (such as **SS Tutor BD** and **Mechanics**) are forked and trained.

```text
                    SS BANGLADESH CORE MODEL
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
        SS Tutor BD        Mechanics        Future Niche
      (Class 6–10 NCTB) (Applied Physics) (Custom Domain)
```

---

## 2. Parameter & Layer Specifications

* **Architecture Class:** `LlamaForCausalLM`
* **Layers:** 10
* **Hidden Dimension ($d_{{\\text{{model}}}}$):** 576
* **Intermediate Dimension ($d_{{\\text{{ffn}}}}$):** 2,304 (SwiGLU activation)
* **Attention Heads:** 8
* **Key-Value Heads:** 8
* **Context Length ($L_{{\\text{{ctx}}}}$):** 256 tokens
* **Total Parameters:** **{total_params:,} ({round(total_params/1e6, 2)}M)**
* **Tokenizer:** 16,000 Byte-level BPE (`tokenizer/`)
* **Initialization Seed:** {seed} (Truncated Normal, $\\sigma = 0.02$)
* **Model Checksum (model.safetensors):** `{checksums.get('model/model.safetensors', 'N/A')}`

---

## 3. Immutability & Domain Neutrality

* This artifact contains **ZERO curriculum bias or hardcoded textbook facts**.
* Downstream training and domain knowledge packs are strictly isolated in downstream specializations.
"""
    with open(MASTER_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Also sync root level bundle copy/pointer if needed
    root_master = PROJECT_ROOT / "ss_bangladesh"
    if root_master.exists():
        shutil.rmtree(root_master)
    shutil.copytree(MASTER_DIR, root_master)

    # Write root CORE_MODEL_MANIFEST.json
    with open(PROJECT_ROOT / "CORE_MODEL_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[Assembly Complete] Core Model Master successfully assembled at:")
    print(f"  - Primary Repository Path: {MASTER_DIR}")
    print(f"  - Root Master Bundle Path: {root_master}")
    print(f"  - Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - Safetensors Checksum: {checksums.get('model/model.safetensors', 'N/A')}")
    print("=" * 75 + "\n")

    return manifest


if __name__ == "__main__":
    assemble_core_master()
