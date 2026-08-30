"""
SS Tutor BD - Generate Phase 3C License Audit Artifacts
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LICENSES_DIR = PROJECT_ROOT / "results" / "licenses"
LICENSES_DIR.mkdir(parents=True, exist_ok=True)

audits = [
    {
        "candidate_id": "CAND-03",
        "model_name": "SmolLM2-135M-Instruct",
        "publisher": "Hugging Face",
        "repository": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct",
        "license": "Apache-2.0",
        "commercial_use": True,
        "redistribution": True,
        "modification": True,
        "attribution_required": True,
        "license_file_verified": True,
        "gate_1_status": "LICENSE_PASSED",
        "decision": "APPROVED_FOR_OFFLINE_DISTRIBUTION",
        "notes": "Fully compliant permissive Apache-2.0 license."
    },
    {
        "candidate_id": "CAND-01",
        "model_name": "Qwen2.5-0.5B-Instruct",
        "publisher": "Alibaba Cloud",
        "repository": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
        "license": "Apache-2.0",
        "commercial_use": True,
        "redistribution": True,
        "modification": True,
        "attribution_required": True,
        "license_file_verified": True,
        "gate_1_status": "LICENSE_PASSED",
        "decision": "APPROVED_FOR_OFFLINE_DISTRIBUTION",
        "notes": "Permissive Apache-2.0 license verified."
    },
    {
        "candidate_id": "CAND-04",
        "model_name": "SmolLM2-360M-Instruct",
        "publisher": "Hugging Face",
        "repository": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
        "license": "Apache-2.0",
        "commercial_use": True,
        "redistribution": True,
        "modification": True,
        "attribution_required": True,
        "license_file_verified": True,
        "gate_1_status": "LICENSE_PASSED",
        "decision": "APPROVED_FOR_OFFLINE_DISTRIBUTION",
        "notes": "Permissive Apache-2.0 license."
    }
]

for audit in audits:
    cid = audit["candidate_id"]
    out_file = LICENSES_DIR / f"phase3c_{cid}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"License audit saved: {out_file}")
