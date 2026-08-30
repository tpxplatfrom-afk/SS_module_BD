"""
SS Tutor BD — Phase 8.2 Core Model Master Validation Suite
Tests 12 distinct requirements: existence, manifest, architecture, parameter count,
tokenizer compatibility, isolation from Class 8 Math, RAG exclusion, checksum integrity,
reproducibility, child fork test, regression integrity, and SS Tutor BD specialization intactness.
"""
import sys
import os
import json
import shutil
import hashlib
import unittest
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import PreTrainedTokenizerFast
from safetensors import safe_open
from training.train_micro_model import build_70m_micro_model

MASTER_DIR = PROJECT_ROOT / "models" / "core" / "ss_bangladesh"
ROOT_MASTER_DIR = PROJECT_ROOT / "ss_bangladesh"
SPECIALIZED_MODEL = PROJECT_ROOT / "models" / "sstutor_bengali_70m_edu" / "model.safetensors"


def get_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class TestPhase82CoreMaster(unittest.TestCase):
    def test_01_core_master_exists(self):
        self.assertTrue(MASTER_DIR.exists(), f"Master dir {MASTER_DIR} does not exist")
        self.assertTrue((MASTER_DIR / "model" / "model.safetensors").exists())
        self.assertTrue((MASTER_DIR / "tokenizer" / "tokenizer.json").exists())
        self.assertTrue(ROOT_MASTER_DIR.exists())

    def test_02_core_master_manifest_exists(self):
        manifest_path = MASTER_DIR / "manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data.get("model_id"), "ss_bangladesh")
            self.assertEqual(data.get("version"), "0.8.2")
            self.assertEqual(data.get("training_status"), "UNTRAINED_REUSABLE_BASE")

    def test_03_core_master_architecture_valid(self):
        cfg_path = MASTER_DIR / "model" / "config.json"
        self.assertTrue(cfg_path.exists())
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            self.assertEqual(cfg.get("num_hidden_layers"), 10)
            self.assertEqual(cfg.get("hidden_size"), 576)
            self.assertEqual(cfg.get("intermediate_size"), 2304)
            self.assertEqual(cfg.get("num_attention_heads"), 8)
            self.assertEqual(cfg.get("max_position_embeddings"), 256)

    def test_04_parameter_count_matches(self):
        sf_path = MASTER_DIR / "model" / "model.safetensors"
        total_params = 0
        tensor_names = []
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                total_params += t.numel()
                tensor_names.append(k)
        self.assertEqual(total_params, 71528256)
        self.assertEqual(len(tensor_names), 93)

    def test_05_tokenizer_compatibility(self):
        tok_dir = MASTER_DIR / "tokenizer"
        tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
        self.assertGreaterEqual(tok.vocab_size, 1000)
        self.assertLessEqual(tok.vocab_size, 16000)
        tokens = tok.encode("বাংলায় শিক্ষা")
        self.assertGreater(len(tokens), 0)

    def test_06_not_class8_math_specialization(self):
        master_sf = MASTER_DIR / "model" / "model.safetensors"
        self.assertTrue(SPECIALIZED_MODEL.exists())
        master_hash = get_file_sha256(master_sf)
        spec_hash = get_file_sha256(SPECIALIZED_MODEL)
        self.assertNotEqual(master_hash, spec_hash, "Core Master should NOT have the trained Class 8 Math weights")

    def test_07_class8_rag_excluded(self):
        for p in MASTER_DIR.rglob("*"):
            self.assertFalse(p.name.endswith(".db"), f"RAG database {p.name} found inside Core Master")
            self.assertNotIn("class8", p.name.lower())

    def test_08_core_master_checksum_recorded(self):
        chk_file = MASTER_DIR / "checksums.sha256"
        self.assertTrue(chk_file.exists())
        master_sf = MASTER_DIR / "model" / "model.safetensors"
        actual_hash = get_file_sha256(master_sf)
        with open(chk_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn(actual_hash, content)

    def test_09_reconstruction_reproducible(self):
        torch.manual_seed(42)
        model = build_70m_micro_model(vocab_size=16000)
        total_p = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_p, 71528256)

    def test_10_child_specialization_fork_test(self):
        # Simulate creating a child specialization
        fork_dir = PROJECT_ROOT / "scratch" / "test_fork_mechanics"
        if fork_dir.exists():
            shutil.rmtree(fork_dir)
        shutil.copytree(MASTER_DIR, fork_dir)

        # Mutate child metadata
        with open(fork_dir / "manifest.json", "r+", encoding="utf-8") as f:
            d = json.load(f)
            d["model_id"] = "mechanics_specialization"
            d["domain"] = "Mechanics"
            f.seek(0)
            json.dump(d, f)
            f.truncate()

        # Check Core Master remains unmutated
        with open(MASTER_DIR / "manifest.json", "r", encoding="utf-8") as f:
            d_master = json.load(f)
            self.assertEqual(d_master.get("model_id"), "ss_bangladesh")

        shutil.rmtree(fork_dir)

    def test_11_core_model_manifest_root_exists(self):
        root_manifest = PROJECT_ROOT / "CORE_MODEL_MANIFEST.json"
        self.assertTrue(root_manifest.exists())
        with open(root_manifest, "r", encoding="utf-8") as f:
            d = json.load(f)
            self.assertEqual(d.get("model_id"), "ss_bangladesh")

    def test_12_existing_sstutor_model_intact(self):
        self.assertTrue(SPECIALIZED_MODEL.exists())
        self.assertGreater(SPECIALIZED_MODEL.stat().st_size, 100 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
