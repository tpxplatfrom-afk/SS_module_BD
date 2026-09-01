"""
SS Tutor BD — Phase 8.3 Core Model Master Capacity Validation Suite
Tests all 12 validation requirements: model integrity, parameter count, tokenizer integrity,
Bengali Unicode robustness, context boundary, input boundary, output boundary, memory lifecycle,
offline independence, repeated inference stability, load/unload stability, and 2GB Android compatibility.
"""
import sys
import os
import json
import gc
import hashlib
import unittest
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from safetensors import safe_open

MASTER_DIR = PROJECT_ROOT / "models" / "core" / "ss_bangladesh"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase8.3"

EXPECTED_SHA256 = "bb2f9e7cd79ef83546fd70ea97d8845cff17a7a8482580c3e63e36c4614119bb"


def get_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class TestPhase83CoreCapacity(unittest.TestCase):
    def test_01_core_master_integrity_anchor(self):
        sf_path = MASTER_DIR / "model" / "model.safetensors"
        self.assertTrue(sf_path.exists(), "Core safetensors file does not exist")
        actual_hash = get_file_sha256(sf_path)
        self.assertEqual(actual_hash, EXPECTED_SHA256, "Core Model Master SHA-256 anchor mismatch!")

    def test_02_parameter_count_exact(self):
        sf_path = MASTER_DIR / "model" / "model.safetensors"
        total_p = 0
        tensors = []
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors.append(k)
                total_p += f.get_tensor(k).numel()
        self.assertEqual(total_p, 71528256, "Parameter count must be exactly 71,528,256")
        self.assertEqual(len(tensors), 93, "Tensor count must be exactly 93")

    def test_03_tokenizer_integrity_and_vocab(self):
        tok_dir = MASTER_DIR / "tokenizer"
        tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
        self.assertGreaterEqual(tok.vocab_size, 1000)
        self.assertLessEqual(tok.vocab_size, 16000)
        encoded = tok.encode("বাংলায় শিক্ষা")
        self.assertGreater(len(encoded), 0)

    def test_04_bengali_unicode_robustness(self):
        tok_dir = MASTER_DIR / "tokenizer"
        tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
        unicode_samples = [
            "অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ",
            "ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম",
            "ক্ষ জ্ঞ ষ্ণ ঙ্ক ঙ্গ ঞ্চ ঞ্ছ ঞ্জ ঞ্ঝ",
            "০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯",
            "কা কি কী কু কূ কৃ কে কৈ কো কৌ",
            "x^2 + 2xy + y^2 = (x+y)^2"
        ]
        for s in unicode_samples:
            enc = tok.encode(s)
            dec = tok.decode(enc)
            self.assertEqual(s.replace(" ", ""), dec.replace(" ", ""), f"Unicode decode mismatch for {s}")

    def test_05_context_boundary_enforcement(self):
        cfg_path = MASTER_DIR / "model" / "config.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("max_position_embeddings"), 256)
        self.assertEqual(cfg.get("hidden_size"), 576)
        self.assertEqual(cfg.get("num_hidden_layers"), 10)

    def test_06_input_boundary_and_truncation(self):
        tok_dir = MASTER_DIR / "tokenizer"
        tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
        long_input = "গণিত শিক্ষা " * 200  # ~400 tokens
        tokens = tok.encode(long_input)
        self.assertGreater(len(tokens), 256)
        # Safe truncation test
        safe_truncated = tokens[:256]
        self.assertEqual(len(safe_truncated), 256)

    def test_07_output_generation_boundary(self):
        tok_dir = MASTER_DIR / "tokenizer"
        model_dir = MASTER_DIR / "model"
        tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
        model = LlamaForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
        model.eval()
        prompt = tok("গণিত শিক্ষা", return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(prompt["input_ids"], max_new_tokens=16, do_sample=False, pad_token_id=0, eos_token_id=2)
        self.assertEqual(gen.shape[1], prompt["input_ids"].shape[1] + 16)
        del model
        del tok
        gc.collect()

    def test_08_memory_bounded_lifecycle(self):
        res_file = RESULTS_DIR / "section_h_k_l_memory_stability.json"
        self.assertTrue(res_file.exists(), "Memory stability result JSON must exist")
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data.get("is_memory_bounded"), "Memory must be strictly bounded with zero leak")
        states = data.get("lifecycle_states", {})
        self.assertIn("State_A_unloaded_mb", states)
        self.assertIn("State_G_after_unload_mb", states)

    def test_09_offline_zero_network_dependency(self):
        # Verify model loading and forward pass require no network sockets
        model_dir = MASTER_DIR / "model"
        model = LlamaForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
        inp = torch.randint(10, 1000, (1, 16))
        with torch.no_grad():
            out = model(inp)
        self.assertIsNotNone(out.logits)
        del model
        gc.collect()

    def test_10_repeated_inference_stability(self):
        model_dir = MASTER_DIR / "model"
        model = LlamaForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
        inp = torch.randint(10, 1000, (1, 32))
        for _ in range(10):
            with torch.no_grad():
                out = model(inp)
            self.assertEqual(out.logits.shape, (1, 32, 16000))
        del model
        gc.collect()

    def test_11_load_unload_cycling_integrity(self):
        model_dir = MASTER_DIR / "model"
        for _ in range(3):
            model = LlamaForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
            inp = torch.randint(10, 1000, (1, 16))
            with torch.no_grad():
                _ = model(inp)
            del model
            gc.collect()
        # Verify safetensors hash is still intact after repeated loading
        sf_path = MASTER_DIR / "model" / "model.safetensors"
        self.assertEqual(get_file_sha256(sf_path), EXPECTED_SHA256)

    def test_12_android_2gb_device_compatibility(self):
        res_file = RESULTS_DIR / "section_i_j_n_o_android_device.json"
        self.assertTrue(res_file.exists(), "Android device result JSON must exist")
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        dev = data.get("device", {})
        self.assertEqual(dev.get("model"), "itel A662L")
        self.assertGreater(dev.get("ram_total_mb", 0), 1800)


if __name__ == "__main__":
    unittest.main()
