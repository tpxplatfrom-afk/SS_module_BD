import os
import shutil
import hashlib

src_base = r"ss_bangladesh_nano_android_module\THSA-2B V1"
dst_base = r"ss_bangladesh_nano_android_module\THSA-2B_V2_helper"

sync_files = [
    r"data\build_bilingual_sharegpt_dataset.py",
    r"data\train_sharegpt.jsonl",
    r"data\test_sharegpt.jsonl",
    r"training\models\ternary_layers.py",
    r"training\train_qat.py",
    r"tools\export_to_nano.py",
    r"notebooks\THSA_2B_Colab_Training.ipynb"
]

print("Synchronizing V1 training and dataset files to THSA-2B_V2_helper...")
for rel in sync_files:
    s = os.path.join(src_base, rel)
    d = os.path.join(dst_base, rel)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    
    # Hash check
    h_s = hashlib.sha256(open(s, "rb").read()).hexdigest()
    h_d = hashlib.sha256(open(d, "rb").read()).hexdigest()
    assert h_s == h_d, f"Mismatch in {rel}"
    print(f"  [OK] {rel} -> SHA256: {h_s[:16]}...")

print("\n100% BYTE-LEVEL SYNCHRONIZATION CONFIRMED!")
