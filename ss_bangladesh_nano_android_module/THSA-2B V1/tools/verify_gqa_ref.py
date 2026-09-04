import struct
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

NANO_PATH = Path("android/src/main/assets/model.nano")
HEADER_FMT = "<4sHHHHIIHHHHIIII20s"
DESC_FMT   = "<IIQQfI"

with open(NANO_PATH, "rb") as f:
    raw = f.read(64)
    magic, version, total_blocks, state_blocks, gqa_blocks, \
    d_model, d_ffn, n_q, n_kv, d_head, pad, vocab_size, max_context, \
    crc32, tensor_count, _res = struct.unpack(HEADER_FMT, raw)
    descs_raw = f.read(tensor_count * 32)
    descs = []
    for i in range(tensor_count):
        tid, qt, off, sz, scale, _pad = struct.unpack(DESC_FMT, descs_raw[i*32:(i+1)*32])
        descs.append(dict(id=tid, qt=qt, off=off, sz=sz, scale=scale))

def load_data(desc):
    with open(NANO_PATH, "rb") as f:
        f.seek(desc["off"])
        return f.read(desc["sz"])

def unpack_ternary(packed, rows, cols, scale):
    k_bytes = cols // 4
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    W = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        row_b = packed_arr[r * k_bytes : (r+1) * k_bytes]
        b0 = (row_b & 1) - ((row_b >> 1) & 1)
        b1 = ((row_b >> 2) & 1) - ((row_b >> 3) & 1)
        b2 = ((row_b >> 4) & 1) - ((row_b >> 5) & 1)
        b3 = ((row_b >> 6) & 1) - ((row_b >> 7) & 1)
        codes = np.stack([b0, b1, b2, b3], axis=1).ravel()[:cols]
        W[r] = codes.astype(np.float32) * scale
    return W

# Load embedding table
emb_desc = descs[0]
emb_data = np.frombuffer(load_data(emb_desc), dtype=np.int8).reshape(65536, 2560).astype(np.float32) * emb_desc["scale"]

# Load Layer 2 (GQA)
base = 19
W_q = unpack_ternary(load_data(descs[base + 0]), 2560, 2560, descs[base + 0]["scale"])
W_k = unpack_ternary(load_data(descs[base + 1]), 512, 2560, descs[base + 1]["scale"])
W_v = unpack_ternary(load_data(descs[base + 2]), 512, 2560, descs[base + 2]["scale"])
W_out = unpack_ternary(load_data(descs[base + 3]), 2560, 2560, descs[base + 3]["scale"])
gamma = np.frombuffer(load_data(descs[base + 4]), dtype=np.float32)

# Load Android dumped ckpt13
android_ckpt13 = np.fromfile("tools/fix12c/android/prompt_0/ckpt13_block_02_gqa_attention.bin", dtype=np.float32)
ref_b_ckpt13 = np.fromfile("tools/fix12c/reference_b/prompt_0/ckpt13_block_02_gqa_attention.bin", dtype=np.float32)
android_q = np.fromfile("tools/fix12c/android/prompt_0/ckpt12a_block_02_gqa_q.bin", dtype=np.float32)
android_k = np.fromfile("tools/fix12c/android/prompt_0/ckpt12b_block_02_gqa_k.bin", dtype=np.float32)
android_v = np.fromfile("tools/fix12c/android/prompt_0/ckpt12c_block_02_gqa_v.bin", dtype=np.float32)

print("Loaded all artifacts.")
print("android_ckpt13 norm:", np.linalg.norm(android_ckpt13))
print("ref_b_ckpt13 norm:  ", np.linalg.norm(ref_b_ckpt13))

# Notice that ref_b_ckpt13 is literally android_v expanded by 5:
ref_b_check = android_v.reshape(4, 128).repeat(5, axis=0).reshape(-1)
print("ref_b vs android_v.repeat(5) cos:", np.dot(ref_b_ckpt13, ref_b_check) / (np.linalg.norm(ref_b_ckpt13)*np.linalg.norm(ref_b_check)))
