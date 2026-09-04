import struct
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

NANO_PATH = Path("android/src/main/assets/model.nano")
DESC_FMT = "<IIQQfI"

with open(NANO_PATH, "rb") as f:
    hdr = f.read(64)
    magic, ver, total_b, st_b, gqa_b, d_m, d_f, n_q, n_kv, d_h, v_s, max_c, crc, tc = struct.unpack("<4sHHHHIIIIIIIII", hdr[:52])
    print(f"model.nano: ver={hex(ver)} total={total_b} gqa={gqa_b} d_model={d_m} n_q={n_q} n_kv={n_kv} d_head={d_h}")
    descs_raw = f.read(tc * 32)

descs = []
for i in range(tc):
    tid, qt, off, sz, scale, _pad = struct.unpack(DESC_FMT, descs_raw[i*32:(i+1)*32])
    descs.append(dict(id=tid, qt=qt, off=off, sz=sz, scale=scale))

print("\nLayer 2 descriptors (base=19):")
for j in range(5):
    d = descs[19+j]
    print(f"  slot {j}: id={d['id']} qt={d['qt']} off={d['off']} sz={d['sz']} scale={d['scale']}")
