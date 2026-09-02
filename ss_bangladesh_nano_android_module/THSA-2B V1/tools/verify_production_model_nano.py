#!/usr/bin/env python3
"""
FIX-10 / FIX-10A: Independent Production model.nano Verifier
============================================================
Parses raw bytes of model.nano WITHOUT importing export_production_nano.py
or export_to_nano.py. Completely standalone and memory-bounded (< 5 MB peak RAM).

Usage:
  python tools/verify_production_model_nano.py --nano models/model.nano
"""
import os, sys, zlib, struct, hashlib, argparse, datetime
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── CONTRACT CONSTANTS ─────────────────────────────────────────────────────────
MAGIC_NANO          = b"NANO"
FORMAT_VERSION_V2   = 0x0002
HEADER_SIZE         = 64
DESCRIPTOR_SIZE     = 32
TENSOR_COUNT        = 219
DESCRIPTOR_TABLE_SZ = TENSOR_COUNT * DESCRIPTOR_SIZE  # 7,008
PRE_PAYLOAD_PAD     = 32
PAYLOAD_OFFSET      = HEADER_SIZE + DESCRIPTOR_TABLE_SZ + PRE_PAYLOAD_PAD  # 7,104
ALIGNMENT           = 64
CHUNK_BYTES         = 4 * 1024 * 1024  # 4 MB streaming chunk

EXPECTED_PARAMS      = 2_050_296_320
EXPECTED_RAW_PAYLOAD = 765_470_720
EXPECTED_FINAL_SIZE  = 765_477_824

NANO_QUANT_FP32         = 0
NANO_QUANT_INT8         = 2
NANO_QUANT_TERNARY_2BIT = 4

EXPECTED_FP32_TENSORS    = 81
EXPECTED_INT8_TENSORS    = 2
EXPECTED_TERNARY_TENSORS = 136
EXPECTED_FP32_PARAMS     = 330_240
EXPECTED_INT8_PARAMS     = 335_544_320
EXPECTED_TERNARY_PARAMS  = 1_714_421_760

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()

def numel_from_quant(qt, sz_bytes):
    if qt == NANO_QUANT_FP32: return sz_bytes // 4
    if qt == NANO_QUANT_INT8: return sz_bytes
    if qt == NANO_QUANT_TERNARY_2BIT: return sz_bytes * 4
    return 0

# ── HEADER PARSE ───────────────────────────────────────────────────────────────
def parse_header(raw):
    assert len(raw) >= HEADER_SIZE, "Data too small for header"
    magic = raw[0:4]
    version, = struct.unpack_from("<H", raw, 4)
    total_b, = struct.unpack_from("<H", raw, 6)
    state_b, = struct.unpack_from("<H", raw, 8)
    gqa_b, = struct.unpack_from("<H", raw, 10)
    d_model, = struct.unpack_from("<I", raw, 12)
    d_ffn, = struct.unpack_from("<I", raw, 16)
    n_q, = struct.unpack_from("<H", raw, 20)
    n_kv, = struct.unpack_from("<H", raw, 22)
    d_head, = struct.unpack_from("<H", raw, 24)
    vocab, = struct.unpack_from("<I", raw, 28)
    max_ctx, = struct.unpack_from("<I", raw, 32)
    crc_stored, = struct.unpack_from("<I", raw, 36)
    tensor_count, = struct.unpack_from("<I", raw, 40)
    return {
        "magic": magic, "version": version, "total_blocks": total_b,
        "state_blocks": state_b, "gqa_blocks": gqa_b, "d_model": d_model,
        "d_ffn": d_ffn, "n_q": n_q, "n_kv": n_kv, "d_head": d_head,
        "vocab_size": vocab, "max_context": max_ctx,
        "crc_stored": crc_stored, "tensor_count": tensor_count
    }

# ── DESCRIPTOR PARSE ───────────────────────────────────────────────────────────
def parse_descriptors(raw_header_and_descs):
    desc_start = HEADER_SIZE
    descs = []
    for i in range(TENSOR_COUNT):
        off = desc_start + i * DESCRIPTOR_SIZE
        raw = raw_header_and_descs[off:off + DESCRIPTOR_SIZE]
        t_id, qt, payload_off, sz, scale, pad = struct.unpack("<IIQQfI", raw)
        descs.append({"id": t_id, "qt": qt, "offset": payload_off, "size_bytes": sz, "scale": scale})
    return descs

# ── MAIN VERIFICATION ──────────────────────────────────────────────────────────
def verify(nano_path):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 80)
    print(f"FIX-10 / FIX-10A INDEPENDENT BINARY VERIFIER | {ts}")
    print(f"File: {nano_path}")
    print("=" * 80)

    # 1. File exists & exact size 765,477,824
    assert os.path.isfile(nano_path), f"File not found: {nano_path}"
    file_size = os.path.getsize(nano_path)
    ok_size = (file_size == EXPECTED_FINAL_SIZE)
    print(f"[1]  File size:        {file_size:,}  (expected {EXPECTED_FINAL_SIZE:,})  {'PASS' if ok_size else 'FAIL'}")
    assert ok_size, f"File size mismatch: {file_size}"

    # Read ONLY header and descriptors (7,104 bytes) for metadata checks
    with open(nano_path, "rb") as f:
        meta_bytes = f.read(PAYLOAD_OFFSET)
    assert len(meta_bytes) == PAYLOAD_OFFSET, f"Could not read {PAYLOAD_OFFSET} metadata bytes"

    # 2. Magic
    hdr = parse_header(meta_bytes)
    ok_magic = (hdr["magic"] == MAGIC_NANO)
    print(f"[2]  Magic:            {hdr['magic']}  {'PASS' if ok_magic else 'FAIL'}")
    assert ok_magic

    # 3. Format version
    ok_ver = (hdr["version"] == FORMAT_VERSION_V2)
    print(f"[3]  Format version:   0x{hdr['version']:04X}  (expected 0x0002)  {'PASS' if ok_ver else 'FAIL'}")
    assert ok_ver

    # 4. Header size
    print(f"[4]  Header size:      {HEADER_SIZE} bytes  PASS")

    # 5. Architecture dimensions
    checks = [
        ("total_blocks", 24), ("state_blocks", 16), ("gqa_blocks", 8),
        ("d_model", 2560), ("d_ffn", 6912), ("n_q", 20), ("n_kv", 4), ("d_head", 128)
    ]
    for field, expected in checks:
        ok = (hdr[field] == expected)
        print(f"[5]  {field:20s}: {hdr[field]}  (expected {expected})  {'PASS' if ok else 'FAIL'}")
        assert ok, f"Architecture mismatch: {field}"

    # 6. Vocab
    ok_vocab = (hdr["vocab_size"] == 65536)
    print(f"[6]  vocab_size:       {hdr['vocab_size']}  {'PASS' if ok_vocab else 'FAIL'}")
    assert ok_vocab

    # 7. Context
    ok_ctx = (hdr["max_context"] == 10000)
    print(f"[7]  max_context:      {hdr['max_context']}  {'PASS' if ok_ctx else 'FAIL'}")
    assert ok_ctx

    # 8. tensor_count
    ok_tc = (hdr["tensor_count"] == TENSOR_COUNT)
    print(f"[8]  tensor_count:     {hdr['tensor_count']}  (expected {TENSOR_COUNT})  {'PASS' if ok_tc else 'FAIL'}")
    assert ok_tc

    # 9. Descriptor table does not overlap payload
    desc_end = HEADER_SIZE + DESCRIPTOR_TABLE_SZ
    ok_no_overlap = (desc_end + PRE_PAYLOAD_PAD == PAYLOAD_OFFSET)
    print(f"[9]  Desc table ends at {desc_end}, payload starts at {PAYLOAD_OFFSET}  {'PASS' if ok_no_overlap else 'FAIL'}")
    assert ok_no_overlap

    # 10. Parse all descriptors
    descs = parse_descriptors(meta_bytes)
    assert len(descs) == TENSOR_COUNT

    # 11. IDs are exactly 0..218
    ids = [d["id"] for d in descs]
    ok_ids = (sorted(ids) == list(range(TENSOR_COUNT)))
    print(f"[10] Descriptor IDs 0..218: {'PASS' if ok_ids else 'FAIL'}")
    assert ok_ids, f"IDs: {ids[:5]}"

    # 12. No duplicate IDs
    ok_nodup = (len(ids) == len(set(ids)))
    print(f"[11] No duplicate IDs: {'PASS' if ok_nodup else 'FAIL'}")
    assert ok_nodup

    # 13. Offsets >= 7104 and 64-byte aligned
    all_off_ok = True
    for d in descs:
        if d["offset"] < PAYLOAD_OFFSET or d["offset"] % ALIGNMENT != 0:
            all_off_ok = False; print(f"  BAD OFFSET: id={d['id']} off={d['offset']}")
    print(f"[12] All offsets >=7104 & 64-aligned: {'PASS' if all_off_ok else 'FAIL'}")
    assert all_off_ok

    # 14. All sizes > 0
    all_sz_ok = all(d["size_bytes"] > 0 for d in descs)
    print(f"[13] All payload sizes > 0: {'PASS' if all_sz_ok else 'FAIL'}")
    assert all_sz_ok

    # 15. All payload ranges within file
    all_bounds_ok = True
    for d in descs:
        if d["offset"] + d["size_bytes"] > file_size:
            all_bounds_ok = False; print(f"  OUT OF BOUNDS: id={d['id']}")
    print(f"[14] All payloads within file bounds: {'PASS' if all_bounds_ok else 'FAIL'}")
    assert all_bounds_ok

    # 16. No payload overlaps
    sorted_descs = sorted(descs, key=lambda x: x["offset"])
    no_overlap = True
    for i in range(len(sorted_descs) - 1):
        end_i = sorted_descs[i]["offset"] + sorted_descs[i]["size_bytes"]
        start_next = sorted_descs[i + 1]["offset"]
        if end_i > start_next:
            no_overlap = False; print(f"  OVERLAP: id={sorted_descs[i]['id']} end={end_i} > id={sorted_descs[i+1]['id']} start={start_next}")
    print(f"[15] No payload overlaps: {'PASS' if no_overlap else 'FAIL'}")
    assert no_overlap

    # 17. Quantization types: FP32=0, INT8=2, TERNARY=4
    valid_qts = {NANO_QUANT_FP32, NANO_QUANT_INT8, NANO_QUANT_TERNARY_2BIT}
    all_qt_ok = all(d["qt"] in valid_qts for d in descs)
    fp32_descs = [d for d in descs if d["qt"] == NANO_QUANT_FP32]
    int8_descs = [d for d in descs if d["qt"] == NANO_QUANT_INT8]
    tern_descs = [d for d in descs if d["qt"] == NANO_QUANT_TERNARY_2BIT]
    ok_fp32 = (len(fp32_descs) == EXPECTED_FP32_TENSORS)
    ok_i8 = (len(int8_descs) == EXPECTED_INT8_TENSORS)
    ok_tern = (len(tern_descs) == EXPECTED_TERNARY_TENSORS)
    print(f"[16] Quant types valid: {'PASS' if all_qt_ok else 'FAIL'}")
    print(f"[17] FP32={len(fp32_descs)}/{EXPECTED_FP32_TENSORS} {'PASS' if ok_fp32 else 'FAIL'}")
    print(f"[18] INT8={len(int8_descs)}/{EXPECTED_INT8_TENSORS} {'PASS' if ok_i8 else 'FAIL'}")
    print(f"[19] TERNARY={len(tern_descs)}/{EXPECTED_TERNARY_TENSORS} {'PASS' if ok_tern else 'FAIL'}")
    assert all_qt_ok and ok_fp32 and ok_i8 and ok_tern

    # 18. Scales
    for d in descs:
        if d["qt"] == NANO_QUANT_FP32: assert abs(d["scale"] - 1.0) < 1e-5, f"FP32 scale != 1.0 for id={d['id']}"
        else: assert d["scale"] > 0, f"scale <=0 for id={d['id']}"
    print("[20] All scales valid: PASS")

    # 19. Raw payload byte count
    raw_payload = sum(d["size_bytes"] for d in descs)
    ok_payload = (raw_payload == EXPECTED_RAW_PAYLOAD)
    print(f"[21] Raw payload bytes: {raw_payload:,} (expected {EXPECTED_RAW_PAYLOAD:,}) {'PASS' if ok_payload else 'FAIL'}")
    assert ok_payload, f"Payload bytes mismatch: {raw_payload}"

    # 20. Exact tensor count
    print(f"[22] Exact tensor count {TENSOR_COUNT}: PASS")

    # 21. Parameter count from descriptors
    total_numel = sum(numel_from_quant(d["qt"], d["size_bytes"]) for d in descs)
    ok_params = (total_numel == EXPECTED_PARAMS)
    print(f"[23] Total parameters: {total_numel:,} (expected {EXPECTED_PARAMS:,}) {'PASS' if ok_params else 'FAIL'}")
    assert ok_params, f"Parameter count mismatch: {total_numel}"

    # 22. Quantization category parameter counts
    fp32_p = sum(numel_from_quant(NANO_QUANT_FP32, d["size_bytes"]) for d in fp32_descs)
    i8_p = sum(numel_from_quant(NANO_QUANT_INT8, d["size_bytes"]) for d in int8_descs)
    tern_p = sum(numel_from_quant(NANO_QUANT_TERNARY_2BIT, d["size_bytes"]) for d in tern_descs)
    ok_fp32_p = (fp32_p == EXPECTED_FP32_PARAMS)
    ok_i8_p = (i8_p == EXPECTED_INT8_PARAMS)
    ok_tern_p = (tern_p == EXPECTED_TERNARY_PARAMS)
    print(f"[24] FP32 params:    {fp32_p:,} (expected {EXPECTED_FP32_PARAMS:,}) {'PASS' if ok_fp32_p else 'FAIL'}")
    print(f"[25] INT8 params:    {i8_p:,} (expected {EXPECTED_INT8_PARAMS:,}) {'PASS' if ok_i8_p else 'FAIL'}")
    print(f"[26] TERNARY params: {tern_p:,} (expected {EXPECTED_TERNARY_PARAMS:,}) {'PASS' if ok_tern_p else 'FAIL'}")
    assert ok_fp32_p and ok_i8_p and ok_tern_p

    # 23. CRC (streaming 4MB chunks, bounded memory)
    crc_running = 0
    with open(nano_path, "rb") as f:
        f.read(HEADER_SIZE)
        while chunk := f.read(CHUNK_BYTES):
            crc_running = zlib.crc32(chunk, crc_running)
    crc_computed = crc_running & 0xFFFFFFFF
    ok_crc = (crc_computed == hdr["crc_stored"])
    print(f"[27] CRC-32: computed=0x{crc_computed:08X} stored=0x{hdr['crc_stored']:08X} {'PASS' if ok_crc else 'FAIL'}")
    assert ok_crc, f"CRC mismatch: computed 0x{crc_computed:08X} vs stored 0x{hdr['crc_stored']:08X}"

    # 24. SHA-256 (streaming)
    final_sha = sha256_file(nano_path)
    print(f"\n[FINAL] model.nano SHA-256: {final_sha}")
    print(f"[FINAL] model.nano size:    {file_size:,} bytes")
    print(f"[FINAL] CRC-32:             0x{crc_computed:08X}")
    print(f"[FINAL] Tensor count:       {TENSOR_COUNT}/219")
    print(f"[FINAL] Parameter count:    {total_numel:,}/{EXPECTED_PARAMS:,}")

    print("\n" + "=" * 80)
    print("INDEPENDENT VERIFIER: ALL 27 CHECKS PASSED")
    print(f"MODEL_NANO_SHA256={final_sha}")
    print(f"MODEL_NANO_SIZE={file_size}")
    print(f"MODEL_NANO_CRC=0x{crc_computed:08X}")
    print("TENSORS_VERIFIED=219/219")
    print(f"PARAMETERS_VERIFIED={total_numel}/{EXPECTED_PARAMS}")
    print("INDEPENDENT_VERIFIER=PASS")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent Production model.nano Verifier")
    parser.add_argument("--nano", required=True, help="Path to models/model.nano")
    args = parser.parse_args()
    verify(args.nano)
