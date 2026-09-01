#!/usr/bin/env python3
"""
THSA-2B Binary Package Inspector & Verification Utility.
Inspects .nano binary files, validates CRC32 integrity, and verifies 64-byte SIMD alignment.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import zlib
import struct
from typing import Dict, Any

MAGIC_NANO = b"NANO"

def inspect_nano_file(file_path: str) -> Dict[str, Any]:
    print("=" * 80)
    print(f"INSPECTING .NANO BINARY PACKAGE: {file_path}")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        header_bytes = f.read(64)
        if len(header_bytes) < 64:
            raise ValueError("Corrupt .nano binary: File is smaller than 64-byte header.")
            
        # Unpack header
        (
            magic,
            version,
            total_blocks,
            state_blocks,
            gqa_blocks,
            d_model,
            d_ffn,
            n_q,
            n_kv,
            d_head,
            _, # pad
            vocab_size,
            max_context,
            stored_crc,
            tensor_count,
            _ # reserved
        ) = struct.unpack("<4sHHHHIIHHHHI I I I 20s", header_bytes)
        
        # 1. Verify Magic Number
        if magic != MAGIC_NANO:
            raise ValueError(f"Invalid magic number: {magic} (Expected {MAGIC_NANO})")
            
        # 2. Read Descriptor Table
        desc_table_size = tensor_count * 32
        desc_bytes = f.read(desc_table_size)
        
        # 3. Read Remaining Payload
        remaining_payload = f.read()
        
        # 4. Verify CRC32 Checksum
        computed_crc = zlib.crc32(desc_bytes + remaining_payload)
        crc_match = (computed_crc == stored_crc)
        
        print(f"Format Version:       0x{version:04X}")
        print(f"Backbone Topology:    {total_blocks} Blocks ({state_blocks} State / {gqa_blocks} GQA)")
        print(f"Model Dimensions:     d_model={d_model}, d_ffn={d_ffn}, d_head={d_head}")
        print(f"Attention Heads:      n_query={n_q}, n_kv={n_kv} (GQA {n_q // n_kv}:1)")
        print(f"Vocabulary Size (V):  {vocab_size} tokens")
        print(f"Context Horizon:      {max_context} tokens")
        print(f"Tensor Count:         {tensor_count} tensors")
        print(f"Total File Size:      {file_size / (1024*1024):.2f} MB")
        print(f"CRC32 Checksum:       Stored=0x{stored_crc:08X}, Computed=0x{computed_crc:08X} ({'✅ MATCH' if crc_match else '❌ MISMATCH'})")
        
        if not crc_match:
            raise ValueError("CRC32 Checksum mismatch! Binary package is corrupted.")
            
        # 5. Check 64-byte Alignment of Descriptors
        unaligned_count = 0
        for i in range(tensor_count):
            desc = desc_bytes[i*32 : (i+1)*32]
            t_id, q_type, offset, size_bytes, scale, _ = struct.unpack("<IIQQfI", desc)
            if offset % 64 != 0:
                unaligned_count += 1
                print(f"  ⚠️ Warning: Tensor {t_id} at offset {offset} is not 64-byte aligned!")
                
        print(f"64-Byte Alignment:    {'✅ 100% ALIGNED (0 violations)' if unaligned_count == 0 else f'❌ {unaligned_count} UNALIGNED'}")
        print("=" * 80)
        
        return {
            "version": version,
            "total_blocks": total_blocks,
            "d_model": d_model,
            "crc_match": crc_match,
            "all_aligned": (unaligned_count == 0),
            "file_size": file_size
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_nano_file(sys.argv[1])
    else:
        print("Usage: python inspect_nano_binary.py <model.nano>")
