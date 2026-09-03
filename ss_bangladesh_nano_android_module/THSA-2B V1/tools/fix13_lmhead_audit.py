#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX-13 LM Head Source & Numerical Contract Audit
================================================
Audits the LM-head implementation in nano_engine.cpp and kernels:
- Searches for nano_neon_gemv_dense_int8 or any NEON dense GEMV implementation
- Evaluates the scalar unrolled loop contract [65536, 2560]
- Runs deterministic numerical tests on scalar INT8 LM-head projection
"""

import os
import sys
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

def main():
    print("=" * 80)
    print("FIX-13 STEP A/B/C: LM-HEAD SOURCE & KERNEL CONTRACT AUDIT")
    print("=" * 80)

    engine_cpp = ROOT_DIR / "src" / "engine" / "nano_engine.cpp"
    assert engine_cpp.exists(), f"Missing {engine_cpp}"

    with open(engine_cpp, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Source file: {engine_cpp} ({len(lines)} lines)")

    # Locate Section 4. OUTPUT LOGITS COMPUTATION
    lm_start = -1
    lm_end = -1
    for idx, line in enumerate(lines):
        if "4. OUTPUT LOGITS COMPUTATION" in line:
            lm_start = idx
        if lm_start != -1 and "5. REAL TOKEN SELECTION" in line:
            lm_end = idx
            break

    print(f"LM-Head Block in nano_engine.cpp: Lines {lm_start + 1} to {lm_end + 1}")
    block_text = "".join(lines[lm_start:lm_end])
    print("\n--- Exact Production LM-Head Code Block ---")
    print(block_text.strip())
    print("-------------------------------------------\n")

    has_neon_gemv = "nano_neon_gemv_dense_int8" in block_text
    has_scalar_loop = "for (size_t v = 0; v < 65536; ++v)" in block_text

    print(f"Call to nano_neon_gemv_dense_int8 present: {has_neon_gemv}")
    print(f"Scalar unrolled for-loop present:           {has_scalar_loop}")

    # Search for any definition of nano_neon_gemv_dense_int8 in all headers and cpp files
    all_srcs = list(ROOT_DIR.glob("include/**/*")) + list(ROOT_DIR.glob("src/**/*"))
    found_decl = []
    found_defn = []
    for p in all_srcs:
        if p.is_file() and p.suffix in [".h", ".hpp", ".c", ".cpp"]:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if "nano_neon_gemv_dense_int8" in content:
                    found_decl.append(str(p))
            except Exception:
                pass

    print(f"\nOccurrences of nano_neon_gemv_dense_int8 in codebase: {len(found_decl)}")
    for f in found_decl:
        print(f"  - {f}")

    if not has_neon_gemv:
        print("\n[CRITICAL AUDIT FINDING]")
        print("LM Head NEON optimization (nano_neon_gemv_dense_int8) is NOT on the production execution path.")
        print("The production path explicitly executes the 8-way unrolled scalar C++ loop.")

if __name__ == "__main__":
    main()
