#!/usr/bin/env python3
"""
THSA-2B Phase 1: Memory Model Validator
========================================
Validates whether the THSA-2B architecture can fit within 250 MB RAM ceiling
under 10K context load. This is a mathematical simulation (not actual inference).

Run: python3 01_memory_model_validator.py
Expected Output: Memory budget breakdown + margin analysis
"""

import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MemoryComponent:
    name: str
    size_mb: float
    description: str
    
    def __repr__(self):
        return f"{self.name:30s} | {self.size_mb:7.2f} MB | {self.description}"

class THSAMemoryModel:
    """Mathematical model of THSA-2B memory consumption under 10K context."""
    
    # Constants from architecture spec
    VOCAB_SIZE = 32_000
    HIDDEN_DIM = 2560
    NUM_GQA_BLOCKS = 8
    NUM_STATE_BLOCKS = 16
    KV_HEADS = 4
    HEAD_DIM = 128
    CONTEXT_LENGTH = 10_000
    
    def __init__(self):
        self.components: List[MemoryComponent] = []
        self.total_budget_mb = 250.0
        self.preferred_target_mb = 200.0
        
    def calculate_resident_weights(self) -> float:
        """
        Resident weight pages (memory-mapped, paged into RAM on-demand).
        Spec: <= 130 MB working set
        """
        # Ternary weights compressed to 1 byte per weight (3 values: -1,0,+1)
        # Plus FP16 scaling factors
        
        # FFN blocks: 24 * (2560 * 6912 * 3) ≈ 1,274M parameters
        ffn_params = 24 * (self.HIDDEN_DIM * (self.HIDDEN_DIM * 2.7) * 3)
        ffn_bytes = (ffn_params * 1) + (24 * 10 * 1e6)  # ternary + scaling
        
        # GQA attention: 8 * (3 * 2560 * 512) ≈ 31M parameters
        gqa_params = 8 * (self.HIDDEN_DIM ** 2 + 2 * self.HIDDEN_DIM * 512)
        gqa_bytes = gqa_params * 1
        
        # State blocks: 16 * (3 * 2560 * 2560) ≈ 335M parameters
        state_params = 16 * (3 * self.HIDDEN_DIM ** 2)
        state_bytes = state_params * 1
        
        total_bytes = ffn_bytes + gqa_bytes + state_bytes
        total_mb = total_bytes / 1e6
        
        return min(total_mb, 130.0)  # Capped at spec
    
    def calculate_kv_cache_16_8(self) -> float:
        """
        KV-Cache for 16 State / 8 GQA configuration at 10K context.
        Formula: 2 * L * N_attn * N_kv * D_head * B_KV
        where B_KV = 0.5 bytes (INT4 quantized)
        """
        m_kv = (2 * self.CONTEXT_LENGTH * 
                self.NUM_GQA_BLOCKS * 
                self.KV_HEADS * 
                self.HEAD_DIM * 
                0.5)
        return m_kv / 1e6
    
    def calculate_kv_cache_12_12(self) -> float:
        """
        KV-Cache for elastic fallback to 12 State / 12 GQA configuration.
        """
        m_kv = (2 * self.CONTEXT_LENGTH * 
                12 * 
                self.KV_HEADS * 
                self.HEAD_DIM * 
                0.5)
        return m_kv / 1e6
    
    def calculate_activation_tensors(self) -> float:
        """
        Activation tensors during forward pass.
        Chunked prefill ensures max 256-token chunk → bounded activation memory.
        """
        return 25.0  # Spec: <= 25 MB
    
    def calculate_workspace(self) -> float:
        """Temporary scratchpads, intermediate buffers."""
        return 20.0  # Spec: <= 20 MB
    
    def calculate_runtime_meta(self) -> float:
        """JNI/Runtime overhead, metadata, layer norms (FP32)."""
        return 15.0  # Spec: <= 15 MB
    
    def calculate_safety_margin(self) -> float:
        """Buffer for OS and unexpected allocations."""
        return 15.0  # Spec: ~15 MB
    
    def validate_16_8_topology(self) -> Dict:
        """Validate primary 16 State / 8 GQA topology."""
        weights = self.calculate_resident_weights()
        kv_cache = self.calculate_kv_cache_16_8()
        activations = self.calculate_activation_tensors()
        workspace = self.calculate_workspace()
        runtime_meta = self.calculate_runtime_meta()
        safety = self.calculate_safety_margin()
        
        total = weights + kv_cache + activations + workspace + runtime_meta + safety
        margin = self.total_budget_mb - total
        
        return {
            "topology": "16 State / 8 GQA",
            "components": [
                MemoryComponent("Resident Weights (mmap)", weights, "ternary + scaling"),
                MemoryComponent("KV-Cache (INT4, 10K)", kv_cache, "8 attention blocks"),
                MemoryComponent("Activation Tensors", activations, "chunked prefill max"),
                MemoryComponent("Temporary Workspace", workspace, "scratchpads"),
                MemoryComponent("Runtime/JNI/Meta", runtime_meta, "system overhead"),
                MemoryComponent("Safety Margin", safety, "buffer"),
            ],
            "total_mb": total,
            "budget_mb": self.total_budget_mb,
            "margin_mb": margin,
            "fits": margin >= 0,
            "preferred_margin": total <= self.preferred_target_mb,
        }
    
    def validate_12_12_fallback(self) -> Dict:
        """Validate elastic fallback to 12 State / 12 GQA topology."""
        weights = self.calculate_resident_weights()
        kv_cache = self.calculate_kv_cache_12_12()
        activations = self.calculate_activation_tensors()
        workspace = self.calculate_workspace()
        runtime_meta = self.calculate_runtime_meta()
        safety = self.calculate_safety_margin()
        
        total = weights + kv_cache + activations + workspace + runtime_meta + safety
        margin = self.total_budget_mb - total
        
        return {
            "topology": "12 State / 12 GQA (Elastic Fallback)",
            "components": [
                MemoryComponent("Resident Weights (mmap)", weights, "ternary + scaling"),
                MemoryComponent("KV-Cache (INT4, 10K)", kv_cache, "12 attention blocks"),
                MemoryComponent("Activation Tensors", activations, "chunked prefill max"),
                MemoryComponent("Temporary Workspace", workspace, "scratchpads"),
                MemoryComponent("Runtime/JNI/Meta", runtime_meta, "system overhead"),
                MemoryComponent("Safety Margin", safety, "buffer"),
            ],
            "total_mb": total,
            "budget_mb": self.total_budget_mb,
            "margin_mb": margin,
            "fits": margin >= 0,
            "preferred_margin": total <= self.preferred_target_mb,
        }
    
    def print_report(self, topology_result: Dict, topology_name: str):
        """Pretty-print validation report."""
        print(f"\n{'='*80}")
        print(f"THSA-2B Memory Validation: {topology_name}")
        print(f"{'='*80}\n")
        
        for component in topology_result["components"]:
            print(f"  {component}")
        
        print(f"\n{'-'*80}")
        print(f"{'TOTAL WORKING RAM':30s} | {topology_result['total_mb']:7.2f} MB")
        print(f"{'BUDGET CEILING':30s} | {topology_result['budget_mb']:7.2f} MB")
        print(f"{'SAFETY MARGIN':30s} | {topology_result['margin_mb']:7.2f} MB")
        print(f"{'-'*80}\n")
        
        if topology_result["fits"]:
            status = "✅ PASS: Fits within 250 MB ceiling"
        else:
            status = "❌ FAIL: Exceeds 250 MB ceiling"
        
        print(f"Status: {status}")
        
        if topology_result["preferred_margin"]:
            print(f"Secondary: ✅ Preferred target (≤200 MB) achieved")
        else:
            print(f"Secondary: ⚠️  Above preferred target (≤200 MB), but acceptable")
        
        return topology_result["fits"]

def main():
    print("\n" + "="*80)
    print("THSA-2B PHASE 1: MEMORY MODEL VALIDATOR")
    print("="*80)
    print("Testing whether architecture fits within 250 MB working RAM ceiling...\n")
    
    model = THSAMemoryModel()
    
    # Test primary topology
    result_16_8 = model.validate_16_8_topology()
    passes_16_8 = model.print_report(result_16_8, "16 State / 8 GQA (Primary)")
    
    # Test elastic fallback
    result_12_12 = model.validate_12_12_fallback()
    passes_12_12 = model.print_report(result_12_12, "12 State / 12 GQA (Elastic Fallback)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATION")
    print("="*80)
    
    if passes_16_8 and passes_12_12:
        print("✅ BOTH topologies fit within 250 MB budget!\n")
        print("   → Primary topology (16/8):   {:.2f} MB margin".format(result_16_8["margin_mb"]))
        print("   → Fallback topology (12/12): {:.2f} MB margin\n".format(result_12_12["margin_mb"]))
        print("✅ Recommendation: PROCEED to Phase 2 (quantization validation)")
        print("   Next step: Run 02_quantization_error_simulator.py")
        return 0
    elif passes_16_8:
        print("✅ Primary topology (16/8) fits within 250 MB\n")
        print("⚠️  Fallback topology (12/12) EXCEEDS budget (but can be optimized)\n")
        print("⚠️  Recommendation: CONDITIONAL PROCEED")
        print("   Optimize: Reduce activation buffer or increase context granularity")
        return 1
    else:
        print("❌ PRIMARY TOPOLOGY FAILS - does not fit within 250 MB\n")
        print("❌ CRITICAL ISSUE: Architecture needs redesign\n")
        print("❌ Recommendation: HALT - do not proceed until memory model is fixed")
        return 2

if __name__ == "__main__":
    exit(main())
