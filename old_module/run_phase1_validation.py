#!/usr/bin/env python3
"""
THSA-2B Phase 1: Master Test Runner
===================================
Orchestrates all three Phase 1 validation scripts and provides a unified pass/fail
decision on whether to proceed to Phase 2 (350M proxy training).

Run: python3 run_phase1_validation.py
Expected Output: Summary of all 3 validators + overall recommendation
"""

import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, List, Tuple

class Phase1TestRunner:
    """Orchestrates Phase 1 validation tests."""
    
    VALIDATORS = [
        {
            "name": "Memory Model Validator",
            "script": "01_memory_model_validator.py",
            "description": "Validates 250 MB RAM ceiling is feasible",
            "critical": True,
        },
        {
            "name": "Quantization Error Simulator",
            "script": "02_quantization_error_simulator.py",
            "description": "Validates <= 5% perplexity degradation",
            "critical": True,
        },
        {
            "name": "Energy & Thermal Simulator",
            "script": "03_energy_thermal_simulator.py",
            "description": "Validates no thermal throttling at human-paced DVFS",
            "critical": True,
        },
    ]
    
    def __init__(self):
        self.results: List[Dict] = []
        self.all_passed = True
    
    def run_validator(self, validator: Dict) -> Tuple[bool, int]:
        """
        Run a single validator script and capture exit code.
        Exit codes:
          0 = PASS (proceed)
          1 = CONDITIONAL PASS or WARNING
          2 = FAIL (stop)
        """
        print(f"\n{'='*80}")
        print(f"Running: {validator['name']}")
        print(f"Script:  {validator['script']}")
        print(f"{'='*80}\n")
        
        try:
            sub_env = os.environ.copy()
            sub_env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, validator["script"]],
                capture_output=False,
                timeout=30,
                env=sub_env
            )
            return result.returncode == 0, result.returncode
        except subprocess.TimeoutExpired:
            print(f"❌ TIMEOUT: {validator['script']} exceeded 30 seconds")
            return False, 2
        except FileNotFoundError:
            print(f"❌ FILE NOT FOUND: {validator['script']}")
            print(f"   Make sure script is in current directory")
            return False, 2
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return False, 2
    
    def run_all(self) -> int:
        """Run all validators and return overall exit code."""
        
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  THSA-2B PHASE 1: COMPREHENSIVE VALIDATION SUITE".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        print()
        print("This test suite validates THREE critical architectural assumptions:")
        print("  1. Memory: Does 250 MB RAM ceiling hold mathematically?")
        print("  2. Quantization: Does ternary+INT8+INT4 stay <= 5% perplexity loss?")
        print("  3. Energy/Thermal: Does human-paced DVFS prevent throttling?")
        print()
        print("All three MUST pass to proceed to Phase 2 (350M proxy training).")
        print()
        
        # Run all validators
        for i, validator in enumerate(self.VALIDATORS, 1):
            passed, exit_code = self.run_validator(validator)
            
            self.results.append({
                "name": validator["name"],
                "script": validator["script"],
                "passed": passed,
                "exit_code": exit_code,
                "critical": validator["critical"],
            })
            
            if not passed and validator["critical"]:
                self.all_passed = False
        
        # Print summary
        return self.print_summary()
    
    def print_summary(self) -> int:
        """Print summary report and return overall exit code."""
        
        print("\n\n" + "="*80)
        print("PHASE 1 VALIDATION SUMMARY")
        print("="*80 + "\n")
        
        print("Test Results:")
        print("-" * 80)
        for result in self.results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            critical = "[CRITICAL]" if result["critical"] else "[OPTIONAL]"
            print(f"  {status:10s} {critical:12s} {result['name']}")
        
        print()
        print("="*80)
        print("RECOMMENDATION")
        print("="*80 + "\n")
        
        if self.all_passed:
            print("✅ ALL PHASE 1 TESTS PASSED")
            print()
            print("Summary:")
            print("  • 250 MB RAM ceiling is mathematically feasible")
            print("  • Quantization strategy achieves <= 5% perplexity degradation")
            print("  • Human-paced DVFS keeps device thermal-safe (< 45°C)")
            print()
            print("✅ NEXT STEPS:")
            print("  1. Implement ARM NEON ternary GEMV kernel (Phase 2A, 2-3 weeks)")
            print("  2. Benchmark KV-cache INT4 pack/unpack (Phase 2B)")
            print("  3. Validate chunked prefill pipeline (Phase 2C)")
            print("  4. Train 350M proxy on 50B tokens with QAT (Phase 3, 4-6 weeks)")
            print()
            print("✅ DECISION: PROCEED TO PHASE 2 (Micro-kernel Development)")
            print()
            return 0
        
        elif any(r["passed"] and r["critical"] for r in self.results):
            print("⚠️  PARTIAL PASS - SOME TESTS FAILED")
            print()
            failed = [r["name"] for r in self.results if not r["passed"] and r["critical"]]
            print(f"Critical failures: {', '.join(failed)}")
            print()
            print("⚠️  DECISION: CONDITIONAL PROCEED")
            print("  • Investigate failures before proceeding to Phase 2")
            print("  • May need architecture adjustments or re-calibration")
            print()
            return 1
        
        else:
            print("❌ CRITICAL TESTS FAILED")
            print()
            failed = [r["name"] for r in self.results if not r["passed"] and r["critical"]]
            print(f"Critical failures: {', '.join(failed)}")
            print()
            print("❌ DECISION: HALT - DO NOT PROCEED")
            print("  • Architecture assumptions are not validated")
            print("  • Requires significant redesign or re-analysis")
            print("  • Do not commit to Phase 2 work until tests pass")
            print()
            return 2

def main():
    runner = Phase1TestRunner()
    return runner.run_all()

if __name__ == "__main__":
    exit(main())
