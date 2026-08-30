"""
SS Tutor BD - Memory Budget Manager (Phase 3C)
Explicitly defines and monitors subsystem memory budgets to enforce the hard 200 MB production ceiling.
"""

import os
import psutil
from typing import Dict, Any, Tuple


class MemoryBudgetManager:
    # Phase 3C Strict Allocations (in Megabytes)
    MODEL_BUDGET_MB = 80.0
    RUNTIME_BUDGET_MB = 25.0
    KV_CACHE_BUDGET_MB = 15.0
    RAG_BUDGET_MB = 10.0
    COMPRESSOR_BUFFER_MB = 5.0
    MATH_ENGINE_MB = 5.0
    SANITIZER_MB = 5.0
    APP_OVERHEAD_MB = 30.0
    SAFETY_MARGIN_MB = 25.0

    PREFERRED_PEAK_MB = 180.0
    ABSOLUTE_CEILING_MB = 200.0
    SAFETY_CEILING_MB = 220.0

    @staticmethod
    def get_current_rss_mb() -> float:
        """Returns the current process Resident Set Size in MB."""
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)

    @classmethod
    def evaluate_rss(cls, rss_mb: float) -> Dict[str, Any]:
        """Evaluates measured RSS against the Phase 3C hard contract."""
        if rss_mb <= cls.PREFERRED_PEAK_MB:
            status = "PREFERRED"
            passed = True
        elif rss_mb <= cls.ABSOLUTE_CEILING_MB:
            status = "ACCEPTABLE (<= 200 MB)"
            passed = True
        elif rss_mb <= cls.SAFETY_CEILING_MB:
            status = "WARNING_TIER (200 - 220 MB)"
            passed = False
        else:
            status = "DISQUALIFIED / FAIL (> 220 MB)"
            passed = False

        return {
            "current_rss_mb": rss_mb,
            "preferred_peak_mb": cls.PREFERRED_PEAK_MB,
            "absolute_ceiling_mb": cls.ABSOLUTE_CEILING_MB,
            "safety_ceiling_mb": cls.SAFETY_CEILING_MB,
            "status": status,
            "passed_production_ceiling": passed,
            "headroom_to_ceiling_mb": round(cls.ABSOLUTE_CEILING_MB - rss_mb, 2)
        }

    @classmethod
    def get_budget_breakdown(cls) -> Dict[str, float]:
        """Returns target breakdown across all components."""
        return {
            "model_weights": cls.MODEL_BUDGET_MB,
            "runtime_engine": cls.RUNTIME_BUDGET_MB,
            "kv_cache": cls.KV_CACHE_BUDGET_MB,
            "rag_storage": cls.RAG_BUDGET_MB,
            "compressor_buffers": cls.COMPRESSOR_BUFFER_MB,
            "math_engine": cls.MATH_ENGINE_MB,
            "sanitizer_validator": cls.SANITIZER_MB,
            "app_overhead": cls.APP_OVERHEAD_MB,
            "safety_margin": cls.SAFETY_MARGIN_MB,
            "total_budget_sum": (
                cls.MODEL_BUDGET_MB + cls.RUNTIME_BUDGET_MB + cls.KV_CACHE_BUDGET_MB +
                cls.RAG_BUDGET_MB + cls.COMPRESSOR_BUFFER_MB + cls.MATH_ENGINE_MB +
                cls.SANITIZER_MB + cls.APP_OVERHEAD_MB + cls.SAFETY_MARGIN_MB
            )
        }
