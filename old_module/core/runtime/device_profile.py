"""
SS Tutor BD - Device Profiler & Adaptive Policy (Phase 3C)
Detects host and Android hardware capabilities and assigns adaptive operational tiers.
"""

import os
import psutil
import platform
from typing import Dict, Any


class DeviceProfiler:
    TIER_ULTRA_LOW = "TIER_ULTRA_LOW"    # Devices with <= 1.5 GB available RAM (Context <= 256, Template-first)
    TIER_LOW = "TIER_LOW"                # Standard 2 GB devices (Context <= 384, Micro-LLM bounded)
    TIER_STANDARD = "TIER_STANDARD"      # 3 GB+ devices (Context <= 512)

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Detects current hardware environment."""
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "cpu_cores_logical": psutil.cpu_count(logical=True) or 2,
            "cpu_cores_physical": psutil.cpu_count(logical=False) or 2,
            "total_ram_mb": round(mem.total / (1024 * 1024), 2),
            "available_ram_mb": round(mem.available / (1024 * 1024), 2),
            "total_storage_mb": round(disk.total / (1024 * 1024), 2),
            "free_storage_mb": round(disk.free / (1024 * 1024), 2)
        }

    @classmethod
    def determine_device_tier(cls, override_ram_mb: float = None) -> str:
        """Classifies device into performance and memory safety tiers."""
        if override_ram_mb is not None:
            avail_ram = override_ram_mb
        else:
            avail_ram = psutil.virtual_memory().available / (1024 * 1024)

        if avail_ram < 800.0:
            return cls.TIER_ULTRA_LOW
        elif avail_ram <= 2048.0:
            return cls.TIER_LOW
        else:
            return cls.TIER_STANDARD

    @classmethod
    def get_adaptive_policy(cls, tier: str = None) -> Dict[str, Any]:
        """Returns context limits and inference strategy based on tier."""
        tier = tier or cls.determine_device_tier()

        if tier == cls.TIER_ULTRA_LOW:
            return {
                "tier": cls.TIER_ULTRA_LOW,
                "max_context_tokens": 192,
                "max_output_tokens": 64,
                "max_rag_chunks": 1,
                "max_chunk_words": 35,
                "use_deterministic_math": True,
                "neural_generation_enabled": False,  # Pure deterministic template fallback for ultra-low
                "target_rss_mb": 120.0
            }
        elif tier == cls.TIER_LOW:
            return {
                "tier": cls.TIER_LOW,
                "max_context_tokens": 256,
                "max_output_tokens": 96,
                "max_rag_chunks": 1,
                "max_chunk_words": 45,
                "use_deterministic_math": True,
                "neural_generation_enabled": True,
                "target_rss_mb": 180.0
            }
        else:
            return {
                "tier": cls.TIER_STANDARD,
                "max_context_tokens": 512,
                "max_output_tokens": 128,
                "max_rag_chunks": 2,
                "max_chunk_words": 60,
                "use_deterministic_math": True,
                "neural_generation_enabled": True,
                "target_rss_mb": 200.0
            }
