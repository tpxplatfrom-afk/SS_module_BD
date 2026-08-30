"""
SS Tutor BD - Deterministic Unit Converter
Handles NCTB Class 8 standard measurement conversions (Metric & British systems).
"""

from typing import Dict, Any
from core.math.fraction import FractionHelper


class UnitConverter:
    # NCTB Official Conversion Factors
    METER_TO_INCH = 39.37
    INCH_TO_CM = 2.54
    YARD_TO_METER = 0.9144
    MILE_TO_KM = 1.61
    CUBIC_METER_TO_LITER = 1000.0
    ACRE_TO_SQ_YARDS = 4840.0
    ACRE_TO_SQ_METERS = 4046.86
    HECTARE_TO_SQ_METERS = 10000.0

    @staticmethod
    def meters_to_inches(meters: float) -> Dict[str, Any]:
        inches = meters * UnitConverter.METER_TO_INCH
        return {
            "input": meters,
            "result": round(inches, 2),
            "bengali": f"{FractionHelper.to_bengali_number(meters)} মিটার ≈ {FractionHelper.to_bengali_number(round(inches, 2))} ইঞ্চি"
        }

    @staticmethod
    def inches_to_cm(inches: float) -> Dict[str, Any]:
        cm = inches * UnitConverter.INCH_TO_CM
        return {
            "input": inches,
            "result": round(cm, 2),
            "bengali": f"{FractionHelper.to_bengali_number(inches)} ইঞ্চি ≈ {FractionHelper.to_bengali_number(round(cm, 2))} সেন্টিমিটার"
        }

    @staticmethod
    def cubic_meters_to_liters(m3: float) -> Dict[str, Any]:
        liters = m3 * UnitConverter.CUBIC_METER_TO_LITER
        return {
            "input": m3,
            "result": int(liters) if liters.is_integer() else round(liters, 2),
            "bengali": f"{FractionHelper.to_bengali_number(m3)} ঘনমিটার = {FractionHelper.to_bengali_number(int(liters) if liters.is_integer() else round(liters, 2))} লিটার"
        }
