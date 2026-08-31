/**
 * @file nano_telemetry.h
 * @brief Native Runtime Telemetry & Observability Struct for Android Host Monitoring.
 * Standard Compliance: C99 / C++17 compatible header.
 */

#ifndef NANO_TELEMETRY_H
#define NANO_TELEMETRY_H

#include "nano_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Telemetry Degraded Flags Bitmask (Section 9.8)
 */
#define NANO_FLAG_DEGRADED_NONE            0x00000000
#define NANO_FLAG_DEGRADED_KIVI_ENGAGED    0x00000001  /**< 2.5-bit KV compression active */
#define NANO_FLAG_DEGRADED_THERMAL_CLAMP   0x00000002  /**< DVFS frequency clamped due to heat */
#define NANO_FLAG_DEGRADED_CONTEXT_EVICTED 0x00000004  /**< Attention sink rolling window evicting oldest turns */

/**
 * @brief Real-Time Operational Health & Telemetry Struct (Section 9.8)
 */
typedef struct {
    uint64_t resident_ram_bytes;       /**< Current physical RSS of native engine process */
    uint32_t active_kv_tokens;         /**< Current allocated KV-cache slot count (0..10000) */
    float    instantaneous_tok_per_s;  /**< Instantaneous token emission rate (e.g. 11.2) */
    float    estimated_temp_c;         /**< Estimated chassis skin surface temperature */
    uint32_t total_tokens_generated;   /**< Cumulative output tokens generated in session */
    uint32_t degraded_flags;           /**< Operational bitmask (KIVI / Thermal / Eviction) */
} NanoEngineTelemetry;

#ifdef __cplusplus
}
#endif

#endif /* NANO_TELEMETRY_H */
