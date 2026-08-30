# Benchmark Result JSON Schema: SS Tutor BD

**Document Version:** 1.0.0  
**Purpose:** Standardize the output format for automated and semi-automated benchmark runs across all candidate models.

---

## 1. JSON Schema Definition

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SSTutorBenchmarkResult",
  "type": "object",
  "required": [
    "schema_version",
    "timestamp",
    "environment",
    "candidate",
    "static_metrics",
    "runtime_metrics",
    "memory_metrics",
    "language_metrics",
    "reasoning_metrics",
    "tutor_metrics",
    "test_outputs"
  ],
  "properties": {
    "schema_version": { "type": "string", "enum": ["1.0.0"] },
    "timestamp": { "type": "string", "format": "date-time" },
    "environment": {
      "type": "object",
      "required": ["os", "cpu", "threads_used", "ram_total_mb", "runtime_backend"],
      "properties": {
        "os": { "type": "string" },
        "cpu": { "type": "string" },
        "threads_used": { "type": "integer" },
        "ram_total_mb": { "type": "number" },
        "runtime_backend": { "type": "string", "enum": ["llama.cpp", "executorch", "onnxruntime", "transformers_cpu"] }
      }
    },
    "candidate": {
      "type": "object",
      "required": ["model_id", "family", "parameters_billion", "quantization", "license"],
      "properties": {
        "model_id": { "type": "string" },
        "family": { "type": "string" },
        "parameters_billion": { "type": "number" },
        "quantization": { "type": "string" },
        "license": { "type": "string" }
      }
    },
    "static_metrics": {
      "type": "object",
      "required": ["model_file_size_mb"],
      "properties": {
        "model_file_size_mb": { "type": "number" }
      }
    },
    "runtime_metrics": {
      "type": "object",
      "required": ["load_time_ms", "ttft_ms", "tokens_per_second_generation", "tokens_per_second_prompt"],
      "properties": {
        "load_time_ms": { "type": "number" },
        "ttft_ms": { "type": "number" },
        "tokens_per_second_generation": { "type": "number" },
        "tokens_per_second_prompt": { "type": "number" }
      }
    },
    "memory_metrics": {
      "type": "object",
      "required": ["peak_rss_mb", "dirty_heap_mb"],
      "properties": {
        "peak_rss_mb": { "type": "number" },
        "dirty_heap_mb": { "type": "number" },
        "kv_cache_mb": { "type": "number" }
      }
    },
    "language_metrics": {
      "type": "object",
      "required": ["bengali_token_expansion_ratio", "bengali_coherence_score_5pt", "english_leakage_pct"],
      "properties": {
        "bengali_token_expansion_ratio": { "type": "number" },
        "bengali_coherence_score_5pt": { "type": "number", "minimum": 1.0, "maximum": 5.0 },
        "english_leakage_pct": { "type": "number" }
      }
    },
    "reasoning_metrics": {
      "type": "object",
      "required": ["math_final_accuracy_pct", "derivation_step_validity_pct"],
      "properties": {
        "math_final_accuracy_pct": { "type": "number" },
        "derivation_step_validity_pct": { "type": "number" }
      }
    },
    "tutor_metrics": {
      "type": "object",
      "required": ["socratic_scaffolding_score_5pt", "hint_compliance_pct", "hallucination_rejection_pct", "format_compliance_pct"],
      "properties": {
        "socratic_scaffolding_score_5pt": { "type": "number", "minimum": 1.0, "maximum": 5.0 },
        "hint_compliance_pct": { "type": "number" },
        "hallucination_rejection_pct": { "type": "number" },
        "format_compliance_pct": { "type": "number" }
      }
    },
    "test_outputs": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["test_id", "prompt", "generated_response", "latency_ms", "tokens_generated"],
        "properties": {
          "test_id": { "type": "string" },
          "prompt": { "type": "string" },
          "generated_response": { "type": "string" },
          "latency_ms": { "type": "number" },
          "tokens_generated": { "type": "integer" },
          "eval_notes": { "type": "string" }
        }
      }
    },
    "notes": { "type": "string" }
  }
}
```

---

## 2. Example Instance

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2026-08-30T15:00:00Z",
  "environment": {
    "os": "Windows 10 Pro (x64)",
    "cpu": "Intel Core i5-6500 @ 3.20GHz",
    "threads_used": 4,
    "ram_total_mb": 8109,
    "runtime_backend": "llama.cpp"
  },
  "candidate": {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "family": "Qwen2.5",
    "parameters_billion": 0.49,
    "quantization": "Q4_K_M",
    "license": "Apache-2.0"
  },
  "static_metrics": {
    "model_file_size_mb": 348.5
  },
  "runtime_metrics": {
    "load_time_ms": 420.0,
    "ttft_ms": 1150.0,
    "tokens_per_second_generation": 14.2,
    "tokens_per_second_prompt": 68.0
  },
  "memory_metrics": {
    "peak_rss_mb": 512.0,
    "dirty_heap_mb": 95.0,
    "kv_cache_mb": 42.0
  },
  "language_metrics": {
    "bengali_token_expansion_ratio": 1.42,
    "bengali_coherence_score_5pt": 4.2,
    "english_leakage_pct": 1.5
  },
  "reasoning_metrics": {
    "math_final_accuracy_pct": 62.5,
    "derivation_step_validity_pct": 75.0
  },
  "tutor_metrics": {
    "socratic_scaffolding_score_5pt": 4.0,
    "hint_compliance_pct": 100.0,
    "hallucination_rejection_pct": 80.0,
    "format_compliance_pct": 95.0
  },
  "test_outputs": [],
  "notes": "Example mock instance for schema validation only. No live test performed."
}
```
