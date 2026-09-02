# FIX-08 — 219-TENSOR .NANO BINARY REPRESENTATION SPECIFICATION

**Document Version:** `2.0.0`
**Status:** `APPROVED & COMPLETED`
**Format Magic:** `NANO` (`0x4E414E4F`)
**Format Version:** `0x0002`
**Total Trainable Tensors:** `219`
**Total Target Parameters:** `2,050,296,320`
**Projected .nano Package Size:** `765,477,824 bytes (730.02 MB)`

---

## 1. Binary Layout & Header Specification

The binary `.nano` format consists of three contiguous, 64-byte aligned sections:
1. **64-byte Primary Binary Header (`NanoBinaryHeader`)**
2. **7,008-byte Tensor Manifest Descriptor Table (`219 × 32 bytes`)**
3. **Payload Data Stream (Aligned to 64 bytes at offset 7,104)**

```
+-------------------------------------------------------------+ Offset 0x0000 (0)
|                 64-byte NanoBinaryHeader                    |
+-------------------------------------------------------------+ Offset 0x0040 (64)
|               219 × 32-byte Tensor Descriptors              | (7,008 bytes)
+-------------------------------------------------------------+ Offset 0x1BA0 (7,072)
|                     32-byte Alignment Pad                   |
+-------------------------------------------------------------+ Offset 0x1BC0 (7,104) [64-byte Aligned]
|               Tensor 0 Payload (embed_tokens)               |
+-------------------------------------------------------------+ Offset ...
|               Tensor 1 .. 218 Payloads                      |
+-------------------------------------------------------------+ End of File
```

### 1.1 Header Structure (`NanoBinaryHeader`, 64 bytes)
```c
#pragma pack(push, 1)
typedef struct {
    char      magic[4];            /**< "NANO" = 0x4E414E4F */
    uint16_t  version;             /**< Format version (0x0002) */
    uint16_t  total_blocks;        /**< 24 backbone blocks */
    uint16_t  state_blocks;        /**< 16 Short-Conv state blocks */
    uint16_t  gqa_blocks;          /**< 8 Grouped-Query Attention blocks */
    uint32_t  d_model;             /**< 2560 */
    uint32_t  d_ffn;               /**< 6912 */
    uint16_t  n_q;                 /**< 20 query heads */
    uint16_t  n_kv;                /**< 4 key/value heads */
    uint16_t  d_head;              /**< 128 head dimension */
    uint16_t  pad;                 /**< 0 */
    uint32_t  vocab_size;          /**< 65536 */
    uint32_t  max_context;         /**< 10000 tokens */
    uint32_t  crc32;               /**< IEEE 802.3 CRC32 over descriptor table + payload */
    uint32_t  tensor_count;        /**< Exactly 219 */
    uint8_t   reserved[20];        /**< Zero padding */
} NanoBinaryHeader;
#pragma pack(pop)
```

### 1.2 Descriptor Entry (`NanoTensorDescriptor`, 32 bytes)
```c
#pragma pack(push, 1)
typedef struct {
    uint32_t  tensor_id;           /**< ID: 0 .. 218 */
    uint32_t  quant_type;          /**< 0=FP32, 1=INT8, 2=TERNARY_2BIT */
    uint64_t  offset;              /**< Absolute file offset (64-byte aligned) */
    uint64_t  size_bytes;          /**< Byte size of raw payload data */
    float     scale;               /**< Dequantization scale factor */
    uint32_t  pad;                 /**< Zero pad */
} NanoTensorDescriptor;
#pragma pack(pop)
```

---

## 2. Comprehensive 219 Tensor ID & Type Specification

### Modular Indexing Formula:
- **Tensor 0:** `embed_tokens.weight` (INT8, `[65536, 2560]`)
- **Layers 0 .. 23:** Each layer occupies **9 contiguous tensor IDs**:
  - `Base = 1 + (9 * Layer)`
  - `Offset 0:` Mixer Pre-RMSNorm (`mixer.norm.weight`, FP32 `[2560]`)
  - **If State Layer (`(Layer + 1) % 3 != 0`):**
    - `Offset 1:` In-Projection (`mixer.in_proj.weight`, Ternary `[5120, 2560]`)
    - `Offset 2:` Depthwise Conv1D (`mixer.conv1d.weight`, FP32 `[2560, 1, 4]`)
    - `Offset 3:` Conv1D Bias (`mixer.conv1d.bias`, FP32 `[2560]`)
    - `Offset 4:` Out-Projection (`mixer.out_proj.weight`, Ternary `[2560, 2560]`)
  - **If GQA Layer (`(Layer + 1) % 3 == 0`):**
    - `Offset 1:` Query Projection (`mixer.q_proj.weight`, Ternary `[2560, 2560]`)
    - `Offset 2:` Key Projection (`mixer.k_proj.weight`, Ternary `[512, 2560]`)
    - `Offset 3:` Value Projection (`mixer.v_proj.weight`, Ternary `[512, 2560]`)
    - `Offset 4:` Out-Projection (`mixer.out_proj.weight`, Ternary `[2560, 2560]`)
  - `Offset 5:` FFN Pre-RMSNorm (`ffn.norm.weight`, FP32 `[2560]`)
  - `Offset 6:` FFN Gate Projection (`ffn.gate_proj.weight`, Ternary `[6912, 2560]`)
  - `Offset 7:` FFN Up Projection (`ffn.up_proj.weight`, Ternary `[6912, 2560]`)
  - `Offset 8:` FFN Down Projection (`ffn.down_proj.weight`, Ternary `[2560, 6912]`)
- **Tensor 217:** Final RMSNorm (`final_norm.weight`, FP32 `[2560]`)
- **Tensor 218:** LM Head Projection (`lm_head.weight`, INT8 `[65536, 2560]`)

### Quantitative Breakdown by Quantization Category:
1. **FP32 Tensors (67 Tensors, 333,056 Parameters, 1,332,224 bytes):**
   - 24 Mixer RMSNorm weights (`[2560]`, 10,240 bytes each)
   - 24 FFN RMSNorm weights (`[2560]`, 10,240 bytes each)
   - 16 State Conv1D weights (`[2560, 1, 4]`, 40,960 bytes each)
   - 16 State Conv1D biases (`[2560]`, 10,240 bytes each)
   - 1 Final RMSNorm weight (`[2560]`, 10,240 bytes)
2. **Ternary 2-bit Packed Tensors (150 Tensors, 1,714,419,200 Parameters, 428,604,800 bytes):**
   - 16 State In-Projections (`[5120, 2560]`, 3,276,800 bytes each)
   - 16 State Out-Projections (`[2560, 2560]`, 1,638,400 bytes each)
   - 8 GQA Q-Projections (`[2560, 2560]`, 1,638,400 bytes each)
   - 8 GQA K-Projections (`[512, 2560]`, 327,680 bytes each)
   - 8 GQA V-Projections (`[512, 2560]`, 327,680 bytes each)
   - 8 GQA Out-Projections (`[2560, 2560]`, 1,638,400 bytes each)
   - 24 FFN Gate Projections (`[6912, 2560]`, 4,423,680 bytes each)
   - 24 FFN Up Projections (`[6912, 2560]`, 4,423,680 bytes each)
   - 24 FFN Down Projections (`[2560, 6912]`, 4,423,680 bytes each)
3. **INT8 Tensors (2 Tensors, 335,544,320 Parameters, 335,544,320 bytes):**
   - 1 Embedding Matrix (`[65536, 2560]`, 167,772,160 bytes)
   - 1 LM Head Matrix (`[65536, 2560]`, 167,772,160 bytes)

---

## 3. Native C++ Engine Integration

### Memory Architecture & Pointer Mapping
```cpp
struct NanoLayerPointers {
    bool           is_gqa;
    const float*   gamma_mixer;      // RMSNorm (FP32)

    // GQA Attention
    const uint8_t* w_q_packed;       // Ternary 2-bit
    float          scale_q;
    const uint8_t* w_k_packed;       // Ternary 2-bit
    float          scale_k;
    const uint8_t* w_v_packed;       // Ternary 2-bit
    float          scale_v;
    const uint8_t* w_out_packed;     // Ternary 2-bit
    float          scale_out;

    // State Block
    const uint8_t* w_state_in_proj;  // Ternary 2-bit [5120, 2560]
    float          scale_state_in;
    const float*   conv_weights;     // FP32 [4, 2560]
    const float*   conv_bias;        // FP32 [2560]
    const uint8_t* w_state_out_proj; // Ternary 2-bit [2560, 2560]
    float          scale_state_out;

    // FFN Block
    const float*   gamma_ffn;        // RMSNorm (FP32)
    const uint8_t* w_gate_packed;    // Ternary 2-bit [6912, 2560]
    float          scale_gate;
    const uint8_t* w_up_packed;      // Ternary 2-bit [6912, 2560]
    float          scale_up;
    const uint8_t* w_down_packed;    // Ternary 2-bit [2560, 6912]
    float          scale_down;
};
```
Every pointer routes directly to memory-mapped read-only payload segments with zero runtime copy overhead.
