/**
 * INT8 Tensor Core Flash Attention - Header
 *
 * Provides INT8 quantized attention using __dp4a Tensor Core intrinsics.
 * ~4x throughput improvement for INT8 dot products on sm_61+ GPUs.
 *
 * Supported GPUs:
 * - Turing (sm_75+): T4, RTX 20xx
 * - Ampere (sm_80+): A100, RTX 30xx
 * - Ada (sm_89+): RTX 40xx
 *
 * Key optimizations:
 * - __dp4a: 4-element INT8 dot product in single instruction
 * - Vectorized INT8 loads (int8x4 packed as int32)
 * - Async CUDA streams for overlapped copy/compute
 * - GPU-Resident API for zero per-token memcpy
 */

#ifndef FLASH_ATTENTION_INT8_H
#define FLASH_ATTENTION_INT8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Initialization & Cleanup
// ============================================================================

/**
 * Initialize INT8 Flash Attention buffers with async streams
 *
 * @param max_batch_heads  Maximum batch * num_heads
 * @param max_cache_len    Maximum KV cache length
 * @param head_dim         Head dimension (must be multiple of 4 for __dp4a)
 * @return 0 on success, -1 on failure
 */
int flash_attention_int8_init(int max_batch_heads, int max_cache_len, int head_dim);

/**
 * Cleanup INT8 Flash Attention resources (buffers and streams)
 */
void flash_attention_int8_cleanup(void);

/**
 * Reset cache position for new generation sequence
 */
void flash_attention_int8_reset(void);

/**
 * Quantize FP32 tensor to INT8
 *
 * @param input   Input FP32 tensor
 * @param output  Output INT8 tensor (must be pre-allocated)
 * @param scale   Output dequantization scale
 * @param size    Number of elements
 * @return 0 on success
 */
int quantize_tensor_int8(
    const float* input,
    int8_t* output,
    float* scale,
    int size
);

/**
 * Update INT8 KV cache
 *
 * @param K_int8       New key in INT8 [batch_heads, head_dim]
 * @param V_int8       New value in INT8 [batch_heads, head_dim]
 * @param scale_k      Dequantization scale for K
 * @param scale_v      Dequantization scale for V
 * @param batch_heads  Number of batch * heads
 * @param cache_pos    Position to write in cache
 * @return 0 on success
 */
int flash_attention_int8_update_cache(
    const int8_t* K_int8,
    const int8_t* V_int8,
    float scale_k,
    float scale_v,
    int batch_heads,
    int cache_pos
);

/**
 * INT8 Flash Attention Decode (with pre-quantized input)
 *
 * Single-token decode using INT8 quantized QKV.
 *
 * @param Q_int8       Query in INT8 [batch_heads, head_dim]
 * @param K_int8       New key in INT8 [batch_heads, head_dim]
 * @param V_int8       New value in INT8 [batch_heads, head_dim]
 * @param O            Output in FP32 [batch_heads, head_dim]
 * @param scale_q      Dequantization scale for Q
 * @param scale_k      Dequantization scale for K
 * @param scale_v      Dequantization scale for V
 * @param batch_heads  Number of batch * heads
 * @param cache_pos    Current cache position (0-indexed)
 * @param head_dim     Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_int8_decode(
    const int8_t* Q_int8,
    const int8_t* K_int8,
    const int8_t* V_int8,
    float* O,
    float scale_q,
    float scale_k,
    float scale_v,
    int batch_heads,
    int cache_pos,
    int head_dim
);

/**
 * INT8 Flash Attention Decode (with FP32 input)
 *
 * Convenience function that accepts FP32 input and handles quantization internally.
 * Slightly slower than pre-quantized version but easier to use.
 *
 * @param Q           Query in FP32 [batch_heads, head_dim]
 * @param K_new       New key in FP32 [batch_heads, head_dim]
 * @param V_new       New value in FP32 [batch_heads, head_dim]
 * @param O           Output in FP32 [batch_heads, head_dim]
 * @param batch_heads Number of batch * heads
 * @param cache_pos   Current cache position (0-indexed)
 * @param head_dim    Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_int8_decode_fp32(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
);

/**
 * Get INT8 Flash Attention status
 *
 * @param initialized  Output: 1 if initialized
 * @param max_cache    Output: Maximum cache length
 * @param max_bh       Output: Maximum batch*heads
 * @param h_dim        Output: Head dimension
 */
void flash_attention_int8_info(int* initialized, int* max_cache, int* max_bh, int* h_dim);

// ============================================================================
// GPU-Resident API (Zero per-token host-device transfer)
// ============================================================================

/**
 * GPU-Resident INT8 decode (for use when Q, K, V are already on GPU)
 *
 * Eliminates PCIe overhead for maximum throughput. Call after loading
 * Q, K, V to GPU buffers using flash_attention_int8_update_cache.
 *
 * @param batch_heads  Number of batch * heads
 * @param cache_len    Current KV cache length
 * @param scale_q      Dequantization scale for Q
 * @param scale_k      Dequantization scale for K
 * @param scale_v      Dequantization scale for V
 * @return 0 on success
 */
int flash_attention_int8_decode_gpu(
    int batch_heads,
    int cache_len,
    float scale_q,
    float scale_k,
    float scale_v
);

/**
 * Synchronize INT8 compute stream
 *
 * Call after flash_attention_int8_decode_gpu to wait for completion.
 */
void flash_attention_int8_sync(void);

// ============================================================================
// STATELESS Attention API (for Multi-Layer Transformers)
// ============================================================================

/**
 * Stateless FP32 Attention (no internal caches)
 *
 * DETERMINISTIC approach for multi-layer transformers.
 * The caller (Mojo/C++) owns and manages per-layer KV caches.
 * The kernel just computes attention - no state management.
 *
 * This is the recommended API for production inference engines.
 *
 * @param Q           Query [batch_heads, head_dim] - FP32
 * @param K_cache     Key cache [batch_heads, cache_len, head_dim] - FP32, EXTERNAL
 * @param V_cache     Value cache [batch_heads, cache_len, head_dim] - FP32, EXTERNAL
 * @param O           Output [batch_heads, head_dim] - FP32
 * @param batch_heads Number of batch * heads
 * @param cache_len   Number of valid positions in cache (1 to max_seq_len)
 * @param head_dim    Dimension per head
 * @return 0 on success
 */
int attention_stateless_fp32(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim
);

/**
 * Initialize stateless attention buffers
 *
 * Call once at start. Reuses GPU memory across calls for better performance.
 *
 * @param max_batch_heads  Maximum batch * num_heads
 * @param max_cache_len    Maximum KV cache length
 * @param head_dim         Head dimension (must be multiple of 4)
 * @return 0 on success
 */
int attention_stateless_init(int max_batch_heads, int max_cache_len, int head_dim);

/**
 * Cleanup stateless attention buffers
 */
void attention_stateless_cleanup(void);

/**
 * Fast stateless attention with reused buffers
 *
 * More efficient than attention_stateless_fp32 when making multiple calls.
 * Call attention_stateless_init first, or it will auto-initialize.
 * NOTE: Expects contiguous [batch_heads, cache_len, head_dim] layout.
 *
 * @param Q           Query [batch_heads, head_dim] - FP32
 * @param K_cache     Key cache [batch_heads, cache_len, head_dim] - FP32
 * @param V_cache     Value cache [batch_heads, cache_len, head_dim] - FP32
 * @param O           Output [batch_heads, head_dim] - FP32
 * @param batch_heads Number of batch * heads
 * @param cache_len   Number of valid positions in cache
 * @param head_dim    Dimension per head
 * @return 0 on success
 */
int attention_stateless_fast(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim
);

/**
 * Pure FP32 strided attention (no INT8 quantization) - for debugging
 */
int attention_fp32_strided(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim,
    int buffer_seq_len
);

/**
 * Strided stateless attention - handles non-contiguous KV cache
 *
 * Use when KV cache has layout [batch_heads, max_seq_len, head_dim]
 * but only cache_len positions are valid.
 *
 * @param Q              Query [batch_heads, head_dim] - FP32
 * @param K_cache        Key cache [batch_heads, buffer_seq_len, head_dim] - FP32
 * @param V_cache        Value cache [batch_heads, buffer_seq_len, head_dim] - FP32
 * @param O              Output [batch_heads, head_dim] - FP32
 * @param batch_heads    Number of batch * heads
 * @param cache_len      Number of valid positions in cache (used for attention)
 * @param head_dim       Dimension per head
 * @param buffer_seq_len Stride between heads in source buffer (typically max_seq_len)
 * @return 0 on success
 */
int attention_stateless_strided(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim,
    int buffer_seq_len
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_ATTENTION_INT8_H
