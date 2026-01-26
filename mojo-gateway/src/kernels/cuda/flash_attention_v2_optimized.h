/**
 * FlashAttention-2 Optimized - Header
 *
 * 3-point optimization:
 * 1. Tensor Cores (WMMA) - FP16 compute
 * 2. GPU-Resident Inference - No per-token memcpy
 * 3. FlashDecoding++ - Split-K, unified max
 */

#ifndef FLASH_ATTENTION_V2_OPTIMIZED_H
#define FLASH_ATTENTION_V2_OPTIMIZED_H

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GPU-Resident API (Recommended)
// ============================================================================

/**
 * Initialize optimized FA2 with persistent GPU buffers
 *
 * Call once at model load time. Allocates all GPU memory upfront.
 */
int flash_attention_v2_opt_init(int max_batch_heads, int max_cache_len, int head_dim);

/**
 * Cleanup optimized FA2
 */
void flash_attention_v2_opt_cleanup(void);

/**
 * Load Q, K, V to GPU and update cache
 *
 * Call once per token. This is the only host-device transfer.
 *
 * @param Q_fp32   Query [batch_heads, head_dim] FP32 on host
 * @param K_fp32   Key [batch_heads, head_dim] FP32 on host
 * @param V_fp32   Value [batch_heads, head_dim] FP32 on host
 * @param batch_heads  Number of batch * heads
 */
int flash_attention_v2_opt_load_qkv(
    const float* Q_fp32,
    const float* K_fp32,
    const float* V_fp32,
    int batch_heads
);

/**
 * Run attention on GPU (no host-device transfer)
 *
 * @param batch_heads  Number of batch * heads
 * @param use_split_k  Use Split-K parallelism (better for long sequences)
 */
int flash_attention_v2_opt_decode_gpu(int batch_heads, int use_split_k);

/**
 * Get output from GPU
 *
 * Call once at end of generation (or when output is needed).
 *
 * @param O_fp32   Output buffer [batch_heads, head_dim] FP32 on host
 * @param batch_heads  Number of batch * heads
 */
int flash_attention_v2_opt_get_output(float* O_fp32, int batch_heads);

/**
 * Synchronize compute stream
 */
void flash_attention_v2_opt_sync(void);

/**
 * Reset sequence position (for new generation)
 */
void flash_attention_v2_opt_reset(void);

/**
 * Get info about optimized FA2
 */
void flash_attention_v2_opt_info(int* initialized, int* max_cache, int* max_bh, int* h_dim, int* seq_len);

// ============================================================================
// Backward Compatible API
// ============================================================================

/**
 * Backward compatible API - matches original flash_attention_v2_decode signature
 *
 * Internally uses optimized implementation. Slower than GPU-resident API
 * due to per-call host-device transfers, but easier to integrate.
 */
int flash_attention_v2_opt_decode(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_ATTENTION_V2_OPTIMIZED_H
