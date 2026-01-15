/**
 * FlashAttention-2 - Header
 *
 * Optimized attention with tiled processing and online softmax.
 * Minimizes HBM bandwidth for better performance.
 */

#ifndef FLASH_ATTENTION_V2_H
#define FLASH_ATTENTION_V2_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize FlashAttention-2 buffers
 *
 * @param max_batch_heads  Maximum batch * num_heads
 * @param max_cache_len    Maximum KV cache length
 * @param head_dim         Head dimension (must be multiple of 4)
 * @return 0 on success, -1 on failure
 */
int flash_attention_v2_init(int max_batch_heads, int max_cache_len, int head_dim);

/**
 * Cleanup FlashAttention-2 resources
 */
void flash_attention_v2_cleanup(void);

/**
 * Update KV cache at position
 *
 * @param K_new       New key [batch_heads, head_dim]
 * @param V_new       New value [batch_heads, head_dim]
 * @param batch_heads Number of batch * heads
 * @param cache_pos   Position to write in cache
 * @param head_dim    Head dimension
 * @return 0 on success
 */
int flash_attention_v2_update_cache(
    const float* K_new,
    const float* V_new,
    int batch_heads,
    int cache_pos,
    int head_dim
);

/**
 * FlashAttention-2 Decode
 *
 * Single-token decode with tiled K/V processing and online softmax.
 *
 * @param Q           Query [batch_heads, head_dim]
 * @param K_new       New key [batch_heads, head_dim]
 * @param V_new       New value [batch_heads, head_dim]
 * @param O           Output [batch_heads, head_dim]
 * @param batch_heads Number of batch * heads
 * @param cache_pos   Current cache position (0-indexed)
 * @param head_dim    Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_v2_decode(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
);

/**
 * Get FlashAttention-2 status
 */
void flash_attention_v2_info(int* initialized, int* max_cache, int* max_bh, int* h_dim);

#ifdef __cplusplus
}
#endif

#endif // FLASH_ATTENTION_V2_H
