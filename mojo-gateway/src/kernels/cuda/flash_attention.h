/**
 * Flash Attention CUDA Header
 *
 * Memory-efficient attention implementation based on:
 * "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * by Tri Dao et al. (https://arxiv.org/abs/2205.14135)
 *
 * Key benefits:
 * - O(N) memory instead of O(N^2) for attention matrix
 * - 2-4x faster than standard attention
 * - IO-aware: minimizes HBM reads/writes
 */

#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize Flash Attention buffers
 *
 * Allocates device memory for Q, K, V, O tensors.
 *
 * @param max_batch       Maximum batch size
 * @param num_heads       Number of attention heads
 * @param max_sequence_len Maximum sequence length
 * @param head_dim        Dimension per head (typically 64 or 128)
 * @return 0 on success, -1 on failure
 */
int flash_attention_init(int max_batch, int num_heads, int max_sequence_len, int head_dim);

/**
 * Initialize KV cache for inference/decoding
 *
 * Allocates separate buffers for cached keys and values.
 * Required for efficient autoregressive generation.
 *
 * @param max_batch       Maximum batch size
 * @param num_heads       Number of attention heads
 * @param max_cache_len   Maximum cache length (max context window)
 * @param head_dim        Dimension per head
 * @return 0 on success, -1 on failure
 */
int flash_attention_init_kv_cache(int max_batch, int num_heads, int max_cache_len, int head_dim);

/**
 * Cleanup Flash Attention resources
 *
 * Frees all allocated device memory.
 */
void flash_attention_cleanup(void);

/**
 * Flash Attention Forward Pass
 *
 * Computes: O = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Uses tiled computation with online softmax to achieve O(N) memory.
 *
 * @param Q           Query tensor [batch * num_heads, seq_len, head_dim]
 * @param K           Key tensor [batch * num_heads, seq_len, head_dim]
 * @param V           Value tensor [batch * num_heads, seq_len, head_dim]
 * @param O           Output tensor [batch * num_heads, seq_len, head_dim]
 * @param batch_heads batch * num_heads
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension
 * @param causal      1 for causal (decoder) attention, 0 for bidirectional
 * @return 0 on success, -1 on failure
 */
int flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_heads,
    int seq_len,
    int head_dim,
    int causal
);

/**
 * Flash Attention Decode - Single token with KV cache
 *
 * Optimized for autoregressive generation where:
 * - Q has shape [batch * heads, 1, head_dim] (single new token)
 * - K, V are retrieved from cache [batch * heads, cache_len, head_dim]
 *
 * This is the critical path for inference throughput.
 *
 * @param Q           Query for new token [batch * heads, 1, head_dim]
 * @param K_new       New key to append to cache [batch * heads, 1, head_dim]
 * @param V_new       New value to append to cache [batch * heads, 1, head_dim]
 * @param O           Output [batch * heads, 1, head_dim]
 * @param batch_heads batch * num_heads
 * @param cache_pos   Current position in cache (0-indexed, also = num_cached_tokens)
 * @param head_dim    Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_decode(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
);

/**
 * Update KV cache directly
 *
 * Used during prefill phase to populate cache with initial context.
 *
 * @param K           Keys to cache [batch * heads, num_tokens, head_dim]
 * @param V           Values to cache [batch * heads, num_tokens, head_dim]
 * @param batch_heads batch * num_heads
 * @param start_pos   Starting position in cache
 * @param num_tokens  Number of tokens to cache
 * @param head_dim    Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_update_kv_cache(
    const float* K,
    const float* V,
    int batch_heads,
    int start_pos,
    int num_tokens,
    int head_dim
);

/**
 * Get Flash Attention status info
 *
 * @param initialized   Output: 1 if FA buffers initialized
 * @param cache_ready   Output: 1 if KV cache initialized
 * @param max_seq       Output: Maximum sequence length
 * @param head_d        Output: Head dimension
 */
void flash_attention_info(int* initialized, int* cache_ready, int* max_seq, int* head_d);

#ifdef __cplusplus
}
#endif

#endif // FLASH_ATTENTION_H
