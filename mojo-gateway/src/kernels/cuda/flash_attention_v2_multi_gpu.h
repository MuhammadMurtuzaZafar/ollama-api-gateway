/**
 * FlashAttention-2 Multi-GPU - Header
 *
 * Tensor Parallel attention with NCCL communication.
 * Shards attention heads across multiple GPUs for linear speedup.
 *
 * Supported configurations:
 * - 2 GPUs: 1.7-2.0x speedup
 * - 4 GPUs: 2.5-3.1x speedup
 * - 8 GPUs: 3.5-4.0x speedup (with NVLink)
 */

#ifndef FLASH_ATTENTION_V2_MULTI_GPU_H
#define FLASH_ATTENTION_V2_MULTI_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize Multi-GPU FlashAttention-2
 *
 * Sets up NCCL communicators and allocates memory on each GPU.
 * Heads are automatically sharded across available GPUs.
 *
 * @param num_gpus          Number of GPUs to use (0 = auto-detect)
 * @param total_heads       Total number of attention heads
 * @param max_cache_len     Maximum KV cache length
 * @param head_dim          Head dimension (must be multiple of 4)
 * @return 0 on success, -1 on failure
 */
int fa2_multi_gpu_init(int num_gpus, int total_heads, int max_cache_len, int head_dim);

/**
 * Cleanup Multi-GPU resources
 */
void fa2_multi_gpu_cleanup(void);

/**
 * Multi-GPU FlashAttention-2 Decode
 *
 * Performs tensor-parallel attention across multiple GPUs.
 * Each GPU processes a subset of attention heads.
 *
 * @param Q           Query [total_heads, head_dim] (on host)
 * @param K_new       New key [total_heads, head_dim] (on host)
 * @param V_new       New value [total_heads, head_dim] (on host)
 * @param O           Output [total_heads, head_dim] (on host)
 * @param cache_pos   Current cache position (0-indexed)
 * @return 0 on success, -1 on failure
 */
int fa2_multi_gpu_decode(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int cache_pos
);

/**
 * Get Multi-GPU configuration info
 *
 * @param num_gpus      Output: Number of GPUs in use
 * @param heads_per_gpu Output: Heads assigned to each GPU
 * @param total_heads   Output: Total number of heads
 */
void fa2_multi_gpu_info(int* num_gpus, int* heads_per_gpu, int* total_heads);

/**
 * Get per-GPU timing statistics
 *
 * @param gpu_id        GPU index
 * @param compute_ms    Output: Compute time in milliseconds
 * @param comm_ms       Output: Communication time in milliseconds
 */
void fa2_multi_gpu_get_timing(int gpu_id, float* compute_ms, float* comm_ms);

/**
 * Synchronize all GPUs
 * Useful for benchmarking and ensuring completion.
 */
void fa2_multi_gpu_sync(void);

/**
 * Check if multi-GPU is available and initialized
 *
 * @return 1 if initialized with multiple GPUs, 0 otherwise
 */
int fa2_multi_gpu_available(void);

#ifdef __cplusplus
}
#endif

#endif // FLASH_ATTENTION_V2_MULTI_GPU_H
