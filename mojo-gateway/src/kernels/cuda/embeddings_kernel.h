/**
 * EdgeLLM Embeddings CUDA Kernel Header
 *
 * Token embedding lookup and positional encoding for LLM inference.
 */

#ifndef EDGELLM_EMBEDDINGS_KERNEL_H
#define EDGELLM_EMBEDDINGS_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Token Embedding Lookup (FP32)
 *
 * Looks up token embeddings from embedding table.
 *
 * @param output Output tensor [batch_size, seq_len, hidden_dim]
 * @param token_ids Input token IDs [batch_size, seq_len]
 * @param embedding_table Embedding table [vocab_size, hidden_dim]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream
 */
void embedding_lookup_f32(
    float* output,
    const int32_t* token_ids,
    const float* embedding_table,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
);

/**
 * Token Embedding Lookup (FP16)
 */
void embedding_lookup_f16(
    half* output,
    const int32_t* token_ids,
    const half* embedding_table,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
);

/**
 * Fused Embedding + RoPE (FP32)
 *
 * Combines token lookup with Rotary Position Embedding.
 *
 * @param output Output tensor [batch_size, seq_len, hidden_dim]
 * @param token_ids Input token IDs [batch_size, seq_len]
 * @param embedding_table Embedding table [vocab_size, hidden_dim]
 * @param cos_cache RoPE cosine cache [max_seq_len, head_dim/2]
 * @param sin_cache RoPE sine cache [max_seq_len, head_dim/2]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 * @param head_dim Dimension per head (for RoPE)
 * @param start_pos Starting position for RoPE
 * @param stream CUDA stream
 */
void embedding_rope_f32(
    float* output,
    const int32_t* token_ids,
    const float* embedding_table,
    const float* cos_cache,
    const float* sin_cache,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int vocab_size,
    int head_dim,
    int start_pos,
    cudaStream_t stream
);

/**
 * RoPE Application (in-place, FP32)
 *
 * Applies Rotary Position Embedding to Q/K tensors.
 *
 * @param q Query tensor [batch_size, n_heads, seq_len, head_dim]
 * @param k Key tensor [batch_size, n_kv_heads, seq_len, head_dim]
 * @param cos_cache Cosine cache [max_seq_len, head_dim/2]
 * @param sin_cache Sine cache [max_seq_len, head_dim/2]
 * @param batch_size Batch size
 * @param n_heads Number of query heads
 * @param n_kv_heads Number of key/value heads
 * @param seq_len Sequence length
 * @param head_dim Dimension per head
 * @param start_pos Starting position
 * @param stream CUDA stream
 */
void apply_rope_f32(
    float* q,
    float* k,
    const float* cos_cache,
    const float* sin_cache,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int start_pos,
    cudaStream_t stream
);

/**
 * Build RoPE Cache
 *
 * Pre-computes cos/sin values for rotary position embedding.
 *
 * @param cos_cache Output cosine cache [max_seq_len, head_dim/2]
 * @param sin_cache Output sine cache [max_seq_len, head_dim/2]
 * @param max_seq_len Maximum sequence length
 * @param head_dim Dimension per head
 * @param base RoPE base frequency (default: 10000.0)
 * @param stream CUDA stream
 */
void build_rope_cache(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // EDGELLM_EMBEDDINGS_KERNEL_H
