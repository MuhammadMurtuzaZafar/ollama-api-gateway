/**
 * EdgeLLM Linear Projection Kernels - Header
 *
 * High-performance linear projections for transformer models.
 * Uses cuBLAS SGEMM with optional INT8 quantization.
 *
 * Key APIs:
 * - linear_f32(): Generic FP32 linear projection
 * - qkv_projection_f32(): Fused Q/K/V projections
 * - output_projection_f32(): Attention output projection
 * - lm_head_f32(): Language model head (hidden -> vocab)
 * - linear_int8(): INT8 quantized projection (4x memory reduction)
 */

#ifndef LINEAR_PROJECTION_H
#define LINEAR_PROJECTION_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Initialization
// =============================================================================

/**
 * Initialize cuBLAS handle for linear projections
 * @return 0 on success
 */
int linear_projection_init(void);

/**
 * Cleanup cuBLAS resources
 */
void linear_projection_cleanup(void);

// =============================================================================
// FP32 Linear Projections
// =============================================================================

/**
 * Generic FP32 linear projection
 *
 * Computes: output = input @ weight^T + bias
 *
 * @param output      Output tensor [batch_size, out_dim]
 * @param input       Input tensor [batch_size, in_dim]
 * @param weight      Weight matrix [out_dim, in_dim]
 * @param bias        Bias vector [out_dim] (NULL for no bias)
 * @param batch_size  Number of samples
 * @param in_dim      Input dimension
 * @param out_dim     Output dimension
 * @param stream      CUDA stream
 * @return 0 on success
 */
int linear_f32(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size,
    int in_dim,
    int out_dim,
    cudaStream_t stream
);

/**
 * Q/K/V Projections for attention
 *
 * Computes Q, K, V from hidden state with support for GQA.
 *
 * @param Q           Query output [batch, n_heads * head_dim]
 * @param K           Key output [batch, n_kv_heads * head_dim]
 * @param V           Value output [batch, n_kv_heads * head_dim]
 * @param hidden      Hidden state [batch, hidden_dim]
 * @param W_q         Query weight [n_heads * head_dim, hidden_dim]
 * @param W_k         Key weight [n_kv_heads * head_dim, hidden_dim]
 * @param W_v         Value weight [n_kv_heads * head_dim, hidden_dim]
 * @param b_q         Query bias (NULL for no bias)
 * @param b_k         Key bias (NULL for no bias)
 * @param b_v         Value bias (NULL for no bias)
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param n_heads     Number of query heads
 * @param n_kv_heads  Number of key/value heads (for GQA)
 * @param head_dim    Dimension per head
 * @param stream      CUDA stream
 * @return 0 on success
 */
int qkv_projection_f32(
    float* Q,
    float* K,
    float* V,
    const float* hidden,
    const float* W_q,
    const float* W_k,
    const float* W_v,
    const float* b_q,
    const float* b_k,
    const float* b_v,
    int batch_size,
    int hidden_dim,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    cudaStream_t stream
);

/**
 * Attention output projection
 *
 * Projects concatenated attention heads back to hidden dimension.
 *
 * @param output      Output [batch, hidden_dim]
 * @param attn_out    Attention output [batch, n_heads * head_dim]
 * @param W_o         Output weight [hidden_dim, n_heads * head_dim]
 * @param b_o         Output bias (NULL for no bias)
 * @param batch_size  Batch size
 * @param n_heads     Number of attention heads
 * @param head_dim    Dimension per head
 * @param hidden_dim  Hidden dimension
 * @param stream      CUDA stream
 * @return 0 on success
 */
int output_projection_f32(
    float* output,
    const float* attn_out,
    const float* W_o,
    const float* b_o,
    int batch_size,
    int n_heads,
    int head_dim,
    int hidden_dim,
    cudaStream_t stream
);

/**
 * Language model head projection
 *
 * Projects final hidden state to vocabulary logits.
 *
 * @param logits      Output logits [batch, vocab_size]
 * @param hidden      Hidden state [batch, hidden_dim]
 * @param W_lm_head   LM head weight [vocab_size, hidden_dim]
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param vocab_size  Vocabulary size
 * @param stream      CUDA stream
 * @return 0 on success
 */
int lm_head_f32(
    float* logits,
    const float* hidden,
    const float* W_lm_head,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
);

// =============================================================================
// INT8 Quantized Projections
// =============================================================================

/**
 * INT8 quantized linear projection
 *
 * Uses INT8 weights with dp4a for 4x memory bandwidth reduction.
 * Best for memory-bound operations on large weight matrices.
 *
 * @param output        Output [batch, out_dim]
 * @param input         Input FP32 [batch, in_dim]
 * @param weight_int8   INT8 weight [out_dim, in_dim]
 * @param weight_scale  Per-channel scales [out_dim]
 * @param bias          Bias (NULL for no bias)
 * @param batch_size    Batch size
 * @param in_dim        Input dimension
 * @param out_dim       Output dimension
 * @param stream        CUDA stream
 * @return 0 on success
 */
int linear_int8(
    float* output,
    const float* input,
    const int8_t* weight_int8,
    const float* weight_scale,
    const float* bias,
    int batch_size,
    int in_dim,
    int out_dim,
    cudaStream_t stream
);

// =============================================================================
// Batched Projections
// =============================================================================

/**
 * Batched linear projection
 *
 * Process multiple independent projections efficiently using
 * cuBLAS batched GEMM.
 *
 * @param output      Output array [num_batches][batch_size, out_dim]
 * @param input       Input array [num_batches][batch_size, in_dim]
 * @param weight      Weight array [num_batches][out_dim, in_dim]
 * @param num_batches Number of independent projections
 * @param batch_size  Batch size per projection
 * @param in_dim      Input dimension
 * @param out_dim     Output dimension
 * @param stream      CUDA stream
 * @return 0 on success
 */
int linear_batched_f32(
    float** output,
    const float** input,
    const float** weight,
    int num_batches,
    int batch_size,
    int in_dim,
    int out_dim,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // LINEAR_PROJECTION_H
