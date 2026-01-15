/**
 * EdgeLLM Residual Add Kernels - Header
 *
 * Element-wise operations for transformer residual connections.
 *
 * Key APIs:
 * - residual_add_f32(): In-place x += residual
 * - residual_add_out_f32(): Out-of-place output = a + b
 * - residual_add_rmsnorm_f32(): Fused residual + RMSNorm
 */

#ifndef RESIDUAL_ADD_H
#define RESIDUAL_ADD_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * In-place residual addition
 *
 * x = x + residual
 *
 * @param x         Tensor to update (modified in-place)
 * @param residual  Residual tensor
 * @param size      Total number of elements
 * @param stream    CUDA stream
 * @return 0 on success
 */
int residual_add_f32(
    float* x,
    const float* residual,
    int size,
    cudaStream_t stream
);

/**
 * Scaled residual addition
 *
 * x = x + alpha * residual
 *
 * @param x         Tensor to update (modified in-place)
 * @param residual  Residual tensor
 * @param alpha     Scale factor
 * @param size      Total number of elements
 * @param stream    CUDA stream
 * @return 0 on success
 */
int residual_add_scale_f32(
    float* x,
    const float* residual,
    float alpha,
    int size,
    cudaStream_t stream
);

/**
 * Out-of-place residual addition
 *
 * output = a + b
 *
 * @param output    Output tensor
 * @param a         First input tensor
 * @param b         Second input tensor
 * @param size      Total number of elements
 * @param stream    CUDA stream
 * @return 0 on success
 */
int residual_add_out_f32(
    float* output,
    const float* a,
    const float* b,
    int size,
    cudaStream_t stream
);

/**
 * Fused residual add + RMSNorm
 *
 * Combines residual connection with normalization in single kernel.
 * Eliminates one memory round-trip compared to separate operations.
 *
 * output = rmsnorm(x + residual, weight, eps)
 *
 * @param output       Normalized output [batch_size, hidden_dim]
 * @param residual_out Optional: save x + residual before normalization (NULL to skip)
 * @param x            Input tensor [batch_size, hidden_dim]
 * @param residual     Residual tensor [batch_size, hidden_dim]
 * @param weight       RMSNorm weight [hidden_dim]
 * @param batch_size   Batch size
 * @param hidden_dim   Hidden dimension
 * @param eps          Epsilon for numerical stability (typically 1e-6)
 * @param stream       CUDA stream
 * @return 0 on success
 */
int residual_add_rmsnorm_f32(
    float* output,
    float* residual_out,
    const float* x,
    const float* residual,
    const float* weight,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
);

/**
 * Batch residual add for multiple tensors
 *
 * Process residual connections at multiple points efficiently.
 *
 * @param x_tensors        Array of tensors to update
 * @param residual_tensors Array of residual tensors
 * @param num_tensors      Number of tensor pairs
 * @param size_per_tensor  Elements per tensor
 * @param stream           CUDA stream
 * @return 0 on success
 */
int residual_add_batch_f32(
    float** x_tensors,
    const float** residual_tensors,
    int num_tensors,
    int size_per_tensor,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // RESIDUAL_ADD_H
