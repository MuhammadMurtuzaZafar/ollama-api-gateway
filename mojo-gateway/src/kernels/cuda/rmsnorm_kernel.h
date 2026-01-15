/**
 * EdgeLLM RMSNorm CUDA Kernel Header
 *
 * High-performance RMS Layer Normalization for LLaMA-style models.
 */

#ifndef EDGELLM_RMSNORM_KERNEL_H
#define EDGELLM_RMSNORM_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMSNorm (FP32)
 *
 * Computes: output = (x / sqrt(mean(x^2) + eps)) * weight
 *
 * @param output Output tensor [batch_size, hidden_dim]
 * @param input Input tensor [batch_size, hidden_dim]
 * @param weight Weight tensor [hidden_dim]
 * @param batch_size Batch size
 * @param hidden_dim Hidden dimension
 * @param eps Epsilon for numerical stability (default: 1e-6)
 * @param stream CUDA stream
 */
void rmsnorm_f32(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
);

/**
 * RMSNorm (FP16)
 *
 * Same as rmsnorm_f32 but with half precision
 */
void rmsnorm_f16(
    half* output,
    const half* input,
    const half* weight,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
);

/**
 * Fused RMSNorm + Residual Add (FP32)
 *
 * Computes: output = rmsnorm(input + residual) * weight
 *           residual_out = input + residual  (for next layer)
 *
 * @param output Output tensor [batch_size, hidden_dim]
 * @param residual_out Updated residual for next layer
 * @param input Input tensor [batch_size, hidden_dim]
 * @param residual Residual tensor [batch_size, hidden_dim]
 * @param weight Weight tensor [hidden_dim]
 * @param batch_size Batch size
 * @param hidden_dim Hidden dimension
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream
 */
void rmsnorm_residual_f32(
    float* output,
    float* residual_out,
    const float* input,
    const float* residual,
    const float* weight,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // EDGELLM_RMSNORM_KERNEL_H
