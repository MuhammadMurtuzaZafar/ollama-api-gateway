/**
 * EdgeLLM FFN/MLP CUDA Kernel Header
 *
 * High-performance Feed-Forward Network with SwiGLU activation.
 */

#ifndef EDGELLM_FFN_KERNEL_H
#define EDGELLM_FFN_KERNEL_H

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SwiGLU activation (gate + up projection)
 *
 * Computes: output = silu(x @ W_gate) * (x @ W_up)
 *
 * @param output Output tensor [batch_size, intermediate_dim]
 * @param input Input tensor [batch_size, hidden_dim]
 * @param w_gate Gate weights [hidden_dim, intermediate_dim]
 * @param w_up Up projection weights [hidden_dim, intermediate_dim]
 * @param batch_size Batch size
 * @param hidden_dim Input dimension
 * @param intermediate_dim Intermediate dimension
 * @param stream CUDA stream
 */
void swiglu_f32(
    float* output,
    const float* input,
    const float* w_gate,
    const float* w_up,
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream
);

/**
 * Down projection
 *
 * Computes: output = input @ W_down
 *
 * @param output Output tensor [batch_size, hidden_dim]
 * @param input Input tensor [batch_size, intermediate_dim]
 * @param w_down Down projection weights [intermediate_dim, hidden_dim]
 * @param batch_size Batch size
 * @param intermediate_dim Intermediate dimension
 * @param hidden_dim Output dimension
 * @param stream CUDA stream
 */
void down_proj_f32(
    float* output,
    const float* input,
    const float* w_down,
    int batch_size,
    int intermediate_dim,
    int hidden_dim,
    cudaStream_t stream
);

/**
 * Full FFN with SwiGLU (fused when possible)
 *
 * Computes: output = (silu(x @ W_gate) * (x @ W_up)) @ W_down
 *
 * @param output Output tensor [batch_size, hidden_dim]
 * @param intermediate Temporary buffer [batch_size, intermediate_dim]
 * @param input Input tensor [batch_size, hidden_dim]
 * @param w_gate Gate weights [hidden_dim, intermediate_dim]
 * @param w_up Up projection weights [hidden_dim, intermediate_dim]
 * @param w_down Down projection weights [intermediate_dim, hidden_dim]
 * @param batch_size Batch size
 * @param hidden_dim Input/output dimension
 * @param intermediate_dim Intermediate dimension
 * @param stream CUDA stream
 */
void ffn_swiglu_f32(
    float* output,
    float* intermediate,
    const float* input,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream
);

/**
 * INT8 quantized SwiGLU
 *
 * Uses INT8 weights with per-column scaling for memory efficiency
 *
 * @param output Output tensor [batch_size, intermediate_dim]
 * @param input Input tensor [batch_size, hidden_dim] (FP32)
 * @param w_gate_int8 INT8 gate weights [hidden_dim, intermediate_dim]
 * @param w_up_int8 INT8 up weights [hidden_dim, intermediate_dim]
 * @param scale_gate Per-column scales for gate [intermediate_dim]
 * @param scale_up Per-column scales for up [intermediate_dim]
 * @param batch_size Batch size
 * @param hidden_dim Input dimension
 * @param intermediate_dim Intermediate dimension
 * @param stream CUDA stream
 */
void swiglu_int8(
    float* output,
    const float* input,
    const int8_t* w_gate_int8,
    const int8_t* w_up_int8,
    const float* scale_gate,
    const float* scale_up,
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // EDGELLM_FFN_KERNEL_H
