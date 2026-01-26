/**
 * INT4 GEMV Header - EdgeLLM
 *
 * INT4 quantized matrix-vector multiplication with on-the-fly dequantization.
 * Optimized for Turing (sm_75) - T4 GPU.
 */

#ifndef INT4_GEMV_H
#define INT4_GEMV_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Group size for INT4 quantization (must match export script)
#define INT4_GROUP_SIZE 128

/**
 * Initialize INT4 inference
 *
 * @param existing_stream Optional existing CUDA stream to use. Pass nullptr to create new.
 * @return 0 on success
 */
int int4_init(cudaStream_t existing_stream);

/**
 * INT4 GEMV: out = W @ x
 *
 * Performs matrix-vector multiplication with INT4 weights.
 * Weights are dequantized on-the-fly using group-wise FP16 scales.
 *
 * Weight format per row:
 *   - scales: [n_groups] half values
 *   - packed: [in_dim/2] bytes (2 INT4 values per byte)
 *
 * INT4 values are unsigned [0,15], dequantized as: (val - 8) * scale
 *
 * @param out_gpu Output vector [out_dim] (device pointer)
 * @param x_gpu Input vector [in_dim] (device pointer)
 * @param W_q_gpu Packed INT4 weights [out_dim * in_dim / 2] (device pointer)
 * @param scales_gpu FP16 scales [out_dim * n_groups] (device pointer)
 * @param out_dim Number of output elements
 * @param in_dim Number of input elements (should be multiple of GROUP_SIZE)
 * @return 0 on success
 */
int int4_gemv(
    float* out_gpu,
    const float* x_gpu,
    const uint8_t* W_q_gpu,
    const half* scales_gpu,
    int out_dim,
    int in_dim
);

/**
 * INT4 GEMV with fused bias add: out = W @ x + bias
 *
 * @param out_gpu Output vector [out_dim] (device pointer)
 * @param x_gpu Input vector [in_dim] (device pointer)
 * @param W_q_gpu Packed INT4 weights [out_dim * in_dim / 2] (device pointer)
 * @param scales_gpu FP16 scales [out_dim * n_groups] (device pointer)
 * @param bias_gpu Bias vector [out_dim] (device pointer)
 * @param out_dim Number of output elements
 * @param in_dim Number of input elements
 * @return 0 on success
 */
int int4_gemv_bias(
    float* out_gpu,
    const float* x_gpu,
    const uint8_t* W_q_gpu,
    const half* scales_gpu,
    const float* bias_gpu,
    int out_dim,
    int in_dim
);

/**
 * Synchronize INT4 operations
 */
void int4_sync(void);

/**
 * Benchmark INT4 GEMV kernel
 *
 * @param out_gpu Output buffer (device)
 * @param x_gpu Input buffer (device)
 * @param W_q_gpu Packed weights (device)
 * @param scales_gpu Scales (device)
 * @param out_dim Output dimension
 * @param in_dim Input dimension
 * @param n_iters Number of iterations to average
 * @return Average time per GEMV in microseconds
 */
float int4_benchmark(
    float* out_gpu,
    const float* x_gpu,
    const uint8_t* W_q_gpu,
    const half* scales_gpu,
    int out_dim,
    int in_dim,
    int n_iters
);

// ============================================================
// Memory layout helpers
// ============================================================

/**
 * Calculate bytes needed for packed INT4 weights
 */
static inline size_t int4_packed_size(int out_dim, int in_dim) {
    return (size_t)out_dim * (in_dim / 2);
}

/**
 * Calculate bytes needed for scales
 */
static inline size_t int4_scales_size(int out_dim, int in_dim) {
    int n_groups = (in_dim + INT4_GROUP_SIZE - 1) / INT4_GROUP_SIZE;
    return (size_t)out_dim * n_groups * sizeof(half);
}

/**
 * Calculate total bytes for one INT4 weight matrix
 */
static inline size_t int4_total_size(int out_dim, int in_dim) {
    return int4_scales_size(out_dim, in_dim) + int4_packed_size(out_dim, in_dim);
}

#ifdef __cplusplus
}
#endif

#endif  // INT4_GEMV_H
