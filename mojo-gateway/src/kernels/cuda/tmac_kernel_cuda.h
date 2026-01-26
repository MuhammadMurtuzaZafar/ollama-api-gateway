/**
 * EdgeLLM CUDA T-MAC Kernel Header
 *
 * GPU-accelerated lookup table-based inference for BitNet 1.58-bit models.
 * Uses CUDA for parallel computation on NVIDIA GPUs.
 *
 * Target: NVIDIA Jetson Nano/Orin, RTX GPUs
 */

#ifndef TMAC_KERNEL_CUDA_H
#define TMAC_KERNEL_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize CUDA context with default buffer sizes.
 *
 * Uses sensible defaults suitable for models up to 7B parameters.
 * Buffers are dynamically resized as needed during matmul operations.
 *
 * Must be called before any CUDA operations.
 *
 * @return 0 on success, -1 on failure
 */
int cuda_init(void);

/**
 * Initialize CUDA context with explicit buffer sizes.
 *
 * Allows specifying initial buffer sizes for memory optimization.
 * Buffers are still dynamically resized if operations require more space.
 *
 * @param max_weights_bytes   Initial size of weight buffer in bytes
 * @param max_activations     Initial number of activation elements
 * @param max_output          Initial number of output elements
 * @return 0 on success, -1 on failure
 */
int cuda_init_sized(int max_weights_bytes, int max_activations, int max_output);

/**
 * Cleanup CUDA resources.
 *
 * Frees all allocated device memory.
 */
void cuda_cleanup(void);

/**
 * T-MAC Matrix Multiplication (CUDA)
 *
 * GPU-accelerated BitNet inference using table lookups.
 * Each weight byte encodes 4 ternary values {-1, 0, +1}.
 *
 * @param weights     Packed ternary weights (host memory) [M * (K/4)]
 * @param activations Input activations (host memory) [K * N]
 * @param output      Output buffer (host memory) [M * N]
 * @param scales      Per-row scaling factors (host memory) [M]
 * @param M           Number of output rows
 * @param N           Number of output columns (batch size)
 * @param K           Inner dimension
 * @return 0 on success, -1 on failure
 */
int tmac_matmul_cuda(
    const int8_t* weights,
    const float* activations,
    float* output,
    const float* scales,
    int M, int N, int K
);

/**
 * RMSNorm (CUDA)
 *
 * GPU-accelerated Root Mean Square Layer Normalization.
 * output[i] = (input[i] / rms) * weight[i]
 * where rms = sqrt(mean(input^2) + eps)
 *
 * @param output      Output buffer (host memory) [batch_size * size]
 * @param input       Input buffer (host memory) [batch_size * size]
 * @param weight      Weight buffer (host memory) [size]
 * @param batch_size  Number of batches
 * @param size        Vector size per batch
 * @param eps         Epsilon for numerical stability
 * @return 0 on success, -1 on failure
 */
int rmsnorm_cuda(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int size,
    float eps
);

/**
 * Softmax (CUDA)
 *
 * GPU-accelerated numerically stable softmax.
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * @param output      Output buffer (host memory) [batch_size * size]
 * @param input       Input buffer (host memory) [batch_size * size]
 * @param batch_size  Number of batches
 * @param size        Vector size per batch
 * @return 0 on success, -1 on failure
 */
int softmax_cuda(
    float* output,
    const float* input,
    int batch_size,
    int size
);

/**
 * Check if CUDA is available.
 *
 * @return 1 if CUDA device is available, 0 otherwise
 */
int cuda_available(void);

/**
 * Get CUDA device name.
 *
 * @return Device name string (static buffer, do not free)
 */
const char* cuda_device_name(void);

/**
 * Synchronize CUDA device.
 *
 * Blocks until all CUDA operations complete.
 */
void cuda_sync(void);

/**
 * Get CUDA device properties.
 *
 * @param total_memory    Output: Total device memory in bytes (can be NULL)
 * @param sm_count        Output: Number of streaming multiprocessors (can be NULL)
 * @param compute_major   Output: Compute capability major version (can be NULL)
 * @param compute_minor   Output: Compute capability minor version (can be NULL)
 * @return 0 on success, -1 on failure
 */
int cuda_device_info(
    size_t* total_memory,
    int* sm_count,
    int* compute_major,
    int* compute_minor
);

// ============================================================================
// Phase 1: Persistent GPU Memory API
// ============================================================================

/**
 * Load model weights to GPU memory (one-time operation).
 *
 * Weights remain on GPU until cuda_unload_weights() or cuda_cleanup().
 * Subsequent calls to tmac_matmul_cuda_persistent() skip weight transfer.
 *
 * @param weights         Packed ternary weights (host memory)
 * @param scales          Per-row scaling factors (host memory)
 * @param weight_bytes    Size of weights in bytes
 * @param num_rows        Number of rows (for scales)
 * @return 0 on success, -1 on failure
 */
int cuda_load_weights(
    const int8_t* weights,
    const float* scales,
    int weight_bytes,
    int num_rows
);

/**
 * Unload weights from GPU memory.
 *
 * Call this to free GPU memory when switching models.
 */
void cuda_unload_weights(void);

/**
 * Check if weights are loaded on GPU.
 *
 * @return 1 if weights are loaded, 0 otherwise
 */
int cuda_weights_loaded(void);

/**
 * T-MAC Matrix Multiplication with persistent weights (CUDA)
 *
 * Uses pre-loaded weights from cuda_load_weights().
 * Only transfers activations and output - much faster than tmac_matmul_cuda().
 *
 * @param activations Input activations (host memory) [K * N]
 * @param output      Output buffer (host memory) [M * N]
 * @param M           Number of output rows
 * @param N           Number of output columns (batch size)
 * @param K           Inner dimension
 * @return 0 on success, -1 on failure
 */
int tmac_matmul_cuda_persistent(
    const float* activations,
    float* output,
    int M, int N, int K
);

/**
 * RMSNorm with persistent weights (CUDA)
 *
 * Loads normalization weights to GPU once, reuses across calls.
 *
 * @param norm_weights    Normalization weights (host, only used on first call)
 * @param size            Weight vector size
 * @return 0 on success, -1 on failure
 */
int cuda_load_norm_weights(
    const float* norm_weights,
    int size
);

/**
 * RMSNorm using pre-loaded weights (CUDA)
 *
 * @param output      Output buffer (host memory)
 * @param input       Input buffer (host memory)
 * @param batch_size  Number of batches
 * @param size        Vector size per batch
 * @param eps         Epsilon for numerical stability
 * @return 0 on success, -1 on failure
 */
int rmsnorm_cuda_persistent(
    float* output,
    const float* input,
    int batch_size,
    int size,
    float eps
);

// ============================================================================
// Phase 2: Kernel Fusion + CUDA Streams API
// ============================================================================

/**
 * Initialize CUDA streams for async operations.
 *
 * Creates compute and transfer streams for overlapping operations.
 *
 * @return 0 on success, -1 on failure
 */
int cuda_init_streams(void);

/**
 * Cleanup CUDA streams.
 */
void cuda_cleanup_streams(void);

/**
 * Allocate pinned (page-locked) host memory for faster transfers.
 *
 * @param max_activations   Maximum activation buffer size
 * @param max_output        Maximum output buffer size
 * @return 0 on success, -1 on failure
 */
int cuda_alloc_pinned(int max_activations, int max_output);

/**
 * Free pinned host memory.
 */
void cuda_free_pinned(void);

/**
 * Fused RMSNorm + T-MAC MatMul (CUDA)
 *
 * Combines normalization and matrix multiplication in one kernel launch.
 * Requires both norm weights (cuda_load_norm_weights) and matmul weights
 * (cuda_load_weights) to be pre-loaded.
 *
 * @param input     Input activations (host memory) [K * N]
 * @param output    Output buffer (host memory) [M * N]
 * @param M         Number of output rows
 * @param N         Number of output columns (batch size)
 * @param K         Hidden size / input dimension
 * @param eps       Epsilon for RMSNorm numerical stability
 * @return 0 on success, -1 on failure
 */
int fused_rmsnorm_matmul_cuda(
    const float* input,
    float* output,
    int M, int N, int K,
    float eps
);

/**
 * Async T-MAC MatMul using CUDA streams (CUDA)
 *
 * Overlaps data transfer with computation.
 * Call cuda_sync_streams() to wait for completion.
 *
 * @param activations Input activations (host memory) [K * N]
 * @param output      Output buffer (host memory) [M * N]
 * @param M           Number of output rows
 * @param N           Number of output columns (batch size)
 * @param K           Inner dimension
 * @return 0 on success, -1 on failure
 */
int tmac_matmul_cuda_async(
    const float* activations,
    float* output,
    int M, int N, int K
);

/**
 * Wait for all async stream operations to complete.
 */
void cuda_sync_streams(void);

/**
 * Maximum performance fused kernel with all Phase 2 optimizations.
 *
 * Combines:
 * - Fused RMSNorm + MatMul kernel
 * - CUDA streams for async operations
 * - Pinned memory for faster transfers
 *
 * @param input     Input activations (host memory) [K * N]
 * @param output    Output buffer (host memory) [M * N]
 * @param M         Number of output rows
 * @param N         Number of output columns (batch size)
 * @param K         Hidden size / input dimension
 * @param eps       Epsilon for RMSNorm numerical stability
 * @return 0 on success, -1 on failure
 */
int fused_rmsnorm_matmul_cuda_fast(
    const float* input,
    float* output,
    int M, int N, int K,
    float eps
);

// ============================================================================
// Phase 2.1: Optimized Kernels (No Atomics, True Fusion)
// ============================================================================

/**
 * Optimized T-MAC MatMul with warp-private accumulation (Phase 2.1)
 *
 * Key optimizations:
 * - No atomicAdd (each thread accumulates privately)
 * - Warp-level shuffle reduction (fast!)
 * - Direct accumulation without LUT (simpler for batch_size=1)
 *
 * @param activations Input activations (host memory) [K * N]
 * @param output      Output buffer (host memory) [M * N]
 * @param M           Number of output rows
 * @param N           Number of output columns (batch size)
 * @param K           Inner dimension
 * @return 0 on success, -1 on failure
 */
int tmac_matmul_cuda_v3(
    const float* activations,
    float* output,
    int M, int N, int K
);

/**
 * Streaming Fused RMSNorm + T-MAC MatMul (Phase 2.1)
 *
 * True fusion: normalizes on-the-fly without intermediate storage.
 * Two-pass algorithm:
 *   Pass 1: Compute RMS = sqrt(mean(x^2) + eps)
 *   Pass 2: Stream through K, normalize on-the-fly, accumulate
 *
 * Best for batch_size=1 (single token generation).
 *
 * @param input     Input activations (host memory) [K]
 * @param output    Output buffer (host memory) [M]
 * @param M         Output dimension
 * @param K         Hidden size / input dimension
 * @param eps       Epsilon for RMSNorm numerical stability
 * @return 0 on success, -1 on failure
 */
int streaming_fused_rmsnorm_matmul_cuda(
    const float* input,
    float* output,
    int M, int K,
    float eps
);

/**
 * Adaptive T-MAC MatMul dispatch (Phase 2.1)
 *
 * Automatically chooses optimal kernel based on tensor size:
 * - Large tensors (M*K > 50K): v3 kernel (warp-private accumulation)
 * - Small tensors: persistent kernel (avoids overhead)
 *
 * @param activations Input activations (host memory) [K * N]
 * @param output      Output buffer (host memory) [M * N]
 * @param M           Number of output rows
 * @param N           Number of output columns (batch size)
 * @param K           Inner dimension
 * @return 0 on success, -1 on failure
 */
int tmac_matmul_cuda_adaptive(
    const float* activations,
    float* output,
    int M, int N, int K
);

/**
 * Adaptive Fused RMSNorm + MatMul dispatch (Phase 2.1)
 *
 * Automatically chooses optimal kernel based on tensor size and batch:
 * - Batch=1 + large tensors: streaming fused kernel (best performance)
 * - Otherwise: separate RMSNorm + MatMul (avoids fusion overhead)
 *
 * @param input     Input activations (host memory) [K * N]
 * @param output    Output buffer (host memory) [M * N]
 * @param M         Number of output rows
 * @param N         Number of output columns (batch size)
 * @param K         Hidden size / input dimension
 * @param eps       Epsilon for RMSNorm numerical stability
 * @return 0 on success, -1 on failure
 */
int fused_rmsnorm_matmul_cuda_adaptive(
    const float* input,
    float* output,
    int M, int N, int K,
    float eps
);

// ============================================================================
// Phase 3: INT8 Tensor Core API
// ============================================================================

/**
 * Check if INT8 Tensor Cores are available.
 * Requires compute capability >= 7.5 (Turing+)
 *
 * @return 1 if INT8 Tensor Cores available, 0 otherwise
 */
int cuda_has_int8_tensorcore(void);

/**
 * Get compute capability of the current CUDA device.
 *
 * @return Compute capability as integer (e.g., 75 for sm_75, 86 for sm_86)
 */
int cuda_get_compute_capability(void);

/**
 * Load weights in INT8 Tensor Core format.
 *
 * Expands 2-bit packed ternary weights to full INT8 format.
 * This is a one-time operation at model load time.
 * Memory usage: 4x the packed weight size.
 *
 * @param packed_weights  Packed ternary weights [M * K/4]
 * @param scales          Per-row scaling factors [M]
 * @param weight_bytes    Size of packed weights in bytes
 * @param num_rows        M dimension (number of output rows)
 * @param K               K dimension (inner dimension)
 * @return 0 on success, -1 on failure
 */
int cuda_load_weights_int8_tc(
    const int8_t* packed_weights,
    const float* scales,
    int weight_bytes,
    int num_rows,
    int K
);

/**
 * Unload INT8 Tensor Core weights from GPU memory.
 */
void cuda_unload_weights_int8_tc(void);

/**
 * Check if INT8 TC weights are loaded.
 *
 * @return 1 if INT8 TC weights loaded, 0 otherwise
 */
int cuda_weights_int8_tc_loaded(void);

/**
 * INT8 Tensor Core Matrix Multiplication.
 *
 * Performs: output = weights @ activations
 * - Automatically quantizes FP32 activations to INT8
 * - Uses Tensor Core WMMA API for fast INT8 GEMM
 * - Dequantizes INT32 accumulator to FP32 output
 *
 * Requires: cuda_load_weights_int8_tc() called first.
 *
 * @param activations  FP32 input activations [K * N]
 * @param output       FP32 output buffer [M * N]
 * @param M            Number of output rows
 * @param N            Batch size (number of output columns)
 * @param K            Inner dimension
 * @return 0 on success, -1 on failure
 */
int tmac_matmul_cuda_int8_tc(
    const float* activations,
    float* output,
    int M, int N, int K
);

/**
 * Streaming Fused RMSNorm + INT8 Tensor Core MatMul.
 *
 * Combines normalization with Tensor Core matrix multiplication.
 * Best for large matrices where Tensor Core overhead is amortized.
 *
 * @param input   Input activations [K]
 * @param output  Output buffer [M]
 * @param M       Output dimension
 * @param K       Hidden size / input dimension
 * @param eps     Epsilon for RMSNorm numerical stability
 * @return 0 on success, -1 on failure
 */
int streaming_fused_rmsnorm_matmul_int8_tc(
    const float* input,
    float* output,
    int M, int K,
    float eps
);

/**
 * Adaptive dispatch v2 with INT8 Tensor Core support.
 *
 * Automatically selects the optimal kernel based on:
 * - Hardware capability (INT8 TC requires sm_75+)
 * - Tensor dimensions (TC needs K divisible by 16)
 * - Tensor size (TC overhead only worthwhile for large matrices)
 *
 * Kernel selection:
 * - INT8 Tensor Core: Large aligned matrices on sm_75+
 * - Streaming Fused: Batch size 1, medium matrices
 * - V3 Warp-Private: Small matrices, fallback
 *
 * @param input   Input activations [K * N]
 * @param output  Output buffer [M * N]
 * @param M       Number of output rows
 * @param N       Batch size
 * @param K       Hidden size / input dimension
 * @param eps     Epsilon for RMSNorm numerical stability
 * @return 0 on success, -1 on failure
 */
int fused_rmsnorm_matmul_cuda_adaptive_v2(
    const float* input,
    float* output,
    int M, int N, int K,
    float eps
);

// ============================================================================
// Phase 3.2: Multi-GPU API (Future)
// ============================================================================

/**
 * Initialize multi-GPU context.
 *
 * @param num_gpus  Number of GPUs to use
 * @return 0 on success, -1 on failure
 */
int cuda_init_multi_gpu(int num_gpus);

/**
 * Get number of available CUDA devices.
 *
 * @return Number of CUDA devices, or 0 if none available
 */
int cuda_get_device_count(void);

#ifdef __cplusplus
}
#endif

#endif // TMAC_KERNEL_CUDA_H
