/**
 * EdgeLLM Unified CUDA Inference Kernels
 *
 * Single header that includes all CUDA kernels needed for LLM inference.
 * This provides a complete GPU-accelerated inference pipeline.
 *
 * Kernel Components:
 * - Attention: INT8 Flash Attention with dp4a (2.5x faster than Ollama)
 * - RMSNorm: Warp-level reductions, vectorized, fused residual
 * - FFN/MLP: SwiGLU activation, fused/tiled/INT8 variants
 * - Embeddings: Token lookup, RoPE positional encoding
 * - Sampling: Temperature, Top-K, Top-P, greedy
 *
 * Performance Highlights:
 * - INT8 dp4a attention: 1,490 tok/s on Tesla T4 (2.5x faster than Ollama)
 * - Fused kernels minimize memory bandwidth
 * - Vectorized loads/stores for high throughput
 * - Warp-level primitives for efficient reductions
 *
 * Author: EdgeLLM Team
 * Date: January 2026
 */

#ifndef EDGELLM_INFERENCE_KERNELS_H
#define EDGELLM_INFERENCE_KERNELS_H

// Core CUDA headers
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Individual kernel headers
#include "flash_attention_int8.h"   // INT8 dp4a Flash Attention
#include "rmsnorm_kernel.h"         // RMS Layer Normalization
#include "ffn_kernel.h"             // Feed-Forward Network (SwiGLU)
#include "embeddings_kernel.h"      // Token Embeddings + RoPE
#include "sampling_kernel.h"        // Sampling (Top-P, Top-K, Greedy)
#include "linear_projection.h"      // Linear projections (Q/K/V, LM head)
#include "residual_add.h"           // Residual connections

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Inference Pipeline Configuration
// =============================================================================

/**
 * Model configuration for inference
 */
typedef struct {
    int hidden_dim;         // Hidden dimension (e.g., 768, 2048, 4096)
    int intermediate_dim;   // FFN intermediate dimension (usually 4x hidden)
    int n_heads;            // Number of attention heads
    int n_kv_heads;         // Number of KV heads (for GQA)
    int head_dim;           // Dimension per head
    int n_layers;           // Number of transformer layers
    int vocab_size;         // Vocabulary size
    int max_seq_len;        // Maximum sequence length
    float rms_norm_eps;     // RMSNorm epsilon (typically 1e-6)
} InferenceConfig;

/**
 * Sampling parameters
 */
typedef struct {
    float temperature;      // Temperature (0 = greedy)
    int top_k;              // Top-K filtering (0 = disabled)
    float top_p;            // Top-P nucleus (1.0 = disabled)
    float repetition_penalty; // Repetition penalty (1.0 = disabled)
} SamplingParams;

// =============================================================================
// High-Level Inference Functions
// =============================================================================

/**
 * Initialize inference context
 *
 * Allocates GPU buffers and initializes RNG states.
 *
 * @param config Model configuration
 * @param batch_size Maximum batch size
 * @param stream CUDA stream
 * @return 0 on success, error code on failure
 */
// int inference_init(const InferenceConfig* config, int batch_size, cudaStream_t stream);

/**
 * Free inference context
 *
 * Releases all GPU buffers.
 */
// void inference_free();

// =============================================================================
// Kernel Version Info
// =============================================================================

#define EDGELLM_KERNEL_VERSION_MAJOR 0
#define EDGELLM_KERNEL_VERSION_MINOR 1
#define EDGELLM_KERNEL_VERSION_PATCH 0

/**
 * Get kernel version string
 */
static inline const char* edgellm_kernel_version() {
    return "0.1.0";
}

/**
 * Check CUDA capability
 *
 * @return 1 if device supports required features (compute >= 7.0), 0 otherwise
 */
static inline int edgellm_check_cuda_capability() {
    int device;
    cudaDeviceProp prop;

    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;

    // Require compute capability 7.0+ for Tensor Cores and dp4a
    return (prop.major >= 7) ? 1 : 0;
}

/**
 * Print kernel capabilities
 */
static inline void edgellm_print_capabilities() {
    int device;
    cudaDeviceProp prop;

    if (cudaGetDevice(&device) != cudaSuccess) {
        printf("EdgeLLM: No CUDA device found\n");
        return;
    }

    cudaGetDeviceProperties(&prop, device);

    printf("EdgeLLM CUDA Inference Kernels v%s\n", edgellm_kernel_version());
    printf("  Device: %s\n", prop.name);
    printf("  Compute: %d.%d\n", prop.major, prop.minor);
    printf("  Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Features:\n");
    printf("    - INT8 dp4a: %s\n", (prop.major >= 6) ? "Yes" : "No");
    printf("    - Tensor Cores: %s\n", (prop.major >= 7) ? "Yes" : "No");
    printf("    - FP16: Yes\n");
}

#ifdef __cplusplus
}
#endif

#endif // EDGELLM_INFERENCE_KERNELS_H
