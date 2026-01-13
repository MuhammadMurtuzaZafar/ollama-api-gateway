/**
 * cuBLAS Matrix Multiplication Kernels for EdgeLLM
 * Header file for FFI integration with Mojo
 */

#ifndef CUBLAS_MATMUL_H
#define CUBLAS_MATMUL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialization
int cublas_init(size_t weight_bytes, size_t activation_bytes);
int cublas_upload_weights(const float* weights_cpu, size_t bytes);
void cublas_cleanup();
void cublas_sync();

// Memory access
float* get_weights_gpu();
float* get_activations_gpu();

// Core matmul operations
int cublas_matvec(float* out_gpu, const float* x_gpu, const float* W_gpu, int out_dim, int in_dim);
int cublas_matvec_batched(float* out_gpu, const float* x_gpu, const float* W_gpu, int batch, int out_dim, int in_dim);
int cublas_add_bias(float* out_gpu, const float* bias_gpu, int size);

// Layer operations
int gpu_rmsnorm(float* out_gpu, const float* x_gpu, const float* weight_gpu, int size, float eps);
int gpu_swiglu(float* out_gpu, const float* gate_gpu, const float* up_gpu, int size);
int gpu_residual_add(float* x_gpu, const float* residual_gpu, int size);
int gpu_rope(float* q_gpu, float* k_gpu, const float* cos_gpu, const float* sin_gpu, int n_heads, int n_kv_heads, int head_dim);

// GQA Attention
int gpu_gqa_attention(float* output_gpu, const float* Q_gpu, const float* K_cache_gpu, const float* V_cache_gpu,
                      int n_heads, int n_kv_heads, int seq_len, int max_seq, int head_dim);
int gpu_kv_cache_update(float* K_cache_gpu, float* V_cache_gpu, const float* K_gpu, const float* V_gpu,
                        int n_kv_heads, int pos, int max_seq, int head_dim);

// Sampling
int gpu_argmax(int* result_gpu, const float* logits_gpu, int size);

// CUDA memory operations (for FFI)
int cuda_memcpy_d2d(float* dst, const float* src, size_t bytes);
int cuda_memcpy_d2h(float* dst_host, const float* src_device, size_t bytes);
int cuda_memcpy_h2d(float* dst_device, const float* src_host, size_t bytes);

// High-level inference API (recommended for FFI)
int gpu_configure(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                  int vocab_size, int seq_len, int has_bias);
int gpu_forward(int token, int pos);

// ============================================================
// INT4 Quantized Inference API
// ============================================================

// INT4 group size (must match quantization script)
#define INT4_GROUP_SIZE 128

/**
 * Calculate buffer sizes for INT4 model.
 * Use these to allocate GPU memory before init.
 */
void int4_calculate_sizes(
    int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
    int vocab_size, int seq_len,
    size_t* fp32_bytes,          // Size for embeddings/norms/biases (FP32)
    size_t* int4_weights_bytes,  // Size for packed INT4 weights
    size_t* int4_scales_bytes    // Size for FP16 scales
);

/**
 * Initialize INT4 inference mode.
 * Allocates separate buffers for packed INT4 weights and FP16 scales.
 */
int cublas_init_int4(size_t fp32_bytes, size_t int4_weights_bytes,
                     size_t int4_scales_bytes, size_t activation_bytes);

/**
 * Upload INT4 weights to GPU.
 *
 * @param fp32_data FP32 data (embeddings, norms, biases)
 * @param fp32_bytes Size of FP32 data
 * @param int4_packed Packed INT4 weights
 * @param int4_packed_bytes Size of packed weights
 * @param int4_scales FP16 scales
 * @param int4_scales_bytes Size of scales
 */
int cublas_upload_int4_weights(
    const float* fp32_data, size_t fp32_bytes,
    const unsigned char* int4_packed, size_t int4_packed_bytes,
    const void* int4_scales, size_t int4_scales_bytes
);

/**
 * Configure INT4 model dimensions.
 */
int gpu_configure_int4(int dim, int hidden_dim, int n_layers, int n_heads,
                       int n_kv_heads, int vocab_size, int seq_len, int has_bias);

/**
 * INT4 forward pass. Returns next token ID.
 * Uses INT4 GEMV kernels for matmuls, FP32 for embeddings/norms.
 */
int gpu_forward_int4(int token, int pos);

/**
 * Cleanup INT4 resources.
 */
void cublas_cleanup_int4(void);

/**
 * Check if running in INT4 mode.
 * @return 1 if INT4 mode, 0 if FP32 mode
 */
int is_int4_mode(void);

// ============================================================
// INT8 Embedding API (for fast logit computation)
// ============================================================

/**
 * Initialize INT8 embedding for fast logit computation.
 * Allocates GPU memory for INT8 embedding table and per-row FP16 scales.
 *
 * Memory usage: vocab_size * dim (INT8) + vocab_size * 2 (FP16 scales)
 * = 4x smaller than FP32 embedding
 */
int int8_embedding_init(int vocab_size, int dim);

/**
 * Upload INT8 quantized embedding table to GPU.
 *
 * @param emb_int8 INT8 quantized embeddings [vocab_size, dim]
 * @param scales FP16 per-row scales [vocab_size]
 * @param vocab_size Vocabulary size
 * @param dim Embedding dimension
 */
int int8_embedding_upload(
    const signed char* emb_int8,
    const void* scales,  // half* but void for C compatibility
    int vocab_size,
    int dim
);

/**
 * Enable/disable INT8 embedding mode for logit computation.
 * When enabled, uses INT8 GEMV instead of cuBLAS FP32.
 */
void set_int8_embedding_mode(int enable);

#ifdef __cplusplus
}
#endif

#endif // CUBLAS_MATMUL_H
