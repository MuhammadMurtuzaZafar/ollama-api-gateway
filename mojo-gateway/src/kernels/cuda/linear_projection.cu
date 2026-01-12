/**
 * EdgeLLM Linear Projection CUDA Kernels
 *
 * High-performance linear projections for transformer models:
 * - Q/K/V projections: hidden_dim -> head_dim * n_heads
 * - Output projection: head_dim * n_heads -> hidden_dim
 * - LM Head: hidden_dim -> vocab_size
 *
 * Uses cuBLAS for optimized GEMM operations with optional:
 * - Fused bias addition
 * - INT8 quantized weights (using dp4a)
 * - Batched projections
 *
 * Author: EdgeLLM Team
 * Date: January 2026
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        return -1; \
    } \
} while(0)

// =============================================================================
// Bias Addition Kernels
// =============================================================================

/**
 * Add bias to output (in-place)
 * output[i] += bias[i % out_dim]
 */
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int out_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_dim;

    if (idx < total) {
        int bias_idx = idx % out_dim;
        output[idx] += bias[bias_idx];
    }
}

/**
 * Vectorized bias addition (float4)
 */
__global__ void add_bias_vec4_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int out_dim
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total = batch_size * out_dim;

    if (idx + 3 < total) {
        int row = idx / out_dim;
        int col = idx % out_dim;

        // Only use vectorized if aligned within row
        if (col + 3 < out_dim) {
            float4 out_vec = *reinterpret_cast<float4*>(output + idx);
            float4 bias_vec = *reinterpret_cast<const float4*>(bias + col);

            out_vec.x += bias_vec.x;
            out_vec.y += bias_vec.y;
            out_vec.z += bias_vec.z;
            out_vec.w += bias_vec.w;

            *reinterpret_cast<float4*>(output + idx) = out_vec;
        } else {
            // Fall back to scalar
            for (int i = 0; i < 4 && idx + i < total; i++) {
                output[idx + i] += bias[(idx + i) % out_dim];
            }
        }
    }
}

// =============================================================================
// INT8 Quantized Linear Projection
// =============================================================================

/**
 * INT8 Linear Projection using dp4a
 *
 * Computes: output = input @ weight^T * scale
 * Where weight is pre-quantized INT8 and input is quantized on-the-fly.
 *
 * For Q/K/V projections where memory bandwidth is the bottleneck,
 * INT8 weights provide 4x memory reduction.
 */
__global__ void linear_int8_kernel(
    float* __restrict__ output,           // [batch, out_dim]
    const float* __restrict__ input,      // [batch, in_dim]
    const int8_t* __restrict__ weight,    // [out_dim, in_dim] INT8
    const float* __restrict__ weight_scale, // [out_dim] per-channel scales
    const float* __restrict__ bias,       // [out_dim] optional
    int batch_size,
    int in_dim,
    int out_dim,
    int use_bias
) {
    int row = blockIdx.x;  // batch index
    int col = blockIdx.y * blockDim.x + threadIdx.x;  // output column

    if (row >= batch_size || col >= out_dim) return;

    const float* x = input + row * in_dim;
    const int8_t* w = weight + col * in_dim;  // weight row for this output

    // Quantize input on-the-fly and compute dot product
    // For simplicity, use fixed scale; production should use dynamic scaling
    float input_scale = 127.0f;

    int32_t acc = 0;
    int vec_dim = in_dim / 4 * 4;

    // Use dp4a for 4-element INT8 dot products
    for (int k = 0; k < vec_dim; k += 4) {
        // Pack input as int8x4
        int32_t x_packed = 0;
        for (int i = 0; i < 4; i++) {
            int8_t xi = (int8_t)fminf(fmaxf(x[k + i] * input_scale, -127.0f), 127.0f);
            x_packed |= ((int32_t)(uint8_t)xi) << (i * 8);
        }

        // Weight is already packed as int8x4
        int32_t w_packed = *reinterpret_cast<const int32_t*>(w + k);

        acc = __dp4a(x_packed, w_packed, acc);
    }

    // Handle remainder
    for (int k = vec_dim; k < in_dim; k++) {
        int8_t xi = (int8_t)fminf(fmaxf(x[k] * input_scale, -127.0f), 127.0f);
        acc += (int32_t)xi * (int32_t)w[k];
    }

    // Dequantize: scale by weight_scale and input_scale
    float result = (float)acc * weight_scale[col] / input_scale;

    if (use_bias && bias != nullptr) {
        result += bias[col];
    }

    output[row * out_dim + col] = result;
}

// =============================================================================
// Host Wrapper Functions
// =============================================================================

// Global cuBLAS handle
static cublasHandle_t cublas_handle = nullptr;
static int cublas_initialized = 0;

extern "C" {

/**
 * Initialize cuBLAS handle
 */
int linear_projection_init(void) {
    if (cublas_initialized) return 0;

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cublas_initialized = 1;

    printf("Linear projection initialized (cuBLAS)\n");
    return 0;
}

/**
 * Cleanup cuBLAS handle
 */
void linear_projection_cleanup(void) {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
    cublas_initialized = 0;
}

/**
 * FP32 Linear Projection using cuBLAS SGEMM
 *
 * Computes: output = input @ weight^T + bias
 *
 * @param output      Output tensor [batch_size, out_dim]
 * @param input       Input tensor [batch_size, in_dim]
 * @param weight      Weight matrix [out_dim, in_dim]
 * @param bias        Bias vector [out_dim] (can be NULL)
 * @param batch_size  Batch size
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
) {
    if (!cublas_initialized) {
        if (linear_projection_init() != 0) return -1;
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    // cuBLAS uses column-major, so we compute:
    // C = alpha * B^T @ A^T + beta * C
    // where A = input [batch, in_dim], B = weight [out_dim, in_dim]
    // Result C = [batch, out_dim]
    //
    // In row-major terms: output = input @ weight^T
    // cuBLAS: output^T = weight @ input^T
    //
    // m = out_dim, n = batch_size, k = in_dim
    // A = weight (m x k), B = input (k x n), C = output (m x n)

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,      // weight transposed
        CUBLAS_OP_N,      // input not transposed
        out_dim,          // m
        batch_size,       // n
        in_dim,           // k
        &alpha,
        weight, in_dim,   // A (weight), lda
        input, in_dim,    // B (input), ldb
        &beta,
        output, out_dim   // C (output), ldc
    ));

    // Add bias if provided
    if (bias != nullptr) {
        int total = batch_size * out_dim;
        int block = 256;
        int grid = (total + block - 1) / block;

        add_bias_kernel<<<grid, block, 0, stream>>>(
            output, bias, batch_size, out_dim
        );
        CUDA_CHECK(cudaGetLastError());
    }

    return 0;
}

/**
 * Q/K/V Projections (fused for efficiency)
 *
 * Computes Q, K, V projections from hidden state:
 * Q = hidden @ W_q + b_q
 * K = hidden @ W_k + b_k
 * V = hidden @ W_v + b_v
 *
 * @param Q           Output Q [batch, n_heads * head_dim]
 * @param K           Output K [batch, n_kv_heads * head_dim]
 * @param V           Output V [batch, n_kv_heads * head_dim]
 * @param hidden      Input hidden state [batch, hidden_dim]
 * @param W_q         Query weight [n_heads * head_dim, hidden_dim]
 * @param W_k         Key weight [n_kv_heads * head_dim, hidden_dim]
 * @param W_v         Value weight [n_kv_heads * head_dim, hidden_dim]
 * @param b_q         Query bias (can be NULL)
 * @param b_k         Key bias (can be NULL)
 * @param b_v         Value bias (can be NULL)
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param n_heads     Number of attention heads
 * @param n_kv_heads  Number of KV heads (for GQA)
 * @param head_dim    Head dimension
 * @param stream      CUDA stream
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
) {
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    // Project Q, K, V (can be done in parallel with multiple streams)
    int ret = 0;
    ret |= linear_f32(Q, hidden, W_q, b_q, batch_size, hidden_dim, q_dim, stream);
    ret |= linear_f32(K, hidden, W_k, b_k, batch_size, hidden_dim, kv_dim, stream);
    ret |= linear_f32(V, hidden, W_v, b_v, batch_size, hidden_dim, kv_dim, stream);

    return ret;
}

/**
 * Attention Output Projection
 *
 * Projects attention output back to hidden dimension:
 * output = attn_output @ W_o + b_o
 *
 * @param output      Output [batch, hidden_dim]
 * @param attn_out    Attention output [batch, n_heads * head_dim]
 * @param W_o         Output weight [hidden_dim, n_heads * head_dim]
 * @param b_o         Output bias (can be NULL)
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
) {
    int attn_dim = n_heads * head_dim;
    return linear_f32(output, attn_out, W_o, b_o, batch_size, attn_dim, hidden_dim, stream);
}

/**
 * LM Head Projection
 *
 * Projects final hidden state to vocabulary logits:
 * logits = hidden @ W_lm_head
 *
 * Note: Usually no bias for LM head
 */
int lm_head_f32(
    float* logits,
    const float* hidden,
    const float* W_lm_head,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
) {
    return linear_f32(logits, hidden, W_lm_head, nullptr, batch_size, hidden_dim, vocab_size, stream);
}

/**
 * INT8 Linear Projection
 *
 * Uses INT8 quantized weights for 4x memory reduction.
 * Best for memory-bandwidth-bound operations.
 *
 * @param output        Output tensor [batch, out_dim]
 * @param input         Input tensor [batch, in_dim]
 * @param weight_int8   INT8 weight matrix [out_dim, in_dim]
 * @param weight_scale  Per-channel dequantization scales [out_dim]
 * @param bias          Bias vector (can be NULL)
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
) {
    dim3 block(256);
    dim3 grid(batch_size, (out_dim + 255) / 256);

    linear_int8_kernel<<<grid, block, 0, stream>>>(
        output, input, weight_int8, weight_scale, bias,
        batch_size, in_dim, out_dim, (bias != nullptr) ? 1 : 0
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/**
 * Batched Linear Projection
 *
 * For processing multiple sequences in parallel.
 * Uses cuBLAS batched GEMM for efficiency.
 *
 * @param output      Output tensors [num_batches][batch_size, out_dim]
 * @param input       Input tensors [num_batches][batch_size, in_dim]
 * @param weight      Weight matrices [num_batches][out_dim, in_dim]
 * @param num_batches Number of independent projections
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
) {
    if (!cublas_initialized) {
        if (linear_projection_init() != 0) return -1;
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemmBatched(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_dim,
        batch_size,
        in_dim,
        &alpha,
        weight, in_dim,
        input, in_dim,
        &beta,
        output, out_dim,
        num_batches
    ));

    return 0;
}

} // extern "C"
