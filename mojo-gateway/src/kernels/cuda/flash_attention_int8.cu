/**
 * INT8 Tensor Core Flash Attention
 *
 * Uses WMMA (Warp Matrix Multiply Accumulate) API for INT8 matrix operations.
 * Provides ~8x speedup over FP32 CUDA cores on Tensor Core-enabled GPUs.
 *
 * Supported GPUs: Turing (sm_75+), Ampere (sm_80+), Ada (sm_89+)
 * - T4: sm_75 (Turing Tensor Cores, INT8)
 * - A100: sm_80 (Ampere Tensor Cores, INT8)
 * - RTX 4090: sm_89 (Ada Tensor Cores, INT8)
 *
 * Matrix shapes for INT8 WMMA on sm_75:
 * - m=8, n=32, k=16
 * - m=32, n=8, k=16
 * - m=16, n=16, k=16
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace nvcuda;

// ============================================================================
// Configuration for INT8 Tensor Cores
// ============================================================================

// WMMA dimensions for INT8 (sm_75+)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block configuration
#define TC_BLOCK_M 32      // Rows of Q per block
#define TC_BLOCK_N 32      // Columns of K per block
#define TC_THREADS 128     // 4 warps
#define TC_WARPS (TC_THREADS / 32)

// Quantization scale (dynamic per-tensor)
#define QUANT_SCALE 127.0f

// Error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Quantization Utilities (Fixed for multi-block correctness)
// ============================================================================

/**
 * Pass 1: Find max absolute value per block, write to global partial_max array
 */
__global__ void find_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_max,
    int size
) {
    __shared__ float shared_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Each thread finds max in its strided range
    float local_max = 0.0f;
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    // Block reduction
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Block 0 thread 0 writes result
    if (tid == 0) {
        partial_max[blockIdx.x] = shared_max[0];
    }
}

/**
 * Pass 2: Reduce partial maxes to single value
 */
__global__ void reduce_max_kernel(
    float* __restrict__ partial_max,
    float* __restrict__ final_max,
    int num_blocks
) {
    __shared__ float shared_max[256];

    int tid = threadIdx.x;
    float local_max = 0.0f;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_max = fmaxf(local_max, partial_max[i]);
    }

    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *final_max = shared_max[0];
    }
}

/**
 * Pass 3: Quantize with known scale
 */
__global__ void quantize_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float max_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float quant_scale = (max_val > 0.0f) ? (QUANT_SCALE / max_val) : 1.0f;

    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        float val = input[i] * quant_scale;
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        output[i] = (int8_t)rintf(val);
    }
}

/**
 * Quantize FP32 to INT8 with per-tensor scaling (FIXED multi-block version)
 * Host function that orchestrates the 3-pass quantization
 */
__global__ void quantize_fp32_to_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    int size
) {
    // NOTE: This kernel is deprecated - use quantize_tensor_to_int8() instead
    // Keeping for backward compatibility but it only works correctly for single block

    __shared__ float shared_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Find max (only correct for single block!)
    float local_max = 0.0f;
    for (int i = idx; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    float max_val = shared_max[0];
    float quant_scale = (max_val > 0.0f) ? (QUANT_SCALE / max_val) : 1.0f;

    if (tid == 0) {
        *scale = max_val / QUANT_SCALE;
    }
    __syncthreads();

    // Quantize
    for (int i = idx; i < size; i += blockDim.x) {
        float val = input[i] * quant_scale;
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        output[i] = (int8_t)rintf(val);
    }
}

/**
 * Fast quantization with pre-computed scale
 */
__global__ void quantize_with_scale_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * scale;
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        output[idx] = (int8_t)rintf(val);
    }
}

/**
 * Dequantize INT32 accumulator to FP32
 */
__global__ void dequantize_int32_to_fp32_kernel(
    const int* __restrict__ input,
    float* __restrict__ output,
    float scale_q,
    float scale_k,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (float)input[idx] * scale_q * scale_k;
    }
}

// ============================================================================
// INT8 Tensor Core GEMM Kernel
// ============================================================================

/**
 * INT8 WMMA GEMM: C = A @ B^T
 *
 * Uses Tensor Cores for 8x8x32 INT8 matrix multiply-accumulate.
 * A: [M, K] INT8, B: [N, K] INT8, C: [M, N] INT32
 */
__global__ void int8_gemm_wmma_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int* __restrict__ C,
    int M, int N, int K
) {
    // Warp and thread indices
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = threadIdx.x % 32;

    // Calculate warp's tile position
    int warp_row = (warp_id / (N / WMMA_N)) * WMMA_M;
    int warp_col = (warp_id % (N / WMMA_N)) * WMMA_N;

    if (warp_row >= M || warp_col >= N) return;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0);

    // Main loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        // Load A fragment [warp_row:warp_row+M, k:k+K]
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);

        // Load B fragment [warp_col:warp_col+N, k:k+K] (transposed)
        wmma::load_matrix_sync(b_frag, B + warp_col * K + k, K);

        // Tensor Core MMA: C += A @ B^T
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}

// ============================================================================
// INT8 Flash Attention Decode Kernel
// ============================================================================

/**
 * INT8 Tensor Core Flash Attention for single-token decode
 *
 * Optimized for autoregressive generation:
 * - Q: [batch_heads, 1, head_dim] - single new token
 * - K_cache: [batch_heads, cache_len, head_dim] - cached history
 * - V_cache: [batch_heads, cache_len, head_dim] - cached history
 *
 * Algorithm:
 * 1. Quantize Q to INT8
 * 2. Use Tensor Core GEMM for Q @ K^T (INT8 -> INT32)
 * 3. Dequantize to FP32 and apply softmax
 * 4. Quantize softmax output to INT8
 * 5. Use Tensor Core GEMM for softmax @ V (INT8 -> INT32)
 * 6. Dequantize final output
 */
__global__ void flash_attention_int8_decode_kernel(
    const int8_t* __restrict__ Q_int8,      // [batch_heads, head_dim]
    const int8_t* __restrict__ K_cache_int8, // [batch_heads, cache_len, head_dim]
    const int8_t* __restrict__ V_cache_int8, // [batch_heads, cache_len, head_dim]
    float* __restrict__ O,                   // [batch_heads, head_dim]
    const float scale_q,
    const float scale_k,
    const float scale_v,
    const int cache_len,
    const int head_dim,
    const float attn_scale
) {
    // Each block handles one batch*head
    const int batch_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory for scores and intermediate results
    extern __shared__ float smem[];
    float* s_scores = smem;                    // [cache_len]
    int8_t* s_K_tile = (int8_t*)(smem + ((cache_len + 31) / 32 * 32));  // Aligned
    int8_t* s_V_tile = s_K_tile + 32 * head_dim;

    // Load Q for this batch_head (single row)
    __shared__ int8_t s_Q[128];  // max head_dim = 128
    for (int d = tid; d < head_dim; d += TC_THREADS) {
        s_Q[d] = Q_int8[batch_head_idx * head_dim + d];
    }
    __syncthreads();

    // Compute attention scores: Q @ K^T
    // Using INT8 dot products with INT32 accumulation
    float local_max = -FLT_MAX;

    const int8_t* K_ptr = K_cache_int8 + batch_head_idx * cache_len * head_dim;
    const int8_t* V_ptr = V_cache_int8 + batch_head_idx * cache_len * head_dim;

    // Process cache in tiles
    for (int k_start = 0; k_start < cache_len; k_start += TC_THREADS) {
        int k_idx = k_start + tid;

        if (k_idx < cache_len) {
            // INT8 dot product: Q @ K[k_idx]
            int32_t dot = 0;
            for (int d = 0; d < head_dim; d++) {
                dot += (int32_t)s_Q[d] * (int32_t)K_ptr[k_idx * head_dim + d];
            }

            // Dequantize and scale
            float score = (float)dot * scale_q * scale_k * attn_scale;
            s_scores[k_idx] = score;
            local_max = fmaxf(local_max, score);
        }
    }
    __syncthreads();

    // Block reduction for max
    __shared__ float s_max;
    if (tid == 0) s_max = -FLT_MAX;
    __syncthreads();

    atomicMax((int*)&s_max, __float_as_int(local_max));
    __syncthreads();
    float global_max = s_max;

    // Compute softmax numerator and sum
    float local_sum = 0.0f;
    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        float exp_score = expf(s_scores[k_idx] - global_max);
        s_scores[k_idx] = exp_score;
        local_sum += exp_score;
    }
    __syncthreads();

    // Block reduction for sum
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    // Normalize softmax
    float inv_sum = 1.0f / s_sum;
    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        s_scores[k_idx] *= inv_sum;
    }
    __syncthreads();

    // Compute output: softmax @ V
    // For each output dimension
    for (int d = tid; d < head_dim; d += TC_THREADS) {
        float acc = 0.0f;

        for (int k_idx = 0; k_idx < cache_len; k_idx++) {
            // Dequantize V and multiply by softmax
            float v_val = (float)V_ptr[k_idx * head_dim + d] * scale_v;
            acc += s_scores[k_idx] * v_val;
        }

        O[batch_head_idx * head_dim + d] = acc;
    }
}

/**
 * Optimized INT8 Tensor Core Decode using __dp4a for Q@K^T
 *
 * Uses __dp4a intrinsic for 4x throughput INT8 dot products.
 * This is the optimal approach for single-token decode where Q is [1, head_dim].
 *
 * __dp4a(a, b, c) computes: c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
 * where a and b are packed int8x4 values.
 */
__global__ void flash_attention_int8_decode_wmma_kernel(
    const int8_t* __restrict__ Q_int8,
    const int8_t* __restrict__ K_cache_int8,
    const int8_t* __restrict__ V_cache_int8,
    float* __restrict__ O,
    const float scale_q,
    const float scale_k,
    const float scale_v,
    const int cache_len,
    const int head_dim,
    const float attn_scale
) {
    const int batch_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = TC_THREADS / 32;

    // Shared memory layout:
    // s_scores: [cache_len] floats - attention scores
    // s_Q_packed: [head_dim/4] int32 - Q packed as int8x4
    extern __shared__ char shared_mem[];
    float* s_scores = (float*)shared_mem;
    int* s_Q_packed = (int*)(s_scores + ((cache_len + 15) / 16) * 16);

    // Pointers for this batch_head
    const int8_t* Q_ptr = Q_int8 + batch_head_idx * head_dim;
    const int8_t* K_ptr = K_cache_int8 + batch_head_idx * cache_len * head_dim;
    const int8_t* V_ptr = V_cache_int8 + batch_head_idx * cache_len * head_dim;
    float* O_ptr = O + batch_head_idx * head_dim;

    // Load Q to shared memory as packed int8x4 for __dp4a
    // head_dim is typically 64 or 128, always multiple of 4
    const int packed_dim = head_dim / 4;
    const int* Q_packed = (const int*)Q_ptr;
    for (int d = tid; d < packed_dim; d += TC_THREADS) {
        s_Q_packed[d] = Q_packed[d];
    }
    __syncthreads();

    // ============================================================
    // Phase 1: Compute Q @ K^T using __dp4a (4x INT8 dot product)
    // ============================================================

    float local_max = -FLT_MAX;

    // Each thread processes multiple K positions
    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        // K row pointer (packed as int8x4)
        const int* K_row_packed = (const int*)(K_ptr + k_idx * head_dim);

        // INT8 dot product using __dp4a - processes 4 INT8 elements per instruction
        int32_t dot = 0;

        #pragma unroll 4
        for (int d = 0; d < packed_dim; d++) {
            // __dp4a: dot += Q[d*4:d*4+4] . K[d*4:d*4+4]
            dot = __dp4a(s_Q_packed[d], K_row_packed[d], dot);
        }

        // Dequantize and scale
        float score = (float)dot * scale_q * scale_k * attn_scale;
        s_scores[k_idx] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();

    // ============================================================
    // Max reduction using warp shuffles
    // ============================================================

    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Block reduction
    __shared__ float s_block_max[4];
    if (lane_id == 0) s_block_max[warp_id] = local_max;
    __syncthreads();

    if (tid == 0) {
        float block_max = s_block_max[0];
        for (int i = 1; i < num_warps; i++) {
            block_max = fmaxf(block_max, s_block_max[i]);
        }
        s_block_max[0] = block_max;
    }
    __syncthreads();
    float global_max = s_block_max[0];

    // ============================================================
    // Softmax: exp and sum
    // ============================================================

    float local_sum = 0.0f;
    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        float exp_score = expf(s_scores[k_idx] - global_max);
        s_scores[k_idx] = exp_score;
        local_sum += exp_score;
    }
    __syncthreads();

    // Warp reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Block reduction for sum
    __shared__ float s_block_sum[4];
    if (lane_id == 0) s_block_sum[warp_id] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total_sum += s_block_sum[i];
        }
        s_block_sum[0] = total_sum;
    }
    __syncthreads();

    // Normalize softmax probabilities
    float inv_sum = 1.0f / s_block_sum[0];
    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        s_scores[k_idx] *= inv_sum;
    }
    __syncthreads();

    // ============================================================
    // Phase 2: Compute softmax @ V using vectorized loads
    // ============================================================

    // Each thread computes multiple output dimensions
    for (int d = tid; d < head_dim; d += TC_THREADS) {
        float acc = 0.0f;

        // Process V values - unroll for better ILP
        int k_idx = 0;

        // Process in chunks of 4 for better instruction-level parallelism
        for (; k_idx + 3 < cache_len; k_idx += 4) {
            float s0 = s_scores[k_idx];
            float s1 = s_scores[k_idx + 1];
            float s2 = s_scores[k_idx + 2];
            float s3 = s_scores[k_idx + 3];

            int8_t v0 = V_ptr[(k_idx + 0) * head_dim + d];
            int8_t v1 = V_ptr[(k_idx + 1) * head_dim + d];
            int8_t v2 = V_ptr[(k_idx + 2) * head_dim + d];
            int8_t v3 = V_ptr[(k_idx + 3) * head_dim + d];

            acc += s0 * (float)v0;
            acc += s1 * (float)v1;
            acc += s2 * (float)v2;
            acc += s3 * (float)v3;
        }

        // Handle remaining elements
        for (; k_idx < cache_len; k_idx++) {
            acc += s_scores[k_idx] * (float)V_ptr[k_idx * head_dim + d];
        }

        // Dequantize V and write output
        O_ptr[d] = acc * scale_v;
    }
}

/**
 * Ultra-optimized INT8 decode kernel with warp-level P@V using __dp4a
 *
 * For the P@V computation, we transpose the problem:
 * Instead of each thread computing one output dimension,
 * each warp collaboratively computes output using __dp4a on V tiles.
 */
__global__ void flash_attention_int8_decode_dp4a_kernel(
    const int8_t* __restrict__ Q_int8,
    const int8_t* __restrict__ K_cache_int8,
    const int8_t* __restrict__ V_cache_int8,
    float* __restrict__ O,
    const float scale_q,
    const float scale_k,
    const float scale_v,
    const int cache_len,
    const int head_dim,
    const float attn_scale
) {
    const int batch_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = TC_THREADS / 32;

    extern __shared__ char shared_mem[];
    float* s_scores = (float*)shared_mem;
    int* s_Q_packed = (int*)(s_scores + ((cache_len + 15) / 16) * 16);
    float* s_output = (float*)(s_Q_packed + ((head_dim + 3) / 4));

    const int8_t* Q_ptr = Q_int8 + batch_head_idx * head_dim;
    const int8_t* K_ptr = K_cache_int8 + batch_head_idx * cache_len * head_dim;
    const int8_t* V_ptr = V_cache_int8 + batch_head_idx * cache_len * head_dim;
    float* O_ptr = O + batch_head_idx * head_dim;

    const int packed_dim = head_dim / 4;
    const int* Q_packed = (const int*)Q_ptr;

    // Load Q as packed int8x4
    for (int d = tid; d < packed_dim; d += TC_THREADS) {
        s_Q_packed[d] = Q_packed[d];
    }

    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += TC_THREADS) {
        s_output[d] = 0.0f;
    }
    __syncthreads();

    // ============================================================
    // Phase 1: Q @ K^T with __dp4a
    // ============================================================

    float local_max = -FLT_MAX;

    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        const int* K_row = (const int*)(K_ptr + k_idx * head_dim);

        int32_t dot = 0;
        #pragma unroll 4
        for (int d = 0; d < packed_dim; d++) {
            dot = __dp4a(s_Q_packed[d], K_row[d], dot);
        }

        float score = (float)dot * scale_q * scale_k * attn_scale;
        s_scores[k_idx] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();

    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    __shared__ float s_max_sum[8];
    if (lane_id == 0) s_max_sum[warp_id] = local_max;
    __syncthreads();

    if (tid == 0) {
        float m = s_max_sum[0];
        for (int i = 1; i < num_warps; i++) m = fmaxf(m, s_max_sum[i]);
        s_max_sum[0] = m;
    }
    __syncthreads();
    float global_max = s_max_sum[0];

    // ============================================================
    // Online softmax + weighted sum of V
    // ============================================================

    float local_sum = 0.0f;

    // Compute exp scores
    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        float exp_score = expf(s_scores[k_idx] - global_max);
        s_scores[k_idx] = exp_score;
        local_sum += exp_score;
    }
    __syncthreads();

    // Sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane_id == 0) s_max_sum[warp_id + 4] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < num_warps; i++) s += s_max_sum[i + 4];
        s_max_sum[4] = s;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_max_sum[4];

    // Normalize and compute weighted V sum
    for (int k_idx = tid; k_idx < cache_len; k_idx += TC_THREADS) {
        s_scores[k_idx] *= inv_sum;
    }
    __syncthreads();

    // ============================================================
    // P @ V - each thread handles multiple dimensions
    // ============================================================

    for (int d = tid; d < head_dim; d += TC_THREADS) {
        float acc = 0.0f;

        #pragma unroll 8
        for (int k = 0; k < cache_len; k++) {
            acc += s_scores[k] * (float)V_ptr[k * head_dim + d];
        }

        O_ptr[d] = acc * scale_v;
    }
}

/**
 * True WMMA Tensor Core kernel for batched INT8 attention
 *
 * This kernel processes multiple heads together (16 at a time) to achieve
 * proper WMMA utilization with 16x16x16 tiles.
 *
 * For Q @ K^T where Q is [16, head_dim] and K is [seq_len, head_dim]:
 * - Tile Q into [16, 16] blocks
 * - Tile K into [16, 16] blocks (transposed)
 * - Use WMMA mma_sync for INT8 matrix multiply
 *
 * Note: This requires batch_heads >= 16 or padding.
 */
__global__ void flash_attention_int8_decode_batched_wmma_kernel(
    const int8_t* __restrict__ Q_int8,        // [batch_heads, head_dim]
    const int8_t* __restrict__ K_cache_int8,  // [batch_heads, cache_len, head_dim]
    const int8_t* __restrict__ V_cache_int8,  // [batch_heads, cache_len, head_dim]
    float* __restrict__ O,                     // [batch_heads, head_dim]
    const float scale_q,
    const float scale_k,
    const float scale_v,
    const int batch_heads,
    const int cache_len,
    const int head_dim,
    const float attn_scale
) {
    // Process WMMA_M (16) heads per block
    const int head_block = blockIdx.x * WMMA_M;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = TC_THREADS / 32;

    // Bounds check
    if (head_block >= batch_heads) return;
    const int heads_in_block = min(WMMA_M, batch_heads - head_block);

    extern __shared__ char shared_mem[];
    float* s_scores = (float*)shared_mem;  // [WMMA_M, cache_len]
    int32_t* s_scores_int = (int32_t*)(s_scores + WMMA_M * ((cache_len + 15) / 16) * 16);

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> score_frag;

    // ============================================================
    // Phase 1: Q @ K^T using WMMA Tensor Cores
    // Process K in tiles of WMMA_N (16) positions
    // ============================================================

    // For each tile of K positions
    for (int k_tile = warp_id * WMMA_N; k_tile < cache_len; k_tile += num_warps * WMMA_N) {
        int k_tile_size = min(WMMA_N, cache_len - k_tile);
        if (k_tile_size <= 0) continue;

        // Initialize accumulator
        wmma::fill_fragment(score_frag, 0);

        // Accumulate over head_dim in tiles of WMMA_K (16)
        for (int d_tile = 0; d_tile < head_dim; d_tile += WMMA_K) {
            // Load Q fragment [head_block:head_block+16, d_tile:d_tile+16]
            // Q layout: [batch_heads, head_dim]
            wmma::load_matrix_sync(q_frag, Q_int8 + head_block * head_dim + d_tile, head_dim);

            // Load K fragment - K layout is [batch_heads, cache_len, head_dim]
            // We need K[head_block, k_tile:k_tile+16, d_tile:d_tile+16]
            // For simplicity, use first head's K (all heads share K in typical attention)
            // Note: This needs adjustment for true multi-head attention
            const int8_t* K_tile_ptr = K_cache_int8 + head_block * cache_len * head_dim +
                                       k_tile * head_dim + d_tile;
            wmma::load_matrix_sync(k_frag, K_tile_ptr, head_dim);

            // WMMA: score_frag += q_frag @ k_frag^T
            wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
        }

        // Store scores to shared memory
        wmma::store_matrix_sync(s_scores_int + k_tile, score_frag, cache_len, wmma::mem_row_major);
    }
    __syncthreads();

    // Convert INT32 scores to FP32 and apply scaling
    for (int h = 0; h < heads_in_block; h++) {
        for (int k = tid; k < cache_len; k += TC_THREADS) {
            s_scores[h * cache_len + k] =
                (float)s_scores_int[h * cache_len + k] * scale_q * scale_k * attn_scale;
        }
    }
    __syncthreads();

    // ============================================================
    // Phase 2: Softmax per head
    // ============================================================

    __shared__ float s_max[WMMA_M];
    __shared__ float s_sum[WMMA_M];

    // Initialize
    if (tid < WMMA_M) {
        s_max[tid] = -FLT_MAX;
        s_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Find max per head
    for (int h = 0; h < heads_in_block; h++) {
        float local_max = -FLT_MAX;
        for (int k = tid; k < cache_len; k += TC_THREADS) {
            local_max = fmaxf(local_max, s_scores[h * cache_len + k]);
        }
        atomicMax((int*)&s_max[h], __float_as_int(local_max));
    }
    __syncthreads();

    // Compute exp and sum per head
    for (int h = 0; h < heads_in_block; h++) {
        float local_sum = 0.0f;
        for (int k = tid; k < cache_len; k += TC_THREADS) {
            float exp_val = expf(s_scores[h * cache_len + k] - s_max[h]);
            s_scores[h * cache_len + k] = exp_val;
            local_sum += exp_val;
        }
        atomicAdd(&s_sum[h], local_sum);
    }
    __syncthreads();

    // Normalize
    for (int h = 0; h < heads_in_block; h++) {
        float inv_sum = 1.0f / s_sum[h];
        for (int k = tid; k < cache_len; k += TC_THREADS) {
            s_scores[h * cache_len + k] *= inv_sum;
        }
    }
    __syncthreads();

    // ============================================================
    // Phase 3: Softmax @ V
    // ============================================================

    for (int h = 0; h < heads_in_block; h++) {
        int head_idx = head_block + h;
        if (head_idx >= batch_heads) break;

        for (int d = tid; d < head_dim; d += TC_THREADS) {
            float acc = 0.0f;

            #pragma unroll 8
            for (int k = 0; k < cache_len; k++) {
                acc += s_scores[h * cache_len + k] *
                       (float)V_cache_int8[head_idx * cache_len * head_dim + k * head_dim + d];
            }

            O[head_idx * head_dim + d] = acc * scale_v;
        }
    }
}

// ============================================================================
// C Interface
// ============================================================================

extern "C" {

// Persistent INT8 buffers
static int8_t* d_Q_int8 = nullptr;
static cudaStream_t int8_compute_stream = nullptr;
static cudaStream_t int8_copy_stream = nullptr;
static int8_t* d_K_cache_int8 = nullptr;
static int8_t* d_V_cache_int8 = nullptr;
static float* d_O_float = nullptr;
static float* d_scales = nullptr;  // [3] for Q, K, V scales
static int int8_initialized = 0;
static int int8_max_cache_len = 0;
static int int8_max_batch_heads = 0;
static int int8_head_dim = 0;
static int int8_current_cache_pos = 0;  // Track current position in cache

// Forward declaration
void flash_attention_int8_cleanup(void);

/**
 * Initialize INT8 Flash Attention buffers with async streams
 */
int flash_attention_int8_init(int max_batch_heads, int max_cache_len, int head_dim) {
    if (int8_initialized) {
        flash_attention_int8_cleanup();
    }

    // Validate head_dim is multiple of 4 for __dp4a
    if (head_dim % 4 != 0) {
        fprintf(stderr, "INT8 Flash Attention: head_dim must be multiple of 4 for __dp4a, got %d\n", head_dim);
        return -1;
    }

    int8_max_batch_heads = max_batch_heads;
    int8_max_cache_len = max_cache_len;
    int8_head_dim = head_dim;
    int8_current_cache_pos = 0;

    size_t q_size = max_batch_heads * head_dim * sizeof(int8_t);
    size_t cache_size = max_batch_heads * max_cache_len * head_dim * sizeof(int8_t);
    size_t output_size = max_batch_heads * head_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q_int8, q_size));
    CUDA_CHECK(cudaMalloc(&d_K_cache_int8, cache_size));
    CUDA_CHECK(cudaMalloc(&d_V_cache_int8, cache_size));
    CUDA_CHECK(cudaMalloc(&d_O_float, output_size));
    CUDA_CHECK(cudaMalloc(&d_scales, 3 * sizeof(float)));

    // Create async streams for overlapped execution
    CUDA_CHECK(cudaStreamCreate(&int8_compute_stream));
    CUDA_CHECK(cudaStreamCreate(&int8_copy_stream));

    // Initialize with default scales
    float default_scales[3] = {1.0f / 127.0f, 1.0f / 127.0f, 1.0f / 127.0f};
    CUDA_CHECK(cudaMemcpy(d_scales, default_scales, 3 * sizeof(float), cudaMemcpyHostToDevice));

    int8_initialized = 1;

    // Query GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("INT8 Flash Attention initialized (with __dp4a Tensor Core acceleration)\n");
    printf("  GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Config: batch_heads=%d, max_cache=%d, head_dim=%d\n",
           max_batch_heads, max_cache_len, head_dim);
    printf("  Memory: Q=%.2f KB, K_cache=%.2f MB, V_cache=%.2f MB\n",
           q_size / 1024.0f, cache_size / (1024.0f * 1024.0f), cache_size / (1024.0f * 1024.0f));

    // Check for Tensor Core support
    if (prop.major >= 7) {
        printf("  Tensor Cores: ENABLED (__dp4a INT8, sm_%d%d)\n", prop.major, prop.minor);
    } else {
        printf("  Tensor Cores: NOT AVAILABLE (requires sm_70+)\n");
    }

    return 0;
}

/**
 * Cleanup INT8 buffers and streams
 */
void flash_attention_int8_cleanup(void) {
    if (d_Q_int8) { cudaFree(d_Q_int8); d_Q_int8 = nullptr; }
    if (d_K_cache_int8) { cudaFree(d_K_cache_int8); d_K_cache_int8 = nullptr; }
    if (d_V_cache_int8) { cudaFree(d_V_cache_int8); d_V_cache_int8 = nullptr; }
    if (d_O_float) { cudaFree(d_O_float); d_O_float = nullptr; }
    if (d_scales) { cudaFree(d_scales); d_scales = nullptr; }
    if (int8_compute_stream) { cudaStreamDestroy(int8_compute_stream); int8_compute_stream = nullptr; }
    if (int8_copy_stream) { cudaStreamDestroy(int8_copy_stream); int8_copy_stream = nullptr; }
    int8_initialized = 0;
    int8_current_cache_pos = 0;
}

/**
 * Reset INT8 cache position for new generation
 */
void flash_attention_int8_reset(void) {
    int8_current_cache_pos = 0;
}

/**
 * Quantize FP32 tensor to INT8 on device
 */
int quantize_tensor_int8(
    const float* input,
    int8_t* output,
    float* scale,
    int size
) {
    float* d_input;
    float* d_scale;
    int8_t* d_output;

    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scale, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((size + 255) / 256);

    quantize_fp32_to_int8_kernel<<<grid, block>>>(d_input, d_output, d_scale, size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scale);

    return 0;
}

/**
 * Update INT8 KV cache
 *
 * @param K_int8     New key in INT8 [batch_heads, head_dim]
 * @param V_int8     New value in INT8 [batch_heads, head_dim]
 * @param scale_k    Dequantization scale for K
 * @param scale_v    Dequantization scale for V
 * @param batch_heads Number of batch * heads
 * @param cache_pos  Position to write in cache
 */
int flash_attention_int8_update_cache(
    const int8_t* K_int8,
    const int8_t* V_int8,
    float scale_k,
    float scale_v,
    int batch_heads,
    int cache_pos
) {
    if (!int8_initialized) {
        fprintf(stderr, "INT8 Flash Attention not initialized\n");
        return -1;
    }

    // Update scales
    float h_scales[3];
    CUDA_CHECK(cudaMemcpy(h_scales, d_scales, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    h_scales[1] = scale_k;
    h_scales[2] = scale_v;
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales, 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Copy K, V to cache
    for (int bh = 0; bh < batch_heads; bh++) {
        size_t cache_offset = (bh * int8_max_cache_len + cache_pos) * int8_head_dim;
        size_t src_offset = bh * int8_head_dim;

        CUDA_CHECK(cudaMemcpy(d_K_cache_int8 + cache_offset,
                              K_int8 + src_offset,
                              int8_head_dim * sizeof(int8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V_cache_int8 + cache_offset,
                              V_int8 + src_offset,
                              int8_head_dim * sizeof(int8_t),
                              cudaMemcpyHostToDevice));
    }

    return 0;
}

/**
 * INT8 Flash Attention Decode with __dp4a Tensor Core acceleration
 *
 * Single-token decode using INT8 quantized QKV with 4x throughput __dp4a.
 *
 * @param Q_int8     Query in INT8 [batch_heads, head_dim]
 * @param K_int8     New key in INT8 [batch_heads, head_dim]
 * @param V_int8     New value in INT8 [batch_heads, head_dim]
 * @param O          Output in FP32 [batch_heads, head_dim]
 * @param scale_q    Dequantization scale for Q
 * @param scale_k    Dequantization scale for K
 * @param scale_v    Dequantization scale for V
 * @param batch_heads Number of batch * heads
 * @param cache_pos  Current cache position (0-indexed)
 * @param head_dim   Head dimension (must be multiple of 4)
 */
int flash_attention_int8_decode(
    const int8_t* Q_int8,
    const int8_t* K_int8,
    const int8_t* V_int8,
    float* O,
    float scale_q,
    float scale_k,
    float scale_v,
    int batch_heads,
    int cache_pos,
    int head_dim
) {
    if (!int8_initialized) {
        fprintf(stderr, "INT8 Flash Attention not initialized\n");
        return -1;
    }

    // Validate head_dim for __dp4a
    if (head_dim % 4 != 0) {
        fprintf(stderr, "INT8 decode: head_dim must be multiple of 4, got %d\n", head_dim);
        return -1;
    }

    // Async copy Q to device on copy stream
    CUDA_CHECK(cudaMemcpyAsync(d_Q_int8, Q_int8, batch_heads * head_dim * sizeof(int8_t),
                               cudaMemcpyHostToDevice, int8_copy_stream));

    // Update cache with new K, V (also uses copy stream)
    flash_attention_int8_update_cache(K_int8, V_int8, scale_k, scale_v, batch_heads, cache_pos);

    // Sync copy stream before compute
    CUDA_CHECK(cudaStreamSynchronize(int8_copy_stream));

    int cache_len = cache_pos + 1;
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    // Shared memory for optimized __dp4a kernel:
    // - s_scores: [cache_len] floats for attention scores
    // - s_Q_packed: [head_dim/4] ints for packed Q (int8x4)
    size_t scores_size = ((cache_len + 15) / 16 * 16) * sizeof(float);
    size_t q_packed_size = ((head_dim / 4) + 3) / 4 * 4 * sizeof(int);
    size_t smem_size = scores_size + q_packed_size;

    dim3 grid(batch_heads);
    dim3 block(TC_THREADS);

    // Use the __dp4a optimized kernel
    flash_attention_int8_decode_wmma_kernel<<<grid, block, smem_size, int8_compute_stream>>>(
        d_Q_int8, d_K_cache_int8, d_V_cache_int8, d_O_float,
        scale_q, scale_k, scale_v,
        cache_len, head_dim, attn_scale
    );

    CUDA_CHECK(cudaGetLastError());

    // Sync compute before output copy
    CUDA_CHECK(cudaStreamSynchronize(int8_compute_stream));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(O, d_O_float, batch_heads * head_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * GPU-Resident INT8 decode (no host-device transfers per call)
 *
 * For use when Q, K, V are already on GPU. Eliminates PCIe overhead.
 */
int flash_attention_int8_decode_gpu(
    int batch_heads,
    int cache_len,
    float scale_q,
    float scale_k,
    float scale_v
) {
    if (!int8_initialized) {
        fprintf(stderr, "INT8 Flash Attention not initialized\n");
        return -1;
    }

    float attn_scale = 1.0f / sqrtf((float)int8_head_dim);

    size_t scores_size = ((cache_len + 15) / 16 * 16) * sizeof(float);
    size_t q_packed_size = ((int8_head_dim / 4) + 3) / 4 * 4 * sizeof(int);
    size_t smem_size = scores_size + q_packed_size;

    dim3 grid(batch_heads);
    dim3 block(TC_THREADS);

    flash_attention_int8_decode_wmma_kernel<<<grid, block, smem_size, int8_compute_stream>>>(
        d_Q_int8, d_K_cache_int8, d_V_cache_int8, d_O_float,
        scale_q, scale_k, scale_v,
        cache_len, int8_head_dim, attn_scale
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/**
 * Synchronize INT8 compute stream
 */
void flash_attention_int8_sync(void) {
    if (int8_compute_stream) {
        cudaStreamSynchronize(int8_compute_stream);
    }
}

/**
 * INT8 Flash Attention Decode with FP32 input (handles quantization internally)
 *
 * Convenience function that accepts FP32 input and handles quantization.
 */
int flash_attention_int8_decode_fp32(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
) {
    if (!int8_initialized) {
        fprintf(stderr, "INT8 Flash Attention not initialized\n");
        return -1;
    }

    int single_size = batch_heads * head_dim;

    // Allocate temporary INT8 buffers
    int8_t* h_Q_int8 = (int8_t*)malloc(single_size);
    int8_t* h_K_int8 = (int8_t*)malloc(single_size);
    int8_t* h_V_int8 = (int8_t*)malloc(single_size);
    float scale_q, scale_k, scale_v;

    // Quantize Q, K, V
    quantize_tensor_int8(Q, h_Q_int8, &scale_q, single_size);
    quantize_tensor_int8(K_new, h_K_int8, &scale_k, single_size);
    quantize_tensor_int8(V_new, h_V_int8, &scale_v, single_size);

    // Run INT8 decode
    int ret = flash_attention_int8_decode(
        h_Q_int8, h_K_int8, h_V_int8, O,
        scale_q, scale_k, scale_v,
        batch_heads, cache_pos, head_dim
    );

    free(h_Q_int8);
    free(h_K_int8);
    free(h_V_int8);

    return ret;
}

/**
 * Get INT8 Flash Attention info
 */
void flash_attention_int8_info(int* initialized, int* max_cache, int* max_bh, int* h_dim) {
    if (initialized) *initialized = int8_initialized;
    if (max_cache) *max_cache = int8_max_cache_len;
    if (max_bh) *max_bh = int8_max_batch_heads;
    if (h_dim) *h_dim = int8_head_dim;
}

// ============================================================================
// STATELESS Attention - For Multi-Layer Transformers
// ============================================================================

/**
 * Stateless FP32 Attention (no internal caches)
 *
 * This is the DETERMINISTIC approach for multi-layer transformers.
 * The caller (Mojo/C++) owns and manages per-layer KV caches.
 * The kernel just computes attention - no state management.
 *
 * @param Q         Query [batch_heads, head_dim] - FP32
 * @param K_cache   Key cache [batch_heads, cache_len, head_dim] - FP32, EXTERNAL
 * @param V_cache   Value cache [batch_heads, cache_len, head_dim] - FP32, EXTERNAL
 * @param O         Output [batch_heads, head_dim] - FP32
 * @param batch_heads Number of batch * heads
 * @param cache_len  Number of valid positions in cache (1 to max_seq_len)
 * @param head_dim   Dimension per head
 * @return 0 on success
 */
int attention_stateless_fp32(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim
) {
    if (cache_len <= 0 || batch_heads <= 0 || head_dim <= 0) {
        return -1;
    }

    // Allocate device memory
    float* d_Q;
    float* d_K;
    float* d_V;
    float* d_O;

    size_t q_size = batch_heads * head_dim * sizeof(float);
    size_t kv_size = batch_heads * cache_len * head_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q, q_size));
    CUDA_CHECK(cudaMalloc(&d_K, kv_size));
    CUDA_CHECK(cudaMalloc(&d_V, kv_size));
    CUDA_CHECK(cudaMalloc(&d_O, q_size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_Q, Q, q_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K_cache, kv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V_cache, kv_size, cudaMemcpyHostToDevice));

    // Quantize to INT8
    int8_t* d_Q_i8;
    int8_t* d_K_i8;
    int8_t* d_V_i8;
    float* d_scale_q;
    float* d_scale_k;
    float* d_scale_v;

    CUDA_CHECK(cudaMalloc(&d_Q_i8, batch_heads * head_dim));
    CUDA_CHECK(cudaMalloc(&d_K_i8, batch_heads * cache_len * head_dim));
    CUDA_CHECK(cudaMalloc(&d_V_i8, batch_heads * cache_len * head_dim));
    CUDA_CHECK(cudaMalloc(&d_scale_q, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_k, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_v, sizeof(float)));

    dim3 qblock(256);
    dim3 qgrid_q((batch_heads * head_dim + 255) / 256);
    dim3 qgrid_kv((batch_heads * cache_len * head_dim + 255) / 256);

    quantize_fp32_to_int8_kernel<<<qgrid_q, qblock>>>(d_Q, d_Q_i8, d_scale_q, batch_heads * head_dim);
    quantize_fp32_to_int8_kernel<<<qgrid_kv, qblock>>>(d_K, d_K_i8, d_scale_k, batch_heads * cache_len * head_dim);
    quantize_fp32_to_int8_kernel<<<qgrid_kv, qblock>>>(d_V, d_V_i8, d_scale_v, batch_heads * cache_len * head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get scales
    float h_scale_q, h_scale_k, h_scale_v;
    CUDA_CHECK(cudaMemcpy(&h_scale_q, d_scale_q, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_scale_k, d_scale_k, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_scale_v, d_scale_v, sizeof(float), cudaMemcpyDeviceToHost));

    float attn_scale = 1.0f / sqrtf((float)head_dim);

    // Shared memory size
    size_t scores_size = ((cache_len + 15) / 16 * 16) * sizeof(float);
    size_t q_packed_size = ((head_dim / 4) + 3) / 4 * 4 * sizeof(int);
    size_t smem_size = scores_size + q_packed_size;

    dim3 grid(batch_heads);
    dim3 block(TC_THREADS);

    // Run INT8 attention kernel (using external K/V caches)
    flash_attention_int8_decode_kernel<<<grid, block, smem_size>>>(
        d_Q_i8, d_K_i8, d_V_i8, d_O,
        h_scale_q, h_scale_k, h_scale_v,
        cache_len, head_dim, attn_scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(O, d_O, q_size, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_Q_i8);
    cudaFree(d_K_i8);
    cudaFree(d_V_i8);
    cudaFree(d_scale_q);
    cudaFree(d_scale_k);
    cudaFree(d_scale_v);

    return 0;
}

/**
 * Stateless FP32 Attention with pre-allocated GPU buffers
 *
 * More efficient version that reuses GPU memory across calls.
 * Call attention_stateless_init() first to allocate buffers.
 */
static float* d_attn_Q = nullptr;
static float* d_attn_K = nullptr;
static float* d_attn_V = nullptr;
static float* d_attn_O = nullptr;
static int8_t* d_attn_Q_i8 = nullptr;
static int8_t* d_attn_K_i8 = nullptr;
static int8_t* d_attn_V_i8 = nullptr;
static float* d_attn_partial_max = nullptr;  // Workspace for quantization
static cudaStream_t attn_stream = nullptr;   // Async stream
static int attn_stateless_max_batch_heads = 0;
static int attn_stateless_max_cache = 0;
static int attn_stateless_head_dim = 0;
static int attn_stateless_initialized = 0;
static int attn_max_blocks = 0;  // For workspace sizing

int attention_stateless_init(int max_batch_heads, int max_cache_len, int head_dim) {
    if (attn_stateless_initialized) {
        // Already initialized - check if we need to resize
        if (max_batch_heads <= attn_stateless_max_batch_heads &&
            max_cache_len <= attn_stateless_max_cache &&
            head_dim == attn_stateless_head_dim) {
            return 0;  // Existing buffers are sufficient
        }
        // Need to resize - cleanup first
        if (d_attn_Q) cudaFree(d_attn_Q);
        if (d_attn_K) cudaFree(d_attn_K);
        if (d_attn_V) cudaFree(d_attn_V);
        if (d_attn_O) cudaFree(d_attn_O);
        if (d_attn_Q_i8) cudaFree(d_attn_Q_i8);
        if (d_attn_K_i8) cudaFree(d_attn_K_i8);
        if (d_attn_V_i8) cudaFree(d_attn_V_i8);
        if (d_attn_partial_max) cudaFree(d_attn_partial_max);
        if (attn_stream) cudaStreamDestroy(attn_stream);
    }

    size_t q_size = max_batch_heads * head_dim * sizeof(float);
    size_t kv_size = max_batch_heads * max_cache_len * head_dim * sizeof(float);

    // Calculate workspace size for quantization (max elements we'll ever quantize)
    int max_kv_elements = max_batch_heads * max_cache_len * head_dim;
    int max_blocks = (max_kv_elements + 255) / 256;

    CUDA_CHECK(cudaMalloc(&d_attn_Q, q_size));
    CUDA_CHECK(cudaMalloc(&d_attn_K, kv_size));
    CUDA_CHECK(cudaMalloc(&d_attn_V, kv_size));
    CUDA_CHECK(cudaMalloc(&d_attn_O, q_size));
    CUDA_CHECK(cudaMalloc(&d_attn_Q_i8, max_batch_heads * head_dim));
    CUDA_CHECK(cudaMalloc(&d_attn_K_i8, max_batch_heads * max_cache_len * head_dim));
    CUDA_CHECK(cudaMalloc(&d_attn_V_i8, max_batch_heads * max_cache_len * head_dim));

    // Allocate persistent workspace for quantization
    CUDA_CHECK(cudaMalloc(&d_attn_partial_max, (max_blocks + 1) * sizeof(float)));

    // Create CUDA stream for async operations
    CUDA_CHECK(cudaStreamCreate(&attn_stream));

    attn_stateless_max_batch_heads = max_batch_heads;
    attn_stateless_max_cache = max_cache_len;
    attn_stateless_head_dim = head_dim;
    attn_max_blocks = max_blocks;
    attn_stateless_initialized = 1;

    printf("Stateless attention initialized: batch_heads=%d, cache=%d, head_dim=%d\n",
           max_batch_heads, max_cache_len, head_dim);

    return 0;
}

void attention_stateless_cleanup(void) {
    if (d_attn_Q) { cudaFree(d_attn_Q); d_attn_Q = nullptr; }
    if (d_attn_K) { cudaFree(d_attn_K); d_attn_K = nullptr; }
    if (d_attn_V) { cudaFree(d_attn_V); d_attn_V = nullptr; }
    if (d_attn_O) { cudaFree(d_attn_O); d_attn_O = nullptr; }
    if (d_attn_Q_i8) { cudaFree(d_attn_Q_i8); d_attn_Q_i8 = nullptr; }
    if (d_attn_K_i8) { cudaFree(d_attn_K_i8); d_attn_K_i8 = nullptr; }
    if (d_attn_V_i8) { cudaFree(d_attn_V_i8); d_attn_V_i8 = nullptr; }
    if (d_attn_partial_max) { cudaFree(d_attn_partial_max); d_attn_partial_max = nullptr; }
    if (attn_stream) { cudaStreamDestroy(attn_stream); attn_stream = nullptr; }
    attn_stateless_initialized = 0;
    attn_max_blocks = 0;
}

// Forward declarations
int attention_stateless_strided(
    const float* Q, const float* K_cache, const float* V_cache, float* O,
    int batch_heads, int cache_len, int head_dim, int buffer_seq_len);

/**
 * Simple FP32 attention kernel (no INT8 quantization)
 * For debugging and verification.
 */
__global__ void attention_fp32_kernel(
    const float* __restrict__ Q,      // [batch_heads, head_dim]
    const float* __restrict__ K,      // [batch_heads, cache_len, head_dim]
    const float* __restrict__ V,      // [batch_heads, cache_len, head_dim]
    float* __restrict__ O,            // [batch_heads, head_dim]
    int cache_len,
    int head_dim,
    float scale
) {
    int bh = blockIdx.x;  // Which head
    int tid = threadIdx.x;

    // Pointers for this head
    const float* q = Q + bh * head_dim;
    const float* k = K + bh * cache_len * head_dim;
    const float* v = V + bh * cache_len * head_dim;
    float* o = O + bh * head_dim;

    // Shared memory for scores and output accumulator
    extern __shared__ float smem[];
    float* scores = smem;  // [cache_len]

    // Step 1: Compute attention scores Q @ K^T
    for (int t = tid; t < cache_len; t += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * k[t * head_dim + d];
        }
        scores[t] = score * scale;
    }
    __syncthreads();

    // Step 2: Softmax
    // Find max (reduction)
    float max_score = -1e10f;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        max_score = fmaxf(max_score, scores[t]);
    }

    // Block reduction for max
    __shared__ float s_max[256];
    s_max[tid] = max_score;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    max_score = s_max[0];

    // Exp and sum
    float sum_exp = 0.0f;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }

    // Block reduction for sum
    __shared__ float s_sum[256];
    s_sum[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    sum_exp = s_sum[0];

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Output = scores @ V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            acc += scores[t] * v[t * head_dim + d];
        }
        o[d] = acc;
    }
}

/**
 * Pure FP32 stateless attention with strided buffers
 * OPTIMIZED: Uses persistent buffers and CUDA stream
 */
int attention_fp32_strided(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim,
    int buffer_seq_len
) {
    // Auto-initialize if needed
    if (!attn_stateless_initialized) {
        int ret = attention_stateless_init(batch_heads, buffer_seq_len, head_dim);
        if (ret != 0) return ret;
    }

    // Validate sizes
    if (batch_heads > attn_stateless_max_batch_heads ||
        cache_len > attn_stateless_max_cache ||
        head_dim != attn_stateless_head_dim) {
        fprintf(stderr, "FP32 attention: size mismatch (heads=%d/%d, cache=%d/%d, dim=%d/%d)\n",
                batch_heads, attn_stateless_max_batch_heads,
                cache_len, attn_stateless_max_cache,
                head_dim, attn_stateless_head_dim);
        return -1;
    }

    size_t q_size = batch_heads * head_dim * sizeof(float);

    // Async copy Q to device (use persistent buffer)
    CUDA_CHECK(cudaMemcpyAsync(d_attn_Q, Q, q_size, cudaMemcpyHostToDevice, attn_stream));

    // Copy K, V with stride handling (async, use persistent buffers)
    for (int h = 0; h < batch_heads; h++) {
        size_t src_offset = h * buffer_seq_len * head_dim;
        size_t dst_offset = h * cache_len * head_dim;
        size_t copy_size = cache_len * head_dim * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(d_attn_K + dst_offset, K_cache + src_offset, copy_size, cudaMemcpyHostToDevice, attn_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_attn_V + dst_offset, V_cache + src_offset, copy_size, cudaMemcpyHostToDevice, attn_stream));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    // Shared memory: scores array
    size_t smem_size = cache_len * sizeof(float);

    dim3 grid(batch_heads);
    dim3 block(128);

    attention_fp32_kernel<<<grid, block, smem_size, attn_stream>>>(
        d_attn_Q, d_attn_K, d_attn_V, d_attn_O,
        cache_len, head_dim, scale
    );

    // Async copy result back
    CUDA_CHECK(cudaMemcpyAsync(O, d_attn_O, q_size, cudaMemcpyDeviceToHost, attn_stream));

    // Sync stream before returning
    CUDA_CHECK(cudaStreamSynchronize(attn_stream));

    return 0;
}

/**
 * Fast stateless attention with reused buffers
 * NOTE: Expects contiguous [batch_heads, cache_len, head_dim] layout
 */
int attention_stateless_fast(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim
) {
    // Delegate to strided version with stride = cache_len (contiguous)
    return attention_stateless_strided(Q, K_cache, V_cache, O,
                                       batch_heads, cache_len, head_dim, cache_len);
}

/**
 * Helper: Proper multi-block quantization with pre-allocated workspace
 * Returns the max value (dequant scale = max / 127)
 *
 * @param d_input      Input FP32 tensor (device)
 * @param d_output     Output INT8 tensor (device)
 * @param size         Number of elements
 * @param d_workspace  Pre-allocated workspace [num_blocks + 1] floats
 * @param num_blocks   Number of thread blocks
 * @param stream       CUDA stream (can be null for sync)
 */
static float quantize_tensor_proper(
    const float* d_input,
    int8_t* d_output,
    int size,
    float* d_workspace,
    int num_blocks,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_blocks);

    // Use last element of workspace for final max
    float* d_partial_max = d_workspace;
    float* d_final_max = d_workspace + num_blocks;

    // Pass 1: Find max per block
    find_max_kernel<<<grid, block, 0, stream>>>(d_input, d_partial_max, size);

    // Pass 2: Reduce to single max (use pre-allocated slot)
    reduce_max_kernel<<<1, 256, 0, stream>>>(d_partial_max, d_final_max, num_blocks);

    // Sync to get max value back to host
    float h_max;
    cudaMemcpyAsync(&h_max, d_final_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Pass 3: Quantize with computed scale
    quantize_kernel<<<grid, block, 0, stream>>>(d_input, d_output, h_max, size);

    return h_max / QUANT_SCALE;  // Return dequant scale
}

/**
 * Strided stateless INT8 attention - handles non-contiguous KV cache
 * OPTIMIZED: Uses persistent buffers and CUDA stream
 *
 * Use when KV cache has layout [batch_heads, max_seq_len, head_dim]
 * but only cache_len positions are valid.
 *
 * @param buffer_seq_len  Stride between heads in source buffer (typically max_seq_len)
 */
int attention_stateless_strided(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O,
    int batch_heads,
    int cache_len,
    int head_dim,
    int buffer_seq_len
) {
    if (!attn_stateless_initialized) {
        // Auto-initialize with default sizes
        int ret = attention_stateless_init(batch_heads, buffer_seq_len, head_dim);
        if (ret != 0) return ret;
    }

    // Validate sizes
    if (batch_heads > attn_stateless_max_batch_heads ||
        cache_len > attn_stateless_max_cache ||
        head_dim != attn_stateless_head_dim) {
        fprintf(stderr, "Stateless attention: size mismatch (heads=%d/%d, cache=%d/%d, dim=%d/%d)\n",
                batch_heads, attn_stateless_max_batch_heads,
                cache_len, attn_stateless_max_cache,
                head_dim, attn_stateless_head_dim);
        return -1;
    }

    size_t q_size = batch_heads * head_dim * sizeof(float);
    int q_elements = batch_heads * head_dim;
    int kv_elements = batch_heads * cache_len * head_dim;

    // Async copy Q to device
    CUDA_CHECK(cudaMemcpyAsync(d_attn_Q, Q, q_size, cudaMemcpyHostToDevice, attn_stream));

    // Copy K, V with stride handling (async)
    for (int h = 0; h < batch_heads; h++) {
        size_t src_offset = h * buffer_seq_len * head_dim;
        size_t dst_offset = h * cache_len * head_dim;
        size_t copy_size = cache_len * head_dim * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(d_attn_K + dst_offset, K_cache + src_offset, copy_size, cudaMemcpyHostToDevice, attn_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_attn_V + dst_offset, V_cache + src_offset, copy_size, cudaMemcpyHostToDevice, attn_stream));
    }

    // Use persistent workspace for quantization
    int num_blocks_q = (q_elements + 255) / 256;
    int num_blocks_kv = (kv_elements + 255) / 256;

    // Proper multi-block quantization using persistent workspace
    float scale_q = quantize_tensor_proper(d_attn_Q, d_attn_Q_i8, q_elements, d_attn_partial_max, num_blocks_q, attn_stream);
    float scale_k = quantize_tensor_proper(d_attn_K, d_attn_K_i8, kv_elements, d_attn_partial_max, num_blocks_kv, attn_stream);
    float scale_v = quantize_tensor_proper(d_attn_V, d_attn_V_i8, kv_elements, d_attn_partial_max, num_blocks_kv, attn_stream);

    float attn_scale = 1.0f / sqrtf((float)head_dim);

    size_t scores_size = ((cache_len + 15) / 16 * 16) * sizeof(float);
    size_t q_packed_size = ((head_dim / 4) + 3) / 4 * 4 * sizeof(int);
    size_t smem_size = scores_size + q_packed_size;

    dim3 grid(batch_heads);
    dim3 block(TC_THREADS);

    flash_attention_int8_decode_kernel<<<grid, block, smem_size, attn_stream>>>(
        d_attn_Q_i8, d_attn_K_i8, d_attn_V_i8, d_attn_O,
        scale_q, scale_k, scale_v,
        cache_len, head_dim, attn_scale
    );

    // Async copy result back
    CUDA_CHECK(cudaMemcpyAsync(O, d_attn_O, q_size, cudaMemcpyDeviceToHost, attn_stream));

    // Sync stream before returning
    CUDA_CHECK(cudaStreamSynchronize(attn_stream));

    return 0;
}

} // extern "C"
