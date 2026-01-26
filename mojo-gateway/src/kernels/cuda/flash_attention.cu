/**
 * Flash Attention CUDA Implementation
 *
 * Based on "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * by Tri Dao et al. (https://arxiv.org/abs/2205.14135)
 *
 * Key optimizations:
 * 1. Tiled computation - Q, K, V processed in blocks that fit in shared memory
 * 2. Online softmax - Incremental softmax without materializing N×N attention matrix
 * 3. Memory efficient - O(N) memory instead of O(N²)
 * 4. IO-aware - Minimizes HBM reads/writes
 *
 * For EdgeLLM BitNet inference targeting 630+ tok/s
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32

// Block sizes for Flash Attention
// These are tuned for T4/RTX GPUs with 64KB shared memory
#define FA_BLOCK_M 64      // Rows of Q per block
#define FA_BLOCK_N 64      // Columns of K per block (sequence length tiles)
#define FA_BLOCK_K 64      // Head dimension (typically 64 or 128)

// Thread block configuration: 128 threads = 4 warps
#define FA_THREADS 128
#define FA_WARPS (FA_THREADS / WARP_SIZE)

// For smaller head dimensions (SmolLM uses 64)
#define FA_BLOCK_M_SMALL 32
#define FA_BLOCK_N_SMALL 32

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Online Softmax Helper
// ============================================================================

/**
 * Online softmax state for incremental computation
 * Maintains running max and sum for numerical stability
 */
struct OnlineSoftmax {
    float max_val;
    float sum_exp;

    __device__ __forceinline__ void init() {
        max_val = -FLT_MAX;
        sum_exp = 0.0f;
    }

    __device__ __forceinline__ void update(float new_max, float new_sum) {
        // Combine two softmax states using the log-sum-exp trick
        if (new_max > max_val) {
            sum_exp = sum_exp * expf(max_val - new_max) + new_sum;
            max_val = new_max;
        } else {
            sum_exp = sum_exp + new_sum * expf(new_max - max_val);
        }
    }

    __device__ __forceinline__ float finalize(float val, float local_max) {
        // Convert local softmax value to global
        return val * expf(local_max - max_val) / sum_exp;
    }
};

// ============================================================================
// Flash Attention Forward Kernel
// ============================================================================

/**
 * Flash Attention Forward Pass - Single Head
 *
 * Computes: output = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Algorithm (from FlashAttention paper):
 * 1. Load Q block to shared memory
 * 2. For each K,V block:
 *    a. Load K,V to shared memory
 *    b. Compute S = Q @ K^T / sqrt(d_k) in registers
 *    c. Update running softmax (m, l) using online softmax
 *    d. Compute output contribution O += softmax(S) @ V
 * 3. Rescale output by final softmax denominator
 *
 * @param Q         Query tensor [batch, num_heads, seq_len, head_dim]
 * @param K         Key tensor [batch, num_heads, seq_len, head_dim]
 * @param V         Value tensor [batch, num_heads, seq_len, head_dim]
 * @param O         Output tensor [batch, num_heads, seq_len, head_dim]
 * @param seq_len   Sequence length
 * @param head_dim  Head dimension (d_k)
 * @param scale     1.0 / sqrt(head_dim)
 * @param causal    Whether to apply causal masking
 */
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int causal
) {
    // Shared memory for Q, K, V tiles
    extern __shared__ float smem[];

    // Partition shared memory
    float* s_Q = smem;                                    // [FA_BLOCK_M, head_dim]
    float* s_K = smem + FA_BLOCK_M * head_dim;           // [FA_BLOCK_N, head_dim]
    float* s_V = smem + FA_BLOCK_M * head_dim + FA_BLOCK_N * head_dim;  // [FA_BLOCK_N, head_dim]
    float* s_S = smem + FA_BLOCK_M * head_dim + 2 * FA_BLOCK_N * head_dim;  // [FA_BLOCK_M, FA_BLOCK_N]

    // Block indices
    const int batch_head_idx = blockIdx.x;  // Combined batch and head index
    const int q_block_idx = blockIdx.y;     // Which Q block (row block)

    // Thread indices
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Calculate offsets for this batch/head
    const int qkv_offset = batch_head_idx * seq_len * head_dim;

    // Q block start row
    const int q_start = q_block_idx * FA_BLOCK_M;
    if (q_start >= seq_len) return;

    // Calculate actual block height (handle edge cases)
    const int q_end = min(q_start + FA_BLOCK_M, seq_len);
    const int block_m = q_end - q_start;

    // Initialize output accumulators and online softmax state in registers
    float o_acc[FA_BLOCK_M / FA_WARPS][FA_BLOCK_K / WARP_SIZE];
    OnlineSoftmax softmax_state[FA_BLOCK_M / FA_WARPS];

    // Each warp handles FA_BLOCK_M / FA_WARPS rows
    const int rows_per_warp = FA_BLOCK_M / FA_WARPS;
    const int my_row_start = warp_id * rows_per_warp;

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < rows_per_warp; i++) {
        softmax_state[i].init();
        #pragma unroll
        for (int j = 0; j < FA_BLOCK_K / WARP_SIZE; j++) {
            o_acc[i][j] = 0.0f;
        }
    }

    // ========== Load Q block to shared memory ==========
    // Collaborative loading: each thread loads multiple elements
    for (int i = tid; i < block_m * head_dim; i += FA_THREADS) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (q_start + row < seq_len) {
            s_Q[row * head_dim + col] = Q[qkv_offset + (q_start + row) * head_dim + col];
        } else {
            s_Q[row * head_dim + col] = 0.0f;
        }
    }
    __syncthreads();

    // ========== Iterate over K,V blocks ==========
    const int num_kv_blocks = (seq_len + FA_BLOCK_N - 1) / FA_BLOCK_N;

    // For causal attention, only process K,V blocks up to the diagonal
    const int max_kv_block = causal ?
        min(num_kv_blocks, (q_start + FA_BLOCK_M + FA_BLOCK_N - 1) / FA_BLOCK_N) :
        num_kv_blocks;

    for (int kv_block = 0; kv_block < max_kv_block; kv_block++) {
        const int k_start = kv_block * FA_BLOCK_N;
        const int k_end = min(k_start + FA_BLOCK_N, seq_len);
        const int block_n = k_end - k_start;

        // ========== Load K, V blocks to shared memory ==========
        for (int i = tid; i < block_n * head_dim; i += FA_THREADS) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (k_start + row < seq_len) {
                s_K[row * head_dim + col] = K[qkv_offset + (k_start + row) * head_dim + col];
                s_V[row * head_dim + col] = V[qkv_offset + (k_start + row) * head_dim + col];
            } else {
                s_K[row * head_dim + col] = 0.0f;
                s_V[row * head_dim + col] = 0.0f;
            }
        }
        __syncthreads();

        // ========== Compute S = Q @ K^T / sqrt(d_k) ==========
        // Each warp computes its assigned rows
        for (int i = 0; i < rows_per_warp; i++) {
            int q_row = my_row_start + i;
            if (q_row >= block_m) continue;

            float row_max = -FLT_MAX;
            float row_sum = 0.0f;

            // Compute attention scores for this Q row with all K columns
            for (int j = lane_id; j < block_n; j += WARP_SIZE) {
                float score = 0.0f;

                // Dot product Q[q_row] @ K[j]
                for (int d = 0; d < head_dim; d++) {
                    score += s_Q[q_row * head_dim + d] * s_K[j * head_dim + d];
                }
                score *= scale;

                // Apply causal mask
                if (causal && (q_start + q_row) < (k_start + j)) {
                    score = -FLT_MAX;
                }

                // Store in shared memory for later use
                s_S[q_row * FA_BLOCK_N + j] = score;

                // Track max for softmax
                row_max = fmaxf(row_max, score);
            }

            // Warp-level reduction to find max
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
            }
            row_max = __shfl_sync(0xffffffff, row_max, 0);

            // Compute exp(score - max) and sum
            for (int j = lane_id; j < block_n; j += WARP_SIZE) {
                float score = s_S[q_row * FA_BLOCK_N + j];
                float exp_score = (score > -FLT_MAX + 1.0f) ? expf(score - row_max) : 0.0f;
                s_S[q_row * FA_BLOCK_N + j] = exp_score;
                row_sum += exp_score;
            }

            // Warp-level reduction for sum
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
            }
            row_sum = __shfl_sync(0xffffffff, row_sum, 0);

            // Update online softmax state
            softmax_state[i].update(row_max, row_sum);
        }
        __syncthreads();

        // ========== Compute O += softmax(S) @ V ==========
        for (int i = 0; i < rows_per_warp; i++) {
            int q_row = my_row_start + i;
            if (q_row >= block_m) continue;

            // Accumulate output for each head dimension
            for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
                float acc = 0.0f;

                // Sum over K dimension: acc = sum_j(softmax[j] * V[j, d])
                for (int j = 0; j < block_n; j++) {
                    acc += s_S[q_row * FA_BLOCK_N + j] * s_V[j * head_dim + d];
                }

                // Store in register accumulator
                o_acc[i][d / WARP_SIZE] += acc;
            }
        }
        __syncthreads();
    }

    // ========== Write output with final softmax rescaling ==========
    for (int i = 0; i < rows_per_warp; i++) {
        int q_row = my_row_start + i;
        int global_row = q_start + q_row;

        if (global_row < seq_len && q_row < block_m) {
            for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
                // Rescale by softmax denominator
                float val = o_acc[i][d / WARP_SIZE];
                if (softmax_state[i].sum_exp > 0.0f) {
                    val /= softmax_state[i].sum_exp;
                }
                O[qkv_offset + global_row * head_dim + d] = val;
            }
        }
    }
}

/**
 * Flash Attention Forward - Optimized for small head dimensions (d=64)
 * Uses smaller tile sizes for better occupancy
 */
__global__ void flash_attention_forward_small_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int causal
) {
    // Shared memory - smaller tiles for head_dim <= 64
    __shared__ float s_Q[FA_BLOCK_M_SMALL * 64];     // 32 * 64 = 2KB
    __shared__ float s_K[FA_BLOCK_N_SMALL * 64];     // 32 * 64 = 2KB
    __shared__ float s_V[FA_BLOCK_N_SMALL * 64];     // 32 * 64 = 2KB
    __shared__ float s_S[FA_BLOCK_M_SMALL * FA_BLOCK_N_SMALL];  // 32 * 32 = 1KB

    const int batch_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int qkv_offset = batch_head_idx * seq_len * head_dim;
    const int q_start = q_block_idx * FA_BLOCK_M_SMALL;

    if (q_start >= seq_len) return;

    const int block_m = min(FA_BLOCK_M_SMALL, seq_len - q_start);

    // Output accumulators - each thread handles subset of output
    float o_local[64];  // Max head_dim = 64
    float m_local = -FLT_MAX;  // Running max
    float l_local = 0.0f;      // Running sum

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        o_local[i] = 0.0f;
    }

    // Each thread processes one row of Q
    const int my_row = tid;
    const int global_q_row = q_start + my_row;

    // Load my Q row
    if (my_row < block_m && global_q_row < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            s_Q[my_row * head_dim + d] = Q[qkv_offset + global_q_row * head_dim + d];
        }
    }
    __syncthreads();

    // Iterate over K,V blocks
    const int num_kv_blocks = (seq_len + FA_BLOCK_N_SMALL - 1) / FA_BLOCK_N_SMALL;
    const int max_kv_block = causal ?
        min(num_kv_blocks, (q_start + FA_BLOCK_M_SMALL + FA_BLOCK_N_SMALL - 1) / FA_BLOCK_N_SMALL) :
        num_kv_blocks;

    for (int kv_block = 0; kv_block < max_kv_block; kv_block++) {
        const int k_start = kv_block * FA_BLOCK_N_SMALL;
        const int block_n = min(FA_BLOCK_N_SMALL, seq_len - k_start);

        // Collaborative load K, V
        for (int i = tid; i < block_n * head_dim; i += FA_THREADS) {
            int row = i / head_dim;
            int col = i % head_dim;
            s_K[row * head_dim + col] = K[qkv_offset + (k_start + row) * head_dim + col];
            s_V[row * head_dim + col] = V[qkv_offset + (k_start + row) * head_dim + col];
        }
        __syncthreads();

        if (my_row < block_m && global_q_row < seq_len) {
            // Compute attention scores for my row
            float row_max = -FLT_MAX;
            float scores[FA_BLOCK_N_SMALL];

            for (int j = 0; j < block_n; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += s_Q[my_row * head_dim + d] * s_K[j * head_dim + d];
                }
                score *= scale;

                // Causal mask
                if (causal && global_q_row < (k_start + j)) {
                    score = -FLT_MAX;
                }

                scores[j] = score;
                row_max = fmaxf(row_max, score);
            }

            // Online softmax update
            float row_sum = 0.0f;
            for (int j = 0; j < block_n; j++) {
                scores[j] = (scores[j] > -FLT_MAX + 1.0f) ? expf(scores[j] - row_max) : 0.0f;
                row_sum += scores[j];
            }

            // Rescale previous output and update state
            float scale_factor = expf(m_local - row_max);
            if (row_max > m_local) {
                // New max is larger, rescale previous accumulator
                for (int d = 0; d < head_dim; d++) {
                    o_local[d] *= scale_factor;
                }
                l_local = l_local * scale_factor + row_sum;
                m_local = row_max;
            } else {
                // Old max is larger or equal
                float new_scale = expf(row_max - m_local);
                for (int j = 0; j < block_n; j++) {
                    scores[j] *= new_scale;
                }
                l_local += row_sum * new_scale;
            }

            // Accumulate output: O += scores @ V
            for (int j = 0; j < block_n; j++) {
                for (int d = 0; d < head_dim; d++) {
                    o_local[d] += scores[j] * s_V[j * head_dim + d];
                }
            }
        }
        __syncthreads();
    }

    // Write output with final rescaling
    if (my_row < block_m && global_q_row < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            float val = (l_local > 0.0f) ? o_local[d] / l_local : 0.0f;
            O[qkv_offset + global_q_row * head_dim + d] = val;
        }
    }
}

// ============================================================================
// Flash Attention with KV Cache (for inference/decoding)
// ============================================================================

/**
 * Flash Attention with KV Cache - Single token query
 *
 * Optimized for autoregressive decoding where:
 * - Q has shape [batch, heads, 1, head_dim] (single new token)
 * - K, V have shape [batch, heads, cache_len, head_dim] (cached history)
 *
 * This is the most common case during inference and benefits most from
 * Flash Attention's memory efficiency.
 */
__global__ void flash_attention_decode_kernel(
    const float* __restrict__ Q,         // [batch * heads, 1, head_dim]
    const float* __restrict__ K_cache,   // [batch * heads, cache_len, head_dim]
    const float* __restrict__ V_cache,   // [batch * heads, cache_len, head_dim]
    float* __restrict__ O,               // [batch * heads, 1, head_dim]
    const int cache_len,
    const int head_dim,
    const float scale
) {
    // Each block handles one batch*head
    const int batch_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    // Shared memory for K, V tiles
    extern __shared__ float smem[];
    float* s_K = smem;                          // [TILE_N, head_dim]
    float* s_V = smem + FA_BLOCK_N * head_dim;  // [TILE_N, head_dim]

    // Each thread loads and processes part of Q
    const int q_offset = batch_head_idx * head_dim;
    const int kv_offset = batch_head_idx * cache_len * head_dim;

    // Load Q to registers (each warp loads full Q for its computation)
    float q_reg[4];  // Assuming head_dim <= 128, each thread handles head_dim/32 elements
    for (int i = 0; i < head_dim / WARP_SIZE && i < 4; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
            q_reg[i] = Q[q_offset + d];
        }
    }

    // Output accumulator and softmax state
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    // Process KV cache in tiles
    const int num_tiles = (cache_len + FA_BLOCK_N - 1) / FA_BLOCK_N;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int k_start = tile * FA_BLOCK_N;
        const int tile_size = min(FA_BLOCK_N, cache_len - k_start);

        // Collaborative load K, V tile
        for (int i = tid; i < tile_size * head_dim; i += FA_THREADS) {
            int row = i / head_dim;
            int col = i % head_dim;
            s_K[row * head_dim + col] = K_cache[kv_offset + (k_start + row) * head_dim + col];
            s_V[row * head_dim + col] = V_cache[kv_offset + (k_start + row) * head_dim + col];
        }
        __syncthreads();

        // Compute attention scores Q @ K^T
        float scores[FA_BLOCK_N];
        float tile_max = -FLT_MAX;

        for (int j = 0; j < tile_size; j++) {
            float score = 0.0f;

            // Dot product across head_dim using warp-level parallelism
            for (int i = 0; i < head_dim / WARP_SIZE && i < 4; i++) {
                int d = lane_id + i * WARP_SIZE;
                if (d < head_dim) {
                    score += q_reg[i] * s_K[j * head_dim + d];
                }
            }

            // Warp reduction for dot product
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                score += __shfl_down_sync(0xffffffff, score, offset);
            }
            score = __shfl_sync(0xffffffff, score, 0);

            score *= scale;
            scores[j] = score;
            tile_max = fmaxf(tile_max, score);
        }

        // Compute exp and sum
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_size; j++) {
            scores[j] = expf(scores[j] - tile_max);
            tile_sum += scores[j];
        }

        // Online softmax update
        float new_max = fmaxf(m_prev, tile_max);
        float scale_prev = expf(m_prev - new_max);
        float scale_curr = expf(tile_max - new_max);

        // Rescale previous accumulator
        for (int i = 0; i < 4; i++) {
            o_acc[i] *= scale_prev;
        }
        l_prev = l_prev * scale_prev + tile_sum * scale_curr;
        m_prev = new_max;

        // Scale current scores
        for (int j = 0; j < tile_size; j++) {
            scores[j] *= scale_curr;
        }

        // Accumulate: O += scores @ V
        for (int j = 0; j < tile_size; j++) {
            for (int i = 0; i < head_dim / WARP_SIZE && i < 4; i++) {
                int d = lane_id + i * WARP_SIZE;
                if (d < head_dim) {
                    o_acc[i] += scores[j] * s_V[j * head_dim + d];
                }
            }
        }
        __syncthreads();
    }

    // Final output with softmax normalization
    const int o_offset = batch_head_idx * head_dim;
    for (int i = 0; i < head_dim / WARP_SIZE && i < 4; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim && warp_id == 0) {
            float val = (l_prev > 0.0f) ? o_acc[i] / l_prev : 0.0f;
            O[o_offset + d] = val;
        }
    }
}

// ============================================================================
// C Interface
// ============================================================================

extern "C" {

// Persistent attention buffers
static float* d_Q = nullptr;
static float* d_K = nullptr;
static float* d_V = nullptr;
static float* d_O = nullptr;
static float* d_K_cache = nullptr;
static float* d_V_cache = nullptr;
static int fa_initialized = 0;
static int cache_allocated = 0;
static int max_seq_len = 0;
static int max_batch_heads = 0;
static int stored_head_dim = 0;

/**
 * Initialize Flash Attention buffers
 */
int flash_attention_init(int max_batch, int num_heads, int max_sequence_len, int head_dim) {
    if (fa_initialized) return 0;

    max_seq_len = max_sequence_len;
    max_batch_heads = max_batch * num_heads;
    stored_head_dim = head_dim;

    size_t qkvo_size = max_batch_heads * max_seq_len * head_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q, qkvo_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkvo_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkvo_size));
    CUDA_CHECK(cudaMalloc(&d_O, qkvo_size));

    fa_initialized = 1;

    printf("Flash Attention initialized: batch_heads=%d, max_seq=%d, head_dim=%d (%.2f MB)\n",
           max_batch_heads, max_seq_len, head_dim, 4.0f * qkvo_size / (1024 * 1024));

    return 0;
}

/**
 * Initialize KV cache for inference
 */
int flash_attention_init_kv_cache(int max_batch, int num_heads, int max_cache_len, int head_dim) {
    if (cache_allocated) {
        cudaFree(d_K_cache);
        cudaFree(d_V_cache);
    }

    size_t cache_size = max_batch * num_heads * max_cache_len * head_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_K_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&d_V_cache, cache_size));

    cache_allocated = 1;

    printf("KV Cache initialized: %.2f MB per K/V\n", cache_size / (1024.0f * 1024.0f));

    return 0;
}

/**
 * Cleanup Flash Attention resources
 */
void flash_attention_cleanup(void) {
    if (d_Q) { cudaFree(d_Q); d_Q = nullptr; }
    if (d_K) { cudaFree(d_K); d_K = nullptr; }
    if (d_V) { cudaFree(d_V); d_V = nullptr; }
    if (d_O) { cudaFree(d_O); d_O = nullptr; }
    if (d_K_cache) { cudaFree(d_K_cache); d_K_cache = nullptr; }
    if (d_V_cache) { cudaFree(d_V_cache); d_V_cache = nullptr; }
    fa_initialized = 0;
    cache_allocated = 0;
}

/**
 * Flash Attention Forward Pass
 *
 * @param Q         Query [batch * num_heads, seq_len, head_dim]
 * @param K         Key [batch * num_heads, seq_len, head_dim]
 * @param V         Value [batch * num_heads, seq_len, head_dim]
 * @param O         Output [batch * num_heads, seq_len, head_dim]
 * @param batch_heads  batch * num_heads
 * @param seq_len   Sequence length
 * @param head_dim  Head dimension
 * @param causal    1 for causal masking, 0 otherwise
 */
int flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_heads,
    int seq_len,
    int head_dim,
    int causal
) {
    if (!fa_initialized) {
        fprintf(stderr, "Flash Attention not initialized. Call flash_attention_init() first.\n");
        return -1;
    }

    size_t size = batch_heads * seq_len * head_dim * sizeof(float);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, Q, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)head_dim);

    // Calculate grid dimensions
    int num_q_blocks = (seq_len + FA_BLOCK_M_SMALL - 1) / FA_BLOCK_M_SMALL;
    dim3 grid(batch_heads, num_q_blocks);
    dim3 block(FA_THREADS);

    // Calculate shared memory size
    size_t smem_size = (FA_BLOCK_M_SMALL + 2 * FA_BLOCK_N_SMALL) * head_dim * sizeof(float) +
                       FA_BLOCK_M_SMALL * FA_BLOCK_N_SMALL * sizeof(float);

    // Choose kernel based on head dimension
    if (head_dim <= 64) {
        flash_attention_forward_small_kernel<<<grid, block>>>(
            d_Q, d_K, d_V, d_O,
            seq_len, head_dim, scale, causal
        );
    } else {
        flash_attention_forward_kernel<<<grid, block, smem_size>>>(
            d_Q, d_K, d_V, d_O,
            seq_len, head_dim, scale, causal
        );
    }

    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(O, d_O, size, cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Flash Attention Decode - Single token with KV cache
 *
 * Optimized for autoregressive generation.
 *
 * @param Q         Query [batch * num_heads, 1, head_dim]
 * @param K_new     New key to append [batch * num_heads, 1, head_dim]
 * @param V_new     New value to append [batch * num_heads, 1, head_dim]
 * @param O         Output [batch * num_heads, 1, head_dim]
 * @param batch_heads  batch * num_heads
 * @param cache_pos Current position in cache (0-indexed)
 * @param head_dim  Head dimension
 */
int flash_attention_decode(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
) {
    if (!cache_allocated) {
        fprintf(stderr, "KV Cache not initialized. Call flash_attention_init_kv_cache() first.\n");
        return -1;
    }

    size_t single_size = batch_heads * head_dim * sizeof(float);

    // Copy new Q to device
    CUDA_CHECK(cudaMemcpy(d_Q, Q, single_size, cudaMemcpyHostToDevice));

    // Append new K, V to cache
    for (int bh = 0; bh < batch_heads; bh++) {
        size_t cache_offset = (bh * max_seq_len + cache_pos) * head_dim * sizeof(float);
        size_t src_offset = bh * head_dim * sizeof(float);

        CUDA_CHECK(cudaMemcpy(d_K_cache + (bh * max_seq_len + cache_pos) * head_dim,
                              K_new + bh * head_dim,
                              head_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V_cache + (bh * max_seq_len + cache_pos) * head_dim,
                              V_new + bh * head_dim,
                              head_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    int cache_len = cache_pos + 1;

    // Shared memory for K, V tiles
    size_t smem_size = 2 * FA_BLOCK_N * head_dim * sizeof(float);

    // Launch decode kernel - one block per batch*head
    dim3 grid(batch_heads);
    dim3 block(FA_THREADS);

    flash_attention_decode_kernel<<<grid, block, smem_size>>>(
        d_Q, d_K_cache, d_V_cache, d_O,
        cache_len, head_dim, scale
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(O, d_O, single_size, cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Update KV cache directly (for prefill)
 */
int flash_attention_update_kv_cache(
    const float* K,
    const float* V,
    int batch_heads,
    int start_pos,
    int num_tokens,
    int head_dim
) {
    if (!cache_allocated) {
        fprintf(stderr, "KV Cache not initialized.\n");
        return -1;
    }

    for (int bh = 0; bh < batch_heads; bh++) {
        size_t cache_offset = (bh * max_seq_len + start_pos) * head_dim;
        size_t src_offset = bh * num_tokens * head_dim;

        CUDA_CHECK(cudaMemcpy(d_K_cache + cache_offset,
                              K + src_offset,
                              num_tokens * head_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V_cache + cache_offset,
                              V + src_offset,
                              num_tokens * head_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    return 0;
}

/**
 * Get Flash Attention info
 */
void flash_attention_info(int* initialized, int* cache_ready, int* max_seq, int* head_d) {
    if (initialized) *initialized = fa_initialized;
    if (cache_ready) *cache_ready = cache_allocated;
    if (max_seq) *max_seq = max_seq_len;
    if (head_d) *head_d = stored_head_dim;
}

} // extern "C"
