/**
 * FlashAttention-2 CUDA Implementation
 *
 * Key optimizations over FA1:
 * 1. Outer loop over Q, inner loop over K/V (better for causal)
 * 2. Reduced shared memory reads via register tiling
 * 3. Better warp-level parallelism
 * 4. Minimal HBM bandwidth via online softmax
 *
 * Reference: FlashAttention-2 paper (Dao, 2023)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// ============================================================================
// FlashAttention-2 Configuration
// ============================================================================

// Tile sizes - tuned for T4 (sm_75)
#define FA2_TILE_SIZE 64      // K/V tile size (process 64 keys at a time)
#define FA2_THREADS 256       // Threads per block (8 warps)
#define FA2_WARPS 8

// Maximum supported dimensions
#define FA2_MAX_HEAD_DIM 128

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
// FlashAttention-2 Decode Kernel (Single Token)
// ============================================================================

/**
 * FA2 Decode Kernel - processes one query against entire KV cache
 *
 * Each thread block handles one (batch, head) pair.
 * Uses tiled processing with online softmax.
 *
 * Memory layout:
 * - Q: [batch_heads, head_dim] - single query token
 * - K_cache: [batch_heads, max_seq_len, head_dim]
 * - V_cache: [batch_heads, max_seq_len, head_dim]
 * - O: [batch_heads, head_dim] - output
 */
__global__ void flash_attention_v2_decode_kernel(
    const float* __restrict__ Q,           // [batch_heads, head_dim]
    const float* __restrict__ K_cache,     // [batch_heads, max_seq_len, head_dim]
    const float* __restrict__ V_cache,     // [batch_heads, max_seq_len, head_dim]
    float* __restrict__ O,                 // [batch_heads, head_dim]
    const int batch_heads,
    const int seq_len,                     // Current sequence length (cache_pos + 1)
    const int head_dim,
    const int max_seq_len,
    const float scale                      // 1/sqrt(head_dim)
) {
    const int bh = blockIdx.x;  // batch_head index
    if (bh >= batch_heads) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory for K/V tiles and partial results
    extern __shared__ float smem[];
    float* s_K = smem;                           // [FA2_TILE_SIZE, head_dim]
    float* s_V = s_K + FA2_TILE_SIZE * head_dim; // [FA2_TILE_SIZE, head_dim]
    float* s_scores = s_V + FA2_TILE_SIZE * head_dim; // [FA2_TILE_SIZE]

    // Load query to registers (each thread loads part of Q)
    float q_reg[FA2_MAX_HEAD_DIM / FA2_THREADS + 1];
    const float* q_ptr = Q + bh * head_dim;

    #pragma unroll
    for (int i = tid; i < head_dim; i += FA2_THREADS) {
        q_reg[i / FA2_THREADS] = q_ptr[i];
    }

    // Initialize online softmax state (per thread accumulates partial results)
    float m_prev = -FLT_MAX;  // Running max
    float l_prev = 0.0f;       // Running sum of exp

    // Output accumulator in registers
    float o_reg[FA2_MAX_HEAD_DIM / FA2_THREADS + 1] = {0};

    // Base pointers for K/V cache
    const float* k_base = K_cache + bh * max_seq_len * head_dim;
    const float* v_base = V_cache + bh * max_seq_len * head_dim;

    // Process K/V cache in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += FA2_TILE_SIZE) {
        const int tile_end = min(tile_start + FA2_TILE_SIZE, seq_len);
        const int tile_len = tile_end - tile_start;

        // Load K tile to shared memory (coalesced)
        for (int i = tid; i < tile_len * head_dim; i += FA2_THREADS) {
            int k_idx = i / head_dim;  // Which key in tile
            int d_idx = i % head_dim;  // Which dimension
            s_K[k_idx * head_dim + d_idx] = k_base[(tile_start + k_idx) * head_dim + d_idx];
        }

        // Load V tile to shared memory
        for (int i = tid; i < tile_len * head_dim; i += FA2_THREADS) {
            int v_idx = i / head_dim;
            int d_idx = i % head_dim;
            s_V[v_idx * head_dim + d_idx] = v_base[(tile_start + v_idx) * head_dim + d_idx];
        }

        __syncthreads();

        // Compute attention scores for this tile: S = Q @ K^T * scale
        // Each thread computes scores for a subset of keys
        for (int k = tid; k < tile_len; k += FA2_THREADS) {
            float score = 0.0f;

            #pragma unroll 8
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[d] * s_K[k * head_dim + d];
            }

            s_scores[k] = score * scale;
        }

        __syncthreads();

        // Find max in this tile (for numerical stability)
        float tile_max = -FLT_MAX;
        for (int k = tid; k < tile_len; k += FA2_THREADS) {
            tile_max = fmaxf(tile_max, s_scores[k]);
        }

        // Warp reduction for max
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
        }

        // Share max across warps via shared memory
        __shared__ float warp_max[FA2_WARPS];
        if (lane_id == 0) {
            warp_max[warp_id] = tile_max;
        }
        __syncthreads();

        // First warp finds global max
        if (warp_id == 0) {
            tile_max = (lane_id < FA2_WARPS) ? warp_max[lane_id] : -FLT_MAX;
            #pragma unroll
            for (int offset = FA2_WARPS / 2; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
            if (lane_id == 0) {
                warp_max[0] = tile_max;
            }
        }
        __syncthreads();

        float m_curr = warp_max[0];  // Current tile max

        // Update running max
        float m_new = fmaxf(m_prev, m_curr);

        // Compute exp(score - m_new) and sum
        float tile_sum = 0.0f;
        for (int k = tid; k < tile_len; k += FA2_THREADS) {
            float p = expf(s_scores[k] - m_new);
            s_scores[k] = p;  // Store exp for later use
            tile_sum += p;
        }

        // Warp reduction for sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
        }

        __shared__ float warp_sum[FA2_WARPS];
        if (lane_id == 0) {
            warp_sum[warp_id] = tile_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            tile_sum = (lane_id < FA2_WARPS) ? warp_sum[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = FA2_WARPS / 2; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
            if (lane_id == 0) {
                warp_sum[0] = tile_sum;
            }
        }
        __syncthreads();

        float l_curr = warp_sum[0];

        // Rescale previous output and accumulate new
        // O_new = exp(m_prev - m_new) * O_prev + P @ V
        float rescale = expf(m_prev - m_new);
        float l_new = rescale * l_prev + l_curr;

        // Update output accumulator
        // Each thread handles a subset of dimensions
        for (int d = tid; d < head_dim; d += FA2_THREADS) {
            // Rescale previous output
            float o_val = rescale * o_reg[d / FA2_THREADS];

            // Add P @ V for this dimension
            for (int k = 0; k < tile_len; k++) {
                o_val += s_scores[k] * s_V[k * head_dim + d];
            }

            o_reg[d / FA2_THREADS] = o_val;
        }

        // Update state
        m_prev = m_new;
        l_prev = l_new;

        __syncthreads();
    }

    // Normalize output: O = O / l
    float l_inv = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

    // Write output
    float* o_ptr = O + bh * head_dim;
    for (int d = tid; d < head_dim; d += FA2_THREADS) {
        o_ptr[d] = o_reg[d / FA2_THREADS] * l_inv;
    }
}

/**
 * Optimized FA2 kernel with vectorized loads (float4)
 */
__global__ void flash_attention_v2_decode_vectorized_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    const int batch_heads,
    const int seq_len,
    const int head_dim,
    const int max_seq_len,
    const float scale
) {
    const int bh = blockIdx.x;
    if (bh >= batch_heads) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Smaller tile for better occupancy
    const int TILE = 32;

    extern __shared__ float smem[];
    float* s_K = smem;
    float* s_V = s_K + TILE * head_dim;
    float* s_scores = s_V + TILE * head_dim;

    const float* q_ptr = Q + bh * head_dim;
    const float* k_base = K_cache + bh * max_seq_len * head_dim;
    const float* v_base = V_cache + bh * max_seq_len * head_dim;

    // Load Q to shared memory once
    __shared__ float s_Q[FA2_MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += FA2_THREADS) {
        s_Q[d] = q_ptr[d];
    }
    __syncthreads();

    // Online softmax state
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_acc[4] = {0, 0, 0, 0};  // Accumulator for output (vectorized)

    // Process tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        const int tile_end = min(tile_start + TILE, seq_len);
        const int tile_len = tile_end - tile_start;

        // Vectorized load of K tile
        const int vec_dim = head_dim / 4;
        for (int i = tid; i < tile_len * vec_dim; i += FA2_THREADS) {
            int k_idx = i / vec_dim;
            int v_idx = i % vec_dim;
            float4 kv = reinterpret_cast<const float4*>(k_base + (tile_start + k_idx) * head_dim)[v_idx];
            reinterpret_cast<float4*>(s_K + k_idx * head_dim)[v_idx] = kv;
        }

        // Vectorized load of V tile
        for (int i = tid; i < tile_len * vec_dim; i += FA2_THREADS) {
            int v_idx_row = i / vec_dim;
            int v_idx_col = i % vec_dim;
            float4 vv = reinterpret_cast<const float4*>(v_base + (tile_start + v_idx_row) * head_dim)[v_idx_col];
            reinterpret_cast<float4*>(s_V + v_idx_row * head_dim)[v_idx_col] = vv;
        }

        __syncthreads();

        // Compute scores with vectorized dot product
        for (int k = tid; k < tile_len; k += FA2_THREADS) {
            float score = 0.0f;

            // Vectorized dot product
            for (int d = 0; d < head_dim; d += 4) {
                float4 q4 = *reinterpret_cast<const float4*>(s_Q + d);
                float4 k4 = *reinterpret_cast<const float4*>(s_K + k * head_dim + d);
                score += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }

            s_scores[k] = score * scale;
        }

        __syncthreads();

        // Find tile max
        float tile_max = -FLT_MAX;
        for (int k = 0; k < tile_len; k++) {
            tile_max = fmaxf(tile_max, s_scores[k]);
        }

        float m_new = fmaxf(m_prev, tile_max);

        // Compute softmax and sum
        float tile_sum = 0.0f;
        for (int k = 0; k < tile_len; k++) {
            float p = expf(s_scores[k] - m_new);
            s_scores[k] = p;
            tile_sum += p;
        }

        // Rescale and accumulate
        float rescale = expf(m_prev - m_new);
        float l_new = rescale * l_prev + tile_sum;

        // Update output for this thread's dimensions
        for (int d = tid; d < head_dim; d += FA2_THREADS) {
            float o_val = rescale * o_acc[0];

            for (int k = 0; k < tile_len; k++) {
                o_val += s_scores[k] * s_V[k * head_dim + d];
            }

            o_acc[0] = o_val;
        }

        m_prev = m_new;
        l_prev = l_new;

        __syncthreads();
    }

    // Normalize and write output
    float l_inv = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    float* o_ptr = O + bh * head_dim;

    for (int d = tid; d < head_dim; d += FA2_THREADS) {
        o_ptr[d] = o_acc[0] * l_inv;
    }
}

// ============================================================================
// C Interface
// ============================================================================

extern "C" {

// Persistent buffers
static float* fa2_d_Q = nullptr;
static float* fa2_d_K_cache = nullptr;
static float* fa2_d_V_cache = nullptr;
static float* fa2_d_O = nullptr;
static int fa2_initialized = 0;
static int fa2_max_cache_len = 0;
static int fa2_max_batch_heads = 0;
static int fa2_head_dim = 0;

// Forward declaration
void flash_attention_v2_cleanup(void);

/**
 * Initialize FlashAttention-2 buffers
 */
int flash_attention_v2_init(int max_batch_heads, int max_cache_len, int head_dim) {
    if (fa2_initialized) {
        flash_attention_v2_cleanup();
    }

    fa2_max_batch_heads = max_batch_heads;
    fa2_max_cache_len = max_cache_len;
    fa2_head_dim = head_dim;

    size_t q_size = max_batch_heads * head_dim * sizeof(float);
    size_t cache_size = max_batch_heads * max_cache_len * head_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&fa2_d_Q, q_size));
    CUDA_CHECK(cudaMalloc(&fa2_d_K_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&fa2_d_V_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&fa2_d_O, q_size));

    fa2_initialized = 1;

    printf("FlashAttention-2 initialized: batch_heads=%d, max_cache=%d, head_dim=%d\n",
           max_batch_heads, max_cache_len, head_dim);
    printf("  Memory: Q=%.2f KB, K_cache=%.2f MB, V_cache=%.2f MB\n",
           q_size / 1024.0f, cache_size / (1024.0f * 1024.0f), cache_size / (1024.0f * 1024.0f));

    return 0;
}

/**
 * Cleanup FlashAttention-2 resources
 */
void flash_attention_v2_cleanup(void) {
    if (fa2_d_Q) { cudaFree(fa2_d_Q); fa2_d_Q = nullptr; }
    if (fa2_d_K_cache) { cudaFree(fa2_d_K_cache); fa2_d_K_cache = nullptr; }
    if (fa2_d_V_cache) { cudaFree(fa2_d_V_cache); fa2_d_V_cache = nullptr; }
    if (fa2_d_O) { cudaFree(fa2_d_O); fa2_d_O = nullptr; }
    fa2_initialized = 0;
}

/**
 * Update KV cache at position
 */
int flash_attention_v2_update_cache(
    const float* K_new,
    const float* V_new,
    int batch_heads,
    int cache_pos,
    int head_dim
) {
    if (!fa2_initialized) return -1;

    size_t offset = cache_pos * head_dim * sizeof(float);
    size_t size = batch_heads * head_dim * sizeof(float);

    // Copy to each batch_head's cache position
    for (int bh = 0; bh < batch_heads; bh++) {
        size_t cache_offset = (bh * fa2_max_cache_len * head_dim + cache_pos * head_dim) * sizeof(float);
        size_t src_offset = bh * head_dim * sizeof(float);

        CUDA_CHECK(cudaMemcpy(fa2_d_K_cache + bh * fa2_max_cache_len * head_dim + cache_pos * head_dim,
                              K_new + bh * head_dim,
                              head_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fa2_d_V_cache + bh * fa2_max_cache_len * head_dim + cache_pos * head_dim,
                              V_new + bh * head_dim,
                              head_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    return 0;
}

/**
 * FlashAttention-2 Decode
 */
int flash_attention_v2_decode(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
) {
    if (!fa2_initialized) {
        fprintf(stderr, "FlashAttention-2 not initialized\n");
        return -1;
    }

    if (batch_heads > fa2_max_batch_heads || cache_pos >= fa2_max_cache_len) {
        fprintf(stderr, "FA2: batch_heads=%d > max=%d or cache_pos=%d >= max=%d\n",
                batch_heads, fa2_max_batch_heads, cache_pos, fa2_max_cache_len);
        return -1;
    }

    // Copy Q to device
    CUDA_CHECK(cudaMemcpy(fa2_d_Q, Q, batch_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Update KV cache
    flash_attention_v2_update_cache(K_new, V_new, batch_heads, cache_pos, head_dim);

    int seq_len = cache_pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Calculate shared memory size
    int tile_size = 32;  // Using smaller tile for vectorized kernel
    size_t smem_size = (2 * tile_size * head_dim + tile_size + FA2_MAX_HEAD_DIM) * sizeof(float);

    // Launch kernel
    flash_attention_v2_decode_vectorized_kernel<<<batch_heads, FA2_THREADS, smem_size>>>(
        fa2_d_Q,
        fa2_d_K_cache,
        fa2_d_V_cache,
        fa2_d_O,
        batch_heads,
        seq_len,
        head_dim,
        fa2_max_cache_len,
        scale
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy output back
    CUDA_CHECK(cudaMemcpy(O, fa2_d_O, batch_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Get FA2 status
 */
void flash_attention_v2_info(int* initialized, int* max_cache, int* max_bh, int* h_dim) {
    if (initialized) *initialized = fa2_initialized;
    if (max_cache) *max_cache = fa2_max_cache_len;
    if (max_bh) *max_bh = fa2_max_batch_heads;
    if (h_dim) *h_dim = fa2_head_dim;
}

} // extern "C"
