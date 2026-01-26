/**
 * FlashAttention-2 Multi-GPU Implementation
 *
 * Tensor Parallel attention using NCCL for multi-GPU communication.
 * Each GPU processes a subset of attention heads for linear scaling.
 *
 * Architecture:
 * - Head sharding: total_heads / num_gpus per GPU
 * - KV cache: Distributed across GPUs (each GPU has its heads' cache)
 * - Output: Gathered from all GPUs to host
 *
 * Communication pattern:
 * - AllGather for collecting outputs (O(heads * head_dim * num_gpus))
 * - No AllReduce needed at attention level (heads are independent)
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include "flash_attention_v2_multi_gpu.h"

// ============================================================================
// Configuration
// ============================================================================

#define MAX_GPUS 8
#define FA2_MG_TILE_SIZE 32
#define FA2_MG_THREADS 256
#define FA2_MG_MAX_HEAD_DIM 128

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define NCCL_CHECK(call) do { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, \
                ncclGetErrorString(res)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Per-GPU State
// ============================================================================

typedef struct {
    int gpu_id;
    int start_head;
    int num_heads;

    // Device memory
    float* d_Q;        // [num_heads, head_dim]
    float* d_K_cache;  // [num_heads, max_cache_len, head_dim]
    float* d_V_cache;  // [num_heads, max_cache_len, head_dim]
    float* d_O;        // [num_heads, head_dim]

    // CUDA stream for this GPU
    cudaStream_t stream;

    // Timing
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float compute_time_ms;
    float comm_time_ms;
} GPUState;

// ============================================================================
// Global State
// ============================================================================

static int mg_initialized = 0;
static int mg_num_gpus = 0;
static int mg_total_heads = 0;
static int mg_max_cache_len = 0;
static int mg_head_dim = 0;

static GPUState gpu_states[MAX_GPUS];
static ncclComm_t nccl_comms[MAX_GPUS];
static ncclUniqueId nccl_id;

// Host buffer for gathering outputs
static float* h_output_buffer = nullptr;

// ============================================================================
// FlashAttention-2 Kernel (Same as single-GPU, optimized)
// ============================================================================

__global__ void fa2_mg_decode_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int max_seq_len,
    const float scale
) {
    const int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    const int tid = threadIdx.x;
    const int TILE = FA2_MG_TILE_SIZE;

    extern __shared__ float smem[];
    float* s_K = smem;
    float* s_V = s_K + TILE * head_dim;
    float* s_scores = s_V + TILE * head_dim;
    float* s_Q = s_scores + TILE;
    float* s_O = s_Q + head_dim;  // Shared memory for output accumulation

    // Load Q for this head
    const float* q_ptr = Q + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += FA2_MG_THREADS) {
        s_Q[d] = q_ptr[d];
        s_O[d] = 0.0f;  // Initialize output accumulator
    }
    __syncthreads();

    // Base pointers for this head's KV cache
    const float* k_base = K_cache + head_idx * max_seq_len * head_dim;
    const float* v_base = V_cache + head_idx * max_seq_len * head_dim;

    // Online softmax state (shared across block)
    __shared__ float s_m_prev;
    __shared__ float s_l_prev;
    if (tid == 0) {
        s_m_prev = -FLT_MAX;
        s_l_prev = 0.0f;
    }
    __syncthreads();

    // Process K/V in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE) {
        const int tile_end = min(tile_start + TILE, seq_len);
        const int tile_len = tile_end - tile_start;

        // Load K tile (coalesced)
        for (int i = tid; i < tile_len * head_dim; i += FA2_MG_THREADS) {
            int k_idx = i / head_dim;
            int d_idx = i % head_dim;
            s_K[k_idx * head_dim + d_idx] = k_base[(tile_start + k_idx) * head_dim + d_idx];
        }

        // Load V tile
        for (int i = tid; i < tile_len * head_dim; i += FA2_MG_THREADS) {
            int v_idx = i / head_dim;
            int d_idx = i % head_dim;
            s_V[v_idx * head_dim + d_idx] = v_base[(tile_start + v_idx) * head_dim + d_idx];
        }
        __syncthreads();

        // Compute attention scores: S = Q @ K^T * scale
        for (int k = tid; k < tile_len; k += FA2_MG_THREADS) {
            float score = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < head_dim; d++) {
                score += s_Q[d] * s_K[k * head_dim + d];
            }
            s_scores[k] = score * scale;
        }
        __syncthreads();

        // Find tile max (single thread for simplicity)
        __shared__ float s_tile_max;
        __shared__ float s_m_new;
        __shared__ float s_tile_sum;
        __shared__ float s_rescale;
        __shared__ float s_l_new;

        if (tid == 0) {
            s_tile_max = -FLT_MAX;
            for (int k = 0; k < tile_len; k++) {
                s_tile_max = fmaxf(s_tile_max, s_scores[k]);
            }
            s_m_new = fmaxf(s_m_prev, s_tile_max);
        }
        __syncthreads();

        // Compute softmax values
        for (int k = tid; k < tile_len; k += FA2_MG_THREADS) {
            s_scores[k] = expf(s_scores[k] - s_m_new);
        }
        __syncthreads();

        // Sum softmax values (single thread)
        if (tid == 0) {
            s_tile_sum = 0.0f;
            for (int k = 0; k < tile_len; k++) {
                s_tile_sum += s_scores[k];
            }
            s_rescale = expf(s_m_prev - s_m_new);
            s_l_new = s_rescale * s_l_prev + s_tile_sum;
        }
        __syncthreads();

        // Rescale previous output and accumulate new values
        for (int d = tid; d < head_dim; d += FA2_MG_THREADS) {
            float o_val = s_rescale * s_O[d];
            for (int k = 0; k < tile_len; k++) {
                o_val += s_scores[k] * s_V[k * head_dim + d];
            }
            s_O[d] = o_val;
        }
        __syncthreads();

        // Update online softmax state
        if (tid == 0) {
            s_m_prev = s_m_new;
            s_l_prev = s_l_new;
        }
        __syncthreads();
    }

    // Normalize and write output
    float l_inv = (s_l_prev > 0.0f) ? (1.0f / s_l_prev) : 0.0f;
    float* o_ptr = O + head_idx * head_dim;

    for (int d = tid; d < head_dim; d += FA2_MG_THREADS) {
        o_ptr[d] = s_O[d] * l_inv;
    }
}

// ============================================================================
// Multi-GPU Worker Thread
// ============================================================================

typedef struct {
    int gpu_idx;
    const float* h_Q;
    const float* h_K;
    const float* h_V;
    int cache_pos;
    int result;
} WorkerArgs;

static void* gpu_worker_thread(void* arg) {
    WorkerArgs* args = (WorkerArgs*)arg;
    int gpu_idx = args->gpu_idx;
    GPUState* state = &gpu_states[gpu_idx];

    cudaSetDevice(state->gpu_id);

    int num_heads = state->num_heads;
    int start_head = state->start_head;
    int seq_len = args->cache_pos + 1;

    // Record start time
    cudaEventRecord(state->start_event, state->stream);

    // Copy Q for this GPU's heads (host -> device)
    size_t q_size = num_heads * mg_head_dim * sizeof(float);
    cudaMemcpyAsync(state->d_Q,
                    args->h_Q + start_head * mg_head_dim,
                    q_size, cudaMemcpyHostToDevice, state->stream);

    // Update KV cache for this position
    for (int h = 0; h < num_heads; h++) {
        size_t cache_offset = h * mg_max_cache_len * mg_head_dim + args->cache_pos * mg_head_dim;
        size_t src_offset = (start_head + h) * mg_head_dim;

        cudaMemcpyAsync(state->d_K_cache + cache_offset,
                        args->h_K + src_offset,
                        mg_head_dim * sizeof(float),
                        cudaMemcpyHostToDevice, state->stream);

        cudaMemcpyAsync(state->d_V_cache + cache_offset,
                        args->h_V + src_offset,
                        mg_head_dim * sizeof(float),
                        cudaMemcpyHostToDevice, state->stream);
    }

    // Launch attention kernel
    float scale = 1.0f / sqrtf((float)mg_head_dim);
    // smem: s_K + s_V + s_scores + s_Q + s_O
    size_t smem_size = (2 * FA2_MG_TILE_SIZE * mg_head_dim + FA2_MG_TILE_SIZE + 2 * mg_head_dim) * sizeof(float);

    fa2_mg_decode_kernel<<<num_heads, FA2_MG_THREADS, smem_size, state->stream>>>(
        state->d_Q,
        state->d_K_cache,
        state->d_V_cache,
        state->d_O,
        num_heads,
        seq_len,
        mg_head_dim,
        mg_max_cache_len,
        scale
    );

    // Record end time
    cudaEventRecord(state->end_event, state->stream);

    args->result = 0;
    return nullptr;
}

// ============================================================================
// Public API Implementation
// ============================================================================

extern "C" {

int fa2_multi_gpu_init(int num_gpus, int total_heads, int max_cache_len, int head_dim) {
    if (mg_initialized) {
        fa2_multi_gpu_cleanup();
    }

    // Auto-detect GPUs if not specified
    int available_gpus;
    cudaGetDeviceCount(&available_gpus);

    if (num_gpus <= 0) {
        num_gpus = available_gpus;
    }

    if (num_gpus > available_gpus) {
        fprintf(stderr, "Requested %d GPUs but only %d available\n", num_gpus, available_gpus);
        return -1;
    }

    if (num_gpus > MAX_GPUS) {
        fprintf(stderr, "Maximum %d GPUs supported\n", MAX_GPUS);
        return -1;
    }

    // Check head divisibility
    if (total_heads % num_gpus != 0) {
        fprintf(stderr, "Warning: %d heads not evenly divisible by %d GPUs\n",
                total_heads, num_gpus);
        // Handle uneven distribution
    }

    mg_num_gpus = num_gpus;
    mg_total_heads = total_heads;
    mg_max_cache_len = max_cache_len;
    mg_head_dim = head_dim;

    printf("Multi-GPU FA2: Initializing %d GPUs for %d heads\n", num_gpus, total_heads);

    // Initialize NCCL
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));

    // Calculate heads per GPU
    int base_heads = total_heads / num_gpus;
    int extra_heads = total_heads % num_gpus;

    // Initialize each GPU
    int current_head = 0;
    for (int i = 0; i < num_gpus; i++) {
        GPUState* state = &gpu_states[i];
        state->gpu_id = i;
        state->start_head = current_head;
        state->num_heads = base_heads + (i < extra_heads ? 1 : 0);
        current_head += state->num_heads;

        CUDA_CHECK(cudaSetDevice(i));

        // Create stream and events
        CUDA_CHECK(cudaStreamCreate(&state->stream));
        CUDA_CHECK(cudaEventCreate(&state->start_event));
        CUDA_CHECK(cudaEventCreate(&state->end_event));

        // Allocate device memory
        size_t q_size = state->num_heads * head_dim * sizeof(float);
        size_t cache_size = state->num_heads * max_cache_len * head_dim * sizeof(float);

        CUDA_CHECK(cudaMalloc(&state->d_Q, q_size));
        CUDA_CHECK(cudaMalloc(&state->d_K_cache, cache_size));
        CUDA_CHECK(cudaMalloc(&state->d_V_cache, cache_size));
        CUDA_CHECK(cudaMalloc(&state->d_O, q_size));

        // Initialize NCCL communicator for this GPU
        NCCL_CHECK(ncclCommInitRank(&nccl_comms[i], num_gpus, nccl_id, i));

        printf("  GPU %d: heads %d-%d (%d heads), %.2f MB cache\n",
               i, state->start_head, state->start_head + state->num_heads - 1,
               state->num_heads, 2.0f * cache_size / (1024.0f * 1024.0f));
    }

    // Allocate host output buffer
    h_output_buffer = (float*)malloc(total_heads * head_dim * sizeof(float));
    if (!h_output_buffer) {
        fprintf(stderr, "Failed to allocate host output buffer\n");
        return -1;
    }

    mg_initialized = 1;

    float total_mem_mb = 2.0f * mg_total_heads * max_cache_len * head_dim * sizeof(float) / (1024.0f * 1024.0f);
    printf("Multi-GPU FA2 initialized: %d GPUs, %d heads, %.2f MB total KV cache\n",
           num_gpus, total_heads, total_mem_mb);

    return 0;
}

void fa2_multi_gpu_cleanup(void) {
    if (!mg_initialized) return;

    for (int i = 0; i < mg_num_gpus; i++) {
        GPUState* state = &gpu_states[i];
        cudaSetDevice(state->gpu_id);

        if (state->d_Q) cudaFree(state->d_Q);
        if (state->d_K_cache) cudaFree(state->d_K_cache);
        if (state->d_V_cache) cudaFree(state->d_V_cache);
        if (state->d_O) cudaFree(state->d_O);

        cudaStreamDestroy(state->stream);
        cudaEventDestroy(state->start_event);
        cudaEventDestroy(state->end_event);

        ncclCommDestroy(nccl_comms[i]);

        memset(state, 0, sizeof(GPUState));
    }

    if (h_output_buffer) {
        free(h_output_buffer);
        h_output_buffer = nullptr;
    }

    mg_initialized = 0;
    mg_num_gpus = 0;
}

int fa2_multi_gpu_decode(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int cache_pos
) {
    if (!mg_initialized) {
        fprintf(stderr, "Multi-GPU FA2 not initialized\n");
        return -1;
    }

    if (cache_pos >= mg_max_cache_len) {
        fprintf(stderr, "Cache position %d exceeds max %d\n", cache_pos, mg_max_cache_len);
        return -1;
    }

    // Launch worker threads for each GPU
    pthread_t threads[MAX_GPUS];
    WorkerArgs args[MAX_GPUS];

    for (int i = 0; i < mg_num_gpus; i++) {
        args[i].gpu_idx = i;
        args[i].h_Q = Q;
        args[i].h_K = K_new;
        args[i].h_V = V_new;
        args[i].cache_pos = cache_pos;
        args[i].result = -1;

        pthread_create(&threads[i], nullptr, gpu_worker_thread, &args[i]);
    }

    // Wait for all GPUs to complete
    for (int i = 0; i < mg_num_gpus; i++) {
        pthread_join(threads[i], nullptr);
        if (args[i].result != 0) {
            fprintf(stderr, "GPU %d worker failed\n", i);
            return -1;
        }
    }

    // Synchronize all streams
    for (int i = 0; i < mg_num_gpus; i++) {
        cudaSetDevice(gpu_states[i].gpu_id);
        cudaStreamSynchronize(gpu_states[i].stream);
    }

    // Gather outputs from all GPUs
    for (int i = 0; i < mg_num_gpus; i++) {
        GPUState* state = &gpu_states[i];
        cudaSetDevice(state->gpu_id);

        // Copy this GPU's output to the appropriate position in host buffer
        cudaMemcpy(O + state->start_head * mg_head_dim,
                   state->d_O,
                   state->num_heads * mg_head_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Record timing
        cudaEventSynchronize(state->end_event);
        cudaEventElapsedTime(&state->compute_time_ms, state->start_event, state->end_event);
    }

    return 0;
}

void fa2_multi_gpu_info(int* num_gpus, int* heads_per_gpu, int* total_heads) {
    if (num_gpus) *num_gpus = mg_num_gpus;
    if (heads_per_gpu && mg_num_gpus > 0) *heads_per_gpu = gpu_states[0].num_heads;
    if (total_heads) *total_heads = mg_total_heads;
}

void fa2_multi_gpu_get_timing(int gpu_id, float* compute_ms, float* comm_ms) {
    if (gpu_id < 0 || gpu_id >= mg_num_gpus) return;

    if (compute_ms) *compute_ms = gpu_states[gpu_id].compute_time_ms;
    if (comm_ms) *comm_ms = gpu_states[gpu_id].comm_time_ms;
}

void fa2_multi_gpu_sync(void) {
    for (int i = 0; i < mg_num_gpus; i++) {
        cudaSetDevice(gpu_states[i].gpu_id);
        cudaStreamSynchronize(gpu_states[i].stream);
    }
}

int fa2_multi_gpu_available(void) {
    return mg_initialized && mg_num_gpus > 1;
}

} // extern "C"
