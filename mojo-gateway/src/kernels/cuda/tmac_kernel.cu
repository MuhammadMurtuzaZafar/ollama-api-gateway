/**
 * EdgeLLM CUDA T-MAC Kernels
 *
 * High-performance GPU implementation of Table-lookup Matrix Multiplication (T-MAC)
 * for BitNet 1.58-bit quantized models.
 *
 * Key optimizations:
 * - Lookup tables in shared memory (fast access)
 * - Coalesced global memory reads
 * - Warp-level reductions
 * - Minimal thread divergence
 *
 * Target: NVIDIA Jetson Nano/Orin, RTX GPUs
 */

#include "tmac_kernel_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define LUT_SIZE 16          // 4-bit activation indices (0-15)
#define TILE_K 128           // K dimension tile size
#define TILE_N 64            // N dimension tile size

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/**
 * T-MAC Lookup Table Kernel
 *
 * Instead of: y = W * x (matrix multiplication)
 * We do:      y = LUT[packed_weights] (table lookup)
 *
 * The LUT is precomputed: LUT[i] = sum of activations where ternary weight pattern = i
 */
__global__ void tmac_matmul_kernel(
    const int8_t* __restrict__ packed_weights,  // Packed ternary weights (2 bits each)
    const float* __restrict__ activations,       // Input activations
    float* __restrict__ output,                  // Output buffer
    const float* __restrict__ scales,            // Per-row scales
    int M,                                       // Output rows
    int N,                                       // Output cols (batch)
    int K                                        // Inner dimension
) {
    // Shared memory for lookup tables (one per output row in block)
    __shared__ float lut[LUT_SIZE][TILE_N];

    int row = blockIdx.x;           // Output row
    int col_base = blockIdx.y * TILE_N;  // Output column base
    int tid = threadIdx.x;

    if (row >= M) return;

    // Initialize LUT to zero
    for (int i = tid; i < LUT_SIZE * TILE_N; i += BLOCK_SIZE) {
        int lut_idx = i / TILE_N;
        int col_off = i % TILE_N;
        lut[lut_idx][col_off] = 0.0f;
    }
    __syncthreads();

    // Build lookup table for this row
    // For 4-bit activation quantization, we have 16 possible values
    // LUT[i] = sum of activations where the 4-bit index equals i
    float scale = scales[row];

    // Process K dimension in tiles
    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        int k_end = min(k_base + TILE_K, K);

        // Each thread processes multiple K elements
        for (int k = k_base + tid; k < k_end; k += BLOCK_SIZE) {
            // Get packed weight (4 ternary values per byte)
            int weight_idx = row * ((K + 3) / 4) + (k / 4);
            int8_t packed = packed_weights[weight_idx];

            // Extract ternary value for this k (-1, 0, +1)
            int shift = (k % 4) * 2;
            int ternary = ((packed >> shift) & 0x3) - 1;  // Map 0,1,2 to -1,0,1

            // Update LUT based on activation quantization
            for (int col_off = 0; col_off < TILE_N && (col_base + col_off) < N; col_off++) {
                float act = activations[k * N + col_base + col_off];

                // Quantize activation to 4-bit index
                int act_idx = min(15, max(0, (int)((act + 1.0f) * 8.0f)));

                // Accumulate based on ternary weight
                if (ternary != 0) {
                    atomicAdd(&lut[act_idx][col_off], ternary * act);
                }
            }
        }
    }
    __syncthreads();

    // Reduce LUT to final output
    for (int col_off = tid; col_off < TILE_N && (col_base + col_off) < N; col_off += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < LUT_SIZE; i++) {
            sum += lut[i][col_off];
        }
        output[row * N + col_base + col_off] = sum * scale;
    }
}

/**
 * Optimized T-MAC kernel using warp-level primitives
 * Better memory coalescing and reduced atomic operations
 */
__global__ void tmac_matmul_kernel_v2(
    const uint8_t* __restrict__ packed_weights,  // Packed ternary weights
    const half* __restrict__ activations,        // FP16 activations
    half* __restrict__ output,                   // FP16 output
    const half* __restrict__ scales,             // Per-row scales
    int M,                                       // Output rows
    int N,                                       // Output cols
    int K                                        // Inner dimension
) {
    // Shared memory for partial sums
    __shared__ float smem[BLOCK_SIZE];

    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    float scale = __half2float(scales[row]);

    // Each thread accumulates over K
    int k_per_byte = 4;  // 4 ternary values per byte
    int weight_row_bytes = (K + k_per_byte - 1) / k_per_byte;

    for (int k = tid; k < K; k += BLOCK_SIZE) {
        // Load activation
        float act = __half2float(activations[k * N + col]);

        // Load and unpack weight
        int byte_idx = k / k_per_byte;
        int bit_offset = (k % k_per_byte) * 2;
        uint8_t packed = packed_weights[row * weight_row_bytes + byte_idx];
        int ternary = ((packed >> bit_offset) & 0x3) - 1;  // -1, 0, +1

        // Accumulate
        sum += ternary * act;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps (first warp only)
    if (warp_id == 0) {
        sum = (tid < (BLOCK_SIZE / WARP_SIZE)) ? smem[tid] : 0.0f;

        #pragma unroll
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (tid == 0) {
            output[row * N + col] = __float2half(sum * scale);
        }
    }
}

/**
 * RMSNorm CUDA kernel
 */
__global__ void rmsnorm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int size,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx_base = blockIdx.x * size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        float val = input[idx_base + i];
        sum_sq += val * val;
    }

    // Reduce within block
    smem[tid] = sum_sq;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // Compute normalization factor
    float rms = rsqrtf(smem[0] / size + eps);

    // Apply normalization
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        output[idx_base + i] = input[idx_base + i] * rms * weight[i];
    }
}

/**
 * Softmax CUDA kernel
 */
__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int size
) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx_base = blockIdx.x * size;

    // Find max
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, input[idx_base + i]);
    }

    smem[tid] = max_val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        float exp_val = expf(input[idx_base + i] - max_val);
        output[idx_base + i] = exp_val;
        sum += exp_val;
    }

    smem[tid] = sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];

    // Normalize
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        output[idx_base + i] /= sum;
    }
}

/**
 * RoPE (Rotary Position Embedding) CUDA kernel
 */
__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ freq_cis_real,
    const float* __restrict__ freq_cis_imag,
    int head_dim,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;

    if (idx >= half_dim) return;

    float cos_val = freq_cis_real[pos * half_dim + idx];
    float sin_val = freq_cis_imag[pos * half_dim + idx];

    // Rotate Q
    float q0 = q[idx * 2];
    float q1 = q[idx * 2 + 1];
    q[idx * 2] = q0 * cos_val - q1 * sin_val;
    q[idx * 2 + 1] = q0 * sin_val + q1 * cos_val;

    // Rotate K
    float k0 = k[idx * 2];
    float k1 = k[idx * 2 + 1];
    k[idx * 2] = k0 * cos_val - k1 * sin_val;
    k[idx * 2 + 1] = k0 * sin_val + k1 * cos_val;
}

// ============================================================================
// C Interface for Mojo FFI
// ============================================================================

extern "C" {

// Device memory handles
static float* d_weights = nullptr;
static float* d_activations = nullptr;
static float* d_output = nullptr;
static float* d_scales = nullptr;
static int cuda_initialized = 0;

// Track allocated buffer sizes for dynamic reallocation
static size_t allocated_activations_size = 0;
static size_t allocated_output_size = 0;

// Default sizes for SmolLM-135M (can handle up to 7B models)
#define DEFAULT_MAX_DIM 4096
#define DEFAULT_MAX_WEIGHTS_BYTES (DEFAULT_MAX_DIM * DEFAULT_MAX_DIM / 4)  // 2-bit packed
#define DEFAULT_MAX_ACTIVATIONS DEFAULT_MAX_DIM
#define DEFAULT_MAX_OUTPUT DEFAULT_MAX_DIM

/**
 * Initialize CUDA context (no-argument version with sensible defaults)
 */
int cuda_init() {
    if (cuda_initialized) return 0;

    // Check for CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("EdgeLLM CUDA Engine initialized on: %s\n", prop.name);
    printf("  - Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  - Total memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  - SM count: %d\n", prop.multiProcessorCount);

    // Allocate device memory with sensible defaults
    CUDA_CHECK(cudaMalloc(&d_weights, DEFAULT_MAX_WEIGHTS_BYTES));
    CUDA_CHECK(cudaMalloc(&d_activations, DEFAULT_MAX_ACTIVATIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, DEFAULT_MAX_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, DEFAULT_MAX_OUTPUT * sizeof(float)));

    allocated_activations_size = DEFAULT_MAX_ACTIVATIONS;
    allocated_output_size = DEFAULT_MAX_OUTPUT;

    cuda_initialized = 1;
    return 0;
}

/**
 * Initialize CUDA context and allocate device memory (with explicit sizes)
 */
int cuda_init_sized(int max_weights_bytes, int max_activations, int max_output) {
    if (cuda_initialized) return 0;

    // Validate parameters - use defaults if invalid
    if (max_weights_bytes <= 0) max_weights_bytes = DEFAULT_MAX_WEIGHTS_BYTES;
    if (max_activations <= 0) max_activations = DEFAULT_MAX_ACTIVATIONS;
    if (max_output <= 0) max_output = DEFAULT_MAX_OUTPUT;

    // Check for CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("EdgeLLM CUDA Engine initialized on: %s\n", prop.name);
    printf("  - Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  - Total memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  - SM count: %d\n", prop.multiProcessorCount);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_weights, max_weights_bytes));
    CUDA_CHECK(cudaMalloc(&d_activations, max_activations * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, max_output * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, max_output * sizeof(float)));

    allocated_activations_size = max_activations;
    allocated_output_size = max_output;

    cuda_initialized = 1;
    return 0;
}

/**
 * Cleanup CUDA resources
 */
void cuda_cleanup() {
    if (!cuda_initialized) return;

    cudaFree(d_weights);
    cudaFree(d_activations);
    cudaFree(d_output);
    cudaFree(d_scales);

    d_weights = nullptr;
    d_activations = nullptr;
    d_output = nullptr;
    d_scales = nullptr;

    // Reset allocated sizes
    allocated_activations_size = 0;
    allocated_output_size = 0;

    cuda_initialized = 0;
}

/**
 * T-MAC Matrix Multiplication (GPU)
 *
 * @param weights     Packed ternary weights (host)
 * @param activations Input activations (host)
 * @param output      Output buffer (host)
 * @param scales      Per-row scales (host)
 * @param M           Output rows
 * @param N           Output columns
 * @param K           Inner dimension
 */
int tmac_matmul_cuda(
    const int8_t* weights,
    const float* activations,
    float* output,
    const float* scales,
    int M, int N, int K
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    // Calculate sizes
    int weight_bytes = M * ((K + 3) / 4);
    int act_size = K * N;
    int out_size = M * N;

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_activations, activations, act_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales, M * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);

    tmac_matmul_kernel<<<grid, block>>>(
        (int8_t*)d_weights,
        d_activations,
        d_output,
        d_scales,
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * RMSNorm (GPU)
 */
int rmsnorm_cuda(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int size,
    float eps
) {
    if (!cuda_initialized) return -1;

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_activations, input, batch_size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, weight, size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    rmsnorm_kernel<<<batch_size, BLOCK_SIZE>>>(
        d_output, d_activations, d_scales, size, eps
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy back
    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Softmax (GPU)
 */
int softmax_cuda(
    float* output,
    const float* input,
    int batch_size,
    int size
) {
    if (!cuda_initialized) return -1;

    CUDA_CHECK(cudaMemcpy(d_activations, input, batch_size * size * sizeof(float), cudaMemcpyHostToDevice));

    softmax_kernel<<<batch_size, BLOCK_SIZE>>>(d_output, d_activations, size);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Check if CUDA is available
 */
int cuda_available() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

/**
 * Get CUDA device name
 */
const char* cuda_device_name() {
    static char name[256];
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        strncpy(name, prop.name, 255);
        return name;
    }
    return "Unknown";
}

/**
 * Synchronize CUDA device
 */
void cuda_sync() {
    cudaDeviceSynchronize();
}

/**
 * Get CUDA device properties
 */
int cuda_device_info(
    size_t* total_memory,
    int* sm_count,
    int* compute_major,
    int* compute_minor
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return -1;
    }

    if (total_memory) *total_memory = prop.totalGlobalMem;
    if (sm_count) *sm_count = prop.multiProcessorCount;
    if (compute_major) *compute_major = prop.major;
    if (compute_minor) *compute_minor = prop.minor;

    return 0;
}

// ============================================================================
// Phase 1: Persistent GPU Memory Implementation
// ============================================================================

// Persistent weight storage
static int8_t* d_persistent_weights = nullptr;
static float* d_persistent_scales = nullptr;
static float* d_persistent_norm_weights = nullptr;
static int persistent_weight_bytes = 0;
static int persistent_num_rows = 0;
static int persistent_norm_size = 0;
static int weights_on_gpu = 0;
static int norm_weights_on_gpu = 0;

/**
 * Load model weights to GPU memory (one-time operation)
 */
int cuda_load_weights(
    const int8_t* weights,
    const float* scales,
    int weight_bytes,
    int num_rows
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    // Free existing weights if any
    if (d_persistent_weights) {
        cudaFree(d_persistent_weights);
        d_persistent_weights = nullptr;
    }
    if (d_persistent_scales) {
        cudaFree(d_persistent_scales);
        d_persistent_scales = nullptr;
    }

    // Allocate GPU memory for weights
    CUDA_CHECK(cudaMalloc(&d_persistent_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_persistent_scales, num_rows * sizeof(float)));

    // Copy weights to GPU (one-time transfer)
    CUDA_CHECK(cudaMemcpy(d_persistent_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_persistent_scales, scales, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    persistent_weight_bytes = weight_bytes;
    persistent_num_rows = num_rows;
    weights_on_gpu = 1;

    printf("EdgeLLM: Loaded %d bytes of weights to GPU (%d rows)\n", weight_bytes, num_rows);

    return 0;
}

/**
 * Unload weights from GPU memory
 */
void cuda_unload_weights() {
    if (d_persistent_weights) {
        cudaFree(d_persistent_weights);
        d_persistent_weights = nullptr;
    }
    if (d_persistent_scales) {
        cudaFree(d_persistent_scales);
        d_persistent_scales = nullptr;
    }
    persistent_weight_bytes = 0;
    persistent_num_rows = 0;
    weights_on_gpu = 0;
}

/**
 * Check if weights are loaded on GPU
 */
int cuda_weights_loaded() {
    return weights_on_gpu;
}

/**
 * T-MAC Matrix Multiplication with persistent weights
 *
 * This is the FAST version - only transfers activations, not weights.
 * Weights must be pre-loaded via cuda_load_weights().
 */
int tmac_matmul_cuda_persistent(
    const float* activations,
    float* output,
    int M, int N, int K
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    if (!weights_on_gpu) {
        fprintf(stderr, "Weights not loaded. Call cuda_load_weights() first.\n");
        return -1;
    }

    // Calculate sizes
    size_t act_size = (size_t)K * N;
    size_t out_size = (size_t)M * N;

    // Dynamically resize activation buffer if needed
    if (act_size > allocated_activations_size) {
        if (d_activations) {
            cudaFree(d_activations);
        }
        CUDA_CHECK(cudaMalloc(&d_activations, act_size * sizeof(float)));
        allocated_activations_size = act_size;
    }

    // Dynamically resize output buffer if needed
    if (out_size > allocated_output_size) {
        if (d_output) {
            cudaFree(d_output);
        }
        CUDA_CHECK(cudaMalloc(&d_output, out_size * sizeof(float)));
        allocated_output_size = out_size;
    }

    // Transfer ONLY activations to device (weights already there!)
    CUDA_CHECK(cudaMemcpy(d_activations, activations, act_size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel using persistent weights
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);

    tmac_matmul_kernel<<<grid, block>>>(
        d_persistent_weights,
        d_activations,
        d_output,
        d_persistent_scales,
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    // Transfer output back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Load normalization weights to GPU
 */
int cuda_load_norm_weights(
    const float* norm_weights,
    int size
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    // Free existing norm weights if any
    if (d_persistent_norm_weights) {
        cudaFree(d_persistent_norm_weights);
        d_persistent_norm_weights = nullptr;
    }

    // Allocate and copy
    CUDA_CHECK(cudaMalloc(&d_persistent_norm_weights, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_persistent_norm_weights, norm_weights, size * sizeof(float), cudaMemcpyHostToDevice));

    persistent_norm_size = size;
    norm_weights_on_gpu = 1;

    return 0;
}

/**
 * RMSNorm using pre-loaded weights
 */
int rmsnorm_cuda_persistent(
    float* output,
    const float* input,
    int batch_size,
    int size,
    float eps
) {
    if (!cuda_initialized) return -1;
    if (!norm_weights_on_gpu) {
        fprintf(stderr, "Norm weights not loaded. Call cuda_load_norm_weights() first.\n");
        return -1;
    }

    // Copy only input to device (weights already there!)
    CUDA_CHECK(cudaMemcpy(d_activations, input, batch_size * size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with persistent norm weights
    rmsnorm_kernel<<<batch_size, BLOCK_SIZE>>>(
        d_output, d_activations, d_persistent_norm_weights, size, eps
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy output back
    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

// ============================================================================
// Phase 2: Kernel Fusion + CUDA Streams
// ============================================================================

// CUDA streams for async operations
static cudaStream_t compute_stream = nullptr;
static cudaStream_t transfer_stream = nullptr;
static int streams_initialized = 0;

// Pinned memory buffers for faster transfers
static float* h_pinned_activations = nullptr;
static float* h_pinned_output = nullptr;
static int pinned_buffer_size = 0;

/**
 * Fused RMSNorm + T-MAC MatMul Kernel
 *
 * Combines normalization and matrix multiplication in one kernel launch.
 * Benefits:
 * - Eliminates intermediate buffer
 * - Reduces kernel launch overhead
 * - Better cache utilization
 */
__global__ void fused_rmsnorm_matmul_kernel(
    const float* __restrict__ input,        // Input activations [batch_size * size]
    const float* __restrict__ norm_weights, // Normalization weights [size]
    const int8_t* __restrict__ matmul_weights, // Packed ternary weights [M * K/4]
    const float* __restrict__ matmul_scales,   // Per-row scales [M]
    float* __restrict__ output,             // Output [batch_size * M]
    int size,                               // Input/hidden size (K)
    int M,                                  // Output dimension
    int N,                                  // Batch size
    float eps
) {
    // Shared memory for:
    // 1. Normalized input (reused as LUT base)
    // 2. LUT for T-MAC
    __shared__ float s_normalized[1024];  // Max hidden_size
    __shared__ float s_rms;
    __shared__ float lut[LUT_SIZE][TILE_N];

    int row = blockIdx.x;           // Output row (0 to M-1)
    int col_base = blockIdx.y * TILE_N;  // Output column base
    int tid = threadIdx.x;
    int batch_idx = 0;  // For now, batch_size = 1

    if (row >= M) return;

    // =========== Step 1: RMSNorm (collaborative) ===========
    // All threads participate in computing RMS of input
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        float val = input[batch_idx * size + i];
        sum_sq += val * val;
    }

    // Reduce sum_sq within block
    __shared__ float smem_reduce[BLOCK_SIZE];
    smem_reduce[tid] = sum_sq;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_reduce[tid] += smem_reduce[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 computes final RMS
    if (tid == 0) {
        s_rms = rsqrtf(smem_reduce[0] / size + eps);
    }
    __syncthreads();

    // Apply normalization and store in shared memory
    float rms = s_rms;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        s_normalized[i] = input[batch_idx * size + i] * rms * norm_weights[i];
    }
    __syncthreads();

    // =========== Step 2: T-MAC MatMul (using normalized input) ===========
    // Initialize LUT to zero
    for (int i = tid; i < LUT_SIZE * TILE_N; i += BLOCK_SIZE) {
        int lut_idx = i / TILE_N;
        int col_off = i % TILE_N;
        lut[lut_idx][col_off] = 0.0f;
    }
    __syncthreads();

    // Build lookup table for this output row
    float scale = matmul_scales[row];

    // Process K dimension
    for (int k = tid; k < size; k += BLOCK_SIZE) {
        // Get packed weight (4 ternary values per byte)
        int weight_idx = row * ((size + 3) / 4) + (k / 4);
        int8_t packed = matmul_weights[weight_idx];

        // Extract ternary value for this k (-1, 0, +1)
        int shift = (k % 4) * 2;
        int ternary = ((packed >> shift) & 0x3) - 1;

        // Use normalized activation from shared memory
        for (int col_off = 0; col_off < TILE_N && (col_base + col_off) < N; col_off++) {
            // For batch_size=1, just use s_normalized[k]
            float act = s_normalized[k];

            // Quantize activation to 4-bit index
            int act_idx = min(15, max(0, (int)((act + 1.0f) * 8.0f)));

            // Accumulate based on ternary weight
            if (ternary != 0) {
                atomicAdd(&lut[act_idx][col_off], ternary * act);
            }
        }
    }
    __syncthreads();

    // Reduce LUT to final output
    for (int col_off = tid; col_off < TILE_N && (col_base + col_off) < N; col_off += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < LUT_SIZE; i++) {
            sum += lut[i][col_off];
        }
        output[row * N + col_base + col_off] = sum * scale;
    }
}

/**
 * Initialize CUDA streams for async operations
 */
int cuda_init_streams() {
    if (streams_initialized) return 0;

    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));

    streams_initialized = 1;
    return 0;
}

/**
 * Cleanup CUDA streams
 */
void cuda_cleanup_streams() {
    if (!streams_initialized) return;

    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);
    compute_stream = nullptr;
    transfer_stream = nullptr;
    streams_initialized = 0;
}

/**
 * Allocate pinned (page-locked) memory for faster transfers
 */
int cuda_alloc_pinned(int max_activations, int max_output) {
    if (h_pinned_activations) {
        cudaFreeHost(h_pinned_activations);
    }
    if (h_pinned_output) {
        cudaFreeHost(h_pinned_output);
    }

    CUDA_CHECK(cudaMallocHost(&h_pinned_activations, max_activations * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_output, max_output * sizeof(float)));

    pinned_buffer_size = max_activations;
    printf("EdgeLLM: Allocated pinned memory (%d activations, %d output)\n",
           max_activations, max_output);

    return 0;
}

/**
 * Free pinned memory
 */
void cuda_free_pinned() {
    if (h_pinned_activations) {
        cudaFreeHost(h_pinned_activations);
        h_pinned_activations = nullptr;
    }
    if (h_pinned_output) {
        cudaFreeHost(h_pinned_output);
        h_pinned_output = nullptr;
    }
    pinned_buffer_size = 0;
}

/**
 * Fused RMSNorm + T-MAC MatMul (Phase 2 optimization)
 *
 * Combines normalization and matrix multiplication in a single kernel launch.
 * Requires:
 * - Norm weights loaded via cuda_load_norm_weights()
 * - Matmul weights loaded via cuda_load_weights()
 */
int fused_rmsnorm_matmul_cuda(
    const float* input,
    float* output,
    int M,      // Output dimension
    int N,      // Batch size (usually 1)
    int K,      // Hidden size / input dimension
    float eps
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }
    if (!weights_on_gpu || !norm_weights_on_gpu) {
        fprintf(stderr, "Weights not loaded. Load both norm and matmul weights first.\n");
        return -1;
    }

    int input_size = K * N;
    int out_size = M * N;

    // Transfer input to device
    CUDA_CHECK(cudaMemcpy(d_activations, input, input_size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch fused kernel
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);

    fused_rmsnorm_matmul_kernel<<<grid, block>>>(
        d_activations,
        d_persistent_norm_weights,
        d_persistent_weights,
        d_persistent_scales,
        d_output,
        K, M, N, eps
    );

    CUDA_CHECK(cudaGetLastError());

    // Transfer output back
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Async version of T-MAC MatMul using CUDA streams
 *
 * Overlaps data transfer with computation when possible.
 * Use cuda_sync_streams() to wait for completion.
 */
int tmac_matmul_cuda_async(
    const float* activations,
    float* output,
    int M, int N, int K
) {
    if (!cuda_initialized || !weights_on_gpu) return -1;

    if (!streams_initialized) {
        cuda_init_streams();
    }

    int act_size = K * N;
    int out_size = M * N;

    // Async copy activations to device
    CUDA_CHECK(cudaMemcpyAsync(d_activations, activations, act_size * sizeof(float),
                                cudaMemcpyHostToDevice, transfer_stream));

    // Wait for transfer to complete before computing
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

    // Launch kernel on compute stream
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);

    tmac_matmul_kernel<<<grid, block, 0, compute_stream>>>(
        d_persistent_weights,
        d_activations,
        d_output,
        d_persistent_scales,
        M, N, K
    );

    // Async copy output back
    CUDA_CHECK(cudaMemcpyAsync(output, d_output, out_size * sizeof(float),
                                cudaMemcpyDeviceToHost, compute_stream));

    return 0;
}

/**
 * Wait for all async operations to complete
 */
void cuda_sync_streams() {
    if (compute_stream) cudaStreamSynchronize(compute_stream);
    if (transfer_stream) cudaStreamSynchronize(transfer_stream);
}

/**
 * Fused RMSNorm + MatMul with async streams and pinned memory
 *
 * Maximum performance version combining all Phase 2 optimizations.
 */
int fused_rmsnorm_matmul_cuda_fast(
    const float* input,
    float* output,
    int M, int N, int K,
    float eps
) {
    if (!cuda_initialized || !weights_on_gpu || !norm_weights_on_gpu) return -1;

    if (!streams_initialized) {
        cuda_init_streams();
    }

    int input_size = K * N;
    int out_size = M * N;

    // Use pinned memory if available for faster transfer
    if (h_pinned_activations && input_size <= pinned_buffer_size) {
        memcpy(h_pinned_activations, input, input_size * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(d_activations, h_pinned_activations,
                                    input_size * sizeof(float),
                                    cudaMemcpyHostToDevice, transfer_stream));
    } else {
        CUDA_CHECK(cudaMemcpy(d_activations, input, input_size * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Sync transfer before compute
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

    // Launch fused kernel
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);

    fused_rmsnorm_matmul_kernel<<<grid, block, 0, compute_stream>>>(
        d_activations,
        d_persistent_norm_weights,
        d_persistent_weights,
        d_persistent_scales,
        d_output,
        K, M, N, eps
    );

    // Async copy output using pinned memory
    if (h_pinned_output && out_size <= pinned_buffer_size) {
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_output, d_output,
                                    out_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        memcpy(output, h_pinned_output, out_size * sizeof(float));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    return 0;
}

// ============================================================================
// Phase 2.1: Optimized Kernels (No Atomics, True Fusion)
// ============================================================================

/**
 * Optimized T-MAC kernel with warp-private accumulation (NO atomicAdd)
 *
 * Key optimizations:
 * 1. Each thread accumulates its own partial sum (no atomics)
 * 2. Warp-level reduction using shuffle
 * 3. Direct accumulation without LUT (simpler for batch_size=1)
 * 4. Better memory coalescing
 */
__global__ void tmac_matmul_kernel_v3(
    const int8_t* __restrict__ packed_weights,  // Packed ternary weights
    const float* __restrict__ activations,       // Input activations [K * N]
    float* __restrict__ output,                  // Output buffer [M * N]
    const float* __restrict__ scales,            // Per-row scales [M]
    int M,                                       // Output rows
    int N,                                       // Output cols (batch)
    int K                                        // Inner dimension
) {
    // Each block handles one output row, threads process K in parallel
    int row = blockIdx.x;
    int col = blockIdx.y;  // For batch_size=1, col=0
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (row >= M || col >= N) return;

    // Each thread accumulates its own partial sum
    float partial_sum = 0.0f;
    float scale = scales[row];
    int weight_row_bytes = (K + 3) / 4;

    // Process K dimension - each thread handles strided elements
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        // Load activation (coalesced across threads)
        float act = activations[k * N + col];

        // Load and unpack weight
        int byte_idx = k / 4;
        int bit_offset = (k % 4) * 2;
        int8_t packed = packed_weights[row * weight_row_bytes + byte_idx];
        int ternary = ((packed >> bit_offset) & 0x3) - 1;  // -1, 0, +1

        // Accumulate (no atomic needed - thread-private)
        partial_sum += ternary * act;
    }

    // Warp-level reduction using shuffle (fast!)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Write warp result to shared memory
    __shared__ float warp_sums[8];  // Max 8 warps per block
    if (lane == 0) {
        warp_sums[warp_id] = partial_sum;
    }
    __syncthreads();

    // Final reduction across warps (only first warp)
    if (warp_id == 0 && tid < (BLOCK_SIZE / WARP_SIZE)) {
        partial_sum = warp_sums[tid];

        #pragma unroll
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        if (tid == 0) {
            output[row * N + col] = partial_sum * scale;
        }
    }
}

/**
 * True Streaming Fused RMSNorm + T-MAC MatMul Kernel
 *
 * Key insight: For batch_size=1, we don't need LUT at all!
 * Just compute: output[row] = scale[row] * sum_k(ternary[row,k] * normalized[k])
 *
 * Two-pass algorithm:
 * Pass 1: Compute RMS = sqrt(mean(x^2) + eps)
 * Pass 2: Stream through K, normalize on-the-fly, accumulate with ternary weights
 */
__global__ void streaming_fused_rmsnorm_matmul_kernel(
    const float* __restrict__ input,            // Input [K]
    const float* __restrict__ norm_weights,     // RMSNorm weights [K]
    const int8_t* __restrict__ matmul_weights,  // Packed ternary [M * K/4]
    const float* __restrict__ matmul_scales,    // Per-row scales [M]
    float* __restrict__ output,                 // Output [M]
    int K,                                      // Hidden size
    int M,                                      // Output dimension
    float eps
) {
    // Shared memory for RMS computation
    __shared__ float smem[BLOCK_SIZE];
    __shared__ float s_rms;

    int row = blockIdx.x;  // Output row
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (row >= M) return;

    // =========== Pass 1: Compute RMS (all threads collaborate) ===========
    float sum_sq = 0.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float val = input[k];
        sum_sq += val * val;
    }

    // Reduce within block
    smem[tid] = sum_sq;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_rms = rsqrtf(smem[0] / K + eps);
    }
    __syncthreads();

    float rms = s_rms;

    // =========== Pass 2: Streaming normalize + matmul ===========
    // Each thread accumulates its partial sum
    float partial_sum = 0.0f;
    float scale = matmul_scales[row];
    int weight_row_bytes = (K + 3) / 4;

    for (int k = tid; k < K; k += BLOCK_SIZE) {
        // Load and normalize on-the-fly (no intermediate storage!)
        float val = input[k];
        float normalized = val * rms * norm_weights[k];

        // Load ternary weight
        int byte_idx = k / 4;
        int bit_offset = (k % 4) * 2;
        int8_t packed = matmul_weights[row * weight_row_bytes + byte_idx];
        int ternary = ((packed >> bit_offset) & 0x3) - 1;

        // Accumulate directly
        partial_sum += ternary * normalized;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Write warp results to shared memory (reuse smem)
    if (lane == 0) {
        smem[warp_id] = partial_sum;
    }
    __syncthreads();

    // Final reduction
    if (warp_id == 0 && tid < (BLOCK_SIZE / WARP_SIZE)) {
        partial_sum = smem[tid];

        #pragma unroll
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        if (tid == 0) {
            output[row] = partial_sum * scale;
        }
    }
}

// Dispatch threshold constants
#define DISPATCH_THRESHOLD_FUSED 500000  // M*K > 500K: use fused
#define DISPATCH_THRESHOLD_V3    50000   // M*K > 50K: use v3 kernel

/**
 * Optimized T-MAC MatMul with adaptive dispatch
 *
 * Chooses the best kernel based on tensor size:
 * - Large tensors: streaming fused kernel
 * - Medium tensors: v3 kernel (warp-private)
 * - Small tensors: simple persistent kernel (avoid overhead)
 */
int tmac_matmul_cuda_v3(
    const float* activations,
    float* output,
    int M, int N, int K
) {
    if (!cuda_initialized || !weights_on_gpu) return -1;

    int act_size = K * N;
    int out_size = M * N;

    // Transfer activations
    CUDA_CHECK(cudaMemcpy(d_activations, activations, act_size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch optimized kernel
    dim3 grid(M, N);  // One block per (row, col) for batch_size=1
    dim3 block(BLOCK_SIZE);

    tmac_matmul_kernel_v3<<<grid, block>>>(
        d_persistent_weights,
        d_activations,
        d_output,
        d_persistent_scales,
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    // Transfer output
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Streaming Fused RMSNorm + T-MAC MatMul
 *
 * True fusion: normalizes on-the-fly without intermediate storage.
 * Best for batch_size=1 (single token generation).
 */
int streaming_fused_rmsnorm_matmul_cuda(
    const float* input,
    float* output,
    int M,      // Output dimension
    int K,      // Hidden size
    float eps
) {
    if (!cuda_initialized || !weights_on_gpu || !norm_weights_on_gpu) return -1;

    // Transfer input
    CUDA_CHECK(cudaMemcpy(d_activations, input, K * sizeof(float), cudaMemcpyHostToDevice));

    // Launch streaming fused kernel
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);

    streaming_fused_rmsnorm_matmul_kernel<<<grid, block>>>(
        d_activations,
        d_persistent_norm_weights,
        d_persistent_weights,
        d_persistent_scales,
        d_output,
        K, M, eps
    );

    CUDA_CHECK(cudaGetLastError());

    // Transfer output
    CUDA_CHECK(cudaMemcpy(output, d_output, M * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Adaptive dispatch: automatically choose best kernel
 */
int tmac_matmul_cuda_adaptive(
    const float* activations,
    float* output,
    int M, int N, int K
) {
    if (!cuda_initialized || !weights_on_gpu) return -1;

    int elements = M * K;

    if (elements > DISPATCH_THRESHOLD_V3) {
        // Large/medium tensors: use optimized v3 kernel
        return tmac_matmul_cuda_v3(activations, output, M, N, K);
    } else {
        // Small tensors: use simple persistent kernel (less overhead)
        return tmac_matmul_cuda_persistent(activations, output, M, N, K);
    }
}

/**
 * Adaptive fused: choose between streaming fused and separate kernels
 */
int fused_rmsnorm_matmul_cuda_adaptive(
    const float* input,
    float* output,
    int M, int N, int K,
    float eps
) {
    if (!cuda_initialized || !weights_on_gpu || !norm_weights_on_gpu) return -1;

    int elements = M * K;

    if (N == 1 && elements > DISPATCH_THRESHOLD_V3) {
        // Batch size 1 + large tensor: use streaming fused
        return streaming_fused_rmsnorm_matmul_cuda(input, output, M, K, eps);
    } else {
        // Fall back to separate kernels for small tensors or batched
        // (The overhead of fusion doesn't pay off)
        float* norm_out = (float*)malloc(K * sizeof(float));
        if (!norm_out) return -1;

        // RMSNorm
        int ret = rmsnorm_cuda_persistent(norm_out, input, 1, K, eps);
        if (ret != 0) {
            free(norm_out);
            return ret;
        }

        // MatMul
        ret = tmac_matmul_cuda_adaptive(norm_out, output, M, N, K);
        free(norm_out);
        return ret;
    }
}

} // End extern "C" - WMMA code below uses C++ templates

// ============================================================================
// Phase 3: INT8 Tensor Core Implementation
// ============================================================================
// NOTE: This section is OUTSIDE extern "C" because WMMA uses C++ templates.
// The C-callable API wrappers are defined later in a new extern "C" block.

// Include WMMA header for Tensor Core support
// Note: We include unconditionally for host compilation, but use __CUDA_ARCH__
// guards for device-specific intrinsics inside kernels
#if defined(__CUDACC__)
#include <mma.h>
using namespace nvcuda::wmma;
#endif

// Runtime check for WMMA support (used in host code)
// __CUDA_ARCH__ is only defined during device compilation
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define HAS_WMMA_DEVICE 1
#else
#define HAS_WMMA_DEVICE 0
#endif

// WMMA tile dimensions for INT8
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block configuration for Tensor Core kernel
// With 256 threads = 8 warps, we have 4 warps in M x 2 warps in N
// Each warp computes one 16x16 tile, so block covers 64x32 output
#define TC_BLOCK_M 64   // Output rows per block (4 WMMA tiles) = TC_WARPS_M * WMMA_M
#define TC_BLOCK_N 32   // Output cols per block (2 WMMA tiles) = TC_WARPS_N * WMMA_N
#define TC_WARPS_M 4    // Warps in M dimension
#define TC_WARPS_N 2    // Warps in N dimension

// WMMA requires leading dimension to be multiple of 16 for proper alignment
// Shared memory padding: round up to next multiple of 16
#define WMMA_SMEM_PAD 32  // Padded K dimension (16 * 2 = 32, safe for WMMA_K=16)

// Thresholds for INT8 TC dispatch
#define TC_MIN_ELEMENTS 50000   // M*K threshold for TC benefit
#define TC_MIN_COMPUTE_CAP 75   // Minimum compute capability

// Helper macro to pad dimension to multiple of 16 (WMMA tile size)
#define PAD_TO_WMMA(x) (((x) + WMMA_K - 1) / WMMA_K * WMMA_K)

// Persistent INT8 TC buffers
static int8_t* d_weights_int8_expanded = nullptr;  // [M * K] expanded weights
static float* d_weights_int8_scales = nullptr;     // [M] weight scales
static int8_t* d_activations_int8 = nullptr;       // [max_K * max_N] quantized activations
static float* d_act_scales = nullptr;              // [max_N] activation scales
static int32_t* d_output_int32 = nullptr;          // [max_M * max_N] INT32 accumulator
static int weights_int8_tc_loaded = 0;
static int persistent_K_int8 = 0;
static int persistent_M_int8 = 0;
static int cached_compute_capability = -1;

/**
 * Kernel: Expand 2-bit packed ternary weights to INT8
 *
 * Each byte contains 4 ternary values (2 bits each).
 * This kernel expands them to full INT8 format for Tensor Cores.
 */
__global__ void expand_ternary_to_int8_kernel(
    const int8_t* __restrict__ packed_weights,  // [M * K/4] packed
    int8_t* __restrict__ expanded_weights,      // [M * K] expanded
    int M, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * K;

    if (idx < total_elements) {
        int row = idx / K;
        int k = idx % K;

        int weight_row_bytes = (K + 3) / 4;
        int byte_idx = k / 4;
        int bit_offset = (k % 4) * 2;

        int8_t packed = packed_weights[row * weight_row_bytes + byte_idx];
        int ternary = ((packed >> bit_offset) & 0x3) - 1;  // -1, 0, +1

        expanded_weights[row * K + k] = (int8_t)ternary;
    }
}

/**
 * Kernel: Quantize FP32 activations to INT8 with per-token scaling
 *
 * Uses absmax symmetric quantization:
 * scale = absmax / 127
 * quantized = round(value / scale)
 */
__global__ void quantize_activations_int8_kernel(
    const float* __restrict__ input_fp32,   // [K * N] or [K] for batch=1
    int8_t* __restrict__ output_int8,       // [K * N]
    float* __restrict__ scales,             // [N] per-token scale
    int K, int N
) {
    __shared__ float smem_max[BLOCK_SIZE];

    int col = blockIdx.x;  // Token/batch index
    int tid = threadIdx.x;

    if (col >= N) return;

    // Step 1: Find absmax of this token's activations
    float local_max = 0.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float val = fabsf(input_fp32[k * N + col]);
        local_max = fmaxf(local_max, val);
    }

    // Reduce to find global max
    smem_max[tid] = local_max;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
        }
        __syncthreads();
    }

    float absmax = smem_max[0];
    float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;

    if (tid == 0) {
        scales[col] = scale;
    }
    __syncthreads();

    // Step 2: Quantize activations
    float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 1.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float val = input_fp32[k * N + col];
        int quantized = __float2int_rn(val * inv_scale);
        // Clamp to INT8 range
        quantized = max(-127, min(127, quantized));
        output_int8[k * N + col] = (int8_t)quantized;
    }
}

/**
 * Kernel: Dequantize INT32 accumulator to FP32 output
 *
 * output_fp32 = output_int32 * weight_scale * act_scale
 */
__global__ void dequantize_output_kernel(
    const int32_t* __restrict__ output_int32,  // [M * N]
    float* __restrict__ output_fp32,           // [M * N]
    const float* __restrict__ weight_scales,   // [M]
    const float* __restrict__ act_scales,      // [N]
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < M * N) {
        int row = idx / N;
        int col = idx % N;

        float scale = weight_scales[row] * act_scales[col];
        output_fp32[idx] = (float)output_int32[idx] * scale;
    }
}

// INT8 Tensor Core kernel - requires mma.h which is included above under __CUDACC__
// Runtime compute capability check is done in the host API before launching

/**
 * INT8 Tensor Core Matrix Multiplication Kernel
 *
 * Uses WMMA API for INT8 GEMM with 16x16x16 tiles.
 * Each block computes a TC_BLOCK_M x TC_BLOCK_N output tile.
 *
 * Note: Requires compute capability >= 7.5 (Turing/Ampere)
 */
__global__ void int8_tensorcore_matmul_kernel(
    const int8_t* __restrict__ weights_int8,     // [M, K] row-major
    const int8_t* __restrict__ activations_int8, // [K, N] row-major
    int32_t* __restrict__ output_int32,          // [M, N] row-major
    int M, int N, int K
) {
    // Shared memory for tiles - WMMA requires 16-byte aligned leading dimensions
    // Using WMMA_SMEM_PAD (32) ensures proper alignment for load_matrix_sync
    // TC_BLOCK_N (32) is already a multiple of 16, so no extra padding needed
    __shared__ __align__(16) int8_t smem_A[TC_BLOCK_M][WMMA_SMEM_PAD];  // 64x32, padded K
    __shared__ __align__(16) int8_t smem_B[WMMA_SMEM_PAD][TC_BLOCK_N];  // 32x32, padded K x N

    // Block position
    int block_row = blockIdx.x * TC_BLOCK_M;
    int block_col = blockIdx.y * TC_BLOCK_N;

    // Warp position within block
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_row = (warp_id / TC_WARPS_N) * WMMA_M;
    int warp_col = (warp_id % TC_WARPS_N) * WMMA_N;

    // Declare WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, row_major> frag_A[TC_WARPS_M];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, col_major> frag_B[TC_WARPS_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int> frag_C;

    // Initialize accumulator
    fill_fragment(frag_C, 0);

    // Iterate over K dimension in WMMA_K tiles
    for (int k_base = 0; k_base < K; k_base += WMMA_K) {
        // Collaborative load A tile (weights) into shared memory
        int num_loads_A = (TC_BLOCK_M * WMMA_K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < num_loads_A; i++) {
            int load_idx = threadIdx.x + i * BLOCK_SIZE;
            if (load_idx < TC_BLOCK_M * WMMA_K) {
                int m_off = load_idx / WMMA_K;
                int k_off = load_idx % WMMA_K;
                int m_global = block_row + m_off;
                int k_global = k_base + k_off;

                if (m_global < M && k_global < K) {
                    smem_A[m_off][k_off] = weights_int8[m_global * K + k_global];
                } else {
                    smem_A[m_off][k_off] = 0;
                }
            }
        }

        // Collaborative load B tile (activations) into shared memory
        int num_loads_B = (WMMA_K * TC_BLOCK_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < num_loads_B; i++) {
            int load_idx = threadIdx.x + i * BLOCK_SIZE;
            if (load_idx < WMMA_K * TC_BLOCK_N) {
                int k_off = load_idx / TC_BLOCK_N;
                int n_off = load_idx % TC_BLOCK_N;
                int k_global = k_base + k_off;
                int n_global = block_col + n_off;

                if (k_global < K && n_global < N) {
                    smem_B[k_off][n_off] = activations_int8[k_global * N + n_global];
                } else {
                    smem_B[k_off][n_off] = 0;
                }
            }
        }
        __syncthreads();

        // Load WMMA fragments from shared memory and compute
        // Leading dimensions must be multiples of 16 for WMMA alignment
        if (block_row + warp_row < M && block_col + warp_col < N) {
            load_matrix_sync(frag_A[0], &smem_A[warp_row][0], WMMA_SMEM_PAD);  // ld=32
            load_matrix_sync(frag_B[0], &smem_B[0][warp_col], TC_BLOCK_N);     // ld=64

            // Tensor Core MMA operation
            mma_sync(frag_C, frag_A[0], frag_B[0], frag_C);
        }
        __syncthreads();
    }

    // Store result - use aligned shared memory buffer to avoid global memory alignment issues
    // WMMA requires leading dimension to be multiple of 16
    __shared__ __align__(16) int32_t smem_C[TC_BLOCK_M][TC_BLOCK_N];  // 64x32 output tile

    int out_row = block_row + warp_row;
    int out_col = block_col + warp_col;

    // Store WMMA result to aligned shared memory first
    if (out_row < M && out_col < N) {
        // TC_BLOCK_N (32) is guaranteed multiple of 16
        store_matrix_sync(&smem_C[warp_row][warp_col], frag_C, TC_BLOCK_N, mem_row_major);
    }
    __syncthreads();

    // Copy valid results from shared memory to global memory (unaligned N is OK here)
    int tid = threadIdx.x;
    int total_elements = TC_BLOCK_M * TC_BLOCK_N;

    for (int i = tid; i < total_elements; i += BLOCK_SIZE) {
        int local_row = i / TC_BLOCK_N;
        int local_col = i % TC_BLOCK_N;
        int global_row = block_row + local_row;
        int global_col = block_col + local_col;

        if (global_row < M && global_col < N) {
            output_int32[global_row * N + global_col] = smem_C[local_row][local_col];
        }
    }
}

/**
 * Fallback INT8 matmul kernel for devices without WMMA support
 * Uses standard CUDA cores with INT8 accumulation
 */
__global__ void int8_fallback_matmul_kernel(
    const int8_t* __restrict__ weights_int8,
    const int8_t* __restrict__ activations_int8,
    int32_t* __restrict__ output_int32,
    int M, int N, int K
) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= M || col >= N) return;

    // Each thread accumulates partial sum
    int partial_sum = 0;

    for (int k = tid; k < K; k += BLOCK_SIZE) {
        int8_t w = weights_int8[row * K + k];
        int8_t a = activations_int8[k * N + col];
        partial_sum += (int)w * (int)a;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Write warp results to shared memory
    __shared__ int warp_sums[8];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp_id] = partial_sum;
    }
    __syncthreads();

    // Final reduction
    if (warp_id == 0 && tid < (BLOCK_SIZE / WARP_SIZE)) {
        partial_sum = warp_sums[tid];

        #pragma unroll
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        if (tid == 0) {
            output_int32[row * N + col] = partial_sum;
        }
    }
}

// ============================================================================
// C Interface for INT8 Tensor Core API (extern "C" for FFI)
// ============================================================================
extern "C" {

/**
 * Get compute capability of current device
 */
int cuda_get_compute_capability() {
    if (cached_compute_capability >= 0) {
        return cached_compute_capability;
    }

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return 0;
    }

    cached_compute_capability = prop.major * 10 + prop.minor;
    return cached_compute_capability;
}

/**
 * Check if INT8 Tensor Cores are available
 */
int cuda_has_int8_tensorcore() {
    int cc = cuda_get_compute_capability();
    return (cc >= TC_MIN_COMPUTE_CAP) ? 1 : 0;
}

/**
 * Get device count
 */
int cuda_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

/**
 * Load weights in INT8 Tensor Core format
 */
int cuda_load_weights_int8_tc(
    const int8_t* packed_weights,
    const float* scales,
    int weight_bytes,
    int num_rows,
    int K
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    // Free existing INT8 TC weights
    if (d_weights_int8_expanded) {
        cudaFree(d_weights_int8_expanded);
        d_weights_int8_expanded = nullptr;
    }
    if (d_weights_int8_scales) {
        cudaFree(d_weights_int8_scales);
        d_weights_int8_scales = nullptr;
    }

    int expanded_size = num_rows * K;

    // Allocate expanded weights buffer
    CUDA_CHECK(cudaMalloc(&d_weights_int8_expanded, expanded_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_weights_int8_scales, num_rows * sizeof(float)));

    // Copy packed weights to temporary buffer
    int8_t* d_packed_temp;
    CUDA_CHECK(cudaMalloc(&d_packed_temp, weight_bytes));
    CUDA_CHECK(cudaMemcpy(d_packed_temp, packed_weights, weight_bytes, cudaMemcpyHostToDevice));

    // Launch expansion kernel
    int threads = 256;
    int blocks = (expanded_size + threads - 1) / threads;
    expand_ternary_to_int8_kernel<<<blocks, threads>>>(
        d_packed_temp,
        d_weights_int8_expanded,
        num_rows, K
    );
    CUDA_CHECK(cudaGetLastError());

    // Free temporary packed weights
    cudaFree(d_packed_temp);

    // Copy scales
    CUDA_CHECK(cudaMemcpy(d_weights_int8_scales, scales, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate working buffers for activations and output
    int max_K = K;
    int max_N = 64;  // Support up to batch=64
    int max_M = num_rows;

    if (d_activations_int8) cudaFree(d_activations_int8);
    if (d_act_scales) cudaFree(d_act_scales);
    if (d_output_int32) cudaFree(d_output_int32);

    CUDA_CHECK(cudaMalloc(&d_activations_int8, max_K * max_N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_act_scales, max_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_int32, max_M * max_N * sizeof(int32_t)));

    persistent_K_int8 = K;
    persistent_M_int8 = num_rows;
    weights_int8_tc_loaded = 1;

    printf("EdgeLLM: Loaded INT8 TC weights (%d rows x %d cols = %.2f MB expanded)\n",
           num_rows, K, expanded_size / (1024.0f * 1024.0f));

    return 0;
}

/**
 * Unload INT8 TC weights
 */
void cuda_unload_weights_int8_tc() {
    if (d_weights_int8_expanded) {
        cudaFree(d_weights_int8_expanded);
        d_weights_int8_expanded = nullptr;
    }
    if (d_weights_int8_scales) {
        cudaFree(d_weights_int8_scales);
        d_weights_int8_scales = nullptr;
    }
    if (d_activations_int8) {
        cudaFree(d_activations_int8);
        d_activations_int8 = nullptr;
    }
    if (d_act_scales) {
        cudaFree(d_act_scales);
        d_act_scales = nullptr;
    }
    if (d_output_int32) {
        cudaFree(d_output_int32);
        d_output_int32 = nullptr;
    }
    weights_int8_tc_loaded = 0;
    persistent_K_int8 = 0;
    persistent_M_int8 = 0;
}

/**
 * Check if INT8 TC weights are loaded
 */
int cuda_weights_int8_tc_loaded() {
    return weights_int8_tc_loaded;
}

/**
 * INT8 Tensor Core Matrix Multiplication
 */
int tmac_matmul_cuda_int8_tc(
    const float* activations,
    float* output,
    int M, int N, int K
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized.\n");
        return -1;
    }
    if (!weights_int8_tc_loaded) {
        fprintf(stderr, "INT8 TC weights not loaded. Call cuda_load_weights_int8_tc() first.\n");
        return -1;
    }

    // Step 1: Copy FP32 activations to device
    CUDA_CHECK(cudaMemcpy(d_activations, activations, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Step 2: Quantize activations to INT8
    quantize_activations_int8_kernel<<<N, BLOCK_SIZE>>>(
        d_activations,
        d_activations_int8,
        d_act_scales,
        K, N
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 3: INT8 Matrix Multiplication
    int cc = cuda_get_compute_capability();

    // Runtime check for Tensor Core support (>= sm_75)
    if (cc >= TC_MIN_COMPUTE_CAP) {
        // Use Tensor Core kernel
        dim3 grid((M + TC_BLOCK_M - 1) / TC_BLOCK_M, (N + TC_BLOCK_N - 1) / TC_BLOCK_N);
        dim3 block(BLOCK_SIZE);

        int8_tensorcore_matmul_kernel<<<grid, block>>>(
            d_weights_int8_expanded,
            d_activations_int8,
            d_output_int32,
            M, N, K
        );
    } else {
        // Use fallback kernel for older GPUs
        dim3 grid(M, N);
        dim3 block(BLOCK_SIZE);

        int8_fallback_matmul_kernel<<<grid, block>>>(
            d_weights_int8_expanded,
            d_activations_int8,
            d_output_int32,
            M, N, K
        );
    }
    CUDA_CHECK(cudaGetLastError());

    // Step 4: Dequantize output to FP32
    int total_output = M * N;
    int dequant_blocks = (total_output + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dequantize_output_kernel<<<dequant_blocks, BLOCK_SIZE>>>(
        d_output_int32,
        d_output,
        d_weights_int8_scales,
        d_act_scales,
        M, N
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 5: Copy output back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Streaming Fused RMSNorm + INT8 TC MatMul
 */
int streaming_fused_rmsnorm_matmul_int8_tc(
    const float* input,
    float* output,
    int M, int K,
    float eps
) {
    if (!cuda_initialized || !weights_int8_tc_loaded || !norm_weights_on_gpu) {
        return -1;
    }

    // For now, implement as separate RMSNorm + INT8 TC MatMul
    // True fusion can be added later as optimization

    // Allocate temporary normalized buffer
    float* norm_output;
    CUDA_CHECK(cudaMalloc(&norm_output, K * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_activations, input, K * sizeof(float), cudaMemcpyHostToDevice));

    // RMSNorm
    rmsnorm_kernel<<<1, BLOCK_SIZE>>>(
        norm_output, d_activations, d_persistent_norm_weights, K, eps
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy normalized output back to do quantization
    float* h_norm_output = (float*)malloc(K * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_norm_output, norm_output, K * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(norm_output);

    // INT8 TC MatMul
    int ret = tmac_matmul_cuda_int8_tc(h_norm_output, output, M, 1, K);

    free(h_norm_output);
    return ret;
}

/**
 * Adaptive dispatch v2 with INT8 Tensor Core support
 */
int fused_rmsnorm_matmul_cuda_adaptive_v2(
    const float* input,
    float* output,
    int M, int N, int K,
    float eps
) {
    if (!cuda_initialized) return -1;

    int cc = cuda_get_compute_capability();
    long long elements = (long long)M * K;
    int k_aligned = (K % WMMA_K == 0);

    // Decision logic for kernel selection
    if (cc >= TC_MIN_COMPUTE_CAP &&
        weights_int8_tc_loaded &&
        elements >= TC_MIN_ELEMENTS &&
        k_aligned) {
        // Use INT8 Tensor Core path
        if (N == 1 && norm_weights_on_gpu) {
            return streaming_fused_rmsnorm_matmul_int8_tc(input, output, M, K, eps);
        } else {
            // Separate RMSNorm + INT8 TC MatMul
            float* norm_out = (float*)malloc(K * N * sizeof(float));
            if (!norm_out) return -1;

            int ret = rmsnorm_cuda_persistent(norm_out, input, N, K, eps);
            if (ret != 0) {
                free(norm_out);
                return ret;
            }

            ret = tmac_matmul_cuda_int8_tc(norm_out, output, M, N, K);
            free(norm_out);
            return ret;
        }
    }

    // Fall back to previous adaptive dispatch
    return fused_rmsnorm_matmul_cuda_adaptive(input, output, M, N, K, eps);
}

/**
 * Multi-GPU initialization (stub for future implementation)
 */
int cuda_init_multi_gpu(int num_gpus) {
    int available = cuda_get_device_count();
    if (num_gpus > available) {
        fprintf(stderr, "Requested %d GPUs but only %d available.\n", num_gpus, available);
        return -1;
    }

    printf("EdgeLLM: Multi-GPU support initialized with %d devices\n", num_gpus);
    // TODO: Implement tensor parallelism across GPUs
    return 0;
}

} // extern "C"
