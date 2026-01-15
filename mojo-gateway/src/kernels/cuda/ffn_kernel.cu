/**
 * EdgeLLM FFN/MLP CUDA Kernel
 *
 * High-performance Feed-Forward Network with SwiGLU activation.
 * Optimized for LLaMA-style transformer models.
 *
 * SwiGLU: output = (xW1 * silu(xW_gate)) @ W2
 *
 * Performance optimizations:
 * - Fused gate and up projection
 * - Vectorized memory access
 * - Tiled matrix multiplication
 * - Warp-level primitives
 *
 * Author: EdgeLLM Team
 * Date: January 2026
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// Tile sizes for matrix multiplication
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define WARP_SIZE 32

/**
 * SiLU (Swish) activation function
 * silu(x) = x * sigmoid(x)
 */
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ half silu_half(half x) {
    float fx = __half2float(x);
    return __float2half(fx / (1.0f + expf(-fx)));
}

/**
 * Fused SwiGLU Kernel
 *
 * Computes: output = silu(x @ W_gate) * (x @ W_up)
 *
 * This fuses the gate and up projections with SiLU activation
 * to minimize memory bandwidth.
 *
 * @param output Output tensor [batch_size, intermediate_dim]
 * @param input Input tensor [batch_size, hidden_dim]
 * @param w_gate Gate weight matrix [hidden_dim, intermediate_dim]
 * @param w_up Up projection weight matrix [hidden_dim, intermediate_dim]
 * @param batch_size Batch size
 * @param hidden_dim Input dimension
 * @param intermediate_dim Intermediate dimension (usually 4x hidden_dim)
 */
__global__ void swiglu_fused_kernel_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    int batch_size,
    int hidden_dim,
    int intermediate_dim
) {
    int row = blockIdx.x;  // batch index
    int col = blockIdx.y * blockDim.x + threadIdx.x;  // output column

    if (row >= batch_size || col >= intermediate_dim) return;

    const float* x = input + row * hidden_dim;

    // Compute gate projection: sum(x * w_gate[:, col])
    float gate_val = 0.0f;
    float up_val = 0.0f;

    // Vectorized accumulation
    int vec_dim = hidden_dim / 4 * 4;

    for (int k = 0; k < vec_dim; k += 4) {
        float4 x_vec = *reinterpret_cast<const float4*>(x + k);

        // Gate weights
        float4 wg;
        wg.x = w_gate[k * intermediate_dim + col];
        wg.y = w_gate[(k + 1) * intermediate_dim + col];
        wg.z = w_gate[(k + 2) * intermediate_dim + col];
        wg.w = w_gate[(k + 3) * intermediate_dim + col];

        gate_val += x_vec.x * wg.x + x_vec.y * wg.y + x_vec.z * wg.z + x_vec.w * wg.w;

        // Up weights
        float4 wu;
        wu.x = w_up[k * intermediate_dim + col];
        wu.y = w_up[(k + 1) * intermediate_dim + col];
        wu.z = w_up[(k + 2) * intermediate_dim + col];
        wu.w = w_up[(k + 3) * intermediate_dim + col];

        up_val += x_vec.x * wu.x + x_vec.y * wu.y + x_vec.z * wu.z + x_vec.w * wu.w;
    }

    // Handle remaining elements
    for (int k = vec_dim; k < hidden_dim; k++) {
        float xk = x[k];
        gate_val += xk * w_gate[k * intermediate_dim + col];
        up_val += xk * w_up[k * intermediate_dim + col];
    }

    // Apply SwiGLU: silu(gate) * up
    output[row * intermediate_dim + col] = silu(gate_val) * up_val;
}

/**
 * Down Projection Kernel
 *
 * Computes: output = input @ W_down
 *
 * @param output Output tensor [batch_size, hidden_dim]
 * @param input Input tensor [batch_size, intermediate_dim]
 * @param w_down Down projection weight [intermediate_dim, hidden_dim]
 * @param batch_size Batch size
 * @param intermediate_dim Intermediate dimension
 * @param hidden_dim Output dimension
 */
__global__ void down_proj_kernel_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ w_down,
    int batch_size,
    int intermediate_dim,
    int hidden_dim
) {
    int row = blockIdx.x;  // batch index
    int col = blockIdx.y * blockDim.x + threadIdx.x;  // output column

    if (row >= batch_size || col >= hidden_dim) return;

    const float* x = input + row * intermediate_dim;

    float sum = 0.0f;

    // Vectorized accumulation
    int vec_dim = intermediate_dim / 4 * 4;

    for (int k = 0; k < vec_dim; k += 4) {
        float4 x_vec = *reinterpret_cast<const float4*>(x + k);

        float4 w;
        w.x = w_down[k * hidden_dim + col];
        w.y = w_down[(k + 1) * hidden_dim + col];
        w.z = w_down[(k + 2) * hidden_dim + col];
        w.w = w_down[(k + 3) * hidden_dim + col];

        sum += x_vec.x * w.x + x_vec.y * w.y + x_vec.z * w.z + x_vec.w * w.w;
    }

    // Handle remaining elements
    for (int k = vec_dim; k < intermediate_dim; k++) {
        sum += x[k] * w_down[k * hidden_dim + col];
    }

    output[row * hidden_dim + col] = sum;
}

/**
 * Tiled SwiGLU Kernel with Shared Memory
 *
 * Uses shared memory tiling for better memory access patterns
 */
__global__ void swiglu_tiled_kernel_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    int batch_size,
    int hidden_dim,
    int intermediate_dim
) {
    __shared__ float tile_x[TILE_K];
    __shared__ float tile_wg[TILE_K];
    __shared__ float tile_wu[TILE_K];

    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= intermediate_dim) return;

    float gate_val = 0.0f;
    float up_val = 0.0f;

    // Process in tiles
    for (int tile = 0; tile < hidden_dim; tile += TILE_K) {
        // Load tile of input
        if (threadIdx.x < TILE_K && tile + threadIdx.x < hidden_dim) {
            tile_x[threadIdx.x] = input[row * hidden_dim + tile + threadIdx.x];
        }
        __syncthreads();

        // Accumulate
        int tile_end = min(TILE_K, hidden_dim - tile);
        for (int k = 0; k < tile_end; k++) {
            float xk = tile_x[k];
            int global_k = tile + k;
            gate_val += xk * w_gate[global_k * intermediate_dim + col];
            up_val += xk * w_up[global_k * intermediate_dim + col];
        }
        __syncthreads();
    }

    output[row * intermediate_dim + col] = silu(gate_val) * up_val;
}

/**
 * Fused FFN Kernel (SwiGLU + Down Projection)
 *
 * Combines both SwiGLU and down projection for models with small intermediate dim
 */
__global__ void ffn_fused_small_kernel_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    int batch_size,
    int hidden_dim,
    int intermediate_dim
) {
    extern __shared__ float shared_intermediate[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch_size) return;

    const float* x = input + row * hidden_dim;

    // Step 1: Compute SwiGLU for each intermediate dimension
    for (int i = tid; i < intermediate_dim; i += blockDim.x) {
        float gate_val = 0.0f;
        float up_val = 0.0f;

        for (int k = 0; k < hidden_dim; k++) {
            float xk = x[k];
            gate_val += xk * w_gate[k * intermediate_dim + i];
            up_val += xk * w_up[k * intermediate_dim + i];
        }

        shared_intermediate[i] = silu(gate_val) * up_val;
    }
    __syncthreads();

    // Step 2: Down projection
    float* out = output + row * hidden_dim;
    for (int j = tid; j < hidden_dim; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < intermediate_dim; i++) {
            sum += shared_intermediate[i] * w_down[i * hidden_dim + j];
        }
        out[j] = sum;
    }
}

/**
 * INT8 Quantized SwiGLU Kernel
 *
 * Uses INT8 weights with FP32 accumulation for memory efficiency
 * Requires compute capability 6.1+ for __dp4a intrinsic
 * Note: Build with DP4A_CUDA_ARCH (sm_61+) to ensure __dp4a availability
 */
__global__ void swiglu_int8_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int8_t* __restrict__ w_gate_int8,
    const int8_t* __restrict__ w_up_int8,
    const float* __restrict__ scale_gate,
    const float* __restrict__ scale_up,
    int batch_size,
    int hidden_dim,
    int intermediate_dim
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= intermediate_dim) return;

    const float* x = input + row * hidden_dim;
    float sg = scale_gate[col];
    float su = scale_up[col];

    int32_t gate_acc = 0;
    int32_t up_acc = 0;

    // Use dp4a for INT8 dot product
    int vec_dim = hidden_dim / 4;

    for (int k = 0; k < vec_dim; k++) {
        // Pack 4 floats to approximate INT8
        int32_t x_packed = 0;
        for (int i = 0; i < 4; i++) {
            int8_t xi = (int8_t)__float2int_rn(x[k * 4 + i] * 127.0f);
            x_packed |= ((int32_t)(uint8_t)xi) << (i * 8);
        }

        int32_t wg_packed = *reinterpret_cast<const int32_t*>(w_gate_int8 + (k * 4) * intermediate_dim + col * 4);
        int32_t wu_packed = *reinterpret_cast<const int32_t*>(w_up_int8 + (k * 4) * intermediate_dim + col * 4);

        gate_acc = __dp4a(x_packed, wg_packed, gate_acc);
        up_acc = __dp4a(x_packed, wu_packed, up_acc);
    }

    // Scale and apply SwiGLU
    float gate_val = (float)gate_acc * sg / (127.0f * 127.0f);
    float up_val = (float)up_acc * su / (127.0f * 127.0f);

    output[row * intermediate_dim + col] = silu(gate_val) * up_val;
}

// =============================================================================
// Host wrapper functions
// =============================================================================

extern "C" {

/**
 * Launch SwiGLU (gate + up projection with SiLU) kernel
 */
void swiglu_f32(
    float* output,
    const float* input,
    const float* w_gate,
    const float* w_up,
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size, (intermediate_dim + 255) / 256);

    swiglu_fused_kernel_f32<<<grid, block, 0, stream>>>(
        output, input, w_gate, w_up,
        batch_size, hidden_dim, intermediate_dim
    );
}

/**
 * Launch down projection kernel
 */
void down_proj_f32(
    float* output,
    const float* input,
    const float* w_down,
    int batch_size,
    int intermediate_dim,
    int hidden_dim,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size, (hidden_dim + 255) / 256);

    down_proj_kernel_f32<<<grid, block, 0, stream>>>(
        output, input, w_down,
        batch_size, intermediate_dim, hidden_dim
    );
}

/**
 * Launch full FFN (SwiGLU + down projection)
 *
 * For models with small intermediate dim, uses fused kernel
 * For larger models, uses separate kernels with intermediate buffer
 */
void ffn_swiglu_f32(
    float* output,
    float* intermediate,  // Temporary buffer [batch_size, intermediate_dim]
    const float* input,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream
) {
    // For small models, use fused kernel
    if (intermediate_dim <= 2048 && batch_size == 1) {
        int block_size = 256;
        int shared_mem = intermediate_dim * sizeof(float);

        ffn_fused_small_kernel_f32<<<batch_size, block_size, shared_mem, stream>>>(
            output, input, w_gate, w_up, w_down,
            batch_size, hidden_dim, intermediate_dim
        );
    } else {
        // Step 1: SwiGLU
        swiglu_f32(intermediate, input, w_gate, w_up,
                   batch_size, hidden_dim, intermediate_dim, stream);

        // Step 2: Down projection
        down_proj_f32(output, intermediate, w_down,
                      batch_size, intermediate_dim, hidden_dim, stream);
    }
}

/**
 * Launch INT8 quantized SwiGLU kernel
 */
void swiglu_int8(
    float* output,
    const float* input,
    const int8_t* w_gate_int8,
    const int8_t* w_up_int8,
    const float* scale_gate,
    const float* scale_up,
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size, (intermediate_dim + 255) / 256);

    swiglu_int8_kernel<<<grid, block, 0, stream>>>(
        output, input, w_gate_int8, w_up_int8, scale_gate, scale_up,
        batch_size, hidden_dim, intermediate_dim
    );
}

} // extern "C"
