/**
 * EdgeLLM Residual Add CUDA Kernels
 *
 * High-performance element-wise operations for transformer residual connections.
 * Essential for pre-norm and post-norm transformer architectures.
 *
 * Operations:
 * - residual_add: x = x + residual
 * - residual_add_scale: x = x + alpha * residual
 * - fused_residual_norm: x = rmsnorm(x + residual)
 *
 * Optimizations:
 * - Vectorized float4 access (4x memory bandwidth)
 * - In-place operation to minimize memory footprint
 * - Fused variants to reduce kernel launch overhead
 *
 * Author: EdgeLLM Team
 * Date: January 2026
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// =============================================================================
// Basic Residual Add Kernels
// =============================================================================

/**
 * Element-wise residual addition
 * x[i] = x[i] + residual[i]
 */
__global__ void residual_add_kernel(
    float* __restrict__ x,
    const float* __restrict__ residual,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += residual[idx];
    }
}

/**
 * Vectorized residual addition (float4 = 4x throughput)
 */
__global__ void residual_add_vec4_kernel(
    float* __restrict__ x,
    const float* __restrict__ residual,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 x_vec = *reinterpret_cast<float4*>(x + idx);
        float4 r_vec = *reinterpret_cast<const float4*>(residual + idx);

        x_vec.x += r_vec.x;
        x_vec.y += r_vec.y;
        x_vec.z += r_vec.z;
        x_vec.w += r_vec.w;

        *reinterpret_cast<float4*>(x + idx) = x_vec;
    } else if (idx < size) {
        // Handle remainder
        for (int i = 0; i < 4 && idx + i < size; i++) {
            x[idx + i] += residual[idx + i];
        }
    }
}

/**
 * Scaled residual addition
 * x[i] = x[i] + alpha * residual[i]
 */
__global__ void residual_add_scale_kernel(
    float* __restrict__ x,
    const float* __restrict__ residual,
    float alpha,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += alpha * residual[idx];
    }
}

/**
 * Out-of-place residual addition
 * output[i] = a[i] + b[i]
 */
__global__ void residual_add_out_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

/**
 * Vectorized out-of-place residual addition
 */
__global__ void residual_add_out_vec4_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 a_vec = *reinterpret_cast<const float4*>(a + idx);
        float4 b_vec = *reinterpret_cast<const float4*>(b + idx);

        float4 out_vec;
        out_vec.x = a_vec.x + b_vec.x;
        out_vec.y = a_vec.y + b_vec.y;
        out_vec.z = a_vec.z + b_vec.z;
        out_vec.w = a_vec.w + b_vec.w;

        *reinterpret_cast<float4*>(output + idx) = out_vec;
    } else if (idx < size) {
        for (int i = 0; i < 4 && idx + i < size; i++) {
            output[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// =============================================================================
// Fused Residual + RMSNorm Kernel
// =============================================================================

#define WARP_SIZE 32

/**
 * Fused residual add + RMSNorm
 *
 * output = rmsnorm(x + residual)
 *
 * Fusing these operations eliminates one memory round-trip.
 * This is the common pattern in pre-norm transformers:
 * - After attention: output = rmsnorm(attn_output + residual)
 * - After FFN: output = rmsnorm(ffn_output + residual)
 */
__global__ void residual_add_rmsnorm_kernel(
    float* __restrict__ output,
    float* __restrict__ residual_out,  // Optional: save x + residual
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    int batch_size,
    int hidden_dim,
    float eps
) {
    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x_row = x + batch_idx * hidden_dim;
    const float* r_row = residual + batch_idx * hidden_dim;
    float* out_row = output + batch_idx * hidden_dim;

    // Step 1: Compute sum of squares for RMS
    float local_sum_sq = 0.0f;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = x_row[i] + r_row[i];

        // Optionally save the residual sum
        if (residual_out != nullptr) {
            residual_out[batch_idx * hidden_dim + i] = val;
        }

        local_sum_sq += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    // Store warp results
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (tid < num_warps) {
        local_sum_sq = s_sum_sq[tid];
    } else {
        local_sum_sq = 0.0f;
    }

    if (tid < WARP_SIZE) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }
    }

    // Broadcast RMS scale
    __shared__ float s_rms_scale;
    if (tid == 0) {
        float rms = sqrtf(local_sum_sq / hidden_dim + eps);
        s_rms_scale = 1.0f / rms;
    }
    __syncthreads();

    float rms_scale = s_rms_scale;

    // Step 2: Apply normalization
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = x_row[i] + r_row[i];
        out_row[i] = val * rms_scale * weight[i];
    }
}

/**
 * Vectorized fused residual + RMSNorm
 */
__global__ void residual_add_rmsnorm_vec4_kernel(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    int batch_size,
    int hidden_dim,
    float eps
) {
    extern __shared__ float shared[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x_row = x + batch_idx * hidden_dim;
    const float* r_row = residual + batch_idx * hidden_dim;
    float* out_row = output + batch_idx * hidden_dim;

    int vec_dim = hidden_dim / 4;

    // Sum of squares with vectorized loads
    float local_sum_sq = 0.0f;

    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 x_vec = reinterpret_cast<const float4*>(x_row)[i];
        float4 r_vec = reinterpret_cast<const float4*>(r_row)[i];

        float4 sum_vec;
        sum_vec.x = x_vec.x + r_vec.x;
        sum_vec.y = x_vec.y + r_vec.y;
        sum_vec.z = x_vec.z + r_vec.z;
        sum_vec.w = x_vec.w + r_vec.w;

        local_sum_sq += sum_vec.x * sum_vec.x;
        local_sum_sq += sum_vec.y * sum_vec.y;
        local_sum_sq += sum_vec.z * sum_vec.z;
        local_sum_sq += sum_vec.w * sum_vec.w;
    }

    // Handle remainder
    for (int i = vec_dim * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float val = x_row[i] + r_row[i];
        local_sum_sq += val * val;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    if (lane_id == 0) {
        shared[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (tid < num_warps) {
        local_sum_sq = shared[tid];
    } else {
        local_sum_sq = 0.0f;
    }

    if (tid < WARP_SIZE) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }
    }

    __shared__ float s_rms_scale;
    if (tid == 0) {
        float rms = sqrtf(local_sum_sq / hidden_dim + eps);
        s_rms_scale = 1.0f / rms;
    }
    __syncthreads();

    float rms_scale = s_rms_scale;

    // Apply normalization with vectorized writes
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 x_vec = reinterpret_cast<const float4*>(x_row)[i];
        float4 r_vec = reinterpret_cast<const float4*>(r_row)[i];
        float4 w_vec = reinterpret_cast<const float4*>(weight)[i];

        float4 out_vec;
        out_vec.x = (x_vec.x + r_vec.x) * rms_scale * w_vec.x;
        out_vec.y = (x_vec.y + r_vec.y) * rms_scale * w_vec.y;
        out_vec.z = (x_vec.z + r_vec.z) * rms_scale * w_vec.z;
        out_vec.w = (x_vec.w + r_vec.w) * rms_scale * w_vec.w;

        reinterpret_cast<float4*>(out_row)[i] = out_vec;
    }

    // Handle remainder
    for (int i = vec_dim * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float val = x_row[i] + r_row[i];
        out_row[i] = val * rms_scale * weight[i];
    }
}

// =============================================================================
// Host Wrapper Functions
// =============================================================================

extern "C" {

/**
 * In-place residual addition
 *
 * x = x + residual
 *
 * @param x         Tensor to update [batch_size, hidden_dim]
 * @param residual  Residual tensor [batch_size, hidden_dim]
 * @param size      Total number of elements
 * @param stream    CUDA stream
 */
int residual_add_f32(
    float* x,
    const float* residual,
    int size,
    cudaStream_t stream
) {
    // Use vectorized kernel if size is large and aligned
    if (size >= 1024 && size % 4 == 0) {
        int vec_size = size / 4;
        int block = 256;
        int grid = (vec_size + block - 1) / block;

        residual_add_vec4_kernel<<<grid, block, 0, stream>>>(x, residual, size);
    } else {
        int block = 256;
        int grid = (size + block - 1) / block;

        residual_add_kernel<<<grid, block, 0, stream>>>(x, residual, size);
    }

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/**
 * Scaled residual addition
 *
 * x = x + alpha * residual
 */
int residual_add_scale_f32(
    float* x,
    const float* residual,
    float alpha,
    int size,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (size + block - 1) / block;

    residual_add_scale_kernel<<<grid, block, 0, stream>>>(x, residual, alpha, size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/**
 * Out-of-place residual addition
 *
 * output = a + b
 */
int residual_add_out_f32(
    float* output,
    const float* a,
    const float* b,
    int size,
    cudaStream_t stream
) {
    if (size >= 1024 && size % 4 == 0) {
        int vec_size = size / 4;
        int block = 256;
        int grid = (vec_size + block - 1) / block;

        residual_add_out_vec4_kernel<<<grid, block, 0, stream>>>(output, a, b, size);
    } else {
        int block = 256;
        int grid = (size + block - 1) / block;

        residual_add_out_kernel<<<grid, block, 0, stream>>>(output, a, b, size);
    }

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/**
 * Fused residual add + RMSNorm
 *
 * output = rmsnorm(x + residual, weight, eps)
 *
 * @param output       Normalized output [batch_size, hidden_dim]
 * @param residual_out Optional: save x + residual (NULL to skip)
 * @param x            Input tensor [batch_size, hidden_dim]
 * @param residual     Residual tensor [batch_size, hidden_dim]
 * @param weight       RMSNorm weight [hidden_dim]
 * @param batch_size   Batch size
 * @param hidden_dim   Hidden dimension
 * @param eps          Epsilon for numerical stability
 * @param stream       CUDA stream
 */
int residual_add_rmsnorm_f32(
    float* output,
    float* residual_out,
    const float* x,
    const float* residual,
    const float* weight,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    int block = 256;
    int num_warps = block / WARP_SIZE;
    int smem_size = num_warps * sizeof(float);

    // Use vectorized kernel for large hidden dims
    if (hidden_dim >= 256 && hidden_dim % 4 == 0) {
        residual_add_rmsnorm_vec4_kernel<<<batch_size, block, smem_size, stream>>>(
            output, x, residual, weight, batch_size, hidden_dim, eps
        );
    } else {
        residual_add_rmsnorm_kernel<<<batch_size, block, smem_size, stream>>>(
            output, residual_out, x, residual, weight, batch_size, hidden_dim, eps
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/**
 * Batch residual add for multiple tensors
 *
 * For processing residuals at multiple layers efficiently.
 */
int residual_add_batch_f32(
    float** x_tensors,
    const float** residual_tensors,
    int num_tensors,
    int size_per_tensor,
    cudaStream_t stream
) {
    for (int i = 0; i < num_tensors; i++) {
        int ret = residual_add_f32(x_tensors[i], residual_tensors[i], size_per_tensor, stream);
        if (ret != 0) return ret;
    }
    return 0;
}

} // extern "C"
