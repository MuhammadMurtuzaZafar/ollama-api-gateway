/**
 * INT8 Embedding Kernels for EdgeLLM
 *
 * Optimized for T4 GPU (sm_75)
 * Target: Replace 3.6ms FP32 logit GEMV with faster INT8 version
 *
 * Embedding table: [vocab_size, dim] = [151936, 1536]
 * FP32: 894 MB -> INT8: 224 MB (4x reduction)
 *
 * Operations:
 * 1. Token embedding lookup: out[dim] = embedding[token_id, :]
 * 2. Logit GEMV: logits[vocab] = embedding @ hidden_state
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/**
 * INT8 Embedding Lookup Kernel
 *
 * Dequantizes a single row from INT8 embedding table to FP32 output.
 * Each row has its own FP16 scale for quality.
 */
__global__ void int8_embedding_lookup_kernel(
    float* __restrict__ out,           // [dim]
    const int8_t* __restrict__ emb_int8, // [vocab_size, dim]
    const half* __restrict__ scales,   // [vocab_size]
    int token_id,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float scale = __half2float(scales[token_id]);
    int8_t val = emb_int8[token_id * dim + i];
    out[i] = (float)val * scale;
}

/**
 * INT8 Logit GEMV Kernel
 *
 * Computes logits = embedding^T @ x (vocab_size outputs)
 * Uses INT8 quantized embeddings with per-row scales.
 *
 * This is the bottleneck operation (currently 3.6ms with FP32 cuBLAS).
 * Target: <1ms with INT8.
 */
__global__ void int8_logit_gemv_kernel(
    float* __restrict__ logits,        // [vocab_size]
    const float* __restrict__ x,       // [dim] hidden state
    const int8_t* __restrict__ emb_int8, // [vocab_size, dim]
    const half* __restrict__ scales,   // [vocab_size]
    int vocab_size,
    int dim
) {
    int row = blockIdx.x;  // Each block handles one vocab entry
    if (row >= vocab_size) return;

    extern __shared__ float smem[];
    float* s_x = smem;

    // Load hidden state to shared memory
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        s_x[i] = x[i];
    }
    __syncthreads();

    // Compute dot product
    float sum = 0.0f;
    float scale = __half2float(scales[row]);

    const int8_t* row_emb = emb_int8 + row * dim;

    // Process 8 elements at a time with FMA
    int n_oct = dim / 8;

    for (int p = threadIdx.x; p < n_oct; p += blockDim.x) {
        int base = p * 8;

        // Load 8 INT8 values (8 bytes = 2 uint32)
        uint32_t packed0 = *reinterpret_cast<const uint32_t*>(&row_emb[base]);
        uint32_t packed1 = *reinterpret_cast<const uint32_t*>(&row_emb[base + 4]);

        // Unpack INT8 values
        int8_t v0 = (int8_t)(packed0 & 0xFF);
        int8_t v1 = (int8_t)((packed0 >> 8) & 0xFF);
        int8_t v2 = (int8_t)((packed0 >> 16) & 0xFF);
        int8_t v3 = (int8_t)((packed0 >> 24) & 0xFF);
        int8_t v4 = (int8_t)(packed1 & 0xFF);
        int8_t v5 = (int8_t)((packed1 >> 8) & 0xFF);
        int8_t v6 = (int8_t)((packed1 >> 16) & 0xFF);
        int8_t v7 = (int8_t)((packed1 >> 24) & 0xFF);

        // FMA accumulation
        sum = fmaf(s_x[base + 0], (float)v0 * scale, sum);
        sum = fmaf(s_x[base + 1], (float)v1 * scale, sum);
        sum = fmaf(s_x[base + 2], (float)v2 * scale, sum);
        sum = fmaf(s_x[base + 3], (float)v3 * scale, sum);
        sum = fmaf(s_x[base + 4], (float)v4 * scale, sum);
        sum = fmaf(s_x[base + 5], (float)v5 * scale, sum);
        sum = fmaf(s_x[base + 6], (float)v6 * scale, sum);
        sum = fmaf(s_x[base + 7], (float)v7 * scale, sum);
    }

    // Handle remainder
    int remainder_start = n_oct * 8;
    for (int i = remainder_start + threadIdx.x; i < dim; i += blockDim.x) {
        sum = fmaf(s_x[i], (float)row_emb[i] * scale, sum);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block reduction
    __shared__ float s_partial[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        s_partial[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            logits[row] = sum;
        }
    }
}

/**
 * Optimized INT8 Logit GEMV using dp4a intrinsic
 *
 * dp4a computes 4 INT8 dot products in a single instruction.
 * This is the fastest path for INT8 matmul on Turing.
 */
__global__ void int8_logit_gemv_dp4a_kernel(
    float* __restrict__ logits,
    const float* __restrict__ x,       // FP32 input (will convert)
    const int8_t* __restrict__ emb_int8,
    const half* __restrict__ scales,
    int vocab_size,
    int dim
) {
    int row = blockIdx.x;
    if (row >= vocab_size) return;

    extern __shared__ char smem_raw[];
    int8_t* s_x_int8 = reinterpret_cast<int8_t*>(smem_raw);
    float* s_x_scale = reinterpret_cast<float*>(smem_raw + dim);

    // Convert input to INT8 with shared scale (simple quantization)
    // First pass: find max for scaling
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(x[i]));
    }

    // Reduce to find global max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = local_max;
    }
    __syncthreads();

    float x_scale = s_max / 127.0f;
    float x_scale_inv = (x_scale > 0) ? 1.0f / x_scale : 0.0f;

    // Quantize input to INT8
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        s_x_int8[i] = (int8_t)__float2int_rn(x[i] * x_scale_inv);
    }

    if (threadIdx.x == 0) {
        s_x_scale[0] = x_scale;
    }
    __syncthreads();

    // Compute dot product using dp4a
    int sum_int = 0;
    float emb_scale = __half2float(scales[row]);

    const int8_t* row_emb = emb_int8 + row * dim;

    // Process 4 elements at a time with dp4a
    int n_quads = dim / 4;

    for (int p = threadIdx.x; p < n_quads; p += blockDim.x) {
        int base = p * 4;

        // Load 4 INT8 values as int32
        int32_t a = *reinterpret_cast<const int32_t*>(&row_emb[base]);
        int32_t b = *reinterpret_cast<const int32_t*>(&s_x_int8[base]);

        // dp4a: sum += dot(a[0:3], b[0:3])
        sum_int = __dp4a(a, b, sum_int);
    }

    // Convert back to float with scales
    float sum = (float)sum_int * emb_scale * s_x_scale[0];

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block reduction
    __shared__ float s_partial[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        s_partial[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            logits[row] = sum;
        }
    }
}

extern "C" {

static cudaStream_t g_int8_stream = nullptr;

int int8_emb_init(cudaStream_t stream) {
    g_int8_stream = stream;
    return 0;
}

/**
 * INT8 embedding lookup
 */
int int8_embedding_lookup(
    float* out_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int token_id,
    int dim
) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;

    int8_embedding_lookup_kernel<<<blocks, threads, 0, g_int8_stream>>>(
        out_gpu, emb_int8_gpu, scales_gpu, token_id, dim
    );

    return 0;
}

/**
 * INT8 logit GEMV (standard version)
 */
int int8_logit_gemv(
    float* logits_gpu,
    const float* x_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int vocab_size,
    int dim
) {
    int smem_size = dim * sizeof(float);

    int8_logit_gemv_kernel<<<vocab_size, 256, smem_size, g_int8_stream>>>(
        logits_gpu, x_gpu, emb_int8_gpu, scales_gpu, vocab_size, dim
    );

    return 0;
}

/**
 * INT8 logit GEMV using dp4a (fastest on Turing)
 */
int int8_logit_gemv_dp4a(
    float* logits_gpu,
    const float* x_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int vocab_size,
    int dim
) {
    // Shared memory: INT8 input + scale
    int smem_size = dim * sizeof(int8_t) + sizeof(float);

    int8_logit_gemv_dp4a_kernel<<<vocab_size, 256, smem_size, g_int8_stream>>>(
        logits_gpu, x_gpu, emb_int8_gpu, scales_gpu, vocab_size, dim
    );

    return 0;
}

/**
 * Benchmark INT8 logit GEMV
 */
float int8_logit_benchmark(
    float* logits_gpu,
    const float* x_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int vocab_size,
    int dim,
    int n_iters,
    int use_dp4a
) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        if (use_dp4a) {
            int8_logit_gemv_dp4a(logits_gpu, x_gpu, emb_int8_gpu, scales_gpu, vocab_size, dim);
        } else {
            int8_logit_gemv(logits_gpu, x_gpu, emb_int8_gpu, scales_gpu, vocab_size, dim);
        }
    }
    cudaStreamSynchronize(g_int8_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, g_int8_stream);
    for (int i = 0; i < n_iters; i++) {
        if (use_dp4a) {
            int8_logit_gemv_dp4a(logits_gpu, x_gpu, emb_int8_gpu, scales_gpu, vocab_size, dim);
        } else {
            int8_logit_gemv(logits_gpu, x_gpu, emb_int8_gpu, scales_gpu, vocab_size, dim);
        }
    }
    cudaEventRecord(stop, g_int8_stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / n_iters;  // milliseconds per call
}

}  // extern "C"
