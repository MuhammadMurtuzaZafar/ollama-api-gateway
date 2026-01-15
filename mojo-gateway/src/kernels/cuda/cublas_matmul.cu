/**
 * cuBLAS Matrix Multiplication Kernels for EdgeLLM
 *
 * This is the KEY to 400+ tok/s performance.
 * Matmuls are 90%+ of LLM inference compute.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_fp16.h>

// INT4 kernel declarations (from int4_gemv.cu)
extern "C" int int4_init(cudaStream_t stream);
extern "C" int int4_gemv(float* out, const float* x, const unsigned char* W, const half* scales, int out_dim, int in_dim);
extern "C" void int4_sync();

// Forward declarations (for internal use - must be extern "C" for compatibility)
extern "C" int cublas_init_int4_sizes(size_t fp32_bytes, size_t int4_weights_bytes, size_t int4_scales_bytes, size_t activation_bytes);
extern "C" int gpu_configure_int4(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int head_dim, int kv_dim, int group_size);

// Global cuBLAS handle - reuse across calls
static cublasHandle_t g_cublas_handle = nullptr;
static cudaStream_t g_stream = nullptr;

// Persistent GPU buffers for weights and activations
static float* g_weights_gpu = nullptr;
static size_t g_weights_size = 0;
static float* g_act_gpu = nullptr;
static size_t g_act_size = 0;

extern "C" {

/**
 * Initialize cuBLAS and allocate GPU memory for model weights.
 * Call once at startup.
 */
int cublas_init(size_t weight_bytes, size_t activation_bytes) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS init failed: %d\n", status);
        return -1;
    }

    cudaStreamCreate(&g_stream);
    cublasSetStream(g_cublas_handle, g_stream);

    // Use Tensor Cores when available (TF32 on Ampere+)
    cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);

    // Allocate GPU memory for weights (loaded once)
    if (weight_bytes > 0) {
        cudaMalloc(&g_weights_gpu, weight_bytes);
        g_weights_size = weight_bytes;
        printf("Allocated %.2f GB for weights on GPU\n", weight_bytes / 1e9);
    }

    // Allocate GPU memory for activations (reused each forward pass)
    if (activation_bytes > 0) {
        cudaMalloc(&g_act_gpu, activation_bytes);
        g_act_size = activation_bytes;
        printf("Allocated %.2f MB for activations on GPU\n", activation_bytes / 1e6);
    }

    return 0;
}

/**
 * Upload model weights to GPU (call once after loading model).
 */
int cublas_upload_weights(const float* weights_cpu, size_t bytes) {
    if (!g_weights_gpu || bytes > g_weights_size) {
        printf("Weight buffer not allocated or too small\n");
        return -1;
    }

    cudaError_t err = cudaMemcpy(g_weights_gpu, weights_cpu, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Weight upload failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Uploaded %.2f GB weights to GPU\n", bytes / 1e9);
    return 0;
}

/**
 * Matrix-vector multiplication: out = W @ x
 * W is [out_dim, in_dim], x is [in_dim], out is [out_dim]
 *
 * This is the core operation - called millions of times per second.
 */
int cublas_matvec(
    float* out_gpu,      // Output [out_dim]
    const float* x_gpu,  // Input [in_dim]
    const float* W_gpu,  // Weight [out_dim, in_dim]
    int out_dim,
    int in_dim
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // GEMV: y = alpha * A * x + beta * y
    // A is out_dim x in_dim (row-major in C, so we use CUBLAS_OP_T)
    cublasStatus_t status = cublasSgemv(
        g_cublas_handle,
        CUBLAS_OP_T,        // Transpose because row-major
        in_dim,             // rows of A
        out_dim,            // cols of A
        &alpha,
        W_gpu, in_dim,      // A, lda
        x_gpu, 1,           // x, incx
        &beta,
        out_gpu, 1          // y, incy
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/**
 * Batched matrix multiplication for all layers at once.
 * Uses strided batched GEMM for maximum throughput.
 */
int cublas_matvec_batched(
    float* out_gpu,      // Output [batch, out_dim]
    const float* x_gpu,  // Input [batch, in_dim]
    const float* W_gpu,  // Weight [batch, out_dim, in_dim]
    int batch,           // Number of layers
    int out_dim,
    int in_dim
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    long long strideA = (long long)out_dim * in_dim;
    long long strideB = in_dim;
    long long strideC = out_dim;

    // Strided batched GEMV (treated as GEMM with n=1)
    cublasStatus_t status = cublasSgemmStridedBatched(
        g_cublas_handle,
        CUBLAS_OP_T,        // A transposed (row-major)
        CUBLAS_OP_N,        // B not transposed
        out_dim,            // m
        1,                  // n (single vector)
        in_dim,             // k
        &alpha,
        W_gpu, in_dim, strideA,  // A, lda, strideA
        x_gpu, in_dim, strideB,  // B, ldb, strideB
        &beta,
        out_gpu, out_dim, strideC,  // C, ldc, strideC
        batch
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/**
 * Add bias to output vector (in-place).
 */
__global__ void add_bias_kernel(float* out, const float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] += bias[idx];
    }
}

int cublas_add_bias(float* out_gpu, const float* bias_gpu, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, g_stream>>>(out_gpu, bias_gpu, size);
    return 0;
}

/**
 * RMSNorm on GPU with proper block reduction.
 */
__global__ void rmsnorm_kernel(
    float* out,
    const float* x,
    const float* weight,
    int size,
    float eps
) {
    __shared__ float s_partial[32];  // For warp results

    // Compute local sum of squares
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float v = x[i];
        local_ss += v * v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
    }

    // Store warp results
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_partial[warp_id] = local_ss;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        local_ss = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
        }
        if (lane_id == 0) {
            s_partial[0] = rsqrtf(local_ss / size + eps);
        }
    }
    __syncthreads();

    // Normalize and scale
    float scale = s_partial[0];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = weight[i] * x[i] * scale;
    }
}

int gpu_rmsnorm(
    float* out_gpu,
    const float* x_gpu,
    const float* weight_gpu,
    int size,
    float eps
) {
    rmsnorm_kernel<<<1, 256, 0, g_stream>>>(out_gpu, x_gpu, weight_gpu, size, eps);
    return 0;
}

/**
 * SwiGLU activation: out = silu(gate) * up
 */
__global__ void swiglu_kernel(float* out, const float* gate, const float* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));  // SiLU = x * sigmoid(x)
        out[idx] = silu_g * up[idx];
    }
}

int gpu_swiglu(float* out_gpu, const float* gate_gpu, const float* up_gpu, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, g_stream>>>(out_gpu, gate_gpu, up_gpu, size);
    return 0;
}

/**
 * Residual add: x += residual
 */
__global__ void residual_add_kernel(float* x, const float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += residual[idx];
    }
}

int gpu_residual_add(float* x_gpu, const float* residual_gpu, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, g_stream>>>(x_gpu, residual_gpu, size);
    return 0;
}

/**
 * RoPE (Rotary Position Embedding) on GPU.
 */
__global__ void rope_kernel(
    float* q,           // [n_heads, head_dim]
    float* k,           // [n_kv_heads, head_dim]
    const float* cos,   // [head_dim/2]
    const float* sin,   // [head_dim/2]
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int head = blockIdx.x;
    int j = threadIdx.x * 2;  // Process pairs

    if (j >= head_dim) return;

    float c = cos[j / 2];
    float s = sin[j / 2];

    // Apply RoPE to Q
    if (head < n_heads) {
        int idx = head * head_dim + j;
        float q0 = q[idx];
        float q1 = q[idx + 1];
        q[idx] = q0 * c - q1 * s;
        q[idx + 1] = q0 * s + q1 * c;
    }

    // Apply RoPE to K (fewer heads)
    if (head < n_kv_heads) {
        int idx = head * head_dim + j;
        float k0 = k[idx];
        float k1 = k[idx + 1];
        k[idx] = k0 * c - k1 * s;
        k[idx + 1] = k0 * s + k1 * c;
    }
}

int gpu_rope(
    float* q_gpu,
    float* k_gpu,
    const float* cos_gpu,
    const float* sin_gpu,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;
    rope_kernel<<<max_heads, head_dim / 2, 0, g_stream>>>(
        q_gpu, k_gpu, cos_gpu, sin_gpu, n_heads, n_kv_heads, head_dim
    );
    return 0;
}

/**
 * GQA (Grouped Query Attention) decode kernel with proper block reduction.
 * This handles n_heads != n_kv_heads properly.
 *
 * For Qwen 1.5B: 12 Q heads, 2 KV heads, kv_mul = 6
 */
__global__ void gqa_attention_kernel(
    float* output,           // [n_heads, head_dim]
    const float* Q,          // [n_heads, head_dim]
    const float* K_cache,    // [n_kv_heads, max_seq, head_dim]
    const float* V_cache,    // [n_kv_heads, max_seq, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,             // Current sequence length (pos + 1)
    int max_seq,             // Maximum sequence length (stride)
    int head_dim,
    float scale
) {
    int q_head = blockIdx.x;
    int kv_head = q_head / (n_heads / n_kv_heads);  // Which KV head this Q uses

    extern __shared__ float smem[];
    float* s_scores = smem;                         // [seq_len]
    float* s_partial = smem + seq_len;              // [32] for warp results

    // 1. Compute attention scores: Q @ K^T
    float local_max = -1e10f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float score = 0.0f;
        int k_offset = kv_head * max_seq * head_dim + t * head_dim;
        int q_offset = q_head * head_dim;

        for (int d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K_cache[k_offset + d];
        }
        score *= scale;
        s_scores[t] = score;
        local_max = fmaxf(local_max, score);
    }

    // Block reduction for max
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) s_partial[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : -1e10f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (lane_id == 0) s_partial[0] = local_max;
    }
    __syncthreads();
    float max_score = s_partial[0];

    // 2. Softmax: exp and sum
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float exp_score = expf(s_scores[t] - max_score);
        s_scores[t] = exp_score;
        local_sum += exp_score;
    }

    // Block reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) s_partial[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane_id == 0) s_partial[0] = local_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_partial[0];
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        s_scores[t] *= inv_sum;
    }
    __syncthreads();

    // 3. Weighted sum of V
    int out_offset = q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            int v_offset = kv_head * max_seq * head_dim + t * head_dim + d;
            acc += s_scores[t] * V_cache[v_offset];
        }
        output[out_offset + d] = acc;
    }
}

int gpu_gqa_attention(
    float* output_gpu,
    const float* Q_gpu,
    const float* K_cache_gpu,
    const float* V_cache_gpu,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int max_seq,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int smem_size = seq_len * sizeof(float);

    // One block per Q head
    gqa_attention_kernel<<<n_heads, 128, smem_size, g_stream>>>(
        output_gpu, Q_gpu, K_cache_gpu, V_cache_gpu,
        n_heads, n_kv_heads, seq_len, max_seq, head_dim, scale
    );

    return 0;
}

/**
 * Copy current K, V to cache.
 */
__global__ void kv_cache_update_kernel(
    float* K_cache,      // [n_kv_heads, max_seq, head_dim]
    float* V_cache,
    const float* K,      // [n_kv_heads, head_dim]
    const float* V,
    int n_kv_heads,
    int pos,
    int max_seq,
    int head_dim
) {
    int head = blockIdx.x;
    int d = threadIdx.x;

    if (head < n_kv_heads && d < head_dim) {
        int cache_idx = head * max_seq * head_dim + pos * head_dim + d;
        int src_idx = head * head_dim + d;
        K_cache[cache_idx] = K[src_idx];
        V_cache[cache_idx] = V[src_idx];
    }
}

int gpu_kv_cache_update(
    float* K_cache_gpu,
    float* V_cache_gpu,
    const float* K_gpu,
    const float* V_gpu,
    int n_kv_heads,
    int pos,
    int max_seq,
    int head_dim
) {
    kv_cache_update_kernel<<<n_kv_heads, head_dim, 0, g_stream>>>(
        K_cache_gpu, V_cache_gpu, K_gpu, V_gpu,
        n_kv_heads, pos, max_seq, head_dim
    );
    return 0;
}

/**
 * Argmax for greedy sampling.
 */
__global__ void argmax_kernel(int* result, const float* logits, int size) {
    __shared__ float s_max_val[256];
    __shared__ int s_max_idx[256];

    int tid = threadIdx.x;
    float max_val = -1e10f;
    int max_idx = 0;

    for (int i = tid; i < size; i += blockDim.x) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    s_max_val[tid] = max_val;
    s_max_idx[tid] = max_idx;
    __syncthreads();

    // Reduce
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max_val[tid + s] > s_max_val[tid]) {
            s_max_val[tid] = s_max_val[tid + s];
            s_max_idx[tid] = s_max_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = s_max_idx[0];
    }
}

int gpu_argmax(int* result_gpu, const float* logits_gpu, int size) {
    argmax_kernel<<<1, 256, 0, g_stream>>>(result_gpu, logits_gpu, size);
    return 0;
}

/**
 * Synchronize stream.
 */
void cublas_sync() {
    cudaStreamSynchronize(g_stream);
}

/**
 * Cleanup.
 */
void cublas_cleanup() {
    if (g_weights_gpu) cudaFree(g_weights_gpu);
    if (g_act_gpu) cudaFree(g_act_gpu);
    if (g_stream) cudaStreamDestroy(g_stream);
    if (g_cublas_handle) cublasDestroy(g_cublas_handle);
    g_weights_gpu = nullptr;
    g_act_gpu = nullptr;
    g_stream = nullptr;
    g_cublas_handle = nullptr;
}

/**
 * Get pointers to GPU buffers.
 */
float* get_weights_gpu() { return g_weights_gpu; }
float* get_activations_gpu() { return g_act_gpu; }

// EAGLE-specific getters (for speculative decoding integration)
cublasHandle_t get_cublas_handle() { return g_cublas_handle; }
cudaStream_t get_cuda_stream() { return g_stream; }
// Note: INT4 getters are defined after INT4 variables (see get_int4_weights_gpu, get_int4_scales_gpu)

/**
 * CUDA memory operations (wrappers for FFI).
 */
int cuda_memcpy_d2d(float* dst, const float* src, size_t bytes) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, g_stream);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_d2h(float* dst_host, const float* src_device, size_t bytes) {
    cudaStreamSynchronize(g_stream);  // Ensure all ops complete first
    cudaError_t err = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_h2d(float* dst_device, const float* src_host, size_t bytes) {
    cudaError_t err = cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, g_stream);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Full transformer forward pass on GPU.
 * This is the main entry point - handles all memory management internally.
 */
// Model configuration (set once)
static int g_dim = 0;
static int g_hidden_dim = 0;
static int g_n_layers = 0;
static int g_n_heads = 0;
static int g_n_kv_heads = 0;
static int g_vocab_size = 0;
static int g_seq_len = 0;
static int g_head_dim = 0;
static int g_kv_dim = 0;
static bool g_has_bias = false;

// Weight offsets in global buffer
static size_t g_token_emb_offset = 0;
static size_t g_rms_att_offset = 0;
static size_t g_wq_offset = 0;
static size_t g_wk_offset = 0;
static size_t g_wv_offset = 0;
static size_t g_wo_offset = 0;
static size_t g_rms_ffn_offset = 0;
static size_t g_w1_offset = 0;
static size_t g_w2_offset = 0;
static size_t g_w3_offset = 0;
static size_t g_rms_final_offset = 0;
static size_t g_freq_cos_offset = 0;
static size_t g_freq_sin_offset = 0;
static size_t g_bq_offset = 0;
static size_t g_bk_offset = 0;
static size_t g_bv_offset = 0;

// Activation offsets in activation buffer
static size_t g_x_offset = 0;
static size_t g_xb_offset = 0;
static size_t g_xb2_offset = 0;
static size_t g_q_offset = 0;
static size_t g_k_offset = 0;
static size_t g_v_offset = 0;
static size_t g_hb_offset = 0;
static size_t g_hb2_offset = 0;
static size_t g_logits_offset = 0;
static size_t g_k_cache_offset = 0;
static size_t g_v_cache_offset = 0;
static size_t g_result_offset = 0;

// EAGLE config and offset getters
int get_model_dim() { return g_dim; }
int get_model_hidden_dim() { return g_hidden_dim; }
int get_model_n_layers() { return g_n_layers; }
int get_model_n_heads() { return g_n_heads; }
int get_model_n_kv_heads() { return g_n_kv_heads; }
int get_model_vocab_size() { return g_vocab_size; }
int get_model_seq_len() { return g_seq_len; }
int get_model_head_dim() { return g_head_dim; }
int get_model_kv_dim() { return g_kv_dim; }

size_t get_x_offset() { return g_x_offset; }
size_t get_xb_offset() { return g_xb_offset; }
size_t get_xb2_offset() { return g_xb2_offset; }
size_t get_logits_offset() { return g_logits_offset; }
size_t get_k_cache_offset() { return g_k_cache_offset; }
size_t get_v_cache_offset() { return g_v_cache_offset; }
size_t get_rms_final_offset() { return g_rms_final_offset; }

// Sampling integration getters (returns direct GPU pointers)
float* get_logits_gpu() { return g_act_gpu + g_logits_offset; }
int get_vocab_size() { return g_vocab_size; }

/**
 * Configure model dimensions. Call once after loading model.
 */
int gpu_configure(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                  int vocab_size, int seq_len, int has_bias) {
    g_dim = dim;
    g_hidden_dim = hidden_dim;
    g_n_layers = n_layers;
    g_n_heads = n_heads;
    g_n_kv_heads = n_kv_heads;
    g_vocab_size = vocab_size;
    g_seq_len = seq_len;
    g_head_dim = dim / n_heads;
    g_kv_dim = (n_kv_heads * dim) / n_heads;
    g_has_bias = (has_bias != 0);

    // Calculate weight offsets
    size_t offset = 0;
    g_token_emb_offset = offset; offset += vocab_size * dim;
    g_rms_att_offset = offset; offset += n_layers * dim;
    g_wq_offset = offset; offset += n_layers * dim * dim;
    g_wk_offset = offset; offset += n_layers * g_kv_dim * dim;
    g_wv_offset = offset; offset += n_layers * g_kv_dim * dim;
    g_wo_offset = offset; offset += n_layers * dim * dim;
    g_rms_ffn_offset = offset; offset += n_layers * dim;
    g_w1_offset = offset; offset += n_layers * hidden_dim * dim;
    g_w2_offset = offset; offset += n_layers * dim * hidden_dim;
    g_w3_offset = offset; offset += n_layers * hidden_dim * dim;
    g_rms_final_offset = offset; offset += dim;
    g_freq_cos_offset = offset; offset += seq_len * (g_head_dim / 2);
    g_freq_sin_offset = offset; offset += seq_len * (g_head_dim / 2);

    if (has_bias) {
        g_bq_offset = offset; offset += n_layers * dim;
        g_bk_offset = offset; offset += n_layers * g_kv_dim;
        g_bv_offset = offset; offset += n_layers * g_kv_dim;
    }

    // Calculate activation offsets
    size_t act_offset = 0;
    g_x_offset = act_offset; act_offset += dim;
    g_xb_offset = act_offset; act_offset += dim;
    g_xb2_offset = act_offset; act_offset += dim;
    g_q_offset = act_offset; act_offset += dim;
    g_k_offset = act_offset; act_offset += g_kv_dim;
    g_v_offset = act_offset; act_offset += g_kv_dim;
    g_hb_offset = act_offset; act_offset += hidden_dim;
    g_hb2_offset = act_offset; act_offset += hidden_dim;
    g_logits_offset = act_offset; act_offset += vocab_size;
    g_k_cache_offset = act_offset; act_offset += n_layers * n_kv_heads * seq_len * g_head_dim;
    g_v_cache_offset = act_offset; act_offset += n_layers * n_kv_heads * seq_len * g_head_dim;
    g_result_offset = act_offset; act_offset += 1;

    printf("GPU configured: dim=%d, layers=%d, heads=%d/%d, vocab=%d, seq=%d\n",
           dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len);
    return 0;
}

/**
 * Full forward pass. Returns next token ID.
 */
int gpu_forward(int token, int pos) {
    if (!g_weights_gpu || !g_act_gpu || g_dim == 0) {
        printf("GPU not initialized or configured\n");
        return -1;
    }

    // Debug: Print first forward pass info
    if (pos == 0) {
        printf("Forward: token=%d, pos=%d, dim=%d, layers=%d\n",
               token, pos, g_dim, g_n_layers);
    }

    // Pointer shortcuts
    float* w = g_weights_gpu;
    float* a = g_act_gpu;

    int d = g_dim;
    int hd = g_hidden_dim;
    int kv = g_kv_dim;
    int hs = g_head_dim;

    // Get activation pointers
    float* x = a + g_x_offset;
    float* xb = a + g_xb_offset;
    float* xb2 = a + g_xb2_offset;
    float* q = a + g_q_offset;
    float* k = a + g_k_offset;
    float* v = a + g_v_offset;
    float* hb = a + g_hb_offset;
    float* hb2 = a + g_hb2_offset;
    float* logits = a + g_logits_offset;
    float* k_cache = a + g_k_cache_offset;
    float* v_cache = a + g_v_cache_offset;
    int* result = (int*)(a + g_result_offset);

    // Token embedding lookup (copy to x)
    float* emb = w + g_token_emb_offset + token * d;
    cudaMemcpyAsync(x, emb, d * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);

    // Forward through layers
    for (int layer = 0; layer < g_n_layers; layer++) {
        // Weight pointers for this layer
        float* rms_att = w + g_rms_att_offset + layer * d;
        float* wq = w + g_wq_offset + layer * d * d;
        float* wk = w + g_wk_offset + layer * kv * d;
        float* wv = w + g_wv_offset + layer * kv * d;
        float* wo = w + g_wo_offset + layer * d * d;
        float* rms_ffn = w + g_rms_ffn_offset + layer * d;
        float* w1 = w + g_w1_offset + layer * hd * d;
        float* w2 = w + g_w2_offset + layer * d * hd;
        float* w3 = w + g_w3_offset + layer * hd * d;

        // RMSNorm
        rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_att, d, 1e-6f);

        // QKV projections
        cublas_matvec(q, xb, wq, d, d);
        cublas_matvec(k, xb, wk, kv, d);
        cublas_matvec(v, xb, wv, kv, d);

        // Add biases if present
        if (g_has_bias) {
            float* bq = w + g_bq_offset + layer * d;
            float* bk = w + g_bk_offset + layer * kv;
            float* bv = w + g_bv_offset + layer * kv;
            add_bias_kernel<<<(d+255)/256, 256, 0, g_stream>>>(q, bq, d);
            add_bias_kernel<<<(kv+255)/256, 256, 0, g_stream>>>(k, bk, kv);
            add_bias_kernel<<<(kv+255)/256, 256, 0, g_stream>>>(v, bv, kv);
        }

        // RoPE
        int freq_offset = pos * (hs / 2);
        float* cos = w + g_freq_cos_offset + freq_offset;
        float* sin = w + g_freq_sin_offset + freq_offset;
        int max_heads = (g_n_heads > g_n_kv_heads) ? g_n_heads : g_n_kv_heads;
        rope_kernel<<<max_heads, hs/2, 0, g_stream>>>(q, k, cos, sin, g_n_heads, g_n_kv_heads, hs);

        // Update KV cache
        float* layer_k_cache = k_cache + layer * g_n_kv_heads * g_seq_len * hs;
        float* layer_v_cache = v_cache + layer * g_n_kv_heads * g_seq_len * hs;
        kv_cache_update_kernel<<<g_n_kv_heads, hs, 0, g_stream>>>(
            layer_k_cache, layer_v_cache, k, v, g_n_kv_heads, pos, g_seq_len, hs);

        // GQA Attention
        float scale = 1.0f / sqrtf((float)hs);
        int smem = (pos + 1 + 32) * sizeof(float);  // scores + partial sums
        gqa_attention_kernel<<<g_n_heads, 128, smem, g_stream>>>(
            xb, q, layer_k_cache, layer_v_cache,
            g_n_heads, g_n_kv_heads, pos + 1, g_seq_len, hs, scale);

        // Output projection
        cublas_matvec(xb2, xb, wo, d, d);

        // Residual add
        residual_add_kernel<<<(d+255)/256, 256, 0, g_stream>>>(x, xb2, d);

        // FFN
        rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_ffn, d, 1e-6f);
        cublas_matvec(hb, xb, w1, hd, d);
        cublas_matvec(hb2, xb, w3, hd, d);
        swiglu_kernel<<<(hd+255)/256, 256, 0, g_stream>>>(hb, hb, hb2, hd);
        cublas_matvec(xb, hb, w2, d, hd);
        residual_add_kernel<<<(d+255)/256, 256, 0, g_stream>>>(x, xb, d);
    }

    // Final RMSNorm
    float* rms_final = w + g_rms_final_offset;
    rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_final, d, 1e-6f);

    // Logits
    float* token_emb = w + g_token_emb_offset;
    cublas_matvec(logits, xb, token_emb, g_vocab_size, d);

    // Argmax
    argmax_kernel<<<1, 256, 0, g_stream>>>(result, logits, g_vocab_size);

    // Sync and return result
    cudaStreamSynchronize(g_stream);
    int next_token;
    cudaMemcpy(&next_token, result, sizeof(int), cudaMemcpyDeviceToHost);

    return next_token;
}

// ============================================================
// INT4 Quantized Inference Support
// ============================================================

#include "int4_gemv.h"
#include "int8_embedding.h"

// INT4 mode flag and buffers
static bool g_int4_mode = false;

// INT8 embedding buffers (for fast logit computation)
static int8_t* g_emb_int8_gpu = nullptr;
static half* g_emb_scales_gpu = nullptr;
static bool g_use_int8_embedding = false;
static uint8_t* g_int4_weights_gpu = nullptr;   // Packed INT4 weights
static half* g_int4_scales_gpu = nullptr;       // FP16 scales
static size_t g_int4_weights_size = 0;
static size_t g_int4_scales_size = 0;

// INT4 weight layout offsets (in bytes)
static size_t g_int4_wq_scales_offset = 0;
static size_t g_int4_wq_packed_offset = 0;
static size_t g_int4_wk_scales_offset = 0;
static size_t g_int4_wk_packed_offset = 0;
static size_t g_int4_wv_scales_offset = 0;
static size_t g_int4_wv_packed_offset = 0;
static size_t g_int4_wo_scales_offset = 0;
static size_t g_int4_wo_packed_offset = 0;
static size_t g_int4_w1_scales_offset = 0;
static size_t g_int4_w1_packed_offset = 0;
static size_t g_int4_w2_scales_offset = 0;
static size_t g_int4_w2_packed_offset = 0;
static size_t g_int4_w3_scales_offset = 0;
static size_t g_int4_w3_packed_offset = 0;

// INT4 getters (defined after variables to avoid forward reference)
uint8_t* get_int4_weights_gpu() { return g_int4_weights_gpu; }
half* get_int4_scales_gpu() { return g_int4_scales_gpu; }

/**
 * Calculate INT4 model buffer sizes for allocation.
 */
void int4_calculate_sizes(
    int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
    int vocab_size, int seq_len,
    size_t* fp32_bytes,      // Embeddings, norms, biases (FP32)
    size_t* int4_weights_bytes,  // Packed INT4 weights
    size_t* int4_scales_bytes    // FP16 scales
) {
    int head_size = dim / n_heads;
    int kv_dim = (n_kv_heads * dim) / n_heads;
    int group_size = INT4_GROUP_SIZE;

    // FP32 components (not quantized)
    *fp32_bytes = (
        vocab_size * dim +       // token_embedding
        n_layers * dim +         // rms_att
        n_layers * dim +         // rms_ffn
        dim +                    // rms_final
        seq_len * (head_size/2) + // freq_cos
        seq_len * (head_size/2) + // freq_sin
        n_layers * dim +         // bq
        n_layers * kv_dim +      // bk
        n_layers * kv_dim        // bv
    ) * sizeof(float);

    // INT4 packed weights (large matmul weights)
    // wq: [n_layers, dim, dim] -> packed
    // wk: [n_layers, kv_dim, dim] -> packed
    // wv: [n_layers, kv_dim, dim] -> packed
    // wo: [n_layers, dim, dim] -> packed
    // w1: [n_layers, hidden_dim, dim] -> packed
    // w2: [n_layers, dim, hidden_dim] -> packed
    // w3: [n_layers, hidden_dim, dim] -> packed
    size_t wq_packed = (size_t)n_layers * dim * dim / 2;
    size_t wk_packed = (size_t)n_layers * kv_dim * dim / 2;
    size_t wv_packed = (size_t)n_layers * kv_dim * dim / 2;
    size_t wo_packed = (size_t)n_layers * dim * dim / 2;
    size_t w1_packed = (size_t)n_layers * hidden_dim * dim / 2;
    size_t w2_packed = (size_t)n_layers * dim * hidden_dim / 2;
    size_t w3_packed = (size_t)n_layers * hidden_dim * dim / 2;

    *int4_weights_bytes = wq_packed + wk_packed + wv_packed + wo_packed +
                          w1_packed + w2_packed + w3_packed;

    // FP16 scales for each weight
    int n_groups_dim = (dim + group_size - 1) / group_size;
    int n_groups_hd = (hidden_dim + group_size - 1) / group_size;

    size_t wq_scales = (size_t)n_layers * dim * n_groups_dim * sizeof(half);
    size_t wk_scales = (size_t)n_layers * kv_dim * n_groups_dim * sizeof(half);
    size_t wv_scales = (size_t)n_layers * kv_dim * n_groups_dim * sizeof(half);
    size_t wo_scales = (size_t)n_layers * dim * n_groups_dim * sizeof(half);
    size_t w1_scales = (size_t)n_layers * hidden_dim * n_groups_dim * sizeof(half);
    size_t w2_scales = (size_t)n_layers * dim * n_groups_hd * sizeof(half);
    size_t w3_scales = (size_t)n_layers * hidden_dim * n_groups_dim * sizeof(half);

    *int4_scales_bytes = wq_scales + wk_scales + wv_scales + wo_scales +
                         w1_scales + w2_scales + w3_scales;
}

// Temporary storage for FP32 weights (for old API compatibility)
static float* g_temp_fp32_weights = nullptr;
static size_t g_temp_fp32_size = 0;

/**
 * Initialize INT4 inference mode (OLD API for compatibility).
 * Takes model config parameters and FP32 weights pointer.
 *
 * NOTE: This wrapper now calculates the actual FP32 buffer size based on what
 * is typically in model files (embeddings + rms norms), not the full size
 * including freq_cos/sin and biases which are often computed at runtime.
 */
int cublas_init_int4(
    float* fp32_weights, int dim, int hidden_dim, int n_layers,
    int n_heads, int n_kv_heads, int vocab_size, int seq_len,
    int head_dim, int kv_dim
) {
    // Calculate the ACTUAL fp32 buffer size based on model file format (export_qwen_int4.py):
    // This MUST match what's stored in the model file
    size_t actual_fp32_elements = (size_t)vocab_size * dim;  // embeddings
    actual_fp32_elements += (size_t)n_layers * dim;          // rms_att (stacked)
    actual_fp32_elements += (size_t)n_layers * dim;          // rms_ffn (stacked)
    actual_fp32_elements += (size_t)dim;                     // rms_final
    actual_fp32_elements += (size_t)seq_len * (head_dim / 2);  // freq_cos
    actual_fp32_elements += (size_t)seq_len * (head_dim / 2);  // freq_sin
    actual_fp32_elements += (size_t)n_layers * dim;          // bq (QKV biases)
    actual_fp32_elements += (size_t)n_layers * kv_dim;       // bk
    actual_fp32_elements += (size_t)n_layers * kv_dim;       // bv
    size_t actual_fp32_bytes = actual_fp32_elements * sizeof(float);

    // Calculate INT4 weights and scales sizes using the helper
    size_t fp32_bytes_calculated, int4_weights_bytes, int4_scales_bytes;
    int4_calculate_sizes(dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                         vocab_size, seq_len,
                         &fp32_bytes_calculated, &int4_weights_bytes, &int4_scales_bytes);

    // Use the larger size for allocation (includes freq/biases for runtime computation)
    // but only upload what's actually in the buffer
    size_t fp32_alloc_bytes = fp32_bytes_calculated;

    // Calculate activation size
    size_t activation_bytes = 0;
    activation_bytes += (size_t)dim * sizeof(float);           // x
    activation_bytes += (size_t)dim * sizeof(float);           // xb
    activation_bytes += (size_t)dim * sizeof(float);           // xb2
    activation_bytes += (size_t)dim * sizeof(float);           // q
    activation_bytes += (size_t)kv_dim * sizeof(float);        // k
    activation_bytes += (size_t)kv_dim * sizeof(float);        // v
    activation_bytes += (size_t)hidden_dim * sizeof(float);    // hb
    activation_bytes += (size_t)hidden_dim * sizeof(float);    // hb2
    activation_bytes += (size_t)vocab_size * sizeof(float);    // logits
    activation_bytes += (size_t)n_layers * seq_len * kv_dim * sizeof(float); // k_cache
    activation_bytes += (size_t)n_layers * seq_len * kv_dim * sizeof(float); // v_cache
    activation_bytes += sizeof(int);                            // result

    printf("FP32 buffer: allocating %.2f MB, uploading %.2f MB\n",
           fp32_alloc_bytes / 1e6, actual_fp32_bytes / 1e6);

    // Call new init function with allocation size
    int ret = cublas_init_int4_sizes(fp32_alloc_bytes, int4_weights_bytes, int4_scales_bytes, activation_bytes);
    if (ret != 0) return ret;

    // Upload only the actual FP32 weights from the buffer (not the padded size)
    cudaError_t err = cudaMemcpy(g_weights_gpu, fp32_weights, actual_fp32_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("FP32 upload failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("Uploaded %.2f MB FP32 weights to GPU\n", actual_fp32_bytes / 1e6);

    return 0;
}

/**
 * Initialize INT4 inference mode (new API with explicit sizes).
 * Allocates separate buffers for INT4 packed weights and scales.
 */
int cublas_init_int4_sizes(size_t fp32_bytes, size_t int4_weights_bytes, size_t int4_scales_bytes,
                           size_t activation_bytes) {
    // Initialize standard cuBLAS first
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS init failed: %d\n", status);
        return -1;
    }

    cudaStreamCreate(&g_stream);
    cublasSetStream(g_cublas_handle, g_stream);

    // Initialize INT4 kernels with our stream
    int4_init(g_stream);

    // Allocate FP32 buffer (embeddings, norms, biases, etc.)
    if (fp32_bytes > 0) {
        cudaMalloc(&g_weights_gpu, fp32_bytes);
        g_weights_size = fp32_bytes;
        printf("Allocated %.2f MB for FP32 weights on GPU\n", fp32_bytes / 1e6);
    }

    // Allocate INT4 packed weights buffer
    if (int4_weights_bytes > 0) {
        cudaMalloc(&g_int4_weights_gpu, int4_weights_bytes);
        g_int4_weights_size = int4_weights_bytes;
        printf("Allocated %.2f MB for INT4 packed weights on GPU\n", int4_weights_bytes / 1e6);
    }

    // Allocate INT4 scales buffer
    if (int4_scales_bytes > 0) {
        cudaMalloc(&g_int4_scales_gpu, int4_scales_bytes);
        g_int4_scales_size = int4_scales_bytes;
        printf("Allocated %.2f MB for INT4 scales on GPU\n", int4_scales_bytes / 1e6);
    }

    // Allocate activation buffer
    if (activation_bytes > 0) {
        cudaMalloc(&g_act_gpu, activation_bytes);
        g_act_size = activation_bytes;
        printf("Allocated %.2f MB for activations on GPU\n", activation_bytes / 1e6);
    }

    g_int4_mode = true;
    return 0;
}

/**
 * Upload INT4 weights to GPU (OLD API for compatibility).
 * FP32 weights are already uploaded by cublas_init_int4.
 */
int cublas_upload_int4_weights(
    const uint8_t* int4_packed,
    const half* int4_scales,
    size_t int4_packed_bytes,
    size_t int4_scales_bytes
) {
    cudaError_t err;

    // Upload INT4 packed weights
    if (int4_packed && int4_packed_bytes > 0) {
        err = cudaMemcpy(g_int4_weights_gpu, int4_packed, int4_packed_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("INT4 weights upload failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        printf("Uploaded %.2f MB INT4 packed weights to GPU\n", int4_packed_bytes / 1e6);
    }

    // Upload scales
    if (int4_scales && int4_scales_bytes > 0) {
        err = cudaMemcpy(g_int4_scales_gpu, int4_scales, int4_scales_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("INT4 scales upload failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        printf("Uploaded %.2f MB INT4 scales to GPU\n", int4_scales_bytes / 1e6);
    }

    return 0;
}

/**
 * Configure INT4 model dimensions and calculate weight offsets.
 */

// Internal implementation of gpu_configure_int4 with has_bias parameter
static int gpu_configure_int4_internal(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                                       int vocab_size, int seq_len, int has_bias) {
    // Set basic dimensions (reuse FP32 config)
    g_dim = dim;
    g_hidden_dim = hidden_dim;
    g_n_layers = n_layers;
    g_n_heads = n_heads;
    g_n_kv_heads = n_kv_heads;
    g_vocab_size = vocab_size;
    g_seq_len = seq_len;
    g_head_dim = dim / n_heads;
    g_kv_dim = (n_kv_heads * dim) / n_heads;
    g_has_bias = (has_bias != 0);
    g_int4_mode = true;

    int group_size = INT4_GROUP_SIZE;
    int n_groups_dim = (dim + group_size - 1) / group_size;
    int n_groups_hd = (hidden_dim + group_size - 1) / group_size;

    // FP32 weight offsets (in floats, for embeddings/norms/biases)
    size_t offset = 0;
    g_token_emb_offset = offset; offset += vocab_size * dim;
    g_rms_att_offset = offset; offset += n_layers * dim;
    g_rms_ffn_offset = offset; offset += n_layers * dim;
    g_rms_final_offset = offset; offset += dim;
    g_freq_cos_offset = offset; offset += seq_len * (g_head_dim / 2);
    g_freq_sin_offset = offset; offset += seq_len * (g_head_dim / 2);

    if (has_bias) {
        g_bq_offset = offset; offset += n_layers * dim;
        g_bk_offset = offset; offset += n_layers * g_kv_dim;
        g_bv_offset = offset; offset += n_layers * g_kv_dim;
    }

    // INT4 scales offsets (in bytes)
    size_t scales_offset = 0;
    g_int4_wq_scales_offset = scales_offset;
    scales_offset += (size_t)n_layers * dim * n_groups_dim * sizeof(half);
    g_int4_wk_scales_offset = scales_offset;
    scales_offset += (size_t)n_layers * g_kv_dim * n_groups_dim * sizeof(half);
    g_int4_wv_scales_offset = scales_offset;
    scales_offset += (size_t)n_layers * g_kv_dim * n_groups_dim * sizeof(half);
    g_int4_wo_scales_offset = scales_offset;
    scales_offset += (size_t)n_layers * dim * n_groups_dim * sizeof(half);
    g_int4_w1_scales_offset = scales_offset;
    scales_offset += (size_t)n_layers * hidden_dim * n_groups_dim * sizeof(half);
    g_int4_w2_scales_offset = scales_offset;
    scales_offset += (size_t)n_layers * dim * n_groups_hd * sizeof(half);
    g_int4_w3_scales_offset = scales_offset;

    // INT4 packed weights offsets (in bytes)
    size_t packed_offset = 0;
    g_int4_wq_packed_offset = packed_offset;
    packed_offset += (size_t)n_layers * dim * dim / 2;
    g_int4_wk_packed_offset = packed_offset;
    packed_offset += (size_t)n_layers * g_kv_dim * dim / 2;
    g_int4_wv_packed_offset = packed_offset;
    packed_offset += (size_t)n_layers * g_kv_dim * dim / 2;
    g_int4_wo_packed_offset = packed_offset;
    packed_offset += (size_t)n_layers * dim * dim / 2;
    g_int4_w1_packed_offset = packed_offset;
    packed_offset += (size_t)n_layers * hidden_dim * dim / 2;
    g_int4_w2_packed_offset = packed_offset;
    packed_offset += (size_t)n_layers * dim * hidden_dim / 2;
    g_int4_w3_packed_offset = packed_offset;

    // Activation offsets (same as FP32)
    size_t act_offset = 0;
    g_x_offset = act_offset; act_offset += dim;
    g_xb_offset = act_offset; act_offset += dim;
    g_xb2_offset = act_offset; act_offset += dim;
    g_q_offset = act_offset; act_offset += dim;
    g_k_offset = act_offset; act_offset += g_kv_dim;
    g_v_offset = act_offset; act_offset += g_kv_dim;
    g_hb_offset = act_offset; act_offset += hidden_dim;
    g_hb2_offset = act_offset; act_offset += hidden_dim;
    g_logits_offset = act_offset; act_offset += vocab_size;
    g_k_cache_offset = act_offset; act_offset += n_layers * n_kv_heads * seq_len * g_head_dim;
    g_v_cache_offset = act_offset; act_offset += n_layers * n_kv_heads * seq_len * g_head_dim;
    g_result_offset = act_offset;

    printf("INT4 GPU configured: dim=%d, layers=%d, heads=%d/%d, vocab=%d, seq=%d\n",
           dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len);
    printf("  INT4 group size: %d, n_groups(dim): %d, n_groups(hd): %d\n",
           group_size, n_groups_dim, n_groups_hd);
    return 0;
}

/**
 * Public API: Configure INT4 mode with 10 arguments (edgellm_eagle.cu compatibility).
 * head_dim and kv_dim are ignored (calculated internally).
 */
int gpu_configure_int4(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                       int vocab_size, int seq_len, int head_dim, int kv_dim, int group_size) {
    (void)head_dim;  // Unused - calculated internally
    (void)kv_dim;    // Unused - calculated internally
    (void)group_size; // Unused - uses INT4_GROUP_SIZE constant
    // Qwen models have QKV biases - must enable bias mode
    return gpu_configure_int4_internal(dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                                       vocab_size, seq_len, 1);  // 1 = has bias
}

/**
 * Alternative API: Configure INT4 mode with has_bias parameter.
 * For code that needs to specify bias usage explicitly.
 */
void cublas_configure_int4(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                           int vocab_size, int seq_len, int has_bias) {
    gpu_configure_int4_internal(dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                                vocab_size, seq_len, has_bias);
}

/**
 * INT4 GEMV helper - performs matmul using INT4 kernel
 */
static inline int int4_matvec_layer(
    float* out_gpu,
    const float* x_gpu,
    size_t packed_offset,  // Offset into g_int4_weights_gpu
    size_t scales_offset,  // Offset into g_int4_scales_gpu
    int layer,
    int out_dim,
    int in_dim,
    size_t per_layer_packed,
    size_t per_layer_scales
) {
    const uint8_t* W_q = g_int4_weights_gpu + packed_offset + layer * per_layer_packed;
    const half* scales = (half*)((uint8_t*)g_int4_scales_gpu + scales_offset + layer * per_layer_scales);

    return int4_gemv(out_gpu, x_gpu, W_q, scales, out_dim, in_dim);
}

/**
 * INT4 forward pass. Returns next token ID.
 *
 * Key difference from FP32: Uses int4_gemv for all matmuls
 * Embeddings and norms remain FP32 for quality.
 */
int gpu_forward_int4(int token, int pos) {
    if (!g_weights_gpu || !g_act_gpu || !g_int4_weights_gpu || !g_int4_scales_gpu || g_dim == 0) {
        printf("INT4 GPU not initialized\n");
        return -1;
    }

    // Shortcuts
    float* w = g_weights_gpu;  // FP32 weights (embedding, norms)
    float* a = g_act_gpu;

    int d = g_dim;
    int hd = g_hidden_dim;
    int kv = g_kv_dim;
    int hs = g_head_dim;
    int group_size = INT4_GROUP_SIZE;
    int n_groups_dim = (d + group_size - 1) / group_size;
    int n_groups_hd = (hd + group_size - 1) / group_size;

    // Per-layer sizes for INT4 weights
    size_t wq_per_layer_packed = (size_t)d * d / 2;
    size_t wk_per_layer_packed = (size_t)kv * d / 2;
    size_t wv_per_layer_packed = (size_t)kv * d / 2;
    size_t wo_per_layer_packed = (size_t)d * d / 2;
    size_t w1_per_layer_packed = (size_t)hd * d / 2;
    size_t w2_per_layer_packed = (size_t)d * hd / 2;
    size_t w3_per_layer_packed = (size_t)hd * d / 2;

    size_t wq_per_layer_scales = (size_t)d * n_groups_dim * sizeof(half);
    size_t wk_per_layer_scales = (size_t)kv * n_groups_dim * sizeof(half);
    size_t wv_per_layer_scales = (size_t)kv * n_groups_dim * sizeof(half);
    size_t wo_per_layer_scales = (size_t)d * n_groups_dim * sizeof(half);
    size_t w1_per_layer_scales = (size_t)hd * n_groups_dim * sizeof(half);
    size_t w2_per_layer_scales = (size_t)d * n_groups_hd * sizeof(half);
    size_t w3_per_layer_scales = (size_t)hd * n_groups_dim * sizeof(half);

    // Activation pointers
    float* x = a + g_x_offset;
    float* xb = a + g_xb_offset;
    float* xb2 = a + g_xb2_offset;
    float* q = a + g_q_offset;
    float* k = a + g_k_offset;
    float* v = a + g_v_offset;
    float* hb = a + g_hb_offset;
    float* hb2 = a + g_hb2_offset;
    float* logits = a + g_logits_offset;
    float* k_cache = a + g_k_cache_offset;
    float* v_cache = a + g_v_cache_offset;
    int* result = (int*)(a + g_result_offset);

    // Token embedding lookup (FP32)
    float* emb = w + g_token_emb_offset + token * d;
    cudaMemcpyAsync(x, emb, d * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);

    // Forward through layers
    for (int layer = 0; layer < g_n_layers; layer++) {
        // Weight pointers for norms (FP32)
        float* rms_att = w + g_rms_att_offset + layer * d;
        float* rms_ffn = w + g_rms_ffn_offset + layer * d;

        // RMSNorm
        rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_att, d, 1e-6f);

        // QKV projections using INT4 GEMV
        int4_matvec_layer(q, xb, g_int4_wq_packed_offset, g_int4_wq_scales_offset,
                         layer, d, d, wq_per_layer_packed, wq_per_layer_scales);
        int4_matvec_layer(k, xb, g_int4_wk_packed_offset, g_int4_wk_scales_offset,
                         layer, kv, d, wk_per_layer_packed, wk_per_layer_scales);
        int4_matvec_layer(v, xb, g_int4_wv_packed_offset, g_int4_wv_scales_offset,
                         layer, kv, d, wv_per_layer_packed, wv_per_layer_scales);

        // Add biases if present (FP32)
        if (g_has_bias) {
            float* bq = w + g_bq_offset + layer * d;
            float* bk = w + g_bk_offset + layer * kv;
            float* bv = w + g_bv_offset + layer * kv;
            add_bias_kernel<<<(d+255)/256, 256, 0, g_stream>>>(q, bq, d);
            add_bias_kernel<<<(kv+255)/256, 256, 0, g_stream>>>(k, bk, kv);
            add_bias_kernel<<<(kv+255)/256, 256, 0, g_stream>>>(v, bv, kv);
        }

        // RoPE (FP32 freqs)
        int freq_offset = pos * (hs / 2);
        float* cos = w + g_freq_cos_offset + freq_offset;
        float* sin = w + g_freq_sin_offset + freq_offset;
        int max_heads = (g_n_heads > g_n_kv_heads) ? g_n_heads : g_n_kv_heads;
        rope_kernel<<<max_heads, hs/2, 0, g_stream>>>(q, k, cos, sin, g_n_heads, g_n_kv_heads, hs);

        // KV cache update
        float* layer_k_cache = k_cache + layer * g_n_kv_heads * g_seq_len * hs;
        float* layer_v_cache = v_cache + layer * g_n_kv_heads * g_seq_len * hs;
        kv_cache_update_kernel<<<g_n_kv_heads, hs, 0, g_stream>>>(
            layer_k_cache, layer_v_cache, k, v, g_n_kv_heads, pos, g_seq_len, hs);

        // GQA Attention
        float scale = 1.0f / sqrtf((float)hs);
        int smem = (pos + 1 + 32) * sizeof(float);
        gqa_attention_kernel<<<g_n_heads, 128, smem, g_stream>>>(
            xb, q, layer_k_cache, layer_v_cache,
            g_n_heads, g_n_kv_heads, pos + 1, g_seq_len, hs, scale);

        // Output projection (INT4)
        int4_matvec_layer(xb2, xb, g_int4_wo_packed_offset, g_int4_wo_scales_offset,
                         layer, d, d, wo_per_layer_packed, wo_per_layer_scales);

        // Residual add
        residual_add_kernel<<<(d+255)/256, 256, 0, g_stream>>>(x, xb2, d);

        // Sync after attention block (for multirow debugging)
        #ifdef MULTIROW_ATTN_SYNC
        int4_sync();
        #endif

        // FFN
        rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_ffn, d, 1e-6f);

        // FFN projections (INT4)
        int4_matvec_layer(hb, xb, g_int4_w1_packed_offset, g_int4_w1_scales_offset,
                         layer, hd, d, w1_per_layer_packed, w1_per_layer_scales);
        int4_matvec_layer(hb2, xb, g_int4_w3_packed_offset, g_int4_w3_scales_offset,
                         layer, hd, d, w3_per_layer_packed, w3_per_layer_scales);

        swiglu_kernel<<<(hd+255)/256, 256, 0, g_stream>>>(hb, hb, hb2, hd);

        int4_matvec_layer(xb, hb, g_int4_w2_packed_offset, g_int4_w2_scales_offset,
                         layer, d, hd, w2_per_layer_packed, w2_per_layer_scales);

        residual_add_kernel<<<(d+255)/256, 256, 0, g_stream>>>(x, xb, d);

        // Sync INT4 stream at end of each layer (for multirow kernel debugging)
        #ifdef MULTIROW_LAYER_SYNC
        int4_sync();
        #endif
    }

    // Final RMSNorm (FP32)
    float* rms_final = w + g_rms_final_offset;
    rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_final, d, 1e-6f);

    // Logits computation
    if (g_use_int8_embedding && g_emb_int8_gpu && g_emb_scales_gpu) {
        // Use INT8 embedding for fast logit computation (1.6x faster than FP32)
        // Standard kernel keeps input FP32, avoids quantization overhead
        int8_logit_gemv(logits, xb, g_emb_int8_gpu, g_emb_scales_gpu, g_vocab_size, d);
    } else {
        // Fallback to FP32 cuBLAS (slower but always available)
        float* token_emb = w + g_token_emb_offset;
        cublas_matvec(logits, xb, token_emb, g_vocab_size, d);
    }

    // Debug: check logits for first calls
    // Argmax
    argmax_kernel<<<1, 256, 0, g_stream>>>(result, logits, g_vocab_size);

    // Sync and return
    cudaStreamSynchronize(g_stream);
    int next_token;
    cudaMemcpy(&next_token, result, sizeof(int), cudaMemcpyDeviceToHost);

    return next_token;
}

/**
 * Initialize INT8 embedding for fast logit computation.
 * Call after cublas_init_int4 to enable INT8 logits.
 */
int int8_embedding_init(int vocab_size, int dim) {
    if (g_emb_int8_gpu || g_emb_scales_gpu) {
        printf("INT8 embedding already initialized\n");
        return -1;
    }

    // Allocate INT8 embedding table
    size_t emb_bytes = (size_t)vocab_size * dim * sizeof(int8_t);
    size_t scales_bytes = (size_t)vocab_size * sizeof(half);

    cudaError_t err = cudaMalloc(&g_emb_int8_gpu, emb_bytes);
    if (err != cudaSuccess) {
        printf("Failed to allocate INT8 embedding: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&g_emb_scales_gpu, scales_bytes);
    if (err != cudaSuccess) {
        printf("Failed to allocate INT8 embedding scales: %s\n", cudaGetErrorString(err));
        cudaFree(g_emb_int8_gpu);
        g_emb_int8_gpu = nullptr;
        return -1;
    }

    // Initialize INT8 kernel with stream
    int8_emb_init(g_stream);

    printf("Allocated %.2f MB for INT8 embedding on GPU (4x smaller than FP32)\n",
           (emb_bytes + scales_bytes) / 1e6);
    return 0;
}

/**
 * Upload INT8 quantized embedding to GPU.
 */
int int8_embedding_upload(
    const int8_t* emb_int8,
    const half* scales,
    int vocab_size,
    int dim
) {
    if (!g_emb_int8_gpu || !g_emb_scales_gpu) {
        printf("INT8 embedding not initialized\n");
        return -1;
    }

    size_t emb_bytes = (size_t)vocab_size * dim * sizeof(int8_t);
    size_t scales_bytes = (size_t)vocab_size * sizeof(half);

    cudaError_t err = cudaMemcpy(g_emb_int8_gpu, emb_int8, emb_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to upload INT8 embedding: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(g_emb_scales_gpu, scales, scales_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to upload INT8 embedding scales: %s\n", cudaGetErrorString(err));
        return -1;
    }

    g_use_int8_embedding = true;
    printf("Uploaded INT8 embedding: %.2f MB (INT8) + %.2f KB (scales)\n",
           emb_bytes / 1e6, scales_bytes / 1e3);
    return 0;
}

/**
 * Enable/disable INT8 embedding for logit computation.
 */
void set_int8_embedding_mode(int enable) {
    g_use_int8_embedding = (enable != 0) && g_emb_int8_gpu && g_emb_scales_gpu;
    printf("INT8 embedding mode: %s\n", g_use_int8_embedding ? "ENABLED" : "DISABLED");
}

/**
 * Cleanup INT4 resources
 */
void cublas_cleanup_int4() {
    if (g_int4_weights_gpu) cudaFree(g_int4_weights_gpu);
    if (g_int4_scales_gpu) cudaFree(g_int4_scales_gpu);
    if (g_emb_int8_gpu) cudaFree(g_emb_int8_gpu);
    if (g_emb_scales_gpu) cudaFree(g_emb_scales_gpu);

    g_int4_weights_gpu = nullptr;
    g_int4_scales_gpu = nullptr;
    g_emb_int8_gpu = nullptr;
    g_emb_scales_gpu = nullptr;
    g_int4_mode = false;
    g_use_int8_embedding = false;

    // Call base cleanup
    cublas_cleanup();
}

/**
 * Get INT4 mode status
 */
int is_int4_mode() {
    return g_int4_mode ? 1 : 0;
}

}  // extern "C"
