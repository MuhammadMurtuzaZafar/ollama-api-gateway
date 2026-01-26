/**
 * EdgeLLM Inference with Temperature Sampling
 *
 * Wraps the existing INT4 inference with temperature, top-k, and top-p sampling.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cfloat>

// External functions from cublas_matmul.cu
extern "C" {
    int gpu_forward_int4(int token, int pos);
    int cublas_init_int4(
        float* weights, int dim, int hidden_dim, int n_layers,
        int n_heads, int n_kv_heads, int vocab_size, int seq_len,
        int head_dim, int kv_dim
    );
    int cublas_upload_int4_weights(
        const uint8_t* int4_packed,
        const half* int4_scales,
        size_t packed_bytes,
        size_t scales_bytes
    );
    float* get_activations_gpu();
    float* get_logits_gpu();
    int get_vocab_size();
}

// Global configuration
static float g_temperature = 0.7f;
static int g_top_k = 50;
static float g_top_p = 0.9f;
static float g_repetition_penalty = 1.1f;

// RNG state
static curandState* g_sample_rng = nullptr;
static cudaStream_t g_sample_stream = nullptr;

// Vocabulary size (set during init)
static int g_sample_vocab_size = 0;

// Past tokens for repetition penalty
static int* g_past_tokens = nullptr;
static int g_past_count = 0;
static int g_max_past = 64;  // Track last 64 tokens

// Logits buffer
static float* g_logits_buffer = nullptr;

/**
 * Initialize RNG for sampling.
 */
__global__ void init_sample_rng(curandState* state, unsigned long long seed) {
    curand_init(seed, 0, 0, state);
}

/**
 * Apply temperature scaling.
 */
__global__ void apply_temperature(float* logits, int size, float temperature) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;
    logits[tid] /= temperature;
}

/**
 * Apply repetition penalty.
 */
__global__ void apply_repetition_penalty(
    float* logits,
    const int* past_tokens,
    int n_past,
    float penalty
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_past) return;

    int token = past_tokens[tid];
    if (token >= 0) {
        float val = logits[token];
        if (val > 0) {
            logits[token] = val / penalty;
        } else {
            logits[token] = val * penalty;
        }
    }
}

/**
 * Softmax with numerical stability.
 */
__global__ void softmax_stable(float* logits, int size) {
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_vals[256];

    int tid = threadIdx.x;

    // Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < size; i += blockDim.x) {
        if (logits[i] > local_max) local_max = logits[i];
    }
    s_vals[tid] = local_max;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_vals[tid + s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) s_max = s_vals[0];
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float exp_val = expf(logits[i] - s_max);
        logits[i] = exp_val;
        local_sum += exp_val;
    }
    s_vals[tid] = local_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) s_vals[tid] += s_vals[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_sum = s_vals[0];
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / (s_sum + 1e-10f);
    for (int i = tid; i < size; i += blockDim.x) {
        logits[i] *= inv_sum;
    }
}

/**
 * Top-k filtering using a simple threshold approach.
 */
__global__ void topk_mask(float* logits, int size, int k) {
    // Thread 0 finds the kth largest value
    if (threadIdx.x != 0) return;

    // Simple O(n*k) approach for correctness
    float threshold = -FLT_MAX;

    for (int i = 0; i < k; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < size; j++) {
            if (logits[j] > max_val && (i == 0 || logits[j] < threshold)) {
                max_val = logits[j];
            }
        }
        threshold = max_val;
    }

    // Mask out values below threshold
    for (int i = 0; i < size; i++) {
        if (logits[i] < threshold) {
            logits[i] = -FLT_MAX;
        }
    }
}

/**
 * Top-p nucleus filtering.
 */
__global__ void topp_mask(float* probs, int size, float p) {
    if (threadIdx.x != 0) return;

    // Simple approach: find threshold that captures p probability mass
    float cumsum = 0.0f;
    float threshold = 0.0f;

    // Find tokens that contribute to top-p mass
    // (This is simplified - real implementation would sort)
    for (int iter = 0; iter < 50; iter++) {
        float test_threshold = threshold + 0.01f;
        cumsum = 0.0f;
        for (int i = 0; i < size; i++) {
            if (probs[i] >= test_threshold) {
                cumsum += probs[i];
            }
        }
        if (cumsum >= p) {
            threshold = test_threshold;
        } else {
            break;
        }
    }

    // Mask out values below threshold
    for (int i = 0; i < size; i++) {
        if (probs[i] < threshold * 0.9f) {  // Small buffer
            probs[i] = 0.0f;
        }
    }
}

/**
 * Sample from distribution.
 */
__global__ void sample_token(
    int* result,
    const float* probs,
    int size,
    curandState* rng
) {
    if (threadIdx.x != 0) return;

    float r = curand_uniform(rng);
    float cumsum = 0.0f;

    for (int i = 0; i < size; i++) {
        cumsum += probs[i];
        if (cumsum >= r) {
            *result = i;
            return;
        }
    }
    // Fallback to last token
    *result = size - 1;
}

/**
 * Argmax for greedy decoding.
 */
__global__ void argmax_sample(int* result, const float* logits, int size) {
    __shared__ float s_max_val[256];
    __shared__ int s_max_idx[256];

    int tid = threadIdx.x;
    float max_val = -FLT_MAX;
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

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max_val[tid + s] > s_max_val[tid]) {
            s_max_val[tid] = s_max_val[tid + s];
            s_max_idx[tid] = s_max_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) *result = s_max_idx[0];
}

extern "C" {

/**
 * Initialize sampling with parameters.
 */
int sampling_config_init(
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty
) {
    g_sample_vocab_size = vocab_size;
    g_temperature = temperature;
    g_top_k = top_k;
    g_top_p = top_p;
    g_repetition_penalty = repetition_penalty;

    cudaStreamCreate(&g_sample_stream);
    cudaMalloc(&g_sample_rng, sizeof(curandState));
    cudaMalloc(&g_logits_buffer, vocab_size * sizeof(float));
    cudaMalloc(&g_past_tokens, g_max_past * sizeof(int));
    cudaMemset(g_past_tokens, -1, g_max_past * sizeof(int));

    init_sample_rng<<<1, 1, 0, g_sample_stream>>>(g_sample_rng, 42);
    cudaStreamSynchronize(g_sample_stream);

    printf("Sampling initialized: temp=%.2f, top_k=%d, top_p=%.2f, rep_pen=%.2f\n",
           temperature, top_k, top_p, repetition_penalty);

    return 0;
}

/**
 * Set sampling parameters.
 */
void set_sampling_params(float temperature, int top_k, float top_p, float rep_penalty) {
    g_temperature = temperature;
    g_top_k = top_k;
    g_top_p = top_p;
    g_repetition_penalty = rep_penalty;
}

/**
 * Forward pass with temperature sampling.
 * Returns next token.
 */
int forward_with_sampling(int token, int pos) {
    // Run standard forward pass (which computes logits)
    int greedy_token = gpu_forward_int4(token, pos);

    // If temperature is 0, return greedy result
    if (g_temperature <= 0.0f) {
        // Track token for repetition penalty
        if (g_past_count < g_max_past) {
            cudaMemcpy(g_past_tokens + g_past_count, &greedy_token,
                       sizeof(int), cudaMemcpyHostToDevice);
            g_past_count++;
        }
        return greedy_token;
    }

    // Get logits from activation buffer
    float* logits = get_logits_gpu();

    // Copy logits to our buffer (don't modify original)
    cudaMemcpy(g_logits_buffer, logits, g_sample_vocab_size * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Apply repetition penalty
    if (g_repetition_penalty > 1.0f && g_past_count > 0) {
        int blocks = (g_past_count + 255) / 256;
        apply_repetition_penalty<<<blocks, 256, 0, g_sample_stream>>>(
            g_logits_buffer, g_past_tokens, g_past_count, g_repetition_penalty
        );
    }

    // Apply temperature
    if (g_temperature > 0.0f && g_temperature != 1.0f) {
        int blocks = (g_sample_vocab_size + 255) / 256;
        apply_temperature<<<blocks, 256, 0, g_sample_stream>>>(
            g_logits_buffer, g_sample_vocab_size, g_temperature
        );
    }

    // Apply top-k filtering
    if (g_top_k > 0 && g_top_k < g_sample_vocab_size) {
        topk_mask<<<1, 1, 0, g_sample_stream>>>(
            g_logits_buffer, g_sample_vocab_size, g_top_k
        );
    }

    // Convert to probabilities
    softmax_stable<<<1, 256, 0, g_sample_stream>>>(
        g_logits_buffer, g_sample_vocab_size
    );

    // Apply top-p filtering
    if (g_top_p > 0.0f && g_top_p < 1.0f) {
        topp_mask<<<1, 1, 0, g_sample_stream>>>(
            g_logits_buffer, g_sample_vocab_size, g_top_p
        );
        // Renormalize
        softmax_stable<<<1, 256, 0, g_sample_stream>>>(
            g_logits_buffer, g_sample_vocab_size
        );
    }

    // Sample token
    int* result_gpu;
    cudaMalloc(&result_gpu, sizeof(int));
    sample_token<<<1, 1, 0, g_sample_stream>>>(
        result_gpu, g_logits_buffer, g_sample_vocab_size, g_sample_rng
    );

    cudaStreamSynchronize(g_sample_stream);

    int sampled_token;
    cudaMemcpy(&sampled_token, result_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(result_gpu);

    // Track token for repetition penalty
    if (g_past_count < g_max_past) {
        cudaMemcpy(g_past_tokens + g_past_count, &sampled_token,
                   sizeof(int), cudaMemcpyHostToDevice);
        g_past_count++;
    }

    return sampled_token;
}

/**
 * Reset past tokens buffer.
 */
void reset_past_tokens() {
    g_past_count = 0;
    if (g_past_tokens) {
        cudaMemset(g_past_tokens, -1, g_max_past * sizeof(int));
    }
}

/**
 * Cleanup sampling resources.
 */
void sampling_cleanup_resources() {
    if (g_sample_rng) cudaFree(g_sample_rng);
    if (g_logits_buffer) cudaFree(g_logits_buffer);
    if (g_past_tokens) cudaFree(g_past_tokens);
    if (g_sample_stream) cudaStreamDestroy(g_sample_stream);

    g_sample_rng = nullptr;
    g_logits_buffer = nullptr;
    g_past_tokens = nullptr;
    g_sample_stream = nullptr;
}

}  // extern "C"
