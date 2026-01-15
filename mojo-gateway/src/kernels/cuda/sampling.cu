/**
 * Temperature Sampling Kernels for EdgeLLM
 *
 * Implements:
 * - Temperature scaling
 * - Top-k filtering
 * - Top-p (nucleus) sampling
 * - Repetition penalty
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cfloat>

// Sampling configuration
struct SamplingConfig {
    float temperature;      // Temperature for softmax (0.0 = greedy, 1.0 = default)
    int top_k;             // Top-k tokens to consider (0 = disabled)
    float top_p;           // Top-p cumulative probability (1.0 = disabled)
    float repetition_penalty; // Penalty for repeated tokens (1.0 = disabled)
};

// Global RNG state
static curandState* g_rng_states = nullptr;
static bool g_rng_initialized = false;

/**
 * Initialize RNG states for sampling.
 */
__global__ void init_rng_kernel(curandState* states, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

extern "C" {

int sampling_init(unsigned long long seed) {
    if (g_rng_initialized) return 0;

    cudaMalloc(&g_rng_states, 256 * sizeof(curandState));
    init_rng_kernel<<<1, 256>>>(g_rng_states, seed);
    cudaDeviceSynchronize();

    g_rng_initialized = true;
    return 0;
}

void sampling_cleanup() {
    if (g_rng_states) {
        cudaFree(g_rng_states);
        g_rng_states = nullptr;
    }
    g_rng_initialized = false;
}

}  // extern "C"

/**
 * Apply temperature scaling to logits.
 */
__global__ void temperature_scale_kernel(
    float* logits,
    int vocab_size,
    float temperature
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= vocab_size) return;

    if (temperature > 0.0f) {
        logits[tid] /= temperature;
    }
}

/**
 * Apply repetition penalty to logits.
 */
__global__ void repetition_penalty_kernel(
    float* logits,
    const int* past_tokens,
    int n_past,
    float penalty
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_past) return;

    int token = past_tokens[tid];
    if (token >= 0) {
        if (logits[token] > 0) {
            logits[token] /= penalty;
        } else {
            logits[token] *= penalty;
        }
    }
}

/**
 * Compute softmax probabilities in-place.
 */
__global__ void softmax_kernel(
    float* logits,
    int vocab_size
) {
    __shared__ float s_max;
    __shared__ float s_sum;

    int tid = threadIdx.x;

    // Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] > local_max) local_max = logits[i];
    }

    // Reduce max
    __shared__ float s_max_vals[256];
    s_max_vals[tid] = local_max;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max_vals[tid + s] > s_max_vals[tid]) {
            s_max_vals[tid] = s_max_vals[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) s_max = s_max_vals[0];
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[i] = expf(logits[i] - s_max);
        local_sum += logits[i];
    }

    // Reduce sum
    __shared__ float s_sum_vals[256];
    s_sum_vals[tid] = local_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_vals[tid] += s_sum_vals[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) s_sum = s_sum_vals[0];
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / (s_sum + 1e-10f);
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[i] *= inv_sum;
    }
}

/**
 * Top-k filtering: set all but top-k logits to -inf.
 * Note: This is a simplified version - a full implementation would use
 * partial sorting, but for speed we use a threshold approach.
 */
__global__ void topk_filter_kernel(
    float* logits,
    float* sorted_logits,  // Temporary buffer
    int vocab_size,
    int k
) {
    // Find kth largest value using partial sort in shared memory
    __shared__ float s_threshold;

    int tid = threadIdx.x;

    // Simple approach: find approximate kth value
    // In production, use proper top-k algorithm

    if (tid == 0) {
        // Quick estimate: find threshold that keeps ~k values
        float max_val = -FLT_MAX;
        float min_val = FLT_MAX;

        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_val) max_val = logits[i];
            if (logits[i] < min_val) min_val = logits[i];
        }

        // Binary search for threshold
        float lo = min_val, hi = max_val;
        for (int iter = 0; iter < 20; iter++) {
            float mid = (lo + hi) / 2.0f;
            int count = 0;
            for (int i = 0; i < vocab_size && count <= k; i++) {
                if (logits[i] >= mid) count++;
            }
            if (count > k) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s_threshold = lo;
    }
    __syncthreads();

    // Apply threshold
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] < s_threshold) {
            logits[i] = -FLT_MAX;
        }
    }
}

/**
 * Top-p (nucleus) filtering: keep smallest set of tokens with cumulative prob >= p.
 */
__global__ void topp_filter_kernel(
    float* probs,
    int vocab_size,
    float p
) {
    // This requires sorted probabilities - simplified version
    // In production, use proper cumulative sum approach

    __shared__ float s_threshold;

    int tid = threadIdx.x;

    if (tid == 0) {
        // Find threshold for top-p
        float cumsum = 0.0f;
        float threshold = 0.0f;

        // Simple approach: iterate through all values
        // A proper implementation would sort first
        float max_prob = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] > max_prob) max_prob = probs[i];
        }

        // Keep values above threshold such that cumsum >= p
        float lo = 0.0f, hi = max_prob;
        for (int iter = 0; iter < 20; iter++) {
            float mid = (lo + hi) / 2.0f;
            cumsum = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                if (probs[i] >= mid) cumsum += probs[i];
            }
            if (cumsum > p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s_threshold = lo;
    }
    __syncthreads();

    // Apply threshold
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (probs[i] < s_threshold) {
            probs[i] = 0.0f;
        }
    }
}

/**
 * Sample from probability distribution.
 */
__global__ void sample_kernel(
    int* result,
    const float* probs,
    int vocab_size,
    curandState* rng_states
) {
    // Only thread 0 does the sampling
    if (threadIdx.x != 0) return;

    // Generate random number
    float r = curand_uniform(&rng_states[0]);

    // Find token using cumulative distribution
    float cumsum = 0.0f;
    int selected = 0;

    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= r) {
            selected = i;
            break;
        }
    }

    *result = selected;
}

/**
 * Full temperature sampling with top-k and top-p.
 */
extern "C" {

int sample_with_temperature(
    int* result,
    float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    cudaStream_t stream
) {
    if (!g_rng_initialized) {
        sampling_init(42);
    }

    // Apply temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        int blocks = (vocab_size + 255) / 256;
        temperature_scale_kernel<<<blocks, 256, 0, stream>>>(
            logits, vocab_size, temperature
        );
    }

    // Apply top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        topk_filter_kernel<<<1, 256, 0, stream>>>(
            logits, nullptr, vocab_size, top_k
        );
    }

    // Convert to probabilities
    softmax_kernel<<<1, 256, 0, stream>>>(logits, vocab_size);

    // Apply top-p filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        topp_filter_kernel<<<1, 256, 0, stream>>>(logits, vocab_size, top_p);

        // Renormalize after top-p
        softmax_kernel<<<1, 256, 0, stream>>>(logits, vocab_size);
    }

    // Sample
    if (temperature <= 0.0f) {
        // Greedy: caller should use argmax
        return -1;  // Signal to use argmax
    }

    sample_kernel<<<1, 1, 0, stream>>>(result, logits, vocab_size, g_rng_states);
    return 0;
}

/**
 * Simple temperature sampling (for integration).
 * Returns token ID.
 */
int gpu_sample_temperature(
    int* result_gpu,
    float* logits_gpu,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    cudaStream_t stream
) {
    // Temperature = 0 means greedy
    if (temperature <= 0.0f) {
        return -1;  // Use argmax
    }

    return sample_with_temperature(
        result_gpu, logits_gpu, vocab_size,
        temperature, top_k, top_p, stream
    );
}

}  // extern "C"
