/**
 * EdgeLLM Sampling CUDA Kernel Header
 *
 * GPU-accelerated sampling for LLM text generation.
 * Supports temperature scaling, top-k, top-p (nucleus), and repetition penalty.
 */

#ifndef EDGELLM_SAMPLING_KERNEL_H
#define EDGELLM_SAMPLING_KERNEL_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Temperature Scaling (in-place)
 *
 * Scales logits by temperature: logits = logits / temperature
 *
 * @param logits Logits tensor [batch_size, vocab_size] (modified in-place)
 * @param batch_size Batch size
 * @param vocab_size Vocabulary size
 * @param temperature Temperature (> 0, typical: 0.7-1.0)
 * @param stream CUDA stream
 */
void apply_temperature(
    float* logits,
    int batch_size,
    int vocab_size,
    float temperature,
    cudaStream_t stream
);

/**
 * Softmax (in-place)
 *
 * Converts logits to probabilities: probs = softmax(logits)
 *
 * @param logits Input logits, output probabilities [batch_size, vocab_size]
 * @param batch_size Batch size
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream
 */
void softmax_inplace(
    float* logits,
    int batch_size,
    int vocab_size,
    cudaStream_t stream
);

/**
 * Top-K Filtering (in-place)
 *
 * Sets all logits outside top-k to -inf.
 *
 * @param logits Logits tensor [batch_size, vocab_size]
 * @param batch_size Batch size
 * @param vocab_size Vocabulary size
 * @param k Number of top tokens to keep
 * @param stream CUDA stream
 */
void apply_top_k(
    float* logits,
    int batch_size,
    int vocab_size,
    int k,
    cudaStream_t stream
);

/**
 * Top-P (Nucleus) Sampling
 *
 * Samples from the smallest set of tokens with cumulative probability >= p.
 *
 * @param output_tokens Output sampled token IDs [batch_size]
 * @param logits Input logits [batch_size, vocab_size]
 * @param batch_size Batch size
 * @param vocab_size Vocabulary size
 * @param top_p Nucleus probability threshold (0-1, typical: 0.9)
 * @param temperature Temperature for scaling (applied before sampling)
 * @param rng_states Random number generator states [batch_size]
 * @param stream CUDA stream
 */
void sample_top_p(
    int32_t* output_tokens,
    const float* logits,
    int batch_size,
    int vocab_size,
    float top_p,
    float temperature,
    curandState* rng_states,
    cudaStream_t stream
);

/**
 * Greedy Sampling (Argmax)
 *
 * Returns the token with highest probability.
 *
 * @param output_tokens Output token IDs [batch_size]
 * @param logits Input logits [batch_size, vocab_size]
 * @param batch_size Batch size
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream
 */
void sample_greedy(
    int32_t* output_tokens,
    const float* logits,
    int batch_size,
    int vocab_size,
    cudaStream_t stream
);

/**
 * Initialize RNG States
 *
 * Initializes cuRAND states for sampling.
 *
 * @param states Output RNG states [num_states]
 * @param num_states Number of states to initialize
 * @param seed Random seed
 * @param stream CUDA stream
 */
void init_rng_states(
    curandState* states,
    int num_states,
    unsigned long long seed,
    cudaStream_t stream
);

/**
 * Repetition Penalty (in-place)
 *
 * Applies repetition penalty to previously generated tokens.
 *
 * @param logits Logits tensor [batch_size, vocab_size]
 * @param generated_tokens Previously generated tokens [batch_size, gen_len]
 * @param batch_size Batch size
 * @param vocab_size Vocabulary size
 * @param gen_len Number of previously generated tokens
 * @param penalty Repetition penalty (> 1.0 = discourage repetition)
 * @param stream CUDA stream
 */
void apply_repetition_penalty(
    float* logits,
    const int32_t* generated_tokens,
    int batch_size,
    int vocab_size,
    int gen_len,
    float penalty,
    cudaStream_t stream
);

/**
 * Full Sampling Pipeline
 *
 * Temperature -> Top-K -> Top-P -> Sample
 *
 * @param output_tokens Output sampled token IDs [batch_size]
 * @param logits Input logits [batch_size, vocab_size] (will be modified)
 * @param batch_size Batch size
 * @param vocab_size Vocabulary size
 * @param temperature Temperature (0 = greedy)
 * @param top_k Top-K value (0 = disabled)
 * @param top_p Top-P value (1.0 = disabled)
 * @param rng_states RNG states for sampling
 * @param stream CUDA stream
 */
void sample_logits(
    int32_t* output_tokens,
    float* logits,
    int batch_size,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    curandState* rng_states,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // EDGELLM_SAMPLING_KERNEL_H
