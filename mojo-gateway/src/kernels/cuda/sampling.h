/**
 * Temperature Sampling Header for EdgeLLM
 */

#ifndef SAMPLING_H
#define SAMPLING_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize sampling RNG.
 * @param seed Random seed
 * @return 0 on success
 */
int sampling_init(unsigned long long seed);

/**
 * Cleanup sampling resources.
 */
void sampling_cleanup();

/**
 * Sample from logits with temperature, top-k, and top-p.
 *
 * @param result_gpu Output token ID (device pointer)
 * @param logits_gpu Logits array (device pointer, modified in-place)
 * @param vocab_size Vocabulary size
 * @param temperature Temperature (0 = greedy, 1 = default)
 * @param top_k Number of top tokens to consider (0 = disabled)
 * @param top_p Cumulative probability threshold (1.0 = disabled)
 * @param stream CUDA stream
 * @return 0 on success, -1 if should use argmax instead
 */
int gpu_sample_temperature(
    int* result_gpu,
    float* logits_gpu,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // SAMPLING_H
