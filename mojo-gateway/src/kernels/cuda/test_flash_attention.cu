/**
 * Flash Attention Test & Benchmark
 *
 * Tests:
 * 1. Correctness vs naive attention
 * 2. Causal masking
 * 3. Performance benchmarks
 * 4. Memory usage comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include "flash_attention.h"

#define WARMUP_RUNS 5
#define BENCHMARK_RUNS 50

// ============================================================================
// Reference Implementation (Naive Attention)
// ============================================================================

/**
 * Naive attention for correctness comparison
 * O = softmax(Q @ K^T / sqrt(d)) @ V
 */
void naive_attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_heads,
    int seq_len,
    int head_dim,
    int causal
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int bh = 0; bh < batch_heads; bh++) {
        const float* q = Q + bh * seq_len * head_dim;
        const float* k = K + bh * seq_len * head_dim;
        const float* v = V + bh * seq_len * head_dim;
        float* o = O + bh * seq_len * head_dim;

        // Allocate attention scores matrix
        float* scores = (float*)malloc(seq_len * seq_len * sizeof(float));

        // Compute Q @ K^T
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                dot *= scale;

                // Apply causal mask
                if (causal && j > i) {
                    dot = -1e9f;
                }

                scores[i * seq_len + j] = dot;
            }
        }

        // Softmax per row
        for (int i = 0; i < seq_len; i++) {
            float max_val = -1e9f;
            for (int j = 0; j < seq_len; j++) {
                max_val = fmaxf(max_val, scores[i * seq_len + j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }

            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] /= sum;
            }
        }

        // Compute scores @ V
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    val += scores[i * seq_len + j] * v[j * head_dim + d];
                }
                o[i * head_dim + d] = val;
            }
        }

        free(scores);
    }
}

// ============================================================================
// Test Utilities
// ============================================================================

void fill_random(float* data, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
        data[i] = scale * ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

float compute_max_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = fabsf(a[i] - b[i]);
        max_err = fmaxf(max_err, err);
    }
    return max_err;
}

float compute_mean_error(const float* a, const float* b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum / size;
}

// ============================================================================
// Tests
// ============================================================================

int test_correctness() {
    printf("\n=== Test: Flash Attention Correctness ===\n");

    int batch_heads = 4;
    int seq_len = 64;
    int head_dim = 64;

    int size = batch_heads * seq_len * head_dim;

    float* Q = (float*)malloc(size * sizeof(float));
    float* K = (float*)malloc(size * sizeof(float));
    float* V = (float*)malloc(size * sizeof(float));
    float* O_flash = (float*)malloc(size * sizeof(float));
    float* O_naive = (float*)malloc(size * sizeof(float));

    // Fill with random data
    srand(42);
    fill_random(Q, size, 0.5f);
    fill_random(K, size, 0.5f);
    fill_random(V, size, 0.5f);

    // Initialize Flash Attention
    if (flash_attention_init(1, batch_heads, seq_len, head_dim) != 0) {
        printf("FAILED: Could not initialize Flash Attention\n");
        return -1;
    }

    // Test non-causal attention
    printf("\nNon-causal attention (seq_len=%d, head_dim=%d):\n", seq_len, head_dim);

    naive_attention_cpu(Q, K, V, O_naive, batch_heads, seq_len, head_dim, 0);
    flash_attention_forward(Q, K, V, O_flash, batch_heads, seq_len, head_dim, 0);

    float max_err = compute_max_error(O_flash, O_naive, size);
    float mean_err = compute_mean_error(O_flash, O_naive, size);

    printf("  Max error: %.6f\n", max_err);
    printf("  Mean error: %.6f\n", mean_err);
    printf("  Status: %s\n", max_err < 1e-3f ? "PASS" : "FAIL");

    // Test causal attention
    printf("\nCausal attention (seq_len=%d, head_dim=%d):\n", seq_len, head_dim);

    naive_attention_cpu(Q, K, V, O_naive, batch_heads, seq_len, head_dim, 1);
    flash_attention_forward(Q, K, V, O_flash, batch_heads, seq_len, head_dim, 1);

    max_err = compute_max_error(O_flash, O_naive, size);
    mean_err = compute_mean_error(O_flash, O_naive, size);

    printf("  Max error: %.6f\n", max_err);
    printf("  Mean error: %.6f\n", mean_err);
    printf("  Status: %s\n", max_err < 1e-3f ? "PASS" : "FAIL");

    // Cleanup
    free(Q);
    free(K);
    free(V);
    free(O_flash);
    free(O_naive);
    flash_attention_cleanup();

    return (max_err < 1e-3f) ? 0 : -1;
}

int test_kv_cache() {
    printf("\n=== Test: KV Cache Decode ===\n");

    int batch_heads = 9;  // SmolLM-135M has 9 heads
    int head_dim = 64;
    int max_cache = 512;

    // Initialize
    if (flash_attention_init(1, batch_heads, max_cache, head_dim) != 0) {
        printf("FAILED: Could not initialize Flash Attention\n");
        return -1;
    }
    if (flash_attention_init_kv_cache(1, batch_heads, max_cache, head_dim) != 0) {
        printf("FAILED: Could not initialize KV cache\n");
        return -1;
    }

    int single_size = batch_heads * head_dim;

    float* Q = (float*)malloc(single_size * sizeof(float));
    float* K = (float*)malloc(single_size * sizeof(float));
    float* V = (float*)malloc(single_size * sizeof(float));
    float* O = (float*)malloc(single_size * sizeof(float));

    srand(42);

    // Simulate decoding 10 tokens
    printf("Simulating 10-token decode with KV cache...\n");

    for (int pos = 0; pos < 10; pos++) {
        fill_random(Q, single_size, 0.5f);
        fill_random(K, single_size, 0.5f);
        fill_random(V, single_size, 0.5f);

        int ret = flash_attention_decode(Q, K, V, O, batch_heads, pos, head_dim);
        if (ret != 0) {
            printf("FAILED at position %d\n", pos);
            return -1;
        }
    }

    printf("  Decoded 10 tokens successfully\n");
    printf("  Status: PASS\n");

    free(Q);
    free(K);
    free(V);
    free(O);
    flash_attention_cleanup();

    return 0;
}

int benchmark_flash_attention() {
    printf("\n=== Benchmark: Flash Attention ===\n");

    // SmolLM-135M configuration
    int batch = 1;
    int num_heads = 9;
    int head_dim = 64;
    int batch_heads = batch * num_heads;

    // Test different sequence lengths
    int seq_lengths[] = {64, 128, 256, 512, 1024};
    int num_tests = sizeof(seq_lengths) / sizeof(seq_lengths[0]);

    printf("\n%-12s | %-12s | %-12s | %-14s\n",
           "Seq Length", "FA Time (ms)", "Memory (MB)", "Throughput");
    printf("-------------|--------------|--------------|----------------\n");

    for (int t = 0; t < num_tests; t++) {
        int seq_len = seq_lengths[t];
        int size = batch_heads * seq_len * head_dim;

        float* Q = (float*)malloc(size * sizeof(float));
        float* K = (float*)malloc(size * sizeof(float));
        float* V = (float*)malloc(size * sizeof(float));
        float* O = (float*)malloc(size * sizeof(float));

        fill_random(Q, size);
        fill_random(K, size);
        fill_random(V, size);

        // Initialize for this seq_len
        flash_attention_init(batch, num_heads, seq_len, head_dim);

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; i++) {
            flash_attention_forward(Q, K, V, O, batch_heads, seq_len, head_dim, 1);
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCHMARK_RUNS; i++) {
            flash_attention_forward(Q, K, V, O, batch_heads, seq_len, head_dim, 1);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / BENCHMARK_RUNS;

        // Memory: Flash Attention uses O(N) instead of O(N^2)
        // Naive: N^2 * 4 bytes for attention matrix
        // Flash: 4 * BLOCK_SIZE * head_dim * 4 bytes for tiles
        double naive_memory_mb = (double)seq_len * seq_len * sizeof(float) / (1024 * 1024);
        double flash_memory_mb = 4.0 * 64 * head_dim * sizeof(float) / (1024 * 1024);

        // Throughput: tokens processed per second (rough estimate)
        // For decode, this would be much higher
        double tokens_per_sec = seq_len / (avg_ms / 1000.0);

        printf("%-12d | %-12.3f | %-12.3f | %-14.1f\n",
               seq_len, avg_ms, flash_memory_mb, tokens_per_sec);

        free(Q);
        free(K);
        free(V);
        free(O);
        flash_attention_cleanup();
    }

    return 0;
}

int benchmark_decode() {
    printf("\n=== Benchmark: Flash Attention Decode (Single Token) ===\n");

    int batch = 1;
    int num_heads = 9;
    int head_dim = 64;
    int batch_heads = batch * num_heads;
    int max_cache = 2048;

    // Initialize
    flash_attention_init(batch, num_heads, max_cache, head_dim);
    flash_attention_init_kv_cache(batch, num_heads, max_cache, head_dim);

    int single_size = batch_heads * head_dim;
    float* Q = (float*)malloc(single_size * sizeof(float));
    float* K = (float*)malloc(single_size * sizeof(float));
    float* V = (float*)malloc(single_size * sizeof(float));
    float* O = (float*)malloc(single_size * sizeof(float));

    fill_random(Q, single_size);
    fill_random(K, single_size);
    fill_random(V, single_size);

    printf("\n%-14s | %-12s | %-14s\n",
           "Cache Length", "Time (ms)", "Tokens/sec");
    printf("---------------|--------------|----------------\n");

    // Test decode at different cache positions
    int cache_positions[] = {1, 10, 50, 100, 256, 512, 1024};
    int num_tests = sizeof(cache_positions) / sizeof(cache_positions[0]);

    // Fill cache to max test position
    for (int pos = 0; pos < 1024; pos++) {
        fill_random(K, single_size);
        fill_random(V, single_size);
        flash_attention_update_kv_cache(K, V, batch_heads, pos, 1, head_dim);
    }

    for (int t = 0; t < num_tests; t++) {
        int cache_pos = cache_positions[t] - 1;  // 0-indexed

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; i++) {
            flash_attention_decode(Q, K, V, O, batch_heads, cache_pos, head_dim);
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCHMARK_RUNS * 2; i++) {  // More runs for decode
            flash_attention_decode(Q, K, V, O, batch_heads, cache_pos, head_dim);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / (BENCHMARK_RUNS * 2);
        double tokens_per_sec = 1000.0 / avg_ms;

        printf("%-14d | %-12.4f | %-14.1f\n",
               cache_positions[t], avg_ms, tokens_per_sec);
    }

    free(Q);
    free(K);
    free(V);
    free(O);
    flash_attention_cleanup();

    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("Flash Attention Test Suite\n");
    printf("==========================\n");

    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("ERROR: No CUDA devices found\n");
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("VRAM: %.2f GB\n", prop.totalGlobalMem / 1e9);

    int failed = 0;

    // Run tests
    if (test_correctness() != 0) failed++;
    if (test_kv_cache() != 0) failed++;

    // Run benchmarks
    benchmark_flash_attention();
    benchmark_decode();

    printf("\n==========================\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failed);
    }

    return failed;
}
