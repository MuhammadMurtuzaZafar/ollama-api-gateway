/**
 * FlashAttention-2 Multi-GPU Benchmark and Validation Test
 *
 * Tests tensor parallel attention across multiple GPUs:
 * - Correctness validation against single-GPU reference
 * - Throughput benchmarks at various sequence lengths
 * - Scaling efficiency measurements
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include "flash_attention_v2.h"
#include "flash_attention_v2_multi_gpu.h"

#define WARMUP 20
#define RUNS 100

// ============================================================================
// Utility Functions
// ============================================================================

void fill_random(float* data, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

float max_abs_error(const float* ref, const float* test, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = fabsf(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float cosine_similarity(const float* a, const float* b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    return (denom > 0) ? (dot / denom) : 0.0f;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

double benchmark_single_gpu(float* Q, float* K, float* V, float* O,
                            int batch_heads, int cache_len, int head_dim) {
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        flash_attention_v2_decode(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
    }
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) {
        flash_attention_v2_decode(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / RUNS;
}

double benchmark_multi_gpu(float* Q, float* K, float* V, float* O,
                           int cache_len) {
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        fa2_multi_gpu_decode(Q, K, V, O, cache_len - 1);
    }
    fa2_multi_gpu_sync();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) {
        fa2_multi_gpu_decode(Q, K, V, O, cache_len - 1);
    }
    fa2_multi_gpu_sync();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / RUNS;
}

// ============================================================================
// Main Test
// ============================================================================

int main() {
    printf("\n");
    printf("==================================================\n");
    printf("  FlashAttention-2 Multi-GPU Benchmark\n");
    printf("==================================================\n\n");

    // Check available GPUs
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("Available GPUs: %d\n", device_count);

    if (device_count < 1) {
        printf("ERROR: No CUDA GPUs available!\n");
        return 1;
    }

    // Print GPU info
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  GPU %d: %s (SM %d.%d, %.1f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / 1e9);
    }
    printf("\n");

    // Test configuration (SmolLM-135M)
    int batch_heads = 9;
    int head_dim = 64;
    int max_cache_len = 2048;
    int num_layers = 9;

    printf("Configuration:\n");
    printf("  Heads: %d\n", batch_heads);
    printf("  Head dim: %d\n", head_dim);
    printf("  Max cache: %d\n", max_cache_len);
    printf("  Layers: %d\n\n", num_layers);

    // Allocate host memory
    int q_size = batch_heads * head_dim;
    float* Q = (float*)malloc(q_size * sizeof(float));
    float* K = (float*)malloc(q_size * sizeof(float));
    float* V = (float*)malloc(q_size * sizeof(float));
    float* O_single = (float*)malloc(q_size * sizeof(float));
    float* O_multi = (float*)malloc(q_size * sizeof(float));

    fill_random(Q, q_size, 42);
    fill_random(K, q_size, 123);
    fill_random(V, q_size, 456);

    // Initialize single-GPU FA2
    flash_attention_v2_init(batch_heads, max_cache_len, head_dim);

    // Test different GPU configurations
    int gpu_configs[] = {1, 2, 4, 8};
    int num_configs = sizeof(gpu_configs) / sizeof(gpu_configs[0]);

    printf("==================================================\n");
    printf("  Accuracy Validation\n");
    printf("==================================================\n\n");

    // Fill cache with some data
    int test_cache_len = 256;
    for (int pos = 0; pos < test_cache_len; pos++) {
        fill_random(K, q_size, 100 + pos);
        fill_random(V, q_size, 200 + pos);
        flash_attention_v2_decode(Q, K, V, O_single, batch_heads, pos, head_dim);
    }

    // Reference single-GPU output
    fill_random(K, q_size, 999);
    fill_random(V, q_size, 888);
    flash_attention_v2_decode(Q, K, V, O_single, batch_heads, test_cache_len - 1, head_dim);

    printf("| GPUs | Max Error  | Cosine Sim | Status |\n");
    printf("|------|------------|------------|--------|\n");

    for (int c = 0; c < num_configs; c++) {
        int num_gpus = gpu_configs[c];
        if (num_gpus > device_count) continue;

        // Initialize multi-GPU
        int ret = fa2_multi_gpu_init(num_gpus, batch_heads, max_cache_len, head_dim);
        if (ret != 0) {
            printf("| %4d | INIT FAILED | - | SKIP |\n", num_gpus);
            continue;
        }

        // Fill multi-GPU cache
        for (int pos = 0; pos < test_cache_len; pos++) {
            fill_random(K, q_size, 100 + pos);
            fill_random(V, q_size, 200 + pos);
            fa2_multi_gpu_decode(Q, K, V, O_multi, pos);
        }

        // Test same query
        fill_random(K, q_size, 999);
        fill_random(V, q_size, 888);
        fa2_multi_gpu_decode(Q, K, V, O_multi, test_cache_len - 1);
        fa2_multi_gpu_sync();

        float max_err = max_abs_error(O_single, O_multi, q_size);
        float cos_sim = cosine_similarity(O_single, O_multi, q_size);
        bool pass = (max_err < 1e-3f) && (cos_sim > 0.9999f);

        printf("| %4d | %.2e | %.8f | %s |\n",
               num_gpus, max_err, cos_sim, pass ? "PASS" : "FAIL");

        fa2_multi_gpu_cleanup();
    }

    printf("\n");
    printf("==================================================\n");
    printf("  Performance Benchmark\n");
    printf("==================================================\n\n");

    int cache_lengths[] = {64, 128, 256, 512, 1024};
    int num_cache_lengths = sizeof(cache_lengths) / sizeof(cache_lengths[0]);

    // Re-init single GPU
    flash_attention_v2_cleanup();
    flash_attention_v2_init(batch_heads, max_cache_len, head_dim);

    printf("| Cache | Single GPU |");
    for (int c = 0; c < num_configs && gpu_configs[c] <= device_count; c++) {
        printf(" %d GPUs     |", gpu_configs[c]);
    }
    printf(" Best Speedup |\n");

    printf("|-------|------------|");
    for (int c = 0; c < num_configs && gpu_configs[c] <= device_count; c++) {
        printf("------------|");
    }
    printf("---------------|\n");

    double best_tok_s = 0;
    int best_gpus = 1;

    for (int cl = 0; cl < num_cache_lengths; cl++) {
        int cache_len = cache_lengths[cl];

        // Fill single-GPU cache
        for (int pos = 0; pos < cache_len; pos++) {
            fill_random(K, q_size, 100 + pos);
            fill_random(V, q_size, 200 + pos);
            flash_attention_v2_decode(Q, K, V, O_single, batch_heads, pos, head_dim);
        }

        double single_ms = benchmark_single_gpu(Q, K, V, O_single, batch_heads, cache_len, head_dim);

        printf("| %5d | %8.4f ms |", cache_len, single_ms);

        double best_ms = single_ms;
        int best_config = 1;

        for (int c = 0; c < num_configs; c++) {
            int num_gpus = gpu_configs[c];
            if (num_gpus > device_count) continue;
            if (num_gpus == 1) continue;  // Already have single GPU result

            int ret = fa2_multi_gpu_init(num_gpus, batch_heads, max_cache_len, head_dim);
            if (ret != 0) {
                printf(" %8s   |", "N/A");
                continue;
            }

            // Fill multi-GPU cache
            for (int pos = 0; pos < cache_len; pos++) {
                fill_random(K, q_size, 100 + pos);
                fill_random(V, q_size, 200 + pos);
                fa2_multi_gpu_decode(Q, K, V, O_multi, pos);
            }

            double multi_ms = benchmark_multi_gpu(Q, K, V, O_multi, cache_len);
            printf(" %8.4f ms |", multi_ms);

            if (multi_ms < best_ms) {
                best_ms = multi_ms;
                best_config = num_gpus;
            }

            fa2_multi_gpu_cleanup();
        }

        double speedup = single_ms / best_ms;
        printf(" %5.2fx (%dG)   |\n", speedup, best_config);

        // Calculate throughput for representative cache length
        if (cache_len == 256) {
            double tok_s = 1000.0 / (best_ms * num_layers);
            if (tok_s > best_tok_s) {
                best_tok_s = tok_s;
                best_gpus = best_config;
            }
        }
    }

    printf("\n");
    printf("==================================================\n");
    printf("  Summary\n");
    printf("==================================================\n\n");

    printf("Best configuration (cache_len=256):\n");
    printf("  GPUs: %d\n", best_gpus);
    printf("  Throughput: %.1f tok/s (attention only)\n", best_tok_s);
    printf("  Baseline (single GPU): 708 tok/s\n\n");

    if (device_count >= 2) {
        // Print timing breakdown
        fa2_multi_gpu_init(device_count > 4 ? 4 : device_count, batch_heads, max_cache_len, head_dim);

        int num_gpus, heads_per_gpu, total_heads;
        fa2_multi_gpu_info(&num_gpus, &heads_per_gpu, &total_heads);

        printf("Multi-GPU Configuration:\n");
        printf("  Active GPUs: %d\n", num_gpus);
        printf("  Heads per GPU: %d\n", heads_per_gpu);
        printf("  Total heads: %d\n\n", total_heads);

        // Run one decode to get timing
        fill_random(K, q_size, 999);
        fill_random(V, q_size, 888);
        fa2_multi_gpu_decode(Q, K, V, O_multi, 255);
        fa2_multi_gpu_sync();

        printf("Per-GPU Timing:\n");
        for (int g = 0; g < num_gpus; g++) {
            float compute_ms, comm_ms;
            fa2_multi_gpu_get_timing(g, &compute_ms, &comm_ms);
            printf("  GPU %d: compute=%.4f ms, comm=%.4f ms\n", g, compute_ms, comm_ms);
        }

        fa2_multi_gpu_cleanup();
    }

    // JSON output
    printf("\nJSON: {\"device_count\":%d,\"best_gpus\":%d,\"best_tok_s\":%.1f,\"single_gpu_tok_s\":708}\n",
           device_count, best_gpus, best_tok_s);

    // Cleanup
    flash_attention_v2_cleanup();
    free(Q);
    free(K);
    free(V);
    free(O_single);
    free(O_multi);

    printf("\nBenchmark complete!\n");
    return 0;
}
