/**
 * INT8 Tensor Core Flash Attention Test & Benchmark
 *
 * Tests:
 * 1. Quantization correctness
 * 2. INT8 attention vs FP32 reference
 * 3. Performance benchmarks
 * 4. Comparison with FP32 Flash Attention
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include "flash_attention_int8.h"
#include "flash_attention.h"

#define WARMUP_RUNS 10
#define BENCHMARK_RUNS 100

// ============================================================================
// Reference Implementation
// ============================================================================

/**
 * CPU reference attention for correctness validation
 */
void reference_attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int cache_len,
    int head_dim,
    float scale
) {
    // Compute Q @ K^T
    float* scores = (float*)malloc(cache_len * sizeof(float));
    float max_score = -1e9f;

    for (int j = 0; j < cache_len; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[d] * K[j * head_dim + d];
        }
        scores[j] = dot * scale;
        max_score = fmaxf(max_score, scores[j]);
    }

    // Softmax
    float sum = 0.0f;
    for (int j = 0; j < cache_len; j++) {
        scores[j] = expf(scores[j] - max_score);
        sum += scores[j];
    }
    for (int j = 0; j < cache_len; j++) {
        scores[j] /= sum;
    }

    // Output = scores @ V
    for (int d = 0; d < head_dim; d++) {
        float val = 0.0f;
        for (int j = 0; j < cache_len; j++) {
            val += scores[j] * V[j * head_dim + d];
        }
        O[d] = val;
    }

    free(scores);
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

float compute_relative_error(const float* ref, const float* test, int size) {
    float sum_err = 0.0f;
    float sum_ref = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_err += fabsf(ref[i] - test[i]);
        sum_ref += fabsf(ref[i]);
    }
    return (sum_ref > 0.0f) ? (sum_err / sum_ref) : sum_err;
}

// ============================================================================
// Tests
// ============================================================================

int test_quantization() {
    printf("\n=== Test: INT8 Quantization ===\n");

    int size = 1024;
    float* input = (float*)malloc(size * sizeof(float));
    int8_t* output = (int8_t*)malloc(size * sizeof(int8_t));
    float scale;

    // Fill with random data
    srand(42);
    fill_random(input, size, 2.0f);  // Range [-2, 2]

    // Find expected max
    float expected_max = 0.0f;
    for (int i = 0; i < size; i++) {
        expected_max = fmaxf(expected_max, fabsf(input[i]));
    }

    // Quantize
    quantize_tensor_int8(input, output, &scale, size);

    printf("  Input range: [%.3f, %.3f]\n", -expected_max, expected_max);
    printf("  Quantization scale: %.6f\n", scale);
    printf("  Expected scale: %.6f\n", expected_max / 127.0f);

    // Verify dequantization
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float dequant = (float)output[i] * scale;
        float err = fabsf(input[i] - dequant);
        max_err = fmaxf(max_err, err);
    }

    printf("  Max dequantization error: %.6f\n", max_err);
    printf("  Status: %s\n", max_err < 0.1f ? "PASS" : "FAIL");

    free(input);
    free(output);

    return (max_err < 0.1f) ? 0 : -1;
}

int test_int8_attention_correctness() {
    printf("\n=== Test: INT8 Attention Correctness ===\n");

    int batch_heads = 9;  // SmolLM-135M
    int head_dim = 64;
    int cache_len = 64;

    // Initialize INT8 Flash Attention
    if (flash_attention_int8_init(batch_heads, 2048, head_dim) != 0) {
        printf("FAILED: Could not initialize INT8 Flash Attention\n");
        return -1;
    }

    int single_size = batch_heads * head_dim;
    int cache_size = batch_heads * cache_len * head_dim;

    float* Q = (float*)malloc(single_size * sizeof(float));
    float* K_cache = (float*)malloc(cache_size * sizeof(float));
    float* V_cache = (float*)malloc(cache_size * sizeof(float));
    float* O_int8 = (float*)malloc(single_size * sizeof(float));
    float* O_ref = (float*)malloc(single_size * sizeof(float));

    srand(42);
    fill_random(Q, single_size, 0.5f);
    fill_random(K_cache, cache_size, 0.5f);
    fill_random(V_cache, cache_size, 0.5f);

    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute reference for each head
    for (int bh = 0; bh < batch_heads; bh++) {
        reference_attention_cpu(
            Q + bh * head_dim,
            K_cache + bh * cache_len * head_dim,
            V_cache + bh * cache_len * head_dim,
            O_ref + bh * head_dim,
            cache_len, head_dim, scale
        );
    }

    // Fill cache progressively
    for (int pos = 0; pos < cache_len; pos++) {
        flash_attention_int8_decode_fp32(
            Q,
            K_cache + pos * head_dim,  // Just use same K_cache sequentially for simplicity
            V_cache + pos * head_dim,
            O_int8,
            batch_heads, pos, head_dim
        );
    }

    // Final decode at full cache
    // First, we need to properly fill the cache
    // Reinitialize and fill cache
    flash_attention_int8_cleanup();
    flash_attention_int8_init(batch_heads, 2048, head_dim);

    // Fill cache with all K, V values
    for (int pos = 0; pos < cache_len; pos++) {
        // Create K, V for this position
        float* K_pos = (float*)malloc(single_size * sizeof(float));
        float* V_pos = (float*)malloc(single_size * sizeof(float));

        for (int bh = 0; bh < batch_heads; bh++) {
            for (int d = 0; d < head_dim; d++) {
                K_pos[bh * head_dim + d] = K_cache[bh * cache_len * head_dim + pos * head_dim + d];
                V_pos[bh * head_dim + d] = V_cache[bh * cache_len * head_dim + pos * head_dim + d];
            }
        }

        flash_attention_int8_decode_fp32(Q, K_pos, V_pos, O_int8, batch_heads, pos, head_dim);

        free(K_pos);
        free(V_pos);
    }

    // Compare results
    float max_err = compute_max_error(O_int8, O_ref, single_size);
    float mean_err = compute_mean_error(O_int8, O_ref, single_size);
    float rel_err = compute_relative_error(O_ref, O_int8, single_size);

    printf("  Cache length: %d\n", cache_len);
    printf("  Max error: %.6f\n", max_err);
    printf("  Mean error: %.6f\n", mean_err);
    printf("  Relative error: %.4f%%\n", rel_err * 100.0f);

    // INT8 has more quantization error, allow 5% relative error
    int pass = rel_err < 0.05f;
    printf("  Status: %s\n", pass ? "PASS" : "FAIL");

    free(Q);
    free(K_cache);
    free(V_cache);
    free(O_int8);
    free(O_ref);
    flash_attention_int8_cleanup();

    return pass ? 0 : -1;
}

int benchmark_int8_vs_fp32() {
    printf("\n=== Benchmark: INT8 vs FP32 Flash Attention ===\n");

    int batch = 1;
    int num_heads = 9;
    int head_dim = 64;
    int batch_heads = batch * num_heads;
    int max_cache = 2048;

    // Initialize both
    flash_attention_init(batch, num_heads, max_cache, head_dim);
    flash_attention_init_kv_cache(batch, num_heads, max_cache, head_dim);
    flash_attention_int8_init(batch_heads, max_cache, head_dim);

    int single_size = batch_heads * head_dim;
    float* Q = (float*)malloc(single_size * sizeof(float));
    float* K = (float*)malloc(single_size * sizeof(float));
    float* V = (float*)malloc(single_size * sizeof(float));
    float* O = (float*)malloc(single_size * sizeof(float));

    srand(42);

    // Test at different cache lengths
    int cache_lengths[] = {64, 128, 256, 512};
    int num_tests = sizeof(cache_lengths) / sizeof(cache_lengths[0]);

    printf("\n%-12s | %-12s | %-12s | %-10s\n",
           "Cache Len", "FP32 (ms)", "INT8 (ms)", "Speedup");
    printf("-------------|--------------|--------------|------------\n");

    for (int t = 0; t < num_tests; t++) {
        int cache_len = cache_lengths[t];

        // Fill both caches
        for (int pos = 0; pos < cache_len; pos++) {
            fill_random(K, single_size);
            fill_random(V, single_size);
            flash_attention_update_kv_cache(K, V, batch_heads, pos, 1, head_dim);
            flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, pos, head_dim);
        }

        fill_random(Q, single_size);
        fill_random(K, single_size);
        fill_random(V, single_size);

        // Warmup FP32
        for (int i = 0; i < WARMUP_RUNS; i++) {
            flash_attention_decode(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
        }

        // Benchmark FP32
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCHMARK_RUNS; i++) {
            flash_attention_decode(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double fp32_ms = std::chrono::duration<double, std::milli>(end - start).count() / BENCHMARK_RUNS;

        // Warmup INT8
        for (int i = 0; i < WARMUP_RUNS; i++) {
            flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
        }

        // Benchmark INT8
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCHMARK_RUNS; i++) {
            flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
        }
        end = std::chrono::high_resolution_clock::now();
        double int8_ms = std::chrono::duration<double, std::milli>(end - start).count() / BENCHMARK_RUNS;

        double speedup = fp32_ms / int8_ms;

        printf("%-12d | %-12.4f | %-12.4f | %-10.2fx\n",
               cache_len, fp32_ms, int8_ms, speedup);
    }

    free(Q);
    free(K);
    free(V);
    free(O);
    flash_attention_cleanup();
    flash_attention_int8_cleanup();

    return 0;
}

int benchmark_int8_throughput() {
    printf("\n=== Benchmark: INT8 Tensor Core Decode Throughput ===\n");

    int batch = 1;
    int num_heads = 9;
    int head_dim = 64;
    int batch_heads = batch * num_heads;
    int num_layers = 9;  // SmolLM-135M

    flash_attention_int8_init(batch_heads, 2048, head_dim);

    int single_size = batch_heads * head_dim;
    float* Q = (float*)malloc(single_size * sizeof(float));
    float* K = (float*)malloc(single_size * sizeof(float));
    float* V = (float*)malloc(single_size * sizeof(float));
    float* O = (float*)malloc(single_size * sizeof(float));

    srand(42);

    // Fill cache to 256 tokens
    int cache_len = 256;
    for (int pos = 0; pos < cache_len; pos++) {
        fill_random(K, single_size);
        fill_random(V, single_size);
        flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, pos, head_dim);
    }

    fill_random(Q, single_size);
    fill_random(K, single_size);
    fill_random(V, single_size);

    printf("\nSmolLM-135M Configuration:\n");
    printf("  Heads: %d\n", num_heads);
    printf("  Head dim: %d\n", head_dim);
    printf("  Layers: %d\n", num_layers);
    printf("  Cache length: %d\n", cache_len);

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
    }

    // Benchmark single attention layer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_layer_ms = total_ms / BENCHMARK_RUNS;
    double per_token_attn_ms = per_layer_ms * num_layers;
    double attn_throughput = 1000.0 / per_token_attn_ms;

    // Estimate full inference (attention typically 30-40% of compute)
    double attention_fraction = 0.35;
    double per_token_total_ms = per_token_attn_ms / attention_fraction;
    double estimated_throughput = 1000.0 / per_token_total_ms;

    printf("\nResults:\n");
    printf("  Per-layer attention: %.4f ms\n", per_layer_ms);
    printf("  Per-token attention (9 layers): %.4f ms\n", per_token_attn_ms);
    printf("  Attention-only throughput: %.1f tok/s\n", attn_throughput);
    printf("\nEstimated full inference (attention=35%%):\n");
    printf("  Per-token total: %.3f ms\n", per_token_total_ms);
    printf("  Throughput: %.1f tok/s\n", estimated_throughput);
    printf("\nTargets:\n");
    printf("  Goal: 630 tok/s\n");
    printf("  Ollama baseline: 423 tok/s\n");

    printf("\nJSON Results:\n");
    printf("{\n");
    printf("  \"kernel\": \"INT8 Tensor Core\",\n");
    printf("  \"per_layer_attention_ms\": %.4f,\n", per_layer_ms);
    printf("  \"per_token_attention_ms\": %.4f,\n", per_token_attn_ms);
    printf("  \"attention_throughput\": %.1f,\n", attn_throughput);
    printf("  \"estimated_total_throughput\": %.1f,\n", estimated_throughput);
    printf("  \"target_throughput\": 630,\n");
    printf("  \"ollama_baseline\": 423\n");
    printf("}\n");

    free(Q);
    free(K);
    free(V);
    free(O);
    flash_attention_int8_cleanup();

    return 0;
}

/**
 * Benchmark __dp4a optimized kernel with statistics
 */
int benchmark_dp4a_optimized() {
    printf("\n=== __dp4a Optimized INT8 Attention Benchmark ===\n");

    // Qwen 0.5B configuration
    int num_heads = 14;
    int head_dim = 64;
    int batch = 1;
    int batch_heads = batch * num_heads;
    int num_layers = 24;  // Qwen 0.5B
    int cache_len = 256;

    flash_attention_int8_init(batch_heads, 2048, head_dim);

    int single_size = batch_heads * head_dim;
    float* Q = (float*)malloc(single_size * sizeof(float));
    float* K = (float*)malloc(single_size * sizeof(float));
    float* V = (float*)malloc(single_size * sizeof(float));
    float* O = (float*)malloc(single_size * sizeof(float));

    srand(42);

    // Fill cache
    for (int pos = 0; pos < cache_len; pos++) {
        fill_random(K, single_size);
        fill_random(V, single_size);
        flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, pos, head_dim);
    }

    fill_random(Q, single_size);
    fill_random(K, single_size);
    fill_random(V, single_size);

    printf("\nQwen 2.5 0.5B Configuration:\n");
    printf("  Heads: %d\n", num_heads);
    printf("  Head dim: %d\n", head_dim);
    printf("  Layers: %d\n", num_layers);
    printf("  Cache length: %d\n", cache_len);

    // Warmup
    printf("\nWarmup: %d iterations...\n", WARMUP_RUNS);
    for (int i = 0; i < WARMUP_RUNS; i++) {
        flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
    }
    cudaDeviceSynchronize();

    // Collect per-iteration latencies
    int runs = 200;
    double* latencies = (double*)malloc(runs * sizeof(double));

    printf("Benchmark: %d iterations...\n", runs);
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        flash_attention_int8_decode_fp32(Q, K, V, O, batch_heads, cache_len - 1, head_dim);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        latencies[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    // Sort for percentiles
    for (int i = 0; i < runs - 1; i++) {
        for (int j = i + 1; j < runs; j++) {
            if (latencies[j] < latencies[i]) {
                double tmp = latencies[i];
                latencies[i] = latencies[j];
                latencies[j] = tmp;
            }
        }
    }

    double min_us = latencies[0];
    double max_us = latencies[runs - 1];
    double median_us = latencies[runs / 2];
    double p95_us = latencies[(int)(runs * 0.95)];
    double p99_us = latencies[(int)(runs * 0.99)];

    double mean_us = 0;
    for (int i = 0; i < runs; i++) mean_us += latencies[i];
    mean_us /= runs;

    double variance = 0;
    for (int i = 0; i < runs; i++) {
        double diff = latencies[i] - mean_us;
        variance += diff * diff;
    }
    double std_us = sqrt(variance / runs);

    // Convert to ms
    double median_ms = median_us / 1000.0;
    double per_token_attn_ms = median_ms * num_layers;
    double attn_throughput = 1000.0 / per_token_attn_ms;

    double attention_fraction = 0.35;
    double per_token_total_ms = per_token_attn_ms / attention_fraction;
    double estimated_throughput = 1000.0 / per_token_total_ms;

    printf("\n=== __dp4a Kernel Results ===\n");
    printf("  Layer latency (median): %.2f us (%.4f ms)\n", median_us, median_ms);
    printf("  Layer latency (mean):   %.2f ± %.2f us\n", mean_us, std_us);
    printf("  Layer latency (min):    %.2f us\n", min_us);
    printf("  Layer latency (max):    %.2f us\n", max_us);
    printf("  Layer latency (P95):    %.2f us\n", p95_us);
    printf("  Layer latency (P99):    %.2f us\n", p99_us);
    printf("  Jitter (std/median):    %.2f%%\n", (std_us / median_us) * 100);
    printf("\n");
    printf("  Per-token attention (%d layers): %.3f ms\n", num_layers, per_token_attn_ms);
    printf("  Attention-only throughput:       %.1f tok/s\n", attn_throughput);
    printf("  Estimated total throughput:      %.1f tok/s\n", estimated_throughput);

    printf("\nJSON Output:\n");
    printf("{\n");
    printf("  \"kernel\": \"INT8_dp4a_optimized\",\n");
    printf("  \"model\": \"Qwen_2.5_0.5B\",\n");
    printf("  \"layer_latency_us\": %.2f,\n", median_us);
    printf("  \"layer_latency_ms\": %.4f,\n", median_ms);
    printf("  \"std_us\": %.2f,\n", std_us);
    printf("  \"p95_us\": %.2f,\n", p95_us);
    printf("  \"p99_us\": %.2f,\n", p99_us);
    printf("  \"per_token_attn_ms\": %.3f,\n", per_token_attn_ms);
    printf("  \"attn_throughput\": %.1f,\n", attn_throughput);
    printf("  \"estimated_throughput\": %.1f,\n", estimated_throughput);
    printf("  \"target_throughput\": 630,\n");
    printf("  \"ollama_baseline\": 423\n");
    printf("}\n");

    free(Q);
    free(K);
    free(V);
    free(O);
    free(latencies);
    flash_attention_int8_cleanup();

    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("INT8 __dp4a Tensor Core Flash Attention Test Suite\n");
    printf("===================================================\n");

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

    if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
        printf("WARNING: INT8 Tensor Cores require sm_75+ (Turing or newer)\n");
        printf("         Your GPU is sm_%d%d. Tests may fail or run slowly.\n",
               prop.major, prop.minor);
    }

    int failed = 0;

    // Run tests
    if (test_quantization() != 0) failed++;
    if (test_int8_attention_correctness() != 0) failed++;

    // Run benchmarks
    benchmark_int8_vs_fp32();
    benchmark_int8_throughput();

    // Run optimized __dp4a benchmark
    benchmark_dp4a_optimized();

    printf("\n===================================================\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failed);
    }

    return failed;
}
