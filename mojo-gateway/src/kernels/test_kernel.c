/**
 * T-MAC Kernel Test Suite
 *
 * Tests for verifying correctness of SIMD kernels.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "tmac_kernel.h"

#define EPSILON 1e-5f
#define RED "\033[31m"
#define GREEN "\033[32m"
#define RESET "\033[0m"

static int tests_passed = 0;
static int tests_failed = 0;

void assert_close(float a, float b, const char* test_name) {
    if (fabsf(a - b) < EPSILON) {
        printf(GREEN "[PASS]" RESET " %s: %.6f == %.6f\n", test_name, a, b);
        tests_passed++;
    } else {
        printf(RED "[FAIL]" RESET " %s: %.6f != %.6f (diff: %.6f)\n",
               test_name, a, b, fabsf(a - b));
        tests_failed++;
    }
}

void assert_array_close(const float* a, const float* b, int size, const char* test_name) {
    float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    if (max_diff < EPSILON) {
        printf(GREEN "[PASS]" RESET " %s: max diff = %.6f\n", test_name, max_diff);
        tests_passed++;
    } else {
        printf(RED "[FAIL]" RESET " %s: max diff = %.6f at index %d (%.6f vs %.6f)\n",
               test_name, max_diff, max_idx, a[max_idx], b[max_idx]);
        tests_failed++;
    }
}

// Reference implementations for testing
void rmsnorm_ref(float* output, const float* input, const float* weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += input[i] * input[i];
    }
    float rms = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * rms * weight[i];
    }
}

void softmax_ref(float* output, const float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Test RMSNorm
void test_rmsnorm(void) {
    printf("\n=== Testing RMSNorm ===\n");

    const int sizes[] = {64, 128, 256, 512, 1024, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        float* input = malloc(size * sizeof(float));
        float* weight = malloc(size * sizeof(float));
        float* output_ref = malloc(size * sizeof(float));
        float* output_simd = malloc(size * sizeof(float));

        // Initialize with random values
        for (int i = 0; i < size; i++) {
            input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            weight[i] = (float)rand() / RAND_MAX * 2.0f;
        }

        // Compute reference
        rmsnorm_ref(output_ref, input, weight, size, 1e-6f);

        // Compute SIMD (AVX2 or fallback)
        rmsnorm_avx2(output_simd, input, weight, size, 1e-6f);

        char test_name[64];
        snprintf(test_name, sizeof(test_name), "rmsnorm_avx2 size=%d", size);
        assert_array_close(output_ref, output_simd, size, test_name);

        free(input);
        free(weight);
        free(output_ref);
        free(output_simd);
    }
}

// Test Softmax
void test_softmax(void) {
    printf("\n=== Testing Softmax ===\n");

    const int sizes[] = {64, 128, 256, 512, 1024, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        float* input = malloc(size * sizeof(float));
        float* output_ref = malloc(size * sizeof(float));
        float* output_simd = malloc(size * sizeof(float));

        // Initialize with random logits
        for (int i = 0; i < size; i++) {
            input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
        }

        // Compute reference
        softmax_ref(output_ref, input, size);

        // Compute SIMD
        softmax_avx2(output_simd, input, size);

        char test_name[64];
        snprintf(test_name, sizeof(test_name), "softmax_avx2 size=%d", size);
        assert_array_close(output_ref, output_simd, size, test_name);

        // Verify softmax properties
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += output_simd[i];
        }
        snprintf(test_name, sizeof(test_name), "softmax_sum_to_1 size=%d", size);
        assert_close(sum, 1.0f, test_name);

        free(input);
        free(output_ref);
        free(output_simd);
    }
}

// Test LUT Building
void test_build_lut(void) {
    printf("\n=== Testing LUT Building ===\n");

    const int size = 64;
    const int group_size = 4;
    const int num_groups = size / group_size;

    float* activations = malloc(size * sizeof(float));
    float* lut = malloc(num_groups * 256 * sizeof(float));

    // Initialize activations
    for (int i = 0; i < size; i++) {
        activations[i] = (float)(i + 1) / size;
    }

    // Build LUT
    build_lut(lut, activations, size, group_size);

    // Verify some entries
    // Pattern 0b00000000 = all -1s
    // Pattern 0b10101010 = all 0s
    // Pattern 0b01010101 = alternating -1, 0

    // Group 0, pattern 0 (all -1): sum = -(a0 + a1 + a2 + a3)
    float expected_0 = -(activations[0] + activations[1] + activations[2] + activations[3]);
    assert_close(lut[0 * 256 + 0], expected_0, "lut[0,0] all -1");

    printf("LUT building test completed\n");

    free(activations);
    free(lut);
}

// Benchmark
void benchmark(void) {
    printf("\n=== Benchmark ===\n");

    const int size = 4096;
    const int iterations = 10000;

    float* input = malloc(size * sizeof(float));
    float* weight = malloc(size * sizeof(float));
    float* output = malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        input[i] = (float)rand() / RAND_MAX;
        weight[i] = (float)rand() / RAND_MAX;
    }

    // Benchmark RMSNorm
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        rmsnorm_avx2(output, input, weight, size, 1e-6f);
    }
    clock_t end = clock();
    double rmsnorm_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("RMSNorm (%d x %d): %.3f ms/iter, %.2f GB/s\n",
           iterations, size,
           rmsnorm_time / iterations * 1000,
           (size * 3.0 * sizeof(float) * iterations) / rmsnorm_time / 1e9);

    // Benchmark Softmax
    start = clock();
    for (int i = 0; i < iterations; i++) {
        softmax_avx2(output, input, size);
    }
    end = clock();
    double softmax_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Softmax (%d x %d): %.3f ms/iter, %.2f GB/s\n",
           iterations, size,
           softmax_time / iterations * 1000,
           (size * 2.0 * sizeof(float) * iterations) / softmax_time / 1e9);

    free(input);
    free(weight);
    free(output);
}

// Test CPU features
void test_cpu_features(void) {
    printf("\n=== CPU Features ===\n");

    int features = get_cpu_features();

    printf("AVX2:   %s\n", (features & 1) ? GREEN "Yes" RESET : RED "No" RESET);
    printf("AVX512: %s\n", (features & 2) ? GREEN "Yes" RESET : RED "No" RESET);
    printf("NEON:   %s\n", (features & 4) ? GREEN "Yes" RESET : RED "No" RESET);
}

int main(void) {
    printf("T-MAC Kernel Test Suite\n");
    printf("========================\n");

    srand(42);  // Reproducible tests

    test_cpu_features();
    test_rmsnorm();
    test_softmax();
    test_build_lut();
    benchmark();

    printf("\n========================\n");
    printf("Tests passed: " GREEN "%d" RESET "\n", tests_passed);
    printf("Tests failed: %s%d%s\n",
           tests_failed > 0 ? RED : GREEN,
           tests_failed,
           RESET);

    return tests_failed > 0 ? 1 : 0;
}
