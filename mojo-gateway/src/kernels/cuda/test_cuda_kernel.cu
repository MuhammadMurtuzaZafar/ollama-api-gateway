/**
 * EdgeLLM CUDA Kernel Tests
 *
 * Basic verification tests for CUDA kernel functionality.
 */

#include "tmac_kernel_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", msg); \
        return 1; \
    } \
} while(0)

#define TEST_PASS(msg) printf("PASS: %s\n", msg)

int test_cuda_available() {
    int available = cuda_available();
    if (available) {
        printf("CUDA device: %s\n", cuda_device_name());
        TEST_PASS("CUDA availability check");
    } else {
        printf("SKIP: No CUDA device available\n");
    }
    return 0;
}

int test_cuda_init() {
    if (!cuda_available()) {
        printf("SKIP: No CUDA device\n");
        return 0;
    }

    int ret = cuda_init_sized(1000000, 100000, 100000);
    TEST_ASSERT(ret == 0, "CUDA initialization");
    TEST_PASS("CUDA initialization");

    cuda_cleanup();
    TEST_PASS("CUDA cleanup");
    return 0;
}

int test_cuda_device_info() {
    if (!cuda_available()) {
        printf("SKIP: No CUDA device\n");
        return 0;
    }

    size_t total_memory;
    int sm_count, major, minor;

    int ret = cuda_device_info(&total_memory, &sm_count, &major, &minor);
    TEST_ASSERT(ret == 0, "cuda_device_info");
    TEST_ASSERT(total_memory > 0, "Total memory > 0");
    TEST_ASSERT(sm_count > 0, "SM count > 0");

    printf("  Total Memory: %.2f GB\n", total_memory / 1e9);
    printf("  SM Count: %d\n", sm_count);
    printf("  Compute: %d.%d\n", major, minor);

    TEST_PASS("CUDA device info");
    return 0;
}

int test_rmsnorm() {
    if (!cuda_available()) {
        printf("SKIP: No CUDA device\n");
        return 0;
    }

    int ret = cuda_init_sized(1000000, 100000, 100000);
    if (ret != 0) {
        printf("SKIP: CUDA init failed\n");
        return 0;
    }

    const int size = 256;
    const int batch_size = 2;
    float* input = (float*)malloc(batch_size * size * sizeof(float));
    float* output = (float*)malloc(batch_size * size * sizeof(float));
    float* weight = (float*)malloc(size * sizeof(float));

    // Initialize test data
    for (int i = 0; i < batch_size * size; i++) {
        input[i] = (float)(i % 10) / 10.0f;
    }
    for (int i = 0; i < size; i++) {
        weight[i] = 1.0f;
    }

    ret = rmsnorm_cuda(output, input, weight, batch_size, size, 1e-6f);
    TEST_ASSERT(ret == 0, "rmsnorm_cuda execution");

    // Verify output is not all zeros
    float sum = 0;
    for (int i = 0; i < batch_size * size; i++) {
        sum += fabsf(output[i]);
    }
    TEST_ASSERT(sum > 0, "rmsnorm output non-zero");

    free(input);
    free(output);
    free(weight);
    cuda_cleanup();

    TEST_PASS("RMSNorm CUDA");
    return 0;
}

int test_softmax() {
    if (!cuda_available()) {
        printf("SKIP: No CUDA device\n");
        return 0;
    }

    int ret = cuda_init_sized(1000000, 100000, 100000);
    if (ret != 0) {
        printf("SKIP: CUDA init failed\n");
        return 0;
    }

    const int size = 64;
    const int batch_size = 2;
    float* input = (float*)malloc(batch_size * size * sizeof(float));
    float* output = (float*)malloc(batch_size * size * sizeof(float));

    // Initialize test data
    for (int i = 0; i < batch_size * size; i++) {
        input[i] = (float)(i % 10) - 5.0f;
    }

    ret = softmax_cuda(output, input, batch_size, size);
    TEST_ASSERT(ret == 0, "softmax_cuda execution");

    // Verify output sums to 1 for each batch
    for (int b = 0; b < batch_size; b++) {
        float sum = 0;
        for (int i = 0; i < size; i++) {
            sum += output[b * size + i];
        }
        TEST_ASSERT(fabsf(sum - 1.0f) < 0.01f, "softmax sum equals 1");
    }

    free(input);
    free(output);
    cuda_cleanup();

    TEST_PASS("Softmax CUDA");
    return 0;
}

int main() {
    printf("EdgeLLM CUDA Kernel Tests\n");
    printf("=========================\n\n");

    int failures = 0;

    failures += test_cuda_available();
    failures += test_cuda_init();
    failures += test_cuda_device_info();
    failures += test_rmsnorm();
    failures += test_softmax();

    printf("\n=========================\n");
    if (failures == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed\n", failures);
    }

    return failures;
}
