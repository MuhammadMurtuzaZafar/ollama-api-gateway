/**
 * FlashAttention-2 Accuracy Validation Test
 *
 * Compares FA2 output against naive CPU attention reference
 * to verify numerical correctness across various sequence lengths.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "flash_attention_v2.h"

// ============================================================================
// CPU Reference Implementation (Naive Attention)
// ============================================================================

/**
 * Naive CPU attention: O = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * This is the mathematically exact reference implementation.
 */
void cpu_attention_reference(
    const float* Q,        // [batch_heads, head_dim]
    const float* K_cache,  // [batch_heads, seq_len, head_dim]
    const float* V_cache,  // [batch_heads, seq_len, head_dim]
    float* O,              // [batch_heads, head_dim]
    int batch_heads,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int bh = 0; bh < batch_heads; bh++) {
        const float* q = Q + bh * head_dim;
        float* o = O + bh * head_dim;

        // Allocate scores array
        float* scores = (float*)malloc(seq_len * sizeof(float));

        // Step 1: Compute Q @ K^T * scale
        float max_score = -FLT_MAX;
        for (int s = 0; s < seq_len; s++) {
            const float* k = K_cache + (bh * seq_len + s) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[d] * k[d];
            }
            scores[s] = score * scale;
            max_score = fmaxf(max_score, scores[s]);
        }

        // Step 2: Softmax with numerical stability
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        for (int s = 0; s < seq_len; s++) {
            scores[s] /= sum_exp;
        }

        // Step 3: Compute attention @ V
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int s = 0; s < seq_len; s++) {
                const float* v = V_cache + (bh * seq_len + s) * head_dim;
                val += scores[s] * v[d];
            }
            o[d] = val;
        }

        free(scores);
    }
}

// ============================================================================
// Error Metrics
// ============================================================================

struct ErrorMetrics {
    float max_abs_error;
    float mean_abs_error;
    float rmse;
    float cosine_similarity;
    int max_error_idx;
};

ErrorMetrics compute_error_metrics(
    const float* reference,
    const float* test,
    int size
) {
    ErrorMetrics m = {0, 0, 0, 0, 0};

    float sum_abs_error = 0.0f;
    float sum_sq_error = 0.0f;
    float dot_product = 0.0f;
    float norm_ref = 0.0f;
    float norm_test = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = fabsf(reference[i] - test[i]);
        sum_abs_error += diff;
        sum_sq_error += diff * diff;

        if (diff > m.max_abs_error) {
            m.max_abs_error = diff;
            m.max_error_idx = i;
        }

        dot_product += reference[i] * test[i];
        norm_ref += reference[i] * reference[i];
        norm_test += test[i] * test[i];
    }

    m.mean_abs_error = sum_abs_error / size;
    m.rmse = sqrtf(sum_sq_error / size);

    float denom = sqrtf(norm_ref) * sqrtf(norm_test);
    m.cosine_similarity = (denom > 0) ? (dot_product / denom) : 0.0f;

    return m;
}

// ============================================================================
// Test Functions
// ============================================================================

void fill_random(float* data, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

bool test_accuracy_at_length(int seq_len, int batch_heads, int head_dim, bool verbose) {
    printf("\n--- Testing seq_len=%d ---\n", seq_len);

    int q_size = batch_heads * head_dim;
    int kv_size = batch_heads * seq_len * head_dim;

    // Allocate host memory
    float* Q = (float*)malloc(q_size * sizeof(float));
    float* K_cache = (float*)malloc(kv_size * sizeof(float));
    float* V_cache = (float*)malloc(kv_size * sizeof(float));
    float* O_ref = (float*)malloc(q_size * sizeof(float));
    float* O_fa2 = (float*)malloc(q_size * sizeof(float));

    // Initialize with deterministic random data
    fill_random(Q, q_size, 42);
    fill_random(K_cache, kv_size, 123);
    fill_random(V_cache, kv_size, 456);

    // Compute CPU reference
    cpu_attention_reference(Q, K_cache, V_cache, O_ref, batch_heads, seq_len, head_dim);

    // Initialize FA2
    flash_attention_v2_init(batch_heads, seq_len + 1, head_dim);

    // Fill FA2 KV cache position by position
    for (int pos = 0; pos < seq_len; pos++) {
        float* K_pos = (float*)malloc(batch_heads * head_dim * sizeof(float));
        float* V_pos = (float*)malloc(batch_heads * head_dim * sizeof(float));

        for (int bh = 0; bh < batch_heads; bh++) {
            for (int d = 0; d < head_dim; d++) {
                K_pos[bh * head_dim + d] = K_cache[(bh * seq_len + pos) * head_dim + d];
                V_pos[bh * head_dim + d] = V_cache[(bh * seq_len + pos) * head_dim + d];
            }
        }

        // Use decode to update cache (at pos, cache contains pos+1 entries)
        float* dummy_out = (float*)malloc(q_size * sizeof(float));
        flash_attention_v2_decode(Q, K_pos, V_pos, dummy_out, batch_heads, pos, head_dim);
        free(dummy_out);
        free(K_pos);
        free(V_pos);
    }

    // Run FA2 decode at final position
    float* K_last = (float*)malloc(batch_heads * head_dim * sizeof(float));
    float* V_last = (float*)malloc(batch_heads * head_dim * sizeof(float));
    for (int bh = 0; bh < batch_heads; bh++) {
        for (int d = 0; d < head_dim; d++) {
            K_last[bh * head_dim + d] = K_cache[(bh * seq_len + seq_len - 1) * head_dim + d];
            V_last[bh * head_dim + d] = V_cache[(bh * seq_len + seq_len - 1) * head_dim + d];
        }
    }
    flash_attention_v2_decode(Q, K_last, V_last, O_fa2, batch_heads, seq_len - 1, head_dim);

    // Compute error metrics
    ErrorMetrics metrics = compute_error_metrics(O_ref, O_fa2, q_size);

    // Print results
    printf("  Max Abs Error:     %.6e (at idx %d)\n", metrics.max_abs_error, metrics.max_error_idx);
    printf("  Mean Abs Error:    %.6e\n", metrics.mean_abs_error);
    printf("  RMSE:              %.6e\n", metrics.rmse);
    printf("  Cosine Similarity: %.10f\n", metrics.cosine_similarity);

    // Print sample values if verbose
    if (verbose && q_size >= 4) {
        printf("  Sample outputs (first 4):\n");
        printf("    Ref:  [%.6f, %.6f, %.6f, %.6f]\n", O_ref[0], O_ref[1], O_ref[2], O_ref[3]);
        printf("    FA2:  [%.6f, %.6f, %.6f, %.6f]\n", O_fa2[0], O_fa2[1], O_fa2[2], O_fa2[3]);
    }

    // Determine pass/fail
    bool passed = (metrics.max_abs_error < 1e-4f) && (metrics.cosine_similarity > 0.99999f);
    printf("  Status: %s\n", passed ? "PASS" : "FAIL");

    // Cleanup
    flash_attention_v2_cleanup();
    free(Q);
    free(K_cache);
    free(V_cache);
    free(O_ref);
    free(O_fa2);
    free(K_last);
    free(V_last);

    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("==================================================\n");
    printf("  FlashAttention-2 Accuracy Validation Test\n");
    printf("==================================================\n");

    // SmolLM-135M configuration
    int batch_heads = 9;
    int head_dim = 64;

    printf("\nConfiguration: batch_heads=%d, head_dim=%d\n", batch_heads, head_dim);

    // Test various sequence lengths
    int test_lengths[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
    int num_tests = sizeof(test_lengths) / sizeof(test_lengths[0]);

    int passed = 0;
    int failed = 0;

    printf("\n");
    printf("| Seq Len | Max Error | Mean Error | Cosine Sim | Status |\n");
    printf("|---------|-----------|------------|------------|--------|\n");

    for (int i = 0; i < num_tests; i++) {
        int seq_len = test_lengths[i];

        int q_size = batch_heads * head_dim;
        int kv_size = batch_heads * seq_len * head_dim;

        float* Q = (float*)malloc(q_size * sizeof(float));
        float* K_cache = (float*)malloc(kv_size * sizeof(float));
        float* V_cache = (float*)malloc(kv_size * sizeof(float));
        float* O_ref = (float*)malloc(q_size * sizeof(float));
        float* O_fa2 = (float*)malloc(q_size * sizeof(float));

        fill_random(Q, q_size, 42);
        fill_random(K_cache, kv_size, 123);
        fill_random(V_cache, kv_size, 456);

        // CPU reference
        cpu_attention_reference(Q, K_cache, V_cache, O_ref, batch_heads, seq_len, head_dim);

        // FA2
        flash_attention_v2_init(batch_heads, seq_len + 1, head_dim);

        for (int pos = 0; pos < seq_len; pos++) {
            float* K_pos = (float*)malloc(batch_heads * head_dim * sizeof(float));
            float* V_pos = (float*)malloc(batch_heads * head_dim * sizeof(float));

            for (int bh = 0; bh < batch_heads; bh++) {
                for (int d = 0; d < head_dim; d++) {
                    K_pos[bh * head_dim + d] = K_cache[(bh * seq_len + pos) * head_dim + d];
                    V_pos[bh * head_dim + d] = V_cache[(bh * seq_len + pos) * head_dim + d];
                }
            }

            float* dummy = (float*)malloc(q_size * sizeof(float));
            flash_attention_v2_decode(Q, K_pos, V_pos, dummy, batch_heads, pos, head_dim);
            free(dummy);
            free(K_pos);
            free(V_pos);
        }

        float* K_last = (float*)malloc(batch_heads * head_dim * sizeof(float));
        float* V_last = (float*)malloc(batch_heads * head_dim * sizeof(float));
        for (int bh = 0; bh < batch_heads; bh++) {
            for (int d = 0; d < head_dim; d++) {
                K_last[bh * head_dim + d] = K_cache[(bh * seq_len + seq_len - 1) * head_dim + d];
                V_last[bh * head_dim + d] = V_cache[(bh * seq_len + seq_len - 1) * head_dim + d];
            }
        }
        flash_attention_v2_decode(Q, K_last, V_last, O_fa2, batch_heads, seq_len - 1, head_dim);

        ErrorMetrics m = compute_error_metrics(O_ref, O_fa2, q_size);

        bool pass = (m.max_abs_error < 1e-3f) && (m.cosine_similarity > 0.9999f);

        printf("| %7d | %.2e | %.2e | %.8f | %s |\n",
               seq_len, m.max_abs_error, m.mean_abs_error, m.cosine_similarity,
               pass ? "PASS" : "FAIL");

        if (pass) passed++; else failed++;

        flash_attention_v2_cleanup();
        free(Q);
        free(K_cache);
        free(V_cache);
        free(O_ref);
        free(O_fa2);
        free(K_last);
        free(V_last);
    }

    printf("\n==================================================\n");
    printf("  Results: %d/%d tests passed\n", passed, num_tests);
    printf("==================================================\n");

    // JSON output
    printf("\nJSON: {\"passed\":%d,\"failed\":%d,\"total\":%d}\n", passed, failed, num_tests);

    return (failed > 0) ? 1 : 0;
}
