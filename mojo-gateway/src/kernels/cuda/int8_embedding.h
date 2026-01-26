/**
 * INT8 Embedding Kernel Header
 *
 * Fast INT8 logit GEMV to replace cuBLAS FP32.
 * Target: 3.6ms -> <1ms for logit computation.
 */

#ifndef INT8_EMBEDDING_H
#define INT8_EMBEDDING_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize INT8 embedding kernels with stream.
 */
int int8_emb_init(cudaStream_t stream);

/**
 * INT8 embedding lookup: dequantize single row to FP32.
 */
int int8_embedding_lookup(
    float* out_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int token_id,
    int dim
);

/**
 * INT8 logit GEMV: logits = embedding^T @ x
 * Uses standard INT8 with FP32 accumulation.
 */
int int8_logit_gemv(
    float* logits_gpu,
    const float* x_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int vocab_size,
    int dim
);

/**
 * INT8 logit GEMV using dp4a intrinsic (fastest on Turing).
 */
int int8_logit_gemv_dp4a(
    float* logits_gpu,
    const float* x_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int vocab_size,
    int dim
);

/**
 * Benchmark function for INT8 logit GEMV.
 */
float int8_logit_benchmark(
    float* logits_gpu,
    const float* x_gpu,
    const int8_t* emb_int8_gpu,
    const half* scales_gpu,
    int vocab_size,
    int dim,
    int n_iters,
    int use_dp4a
);

#ifdef __cplusplus
}
#endif

#endif // INT8_EMBEDDING_H
