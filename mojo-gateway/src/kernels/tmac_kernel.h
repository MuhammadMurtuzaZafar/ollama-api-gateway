/**
 * T-MAC Kernel Header
 *
 * High-performance lookup table-based inference kernel for BitNet 1.58-bit models.
 * Uses SIMD intrinsics (AVX2 on x86, NEON on ARM) for register-based LUT operations.
 *
 * Reference: T-MAC Paper (EuroSys 2025) - https://arxiv.org/abs/2407.00088
 */

#ifndef TMAC_KERNEL_H
#define TMAC_KERNEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * T-MAC Matrix Multiplication (AVX2)
 *
 * Performs BitNet inference using pshufb-based lookup tables.
 * Each weight byte encodes 4 ternary values {-1, 0, +1}.
 *
 * @param output    Output buffer [rows]
 * @param weights   Packed ternary weights [rows * cols/4]
 * @param lut       Lookup tables [num_groups * 256]
 * @param scales    Per-row scaling factors [rows]
 * @param rows      Number of output rows
 * @param cols      Number of columns (must be divisible by 4)
 * @param num_groups Number of activation groups for LUT
 */
void tmac_matmul_avx2(
    float* output,
    const uint8_t* weights,
    const float* lut,
    const float* scales,
    int rows,
    int cols,
    int num_groups
);

/**
 * T-MAC Matrix Multiplication (NEON)
 *
 * ARM NEON version using tbl instruction for lookup.
 * Same interface as AVX2 version.
 */
void tmac_matmul_neon(
    float* output,
    const uint8_t* weights,
    const float* lut,
    const float* scales,
    int rows,
    int cols,
    int num_groups
);

/**
 * SIMD RMSNorm (AVX2)
 *
 * Root Mean Square Layer Normalization.
 * output[i] = (input[i] / rms) * weight[i]
 * where rms = sqrt(mean(input^2) + eps)
 *
 * @param output    Output buffer [size]
 * @param input     Input buffer [size]
 * @param weight    Weight buffer [size]
 * @param size      Vector size
 * @param eps       Epsilon for numerical stability (typically 1e-6)
 */
void rmsnorm_avx2(
    float* output,
    const float* input,
    const float* weight,
    int size,
    float eps
);

/**
 * SIMD RMSNorm (NEON)
 * ARM NEON version of RMSNorm.
 */
void rmsnorm_neon(
    float* output,
    const float* input,
    const float* weight,
    int size,
    float eps
);

/**
 * SIMD Softmax (AVX2)
 *
 * Numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * @param output    Output buffer [size]
 * @param input     Input buffer [size]
 * @param size      Vector size
 */
void softmax_avx2(
    float* output,
    const float* input,
    int size
);

/**
 * SIMD Softmax (NEON)
 * ARM NEON version of softmax.
 */
void softmax_neon(
    float* output,
    const float* input,
    int size
);

/**
 * Build LUT for T-MAC
 *
 * Pre-computes lookup tables for a group of activations.
 * Each LUT entry corresponds to a 4-bit weight pattern.
 *
 * @param lut           Output LUT buffer [num_groups * 256]
 * @param activations   Input activations [size]
 * @param size          Activation size
 * @param group_size    Number of activations per group (typically 4)
 */
void build_lut(
    float* lut,
    const float* activations,
    int size,
    int group_size
);

/**
 * Check CPU features
 *
 * @return Bitmask of supported features:
 *         - Bit 0: AVX2
 *         - Bit 1: AVX512
 *         - Bit 2: NEON
 */
int get_cpu_features(void);

#ifdef __cplusplus
}
#endif

#endif // TMAC_KERNEL_H
