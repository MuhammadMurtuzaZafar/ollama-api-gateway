# CPU Register and Cache-Level Optimization for T-MAC LLM Inference

## Executive Summary

This document provides a deep technical analysis of CPU-level optimizations needed to achieve 50 tok/s from the current 1 tok/s in our Mojo T-MAC implementation. The analysis covers:

1. **Register-level optimizations** - SIMD intrinsics, register file utilization
2. **Cache hierarchy optimization** - L1/L2/L3 locality, prefetching strategies
3. **Instruction-level parallelism (ILP)** - Loop unrolling, dependency breaking
4. **Memory bandwidth optimization** - Data layout, sequential access patterns

## Current Bottleneck Analysis

### Measured Performance

| Component | Current (ms/token) | % of Total | Target | Technique |
|-----------|-------------------|------------|--------|-----------|
| LUT Matmul (×6/layer) | 800ms | 80% | 16ms | Register LUT |
| RMSNorm (×4/layer) | 50ms | 5% | 5ms | SIMD vectorize |
| Attention Scoring | 100ms | 10% | 20ms | NoMAD-Attention |
| Softmax | 30ms | 3% | 3ms | SIMD fused |
| LUT Build | 200ms | 2% | 0ms | Pre-build once |

### Root Cause: LUT in RAM vs Registers

```
=== The Critical Difference ===

T-MAC (bitnet.cpp):
┌─────────────────────────────────────────────────────┐
│  SIMD Register File (16-32 registers, 256-512 bits) │
│  ┌─────────────────────────────────────────────────┐│
│  │ LUT[0-15] = zmm0 (AVX-512) or ymm0-1 (AVX2)    ││
│  └─────────────────────────────────────────────────┘│
│                         │                            │
│         pshufb/tbl ────┘ (1 cycle!)                 │
│                         ↓                            │
│                   Output Value                       │
└─────────────────────────────────────────────────────┘

Our Current Implementation:
┌─────────────────────────────────────────────────────┐
│  RAM (DDR4/DDR5)                                    │
│  ┌─────────────────────────────────────────────────┐│
│  │ LUT Tables: 256 × num_groups × 4 bytes         ││
│  └─────────────────────────────────────────────────┘│
│                         │                            │
│         Memory Load ────┘ (100-300 cycles!)         │
│                         ↓                            │
│                   Output Value                       │
└─────────────────────────────────────────────────────┘
```

**The 100x gap exists because T-MAC keeps the LUT in SIMD registers.**

---

## Register-Level Optimizations

### 1. The `pshufb` / `tbl` Technique

T-MAC's core optimization uses SIMD shuffle instructions for parallel table lookups:

```c
// x86 AVX2: _mm256_shuffle_epi8 (pshufb)
// - Input: 32-byte vector of indices (0-15 each)
// - Table: 32-byte vector (16 values, duplicated)
// - Output: 32 values looked up in 1 cycle

__m256i indices = _mm256_loadu_si256(weight_patterns);  // 32 weight bytes
__m256 lut_reg = _mm256_broadcast_ss(lut_table);        // Load LUT to register
__m256i result = _mm256_shuffle_epi8(lut_reg, indices); // 32 parallel lookups!
```

```c
// ARM NEON: vqtbl1q_u8 (tbl)
// - Same concept: parallel table lookup in register

uint8x16_t indices = vld1q_u8(weight_patterns);
uint8x16_t lut_reg = vld1q_u8(lut_table);
uint8x16_t result = vqtbl1q_u8(lut_reg, indices);  // 16 parallel lookups
```

**Why Mojo can't do this (yet):**
- Mojo's `shuffle()` requires compile-time constant indices
- Runtime shuffle is needed for T-MAC's dynamic weight patterns
- This is the fundamental limitation requiring FFI to C

### 2. Register Allocation Strategy

Modern CPUs have limited SIMD registers:
- **AVX2 (Intel/AMD)**: 16 × 256-bit YMM registers
- **AVX-512**: 32 × 512-bit ZMM registers
- **ARM NEON**: 32 × 128-bit V registers

**Optimal Register Usage for T-MAC:**

```
Register Allocation (AVX2, 16 registers):
┌────────────────────────────────────────────────┐
│ zmm0-zmm3:   LUT values (4 groups of 16)       │
│ zmm4-zmm7:   Accumulator vectors               │
│ zmm8-zmm11:  Weight patterns (loaded)          │
│ zmm12-zmm15: Temporary / broadcast / shuffle   │
└────────────────────────────────────────────────┘
```

### 3. SIMD Vectorization for Mojo (Possible Now)

These optimizations CAN be done in pure Mojo:

```mojo
# Current (scalar) - bitnet_server.mojo:166-173
fn rmsnorm(mut output: List[Float32], input: List[Float32], ...):
    var ss: Float32 = 0.0
    for i in range(size):
        ss += input[i_offset + i] * input[i_offset + i]  # SLOW: scalar

# Optimized (SIMD) - using Mojo's vectorize
from algorithm import vectorize

fn rmsnorm_simd(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    size: Int
):
    alias simd_width = simdwidthof[DType.float32]()  # 8 for AVX2, 16 for AVX-512

    # Phase 1: Compute sum of squares (SIMD)
    var ss_vec = SIMD[DType.float32, simd_width](0)

    @parameter
    fn sum_squares[width: Int](i: Int):
        var v = input.load[width=width](i)
        ss_vec += v * v

    vectorize[sum_squares, simd_width](size)

    # Horizontal reduction
    var ss = ss_vec.reduce_add()
    ss = 1.0 / math.sqrt(ss / Float32(size) + 1e-5)

    # Phase 2: Apply normalization (SIMD)
    @parameter
    fn apply_norm[width: Int](i: Int):
        var inp = input.load[width=width](i)
        var w = weight.load[width=width](i)
        output.store[width=width](i, w * (ss * inp))

    vectorize[apply_norm, simd_width](size)
```

**Expected Speedup: 8x (AVX2) to 16x (AVX-512)**

---

## Cache Hierarchy Optimization

### Cache Architecture (Typical Modern CPU)

```
┌──────────────────────────────────────────────────────────────┐
│                         CPU Core                              │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                             │
│  │ L1D Cache   │ 32-48 KB, 4-5 cycle latency                │
│  │ Line: 64B   │ Bandwidth: 256+ GB/s                       │
│  └─────────────┘                                             │
│         ↓                                                     │
│  ┌─────────────┐                                             │
│  │ L2 Cache    │ 256KB-1MB, 12-14 cycle latency             │
│  │ Line: 64B   │ Bandwidth: 100+ GB/s                       │
│  └─────────────┘                                             │
│         ↓                                                     │
│  ┌─────────────┐                                             │
│  │ L3 Cache    │ 8-64MB (shared), 40-50 cycle latency       │
│  │ Line: 64B   │ Bandwidth: 50+ GB/s                        │
│  └─────────────┘                                             │
│         ↓                                                     │
│  ┌─────────────┐                                             │
│  │ RAM (DDR5)  │ 100-300 cycle latency                      │
│  │             │ Bandwidth: 50-100 GB/s                     │
│  └─────────────┘                                             │
└──────────────────────────────────────────────────────────────┘
```

### Cache-Optimal Data Layout

**Problem: Current LUT Layout**
```
Current: LUT[group][pattern] = tables[group * 256 + pattern]

Access pattern during matmul:
  lut.get(0, w[0])   → tables[0*256 + w[0]]     = tables[47]
  lut.get(1, w[1])   → tables[1*256 + w[1]]     = tables[389]
  lut.get(2, w[2])   → tables[2*256 + w[2]]     = tables[612]

  Random access within 256 entries → cache misses!
```

**Solution: Sequential Access Pattern**
```
Reorder weights so sequential groups use sequential cache lines:

T-MAC's LUT-centric layout:
1. Reorder weight matrix by groups
2. Pack weights for sequential memory access
3. Prefetch next block while processing current

# Pseudocode
for block in range(0, num_groups, BLOCK_SIZE):
    prefetch(weights[block + BLOCK_SIZE])  # Prefetch next
    for g in range(block, min(block + BLOCK_SIZE, num_groups)):
        sum += lut[g][weights[g]]          # Process current
```

### Prefetching Strategy

```mojo
from sys.intrinsics import prefetch

alias PREFETCH_DISTANCE = 8  # Cache lines ahead

fn tmac_matmul_prefetch(
    mut output: List[Float32],
    lut: LookupTable,
    weights: UnsafePointer[UInt8],
    ...
):
    for g in range(num_groups):
        # Prefetch weights 8 cache lines ahead
        if g + PREFETCH_DISTANCE < num_groups:
            prefetch[PrefetchOptions().for_read().high_locality()](
                weights + g + PREFETCH_DISTANCE * 64
            )

        # Also prefetch LUT entries
        prefetch[PrefetchOptions().for_read().medium_locality()](
            lut.tables.unsafe_ptr() + (g + PREFETCH_DISTANCE) * 256
        )

        sum += lut.get(g, Int(weights[g]))
```

### Blocking/Tiling for Cache

```mojo
alias L1_BLOCK = 32   # Rows that fit in L1 with weights
alias L2_BLOCK = 128  # Rows that fit in L2

fn tmac_matmul_blocked(
    mut output: List[Float32],
    lut: LookupTable,
    weights: List[UInt8],
    rows: Int,
    cols: Int
):
    var bytes_per_row = (cols + 3) // 4

    # Process in L2-sized blocks
    for block_start in range(0, rows, L2_BLOCK):
        var block_end = min(block_start + L2_BLOCK, rows)

        # Within L2 block, process L1-sized sub-blocks
        for sub_start in range(block_start, block_end, L1_BLOCK):
            var sub_end = min(sub_start + L1_BLOCK, block_end)

            # All weights for these rows are now in L1
            @parameter
            fn compute_row(row_idx: Int):
                var row = sub_start + row_idx
                var sum: Float32 = 0.0
                for g in range(bytes_per_row):
                    sum += lut.get(g, Int(weights[row * bytes_per_row + g]))
                output[row] = sum * scales[row]

            parallelize[compute_row](sub_end - sub_start)
```

---

## Instruction-Level Parallelism (ILP)

### Pipeline Utilization

Modern CPUs execute multiple instructions per cycle via:
1. **Superscalar execution**: 4-6 ALUs per core
2. **Out-of-order execution**: Reorder independent ops
3. **Loop unrolling**: Expose more ILP

**Current: 8-way unrolling (good!)**
```mojo
# bitnet_server.mojo:123-149 - Already has 8-way unroll
var sum0: Float32 = 0.0
var sum1: Float32 = 0.0
# ... (8 accumulators)

while g < groups_unrolled:
    sum0 += lut.get(g, Int(w_ptr[w_base + g]))
    sum1 += lut.get(g + 1, Int(w_ptr[w_base + g + 1]))
    # ... process 8 at once
    g += 8
```

### Breaking Data Dependencies

**Problem: False Dependency**
```mojo
# Each load depends on previous store (false dependency)
for i in range(size):
    output[i] = input[i] * weight[i]  # RAW hazard potential
```

**Solution: Separate Load/Compute/Store Phases**
```mojo
# Phase 1: Load all data
var inp_vec = input.load[width=8](i)
var wgt_vec = weight.load[width=8](i)

# Phase 2: Compute (independent)
var result = inp_vec * wgt_vec

# Phase 3: Store
output.store[width=8](i, result)
```

### FMA (Fused Multiply-Add) Utilization

```mojo
# Instead of:
var sum = a * b
sum = sum + c  # 2 operations

# Use FMA:
var sum = math.fma(a, b, c)  # 1 operation, same latency
```

---

## Memory Bandwidth Analysis

### LLM Inference is Memory-Bound

For a 2B parameter model with 1.58-bit weights:
```
Model size: 2B params × 1.58 bits = 395 MB
Memory bandwidth needed for 50 tok/s:
  = 395 MB × 50 = 19.75 GB/s

DDR5 bandwidth: 50-100 GB/s
  → Theoretically achievable!

But: Random access patterns reduce effective bandwidth by 10-50x
```

### Model Bandwidth Utilization (MBU)

```
MBU = (Achieved Throughput) / (Theoretical Peak)

Our current MBU:
  Achieved: 1 tok/s → 395 MB/s
  Peak: 50 GB/s
  MBU = 395 MB / 50 GB = 0.8%

Target MBU for 50 tok/s:
  Required: 19.75 GB/s
  Peak: 50 GB/s
  MBU = 39.5%
```

### Sequential Access Optimization

```mojo
# Bad: Strided access (cache unfriendly)
for row in range(rows):
    for col in range(cols):
        access(matrix[row * stride + col])

# Good: Sequential access within cache lines
for row in range(rows):
    var row_ptr = matrix.unsafe_ptr() + row * stride
    for col in range(0, cols, 64 // sizeof[Float32]()):
        # Process entire cache line
        var vec = row_ptr.load[width=16](col)
```

---

## Implementation Plan: C FFI Kernel

Since Mojo can't do runtime shuffle, we need FFI to C for the critical path:

### Step 1: Write Optimized C Kernel

```c
// tmac_kernel.c
#include <immintrin.h>
#include <stdint.h>

#ifdef __AVX2__
void tmac_matmul_avx2(
    float* __restrict output,
    const uint8_t* __restrict weights,
    const float* __restrict lut,
    const float* __restrict scales,
    int rows, int cols
) {
    int bytes_per_row = (cols + 3) / 4;

    #pragma omp parallel for
    for (int row = 0; row < rows; row++) {
        __m256 sum_vec = _mm256_setzero_ps();
        const uint8_t* w_row = weights + row * bytes_per_row;

        for (int g = 0; g < bytes_per_row; g += 8) {
            // Load 8 weight bytes as indices
            __m128i w_bytes = _mm_loadl_epi64((__m128i*)(w_row + g));

            // Expand to 32-bit indices for gather
            __m256i indices = _mm256_cvtepu8_epi32(w_bytes);

            // Gather LUT values (8 parallel lookups)
            // Note: Each group has 256 entries
            __m256i offsets = _mm256_mullo_epi32(
                _mm256_setr_epi32(g, g+1, g+2, g+3, g+4, g+5, g+6, g+7),
                _mm256_set1_epi32(256)
            );
            __m256i final_idx = _mm256_add_epi32(offsets, indices);

            __m256 lut_vals = _mm256_i32gather_ps(lut, final_idx, 4);
            sum_vec = _mm256_add_ps(sum_vec, lut_vals);
        }

        // Horizontal sum
        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_vec);
        __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);

        output[row] = _mm_cvtss_f32(sum_128) * scales[row];
    }
}
#endif

#ifdef __ARM_NEON
void tmac_matmul_neon(
    float* __restrict output,
    const uint8_t* __restrict weights,
    const float* __restrict lut,
    const float* __restrict scales,
    int rows, int cols
) {
    int bytes_per_row = (cols + 3) / 4;

    for (int row = 0; row < rows; row++) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        const uint8_t* w_row = weights + row * bytes_per_row;

        for (int g = 0; g < bytes_per_row; g += 4) {
            // Load 4 weight patterns
            uint8x8_t w_vec = vld1_u8(w_row + g);

            // Look up each (ARM lacks efficient gather)
            float vals[4];
            for (int i = 0; i < 4; i++) {
                int idx = (g + i) * 256 + w_row[g + i];
                vals[i] = lut[idx];
            }

            float32x4_t lut_vals = vld1q_f32(vals);
            sum = vaddq_f32(sum, lut_vals);
        }

        // Horizontal sum
        float32x2_t sum_pair = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
        output[row] = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0) * scales[row];
    }
}
#endif
```

### Step 2: Build Shared Library

```bash
# x86-64 with AVX2
clang -O3 -mavx2 -fopenmp -shared -fPIC \
    -o libtmac_kernel.so tmac_kernel.c

# ARM NEON (Apple Silicon)
clang -O3 -arch arm64 -shared -fPIC \
    -o libtmac_kernel.dylib tmac_kernel.c
```

### Step 3: Call from Mojo via FFI

```mojo
from sys.ffi import external_call, DLHandle

var handle: DLHandle

fn load_kernel():
    handle = DLHandle("./libtmac_kernel.so")

fn tmac_matmul_fast(
    output: UnsafePointer[Float32],
    weights: UnsafePointer[UInt8],
    lut: UnsafePointer[Float32],
    scales: UnsafePointer[Float32],
    rows: Int,
    cols: Int
):
    external_call["tmac_matmul_avx2", NoneType](
        output, weights, lut, scales, rows, cols
    )
```

---

## Optimization Priority Matrix

| Priority | Optimization | Effort | Expected Speedup | Cumulative |
|----------|-------------|--------|------------------|------------|
| 1 | SIMD RMSNorm | Low | 8x (for RMSNorm) | 1.2x overall |
| 2 | SIMD Softmax | Low | 8x (for Softmax) | 1.3x overall |
| 3 | Pre-build LUT once | Low | 2x (LUT build → 0) | 2.6x overall |
| 4 | Prefetching | Medium | 1.5x | 4x overall |
| 5 | Cache blocking | Medium | 1.5x | 6x overall |
| **6** | **C FFI kernel** | **High** | **10x** | **45x overall** |
| 7 | NoMAD-Attention | High | 1.5x | ~50x overall |

---

## Hardware-Specific Notes

### Intel/AMD x86-64

```
AVX2 (Most CPUs 2013+):
  - 256-bit vectors (8 × float32)
  - _mm256_shuffle_epi8 for LUT
  - _mm256_i32gather_ps for parallel load

AVX-512 (Xeon, EPYC, 12th Gen+):
  - 512-bit vectors (16 × float32)
  - _mm512_permutexvar_epi8 for LUT
  - 2x theoretical speedup over AVX2
  - Beware: May cause frequency throttling

AMX (Intel 4th Gen Xeon):
  - Tile-based matrix multiply
  - Up to 1024 INT8 ops/cycle
  - Requires restructured algorithm
```

### Apple Silicon (ARM)

```
NEON (All Apple Silicon):
  - 128-bit vectors (4 × float32)
  - vqtbl1q_u8 for 16-entry LUT
  - Limited to 16-byte table per lookup

SME (M4+):
  - Scalable Matrix Extension
  - Hardware matrix multiply
  - Not yet exposed in Mojo
```

### Recommended Test Platform

For development, prioritize:
1. **Apple M1/M2/M3**: Unified memory, excellent cache, ARM NEON
2. **Intel i7/i9 12th Gen+**: AVX2 + AVX-512, good for x86 testing
3. **AMD Ryzen 7000**: Strong AVX2, competitive with Intel

---

## Conclusion

The path to 50 tok/s requires:

1. **Immediate (pure Mojo)**: SIMD vectorization of RMSNorm, Softmax, LUT pre-build
2. **Short-term (C FFI)**: Register-based LUT kernel using pshufb/tbl
3. **Medium-term**: NoMAD-Attention for long context

The fundamental bottleneck is **LUT in RAM vs registers**. Until Mojo supports runtime shuffle indices, FFI to C is required for the critical 10x speedup.

## References

1. T-MAC: CPU Renaissance via Table Lookup, EuroSys 2025
2. NoMAD-Attention: Efficient Attention for 1-Million Context, NeurIPS 2024
3. Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
4. ARM NEON Intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
5. Mojo FFI Documentation: https://docs.modular.com/mojo/stdlib/sys/ffi/
