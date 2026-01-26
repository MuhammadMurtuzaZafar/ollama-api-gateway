# Mojo LLM Optimization Roadmap: Path to 50 tok/s

## Current State

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Throughput | 1.0 tok/s | 50 tok/s | **50x** |
| Model | BitNet 2B | BitNet 2B | - |
| Platform | Fly.io (2 vCPU) | Local (8+ cores) | 4x potential |

## Research Summary

### Key Papers Reviewed

1. **T-MAC** (EuroSys 2025) - [arXiv:2407.00088](https://arxiv.org/abs/2407.00088)
   - 38 tok/s on M2-Ultra (8 cores) for Llama2-7B 4-bit
   - Uses `pshufb` (x86) / `tbl` (ARM) for register-based LUT
   - 4x speedup over llama.cpp

2. **NoMAD-Attention** (NeurIPS 2024) - [arXiv:2403.01273](https://arxiv.org/abs/2403.01273)
   - 9 tok/s on dual Xeon for CodeLlama-7B
   - Replaces attention MAD with `_mm256_shuffle_epi8`
   - 2x speedup for attention at 16k context

3. **bitnet.cpp** (ACL 2025) - [Paper](https://aclanthology.org/2025.acl-long.457.pdf)
   - 6.17x speedup over FP baseline on x86
   - Built on T-MAC kernels
   - 5-7 tok/s for 100B model on single CPU

4. **llama.cpp Optimizations** - [GitHub](https://github.com/ggml-org/llama.cpp)
   - AVX2/AVX512/AMX support
   - 10x speedup from ARM dotprod+fp16
   - Memory mapping, KV cache quantization

### Why T-MAC/bitnet.cpp is Fast

```
Traditional:     output[i] = Σ weight[j] * activation[j]
                              ↑ memory load  ↑ multiply (slow)

T-MAC (CPU):     output[i] = Σ LUT[weight_pattern]
                              ↑ pshufb: 1 cycle, 32 elements

Our Current:     output[i] = Σ tables[group*256 + pattern]
                              ↑ memory load from RAM (100+ cycles)
```

The critical difference: **T-MAC keeps LUT in SIMD registers**, we load from RAM.

## Bottleneck Analysis

### Current Implementation Hotspots

```mojo
# BOTTLENECK 1: LUT in RAM, not registers
fn tmac_matmul_lut(...):
    sum0 += lut.get(g, Int(w_ptr[w_base + g]))  # RAM access every iteration

# BOTTLENECK 2: Scalar RMSNorm
fn rmsnorm(...):
    for i in range(size):
        ss += input[i] * input[i]  # No SIMD

# BOTTLENECK 3: LUT rebuilt for every matmul
fn build_lut(activations, offset, size) -> LookupTable:
    for g in range(num_groups):
        for pattern in range(256):  # 256 iterations per group!
```

### Estimated Time Breakdown (per token)

| Operation | Current | Optimized | Technique |
|-----------|---------|-----------|-----------|
| LUT Matmul (6 per layer) | 800ms | 16ms | Register LUT via FFI |
| RMSNorm (4 per layer) | 50ms | 5ms | SIMD vectorize |
| Attention | 100ms | 20ms | NoMAD-Attention |
| Softmax | 30ms | 3ms | SIMD vectorize |
| LUT Build | 200ms | 0ms | Pre-build once |
| **Total (24 layers)** | **~1000ms** | **~20ms** | **50x improvement** |

## Optimization Roadmap

### Phase 1: Low-Hanging Fruit (2-3x improvement)

#### 1.1 SIMD Vectorize RMSNorm
```mojo
# Current (scalar)
for i in range(size):
    ss += input[i] * input[i]

# Optimized (SIMD)
from algorithm import vectorize
alias simd_width = simdwidthof[DType.float32]()

var ss_vec = SIMD[DType.float32, simd_width](0)
@parameter
fn accumulate[width: Int](i: Int):
    var v = input_ptr.load[width=width](offset + i)
    ss_vec += v * v
vectorize[accumulate, simd_width](size)
```

#### 1.2 SIMD Vectorize Softmax
Same pattern as RMSNorm - vectorize max-finding, exp, and sum.

#### 1.3 Pre-build LUT Once Per Layer
Currently `build_lut` is called for every matmul. Instead:
- Build LUT once when activations change (after each norm)
- Reuse for Q, K, V, O projections

**Expected improvement: 2-3x**

### Phase 2: Memory Optimization (3-5x improvement)

#### 2.1 Prefetch for LUT Access
```mojo
from sys.intrinsics import prefetch

# Prefetch next cache line while processing current
prefetch[PrefetchOptions().for_read().high_locality()](
    lut.tables.unsafe_ptr() + (g + 8) * 256
)
```

#### 2.2 Data Layout Optimization
T-MAC uses "LUT-centric data layout":
- Reorder weights to maximize cache line utilization
- Pack weights for sequential access patterns

#### 2.3 Blocked/Tiled Computation
Process matmul in cache-friendly blocks:
```mojo
alias BLOCK_SIZE = 64  # Fits in L1 cache

for block_start in range(0, rows, BLOCK_SIZE):
    var block_end = min(block_start + BLOCK_SIZE, rows)
    # Process block with better cache locality
```

**Expected improvement: 3-5x (cumulative: 6-15x)**

### Phase 3: FFI to C Kernels (10-20x improvement)

This is the **critical optimization** - use actual `pshufb`/`tbl` instructions.

#### 3.1 Write Optimized C Kernel

```c
// tmac_kernel.c
#include <immintrin.h>

void tmac_matmul_avx2(
    float* output,
    const uint8_t* weights,
    const float* lut,      // 16 entries fit in one AVX2 register
    const float* scales,
    int rows, int cols
) {
    for (int row = 0; row < rows; row++) {
        __m256 sum = _mm256_setzero_ps();

        for (int g = 0; g < cols/4; g += 8) {
            // Load 8 weight bytes
            __m128i w = _mm_loadl_epi64((__m128i*)(weights + row * (cols/4) + g));

            // Use pshufb for parallel LUT lookup
            // This is the key optimization!
            __m256i indices = _mm256_cvtepu8_epi32(w);
            __m256 lut_vals = _mm256_i32gather_ps(lut + g * 256, indices, 4);

            sum = _mm256_add_ps(sum, lut_vals);
        }

        // Horizontal sum and scale
        output[row] = hsum_avx2(sum) * scales[row];
    }
}
```

#### 3.2 Call from Mojo via FFI

```mojo
from sys.ffi import external_call, DLHandle

fn load_tmac_kernel() -> DLHandle:
    return DLHandle("./libtmac_kernel.so")

fn tmac_matmul_fast(
    output: UnsafePointer[Float32],
    weights: UnsafePointer[UInt8],
    lut: UnsafePointer[Float32],
    scales: UnsafePointer[Float32],
    rows: Int, cols: Int
):
    external_call["tmac_matmul_avx2", NoneType](
        output, weights, lut, scales, rows, cols
    )
```

**Expected improvement: 10-20x (cumulative: 30-50x)**

### Phase 4: Advanced Optimizations (Fine-tuning)

#### 4.1 NoMAD-Attention for Long Context
Replace standard attention with NoMAD-Attention:
- Product-quantize keys
- Use `shuffle_epi8` for attention score computation
- 2x speedup for attention

#### 4.2 INT8 Accumulation
T-MAC uses int8 intermediate accumulation:
```c
// Accumulate in int8, convert to float only at end
int8_t partial_sums[8];
for (...) {
    partial_sums[i] += lut_int8[pattern];
}
float result = convert_to_float(partial_sums) * scale;
```

#### 4.3 Multi-threaded LUT Building
Parallelize LUT construction across cores.

#### 4.4 ARM-Specific: Use TBL Instruction
For Apple Silicon / ARM:
```c
// ARM NEON
uint8x16_t tbl_result = vqtbl1q_u8(lut_register, indices);
```

## Implementation Priority

| Priority | Optimization | Effort | Impact | Cumulative |
|----------|-------------|--------|--------|------------|
| 1 | SIMD RMSNorm/Softmax | Low | 1.5x | 1.5x |
| 2 | Pre-build LUT | Low | 1.5x | 2.25x |
| 3 | Prefetch + Tiling | Medium | 2x | 4.5x |
| 4 | **C FFI Kernel** | High | **10x** | **45x** |
| 5 | NoMAD-Attention | High | 1.5x | ~50x |

## Technical Requirements

### For Phase 1-2 (Pure Mojo)
- Mojo 24.x or later
- `vectorize`, `parallelize` from algorithm module
- `prefetch` from sys.intrinsics

### For Phase 3 (FFI)
- C compiler with AVX2/NEON support
- `clang -O3 -mavx2 -c tmac_kernel.c -o tmac_kernel.o`
- `clang -shared -o libtmac_kernel.so tmac_kernel.o`
- Mojo FFI: `sys.ffi.external_call`

### Hardware Recommendations
- **Development**: Apple M1/M2/M3 (ARM NEON + unified memory)
- **Deployment**: Intel Xeon with AVX-512 or AMD EPYC
- **Minimum**: 8 cores, 16GB RAM for 2B model

## Success Metrics

| Milestone | Throughput | Status |
|-----------|------------|--------|
| Baseline | 1.0 tok/s | Current |
| Phase 1 | 2-3 tok/s | Planned |
| Phase 2 | 6-10 tok/s | Planned |
| Phase 3 | 30-50 tok/s | Planned |
| Phase 4 | 50+ tok/s | Target |

## References

1. T-MAC: https://arxiv.org/abs/2407.00088
2. NoMAD-Attention: https://arxiv.org/abs/2403.01273
3. bitnet.cpp: https://aclanthology.org/2025.acl-long.457.pdf
4. Mojo FFI: https://docs.modular.com/mojo/stdlib/sys/ffi/
5. Intel Intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
6. ARM NEON: https://developer.arm.com/architectures/instruction-sets/intrinsics/

## Next Steps

1. **Immediate**: Implement Phase 1 optimizations in pure Mojo
2. **Short-term**: Profile to identify actual bottlenecks
3. **Medium-term**: Write and integrate C FFI kernel
4. **Long-term**: Contribute optimizations back to Mojo ecosystem
