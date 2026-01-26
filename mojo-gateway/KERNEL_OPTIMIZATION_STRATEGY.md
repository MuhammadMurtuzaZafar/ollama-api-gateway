# EdgeLLM Kernel Optimization Strategy

## CRITICAL: Use Definitive Benchmark

**Always use**: `notebooks/edgellm_definitive_benchmark.ipynb`

This ensures consistent, reproducible measurements. Do NOT create ad-hoc benchmarks.

---

## Research-Backed 3-Point Implementation Plan

Based on analysis of the following papers:
- **FlashAttention-3** (Dao et al., NeurIPS 2024) - Asynchrony and Low-precision
- **FlashDecoding++** (Hong et al., MLSys 2024) - Unified Max, Flat GEMM
- **Mirage Persistent Kernel** (arXiv 2024) - GPU-Resident Mega-Kernels
- **PagedAttention/vLLM** (Kwon et al., SOSP 2023) - Memory Management
- **CUTLASS FlashAttention** (NVIDIA, 2023) - Tensor Core Implementation

---

## Problem Analysis: Why Our Kernel is 2.2x Slower Than Ollama

### Current Implementation Issues (flash_attention_v2.cu)

| Issue | Impact | Line Reference |
|-------|--------|----------------|
| **CUDA Cores only** | 15x slower than Tensor Cores | Lines 125-133 (dot product loop) |
| **Host-Device copies per call** | 0.5-2ms latency per call | Lines 492, 520 (cudaMemcpy) |
| **Synchronized softmax** | 20% overhead (FlashDecoding++ finding) | Lines 143-206 (reduction) |
| **FP32 only** | 2x slower than FP16, 4x slower than FP8 | Entire kernel |
| **No pipelining** | Compute waits for memory | Sequential tile loop |

### Ollama/llama.cpp Advantages

```
Ollama uses cuBLAS/cuDNN:
├── Tensor Cores via cuBLAS (300 TFLOPs on A100)
├── FP16/BF16 compute
├── CUDA Graphs for kernel launch elimination
└── Continuous batching for throughput
```

---

## 3-Point Implementation Strategy

---

## Point 1: Tensor Core Integration (WMMA/MMA)

### Research Foundation

From [FlashAttention-3](https://arxiv.org/html/2407.08608v2):
> "GPUs have separate units for matmul (Tensor Cores, ~300 TFLOPs on A100) vs generic math (CUDA cores, ~20 TFLOPs). Every non-matmul operation is 15x slower."

From [CUTLASS FlashAttention Case Study](https://arxiv.org/html/2312.11918v1):
> "FlashAttention-2 leverages the NVIDIA CUTLASS library for efficient tensor core utilization in matrix multiplications."

### Implementation Details

**Current (CUDA Cores - 20 TFLOPs):**
```cuda
// Lines 125-133: Scalar dot product
for (int d = 0; d < head_dim; d++) {
    score += q_ptr[d] * s_K[k * head_dim + d];  // FMA instructions
}
```

**Target (Tensor Cores - 300 TFLOPs):**
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// WMMA fragment declarations
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;  // Q
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;  // K^T
fragment<accumulator, 16, 16, 16, float> c_frag;         // Scores

// Load Q and K tiles
load_matrix_sync(a_frag, s_Q, head_dim);
load_matrix_sync(b_frag, s_K_T, head_dim);

// Tensor Core GEMM: S = Q @ K^T (single instruction)
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**T4 GPU (sm_75) WMMA Support:**
| Matrix Size | A Type | B Type | C Type |
|-------------|--------|--------|--------|
| 16x16x16 | FP16 | FP16 | FP16/FP32 |
| 8x32x16 | FP16 | FP16 | FP16/FP32 |
| 32x8x16 | FP16 | FP16 | FP16/FP32 |

**Expected Speedup: 8-15x** on attention score computation

### Key Implementation Notes

1. **Data must be FP16 for Tensor Cores:**
   ```cuda
   // Convert FP32 -> FP16 on load
   __half h_val = __float2half(f_val);
   ```

2. **Memory alignment required (256-byte for best performance):**
   ```cuda
   __shared__ __align__(256) half s_K[TILE * HEAD_DIM];
   ```

3. **Avoid bank conflicts via padding:**
   ```cuda
   // Add skew to avoid shared memory bank conflicts
   __shared__ half s_K[TILE][HEAD_DIM + SKEW];  // SKEW = 8
   ```

---

## Point 2: GPU-Resident Inference (Eliminate Host-Device Transfers)

### Research Foundation

From [Mirage Persistent Kernel](https://arxiv.org/html/2512.22219v1):
> "MPK is the first compiler and runtime system that automatically transforms multi-GPU model inference into a single high-performance mega-kernel... reducing LLM inference latency by 1.2-6.7x."

From [Mind the Memory Gap](https://arxiv.org/html/2503.08311v2):
> "FlashAttention employs tiling to fuse all attention operations in one CUDA kernel, significantly reducing memory accesses."

### The Problem

**Current Implementation (Lines 492, 520):**
```cuda
// EVERY decode call does this:
cudaMemcpy(fa2_d_Q, Q, batch_heads * head_dim * sizeof(float),
           cudaMemcpyHostToDevice);  // ~0.1-0.5ms

// ... kernel runs ...

cudaMemcpy(O, fa2_d_O, batch_heads * head_dim * sizeof(float),
           cudaMemcpyDeviceToHost);  // ~0.1-0.5ms
```

For 100 tokens: `100 * 0.6ms = 60ms` just in memory transfers!

### Implementation: Persistent GPU Inference

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Input Queue │  │ KV Cache    │  │ Output Queue        │  │
│  │ (ring buf)  │  │ (paged)     │  │ (ring buf)          │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         ▼                ▼                     ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           PERSISTENT MEGA-KERNEL                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ Embed   │─▶│ Attn    │─▶│ FFN     │─▶│ Sample  │  │   │
│  │  │ Kernel  │  │ (FA2)   │  │ Kernel  │  │ Kernel  │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │   │
│  │                    FUSED (no HBM round-trips)         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                                           │
         │ cudaMemcpyAsync                          │ cudaMemcpyAsync
         │ (once at start)                          │ (once at end)
         ▼                                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Host Memory                               │
│         Prompt tokens in ────────────▶ Generated tokens out │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cuda
// New API: Queue-based inference
int fa2_enqueue_tokens(const int* token_ids, int num_tokens);
int fa2_launch_persistent_kernel(void);  // Runs until stop token
int fa2_dequeue_output(int* output_tokens, int max_tokens);

// Persistent kernel structure
__global__ void fa2_mega_kernel(
    int* __restrict__ input_queue,    // Ring buffer on GPU
    int* __restrict__ output_queue,   // Ring buffer on GPU
    half* __restrict__ kv_cache,      // Never leaves GPU
    half* __restrict__ weights,       // Pre-loaded model weights
    volatile int* control             // Atomic control flags
) {
    while (atomicAdd(control + STOP_FLAG, 0) == 0) {
        // 1. Check input queue (device-side polling)
        int token = dequeue_token(input_queue);
        if (token < 0) continue;

        // 2. Full forward pass (all fused)
        embed_kernel(...);
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            fa2_attention_fused(layer, ...);  // No HBM round-trip
            ffn_fused(layer, ...);
        }

        // 3. Sample and enqueue output (device-side)
        int out_token = sample_token(...);
        enqueue_token(output_queue, out_token);
    }
}
```

**Expected Speedup: 1.5-3x** from eliminating PCIe latency

### Key Implementation Notes

1. **CUDA Graphs for static parts:**
   ```cuda
   cudaGraph_t graph;
   cudaGraphCreate(&graph, 0);
   // Capture kernel launches
   cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
   // ... kernel launches ...
   cudaStreamEndCapture(stream, &graph);
   cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

   // Execute entire graph with single launch
   cudaGraphLaunch(graphExec, stream);
   ```

2. **Async memory copies (overlap with compute):**
   ```cuda
   // Double-buffered: copy N+1 while computing N
   cudaMemcpyAsync(buffer[1], src, size, cudaMemcpyHostToDevice, stream1);
   kernel<<<...>>>(buffer[0], ...);  // On stream2
   cudaStreamSynchronize(stream2);
   // Swap buffers
   ```

---

## Point 3: FlashDecoding++ Optimizations

### Research Foundation

From [FlashDecoding++](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf) (MLSys 2024):
> "The softmax operation requires a synchronized update operation among each partial softmax result, leading to ~20% overheads."
> "FlashDecoding++ introduces a unified max value technique for different partial softmax computations to avoid synchronization."

### 3A: Unified Max Value (Asynchronous Softmax)

**Current (Synchronized - Lines 143-206):**
```cuda
// Each tile: find max, broadcast, recompute
float tile_max = -FLT_MAX;
for (int k = tid; k < tile_len; k += FA2_THREADS) {
    tile_max = fmaxf(tile_max, s_scores[k]);
}
// ... warp reduction ...
// ... broadcast across warps ...  <-- SYNC OVERHEAD
__syncthreads();  // Barrier!
```

**Target (Unified Max - No Sync):**
```cuda
// Pre-compute unified max from first pass or heuristic
// Key insight: use a safe upper bound that doesn't require sync
__device__ float unified_max(float* scores, int len) {
    // Option 1: Use theoretical max (scale * sqrt(head_dim))
    return 1.0f / sqrtf((float)head_dim) * sqrtf((float)head_dim);  // = 1.0

    // Option 2: Use running max from previous token (warm cache)
    // Option 3: Two-pass with async copy overlap
}

// Each thread computes independently, NO __syncthreads() for max!
for (int k = tid; k < tile_len; k += FA2_THREADS) {
    float p = expf(s_scores[k] - UNIFIED_MAX);  // Safe approximation
    // ...
}
```

**Expected Speedup: 1.15-1.2x** from removing sync barriers

### 3B: Flat GEMM Optimization

**The Problem:** Decode-phase attention has shape `[1, seq_len] @ [seq_len, head_dim]` - extremely "flat" matrices that underutilize Tensor Cores.

From FlashDecoding++:
> "FlashDecoding++ only pads the matrix size to 8 rather than 64 in previous designs for flat-shaped GEMM to improve computation utilization."

**Current (Over-padding):**
```cuda
#define FA2_TILE_SIZE 64  // Line 24 - wastes compute for short sequences
```

**Target (Adaptive Tiling):**
```cuda
// Select tile size based on sequence length
__device__ int get_optimal_tile(int seq_len) {
    if (seq_len <= 64) return 8;    // Minimal padding
    if (seq_len <= 256) return 16;
    if (seq_len <= 1024) return 32;
    return 64;
}

// Double-buffering for flat GEMM
// Load tile N+1 while computing tile N
__shared__ half tile_A[2][TILE][HEAD_DIM];
__shared__ half tile_B[2][TILE][HEAD_DIM];

int ping = 0;
for (int tile = 0; tile < num_tiles; tile++) {
    // Async load next tile
    load_tile_async(tile_A[1-ping], tile + 1);

    // Compute on current tile (overlapped with load)
    wmma_gemm(tile_A[ping], tile_B[ping], accum);

    __syncthreads();
    ping = 1 - ping;  // Swap buffers
}
```

**Expected Speedup: 1.3-1.5x** for short sequences

### 3C: FlashDecoding (Split-K Parallelism)

**The Problem:** Decode attention has `batch_size=1, seq_len=large` - only 1 query but thousands of keys. Standard parallelization is over batch/heads, leaving SMs underutilized.

From [FlashDecoding](https://pytorch.org/blog/flexattention-for-inference/):
> "FlashDecoding partitions the queries and KV cache then calculates the attention in parallel... The bottleneck is to load KV cache as fast as possible."

**Current (One block per head):**
```cuda
// Line 68: Only batch_heads blocks = 14-28 blocks
// T4 has 40 SMs - most are idle!
flash_attention_v2_decode_vectorized_kernel<<<batch_heads, FA2_THREADS, smem_size>>>(...)
```

**Target (Split-K over sequence length):**
```cuda
// Split KV cache across multiple blocks
#define SPLITS 4  // Each head processed by 4 blocks

__global__ void fa2_flash_decoding_kernel(
    // ... inputs ...
    float* partial_out,   // [batch_heads, SPLITS, head_dim]
    float* partial_lse    // [batch_heads, SPLITS] (log-sum-exp)
) {
    int bh = blockIdx.x / SPLITS;
    int split_id = blockIdx.x % SPLITS;

    // Each block processes 1/SPLITS of the KV cache
    int start = split_id * (seq_len / SPLITS);
    int end = (split_id == SPLITS-1) ? seq_len : start + (seq_len / SPLITS);

    // Compute partial attention
    // ... standard FA2 loop over [start, end) ...

    // Store partial result (NO global reduction yet)
    partial_out[bh * SPLITS * head_dim + split_id * head_dim + d] = o_val;
    partial_lse[bh * SPLITS + split_id] = log_sum_exp;
}

// Separate reduction kernel (or fused)
__global__ void fa2_reduce_kernel(
    float* partial_out,
    float* partial_lse,
    float* final_out
) {
    // Combine partial results using log-sum-exp trick
    // O_final = sum_i(exp(lse_i - lse_max) * O_i) / sum_i(exp(lse_i - lse_max))
}
```

**Expected Speedup: 1.5-2x** for long sequences (>1024 tokens)

---

## Implementation Roadmap

### Phase 1: Tensor Cores (Highest Impact)

```
Week 1:
├── Convert kernel to FP16 (half precision)
├── Implement WMMA for Q @ K^T
├── Implement WMMA for Softmax @ V
└── Benchmark: Target 4-6x speedup on attention
```

### Phase 2: GPU-Resident Inference

```
Week 2:
├── Implement ring buffer queues on GPU
├── Create persistent kernel wrapper
├── Add CUDA Graph support for static parts
└── Benchmark: Target 2x speedup from eliminating memcpy
```

### Phase 3: FlashDecoding++ Optimizations

```
Week 3:
├── Implement unified max value
├── Add adaptive tile sizing for flat GEMM
├── Implement Split-K parallelism
└── Benchmark: Target 1.5x additional speedup
```

### Combined Expected Result

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Baseline (current) | 1.0x | 1.0x |
| Tensor Cores (WMMA) | 4-6x | 4-6x |
| GPU-Resident | 1.5-2x | 6-12x |
| FlashDecoding++ | 1.3-1.5x | **8-18x** |

**Target: 320 tok/s * 1.5 = 480+ tok/s** (beat Ollama by 50%)

---

## Hardware-Specific Notes

### T4 GPU (sm_75, Turing)

- WMMA: 16x16x16 FP16 → FP16/FP32
- No TMA (Hopper-only)
- No WGMMA (Hopper-only)
- 65 TFLOPs FP16 Tensor Core
- 320 GB/s memory bandwidth

### Upgrade Path to Hopper (H100)

For future H100 support:
- Replace WMMA with WGMMA (warpgroup-level)
- Use TMA for async memory copies
- Enable FP8 compute (2x throughput)
- Expected: 740+ TFLOPs, ~1.2 PFLOPs with FP8

---

## Sources

1. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/html/2407.08608v2)
2. [FlashDecoding++: Faster LLM Inference](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf)
3. [Mirage Persistent Kernel: Mega-Kernelizing Tensor Programs](https://arxiv.org/html/2512.22219v1)
4. [CUTLASS FlashAttention Case Study](https://arxiv.org/html/2312.11918v1)
5. [PagedAttention/vLLM Paper](https://arxiv.org/pdf/2309.06180)
6. [NVIDIA Tensor Core Programming Guide](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
7. [WGMMA Tutorial (Hopper)](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
