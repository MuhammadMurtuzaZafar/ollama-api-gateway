# EdgeLLM Performance Research Report
## January 11, 2026

---

## Executive Summary

EdgeLLM currently achieves **630 tok/s** on Tesla T4, beating Ollama's **423 tok/s** by **49%**. This report analyzes state-of-the-art research to identify techniques that could push EdgeLLM to **1000+ tok/s** (2.4x Ollama).

---

## 1. Current EdgeLLM Architecture

### What's Working (Phase 2.1)
| Technique | Speedup | Description |
|-----------|---------|-------------|
| Warp-private LUT | 8.57x | Eliminates atomicAdd contention |
| Streaming Fusion | 11.49x | RMSNorm on-the-fly, no intermediate storage |
| Adaptive Dispatch | 11.52x | Auto-selects best kernel per tensor size |
| Persistent Memory | 3.93x | Weights stay on GPU |

### Current Bottlenecks
1. **Kernel launch overhead** - Multiple small kernels per token
2. **Memory bandwidth** - T4 has 320 GB/s, we're not saturating it
3. **Single-token batch** - Can't leverage Tensor Core parallelism

---

## 2. Research Landscape (2025)

### 2.1 Ternary/BitNet Inference

#### Microsoft bitnet.cpp
- **Performance**: 250 tok/s on A100 (2B model)
- **Technique**: Custom CUDA W1.58A8 kernel with dp4a dot products
- **Key insight**: GPU W2A8 path with weight permutation + fast decoding
- **Source**: [microsoft/BitNet](https://github.com/microsoft/BitNet)

#### TriRun Kernel (ACL 2025)
- **Performance**: 8x speedup over FP16 PyTorch
- **Technique**: Optimized 2-bit ternary matmul for GPUs
- **Sweet spot**: Batch sizes 16-32 for peak speedup
- **Limitation**: Speedup diminishes at batch=1
- **Source**: [Scaling Laws for Ternary LLMs](https://arxiv.org/pdf/2506.23025)

#### T-SAR Framework
- **Performance**: 86.2x GEMV throughput on CPUs
- **Technique**: In-register LUT generation in SIMD units
- **Key insight**: Repurpose SIMD registers for dynamic LUT, eliminate memory access
- **Overhead**: Only 3.2% power, 1.4% area overhead
- **Source**: [T-SAR Research](https://quantumzeitgeist.com/2x-5x-cpu-sar-achieves-gemv-throughput-gemm-speedup-only-ternary/)

### 2.2 Kernel Optimization

#### Mirage Persistent Kernel (CMU/NVIDIA, Dec 2025)
- **Performance**: 14.5ms → 12.5ms latency (approaching 10ms theoretical limit)
- **Technique**: Fuse entire LLM forward pass into single megakernel
- **Key insight**: SM-level graph representation enables cross-operator pipelining
- **Challenge**: Requires compiler support, complex implementation
- **Source**: [Mirage Project](https://github.com/mirage-project/mirage)

#### Stanford Hazy Research Megakernel (May 2025)
- **Finding**: vLLM/SGLang only use 50% GPU bandwidth at batch=1
- **Solution**: Fuse whole Llama forward pass into single kernel
- **Challenge**: Synchronization across thread blocks
- **Source**: [No Bubbles Blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)

#### CUDA Graphs
- **Performance**: 1.2x speedup for llama.cpp on H100
- **Technique**: Record kernel sequences, replay as single graph
- **Key insight**: Reduces CPU-side launch overhead from ~10μs to ~2.5μs per kernel
- **Source**: [NVIDIA CUDA Graphs Blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)

### 2.3 Attention Optimization

#### FlashInfer (MLSys 2025 Best Paper)
- **Technique**: Block-sparse KV cache with composable formats
- **Key features**: Paged attention, load-balanced scheduling, CUDAGraph compatible
- **Integration**: vLLM, SGLang, MLC-Engine
- **Source**: [FlashInfer Paper](https://arxiv.org/abs/2501.01005)

#### ThunderMLA
- **Performance**: 20-35% faster than FlashMLA
- **Technique**: Virtual instruction set + megakernel fusion
- **Latency**: 41μs achieving 183 TFLOPs, 1520 GB/s
- **Source**: [ThunderMLA Blog](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)

### 2.4 Speculative Decoding

#### EAGLE-3
- **Technique**: Autoregressive prediction head attached to target model
- **Benefit**: No separate draft model needed
- **Source**: [NVIDIA Speculative Decoding Blog](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

#### Google AI Overviews
- **Production use**: 3x faster inference with speculative decoding
- **Key insight**: Draft model latency matters more than capability
- **Source**: [Google Research Blog](https://research.google/blog/looking-back-at-speculative-decoding/)

---

## 3. Gap Analysis: EdgeLLM vs State-of-the-Art

| Technique | State-of-Art | EdgeLLM | Gap |
|-----------|--------------|---------|-----|
| Ternary Kernel | TriRun (8x) | T-MAC LUT | Similar approach |
| Kernel Fusion | Megakernel (single) | Streaming fused (partial) | **Opportunity** |
| Launch Overhead | CUDA Graphs | None | **Opportunity** |
| Memory Access | In-register LUT (T-SAR) | Shared memory LUT | **Opportunity** |
| Batch Optimization | Tensor Cores | Disabled | Not needed for batch=1 |
| Attention | FlashInfer | Custom | Already efficient |

---

## 4. Recommended Optimizations (Priority Order)

### Priority 1: CUDA Graphs (Expected: 1.2-1.5x)
**Why**: Eliminates kernel launch overhead, easy to implement
```cuda
// Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... launch kernels ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Replay (per token)
cudaGraphLaunch(graphExec, stream);
```
**Estimated gain**: 630 → 756-945 tok/s

### Priority 2: Register-Level LUT (Expected: 1.5-2x)
**Why**: T-SAR shows 86x GEMV improvement with in-register LUT
**Approach**: Use CUDA shuffle instructions for warp-level LUT
```cuda
// Instead of shared memory LUT
__shared__ float lut[16][TILE_N];

// Use warp shuffles for register-level LUT
float lut_val = __shfl_sync(0xffffffff, warp_lut[idx], lane);
```
**Estimated gain**: 756 → 1134-1512 tok/s

### Priority 3: Persistent Megakernel (Expected: 1.3-1.5x)
**Why**: Eliminates all kernel launch overhead, maximizes occupancy
**Challenge**: Complex implementation, requires careful synchronization
**Approach**: Fuse RMSNorm + T-MAC + attention into single kernel
**Estimated gain**: 1134 → 1474-1701 tok/s

### Priority 4: Speculative Decoding (Expected: 2-3x)
**Why**: Generate multiple tokens per forward pass
**Approach**: Small BitNet draft model + verify with main model
**Challenge**: Need to train/find suitable draft model
**Estimated gain**: 1474 → 2948-4422 tok/s (with 2x acceptance rate)

---

## 5. Implementation Roadmap

### Phase 4: CUDA Graphs (1-2 days)
- [ ] Wrap forward pass in CUDA graph capture
- [ ] Handle dynamic shapes with graph variants
- [ ] Benchmark latency reduction
- **Target**: 800+ tok/s

### Phase 5: Register-Level LUT (3-5 days)
- [ ] Study T-SAR register layout
- [ ] Implement warp-shuffle based LUT
- [ ] Optimize for T4's 32 registers/thread
- **Target**: 1200+ tok/s

### Phase 6: Megakernel (1-2 weeks)
- [ ] Design SM-level scheduling
- [ ] Fuse all layers into persistent kernel
- [ ] Implement inter-block synchronization
- **Target**: 1500+ tok/s

### Phase 7: Speculative Decoding (2-4 weeks)
- [ ] Train small draft model (or use existing)
- [ ] Implement parallel verification
- [ ] Optimize acceptance rate
- **Target**: 2000+ tok/s

---

## 6. Competitive Analysis

### vs Ollama (llama.cpp backend)
| Metric | Ollama | EdgeLLM Current | EdgeLLM Target |
|--------|--------|-----------------|----------------|
| Throughput | 423 tok/s | 630 tok/s (+49%) | 1500 tok/s (+255%) |
| Model Size | ~91 MB | 53 MB | 53 MB |
| Quantization | Q4/Q8 | 1.58-bit | 1.58-bit |
| CUDA Graphs | Yes | No | Yes (planned) |

### vs bitnet.cpp
| Metric | bitnet.cpp | EdgeLLM |
|--------|------------|---------|
| Model | 2B params | 135M params |
| GPU | A100 | T4 |
| Throughput | 250 tok/s | 630 tok/s |
| Approach | Custom W1.58A8 CUDA | T-MAC LUT |

### vs TriRun
| Metric | TriRun | EdgeLLM |
|--------|--------|---------|
| Speedup | 8x vs FP16 | 11.5x vs naive |
| Batch sweet spot | 16-32 | 1 (edge inference) |
| Target | Datacenter | Edge devices |

---

## 7. Key Research Papers

1. **T-MAC** (EuroSys 2025) - LUT-based ternary inference
   - [arXiv:2407.00088](https://arxiv.org/abs/2407.00088)

2. **BitNet b1.58 2B4T** (April 2025) - Native 1.58-bit LLMs
   - [arXiv:2504.12285](https://arxiv.org/abs/2504.12285)

3. **FlashInfer** (MLSys 2025 Best Paper) - Efficient attention engine
   - [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

4. **Mirage Persistent Kernel** (Dec 2025) - Megakernel compiler
   - [arXiv:2512.22219](https://arxiv.org/abs/2512.22219)

5. **TriRun** (ACL 2025) - Ternary LLM scaling laws
   - [arXiv:2506.23025](https://arxiv.org/abs/2506.23025)

6. **T-SAR** (2025) - In-register LUT for CPUs
   - [Quantum Zeitgeist Article](https://quantumzeitgeist.com/2x-5x-cpu-sar-achieves-gemv-throughput-gemm-speedup-only-ternary/)

---

## 8. Conclusion

EdgeLLM's current 630 tok/s already beats Ollama by 49%. The most promising optimizations are:

1. **CUDA Graphs** - Low effort, immediate 1.2x gain
2. **Register-Level LUT** - Medium effort, potential 2x gain
3. **Megakernel** - High effort, eliminates all overhead
4. **Speculative Decoding** - Requires draft model, but 2-3x potential

**Conservative target**: 1000 tok/s (2.4x Ollama) with Phases 4-5
**Aggressive target**: 2000+ tok/s (4.7x Ollama) with all phases

---

## Sources

- [NVIDIA LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [microsoft/BitNet GitHub](https://github.com/microsoft/BitNet)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [Mirage Project GitHub](https://github.com/mirage-project/mirage)
- [vLLM Anatomy Blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [LLM Inference Paper Collection](https://github.com/chenhongyu2048/LLM-inference-optimization-paper)
- [Stanford Hazy Research - No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [Google Speculative Decoding](https://research.google/blog/looking-back-at-speculative-decoding/)
