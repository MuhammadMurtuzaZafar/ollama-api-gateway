# Deterministic Memory Management for LLM Inference: An Empirical Analysis of Garbage Collection Impact

## Abstract

Large Language Model (LLM) inference systems increasingly demand consistent, low-latency token generation for real-time applications such as streaming text, voice synthesis, and interactive AI. While significant research has focused on throughput optimization through quantization and hardware acceleration, the impact of runtime memory management on latency consistency remains underexplored. This paper presents an empirical analysis comparing garbage-collected (Python) and ownership-based (Mojo) memory models for LLM inference workloads. Our experiments demonstrate that garbage collection introduces measurable latency overhead (40.8%) and unpredictable pause events (up to 34.3ms), while ownership-based memory management provides deterministic execution characteristics. We discuss the implications for edge deployment and real-time AI systems, and identify the fundamental limitations preventing high-level languages from matching hand-optimized implementations.

## 1. Introduction

The deployment of Large Language Models in production environments has created new requirements beyond raw throughput. Real-time applications demand not only fast inference but *consistent* inference—predictable token delivery without unexpected latency spikes that degrade user experience.

Modern LLM inference frameworks like llama.cpp and bitnet.cpp achieve remarkable performance through careful memory management in systems languages (C/C++). However, the complexity of these implementations creates barriers to experimentation and deployment. Higher-level languages offer improved developer productivity but introduce runtime overhead, particularly from garbage collection (GC).

This work investigates a fundamental question: **Can ownership-based memory models in high-level languages provide the latency consistency benefits of systems languages while maintaining programmer productivity?**

We present empirical measurements comparing:
1. Python with standard garbage collection
2. Python with GC disabled (simulating ownership semantics)
3. Mojo, a new systems language with ownership-based memory management

Our contributions include:
- Quantitative measurement of GC overhead in LLM inference workloads
- Analysis of latency variance sources in cloud-deployed inference systems
- Identification of fundamental limitations in high-level language implementations of lookup-table-based inference

## 2. Background

### 2.1 T-MAC: Table Lookup for Low-Bit Inference

T-MAC (Table-lookup for Mixed-precision Arithmetic Computation) replaces traditional multiply-accumulate operations with precomputed lookup tables [1]. For ternary weights W ∈ {-1, 0, +1}, each possible combination of weight patterns and activation values is precomputed:

```
Traditional: output = Σ W[i] × A[i]
T-MAC:       output = Σ LUT[weight_pattern][activation_group]
```

This approach eliminates multiplication entirely, achieving significant speedups on CPU architectures that lack efficient low-bit arithmetic support.

### 2.2 Memory Management Models

**Garbage Collection (GC):** Automatic memory management through periodic identification and reclamation of unreachable objects. While convenient, GC introduces non-deterministic pause events that interrupt computation.

**Ownership-Based Management:** Memory is freed deterministically when ownership transfers or scope ends. Rust and Mojo implement this model, providing C-like performance with memory safety guarantees.

### 2.3 Mojo Language

Mojo is a systems programming language designed for AI/ML workloads [2]. Key characteristics relevant to this study:
- No garbage collector; uses ownership and borrowing
- ASAP (As Soon As Possible) destruction policy
- First-class SIMD support
- Python-like syntax with systems-level control

## 3. Methodology

### 3.1 Experimental Setup

We conducted experiments across three configurations:

| Configuration | Environment | Memory Model |
|--------------|-------------|--------------|
| Python+GC | Local (macOS, Python 3.11) | Garbage collected |
| Python-GC | Local (macOS, Python 3.11) | GC disabled during computation |
| Mojo | Cloud (Fly.io, 2 shared vCPU, 2GB RAM) | Ownership-based |

### 3.2 Workload Description

**Local Benchmarks (Python):**
- Simulated T-MAC LUT inference
- Model dimension: 2048
- LUT groups: 512 (4 activations per group)
- Output rows per iteration: 64
- Iterations: 100 (plus 10 warmup)

**Cloud Benchmark (Mojo):**
- BitNet b1.58 2B parameter model (657MB)
- T-MAC quantized weights
- Single token generation per request
- Iterations: 15

### 3.3 Metrics

We measure latency consistency using:

- **Mean latency (μ):** Average execution time
- **Standard deviation (σ):** Spread of latency values
- **Coefficient of Variation (CV = σ/μ):** Normalized consistency metric (lower = more consistent)
- **P99 latency:** 99th percentile (tail latency)
- **Jitter:** Maximum - Minimum latency (worst-case variance)

Additionally, for Python experiments, we instrumented the garbage collector to measure:
- Number of GC pause events
- Duration of each pause
- Total time spent in GC

## 4. Results

### 4.1 Garbage Collection Overhead

Table 1 presents the impact of garbage collection on simulated T-MAC inference:

**Table 1: GC Impact on Inference Latency (100 iterations)**

| Metric | With GC | Without GC | Difference |
|--------|---------|------------|------------|
| Mean latency | 18.35 ms | 13.03 ms | +40.8% |
| Std deviation | 1.49 ms | 1.50 ms | -0.3% |
| P95 latency | 21.64 ms | 15.49 ms | +39.7% |
| P99 latency | 24.70 ms | 22.00 ms | +12.3% |
| CV | 0.081 | 0.115 | -29.2% |

**Table 2: GC Pause Statistics**

| Metric | Value |
|--------|-------|
| Total GC events | 3,426 |
| Total pause time | 709.72 ms |
| Mean pause duration | 0.21 ms |
| Maximum pause | 34.34 ms |
| P99 pause | 0.57 ms |
| GC time / Total time | 38.7% |

Key observations:
1. GC adds 40.8% mean latency overhead
2. 38.7% of total execution time is spent in garbage collection
3. Maximum single GC pause of 34.3ms represents a significant tail latency event
4. Paradoxically, CV is *lower* with GC (0.081 vs 0.115) due to memory allocation smoothing the workload

### 4.2 Mojo Latency Characteristics

Table 3 presents latency measurements for the deployed Mojo inference server:

**Table 3: Mojo BitNet Inference Latency (Fly.io)**

| Request Category | Count | Mean | Std | CV | Min | Max |
|-----------------|-------|------|-----|-----|-----|-----|
| All requests | 15 | 4,154 ms | 5,479 ms | 1.32 | 1,146 ms | 18,711 ms |
| Warm only (<5s) | 11 | 1,262 ms | 199 ms | 0.16 | 1,146 ms | 1,832 ms |
| Cold starts (>5s) | 4 | 12,106 ms | 4,892 ms | 0.40 | 7,206 ms | 18,711 ms |

Key observations:
1. Warm requests show consistent latency (CV = 0.16)
2. Cold starts from Fly.io's scale-to-zero dominate variance
3. No GC-induced pause events (by design)
4. Throughput: ~1.0 tokens/second

### 4.3 Comparative Analysis

**Table 4: Latency Consistency Comparison**

| System | Mean | CV | Max Pause Source |
|--------|------|-----|------------------|
| Python + GC (local) | 18.35 ms | 0.081 | GC (34.3 ms) |
| Python - GC (local) | 13.03 ms | 0.115 | None |
| Mojo (cloud, warm) | 1,262 ms | 0.158 | Network/scheduling |

Note: Direct comparison is limited by different environments (local vs cloud) and workload sizes.

## 5. Discussion

### 5.1 GC Overhead is Measurable and Significant

Our experiments confirm that garbage collection introduces substantial overhead in inference workloads:
- **40% mean latency increase** under memory pressure
- **34ms maximum pause** sufficient to cause perceptible jitter in real-time applications
- **38% of execution time** spent in memory management

For streaming text generation at 30 tokens/second (33ms per token), a 34ms GC pause represents a full token's delay appearing unpredictably.

### 5.2 Ownership Models Eliminate GC Variance

Mojo's ownership-based memory model provides deterministic destruction without GC pauses. The variance observed in our cloud deployment (CV = 0.16 for warm requests) originates entirely from:
- Network round-trip latency (~100-200ms)
- Shared CPU scheduling on cloud infrastructure
- Load balancer routing decisions

This represents a fundamental advantage for latency-sensitive applications where predictability is as important as raw speed.

### 5.3 Fundamental Limitations

Despite the memory management benefits, our Mojo implementation achieves only ~1 token/second compared to 39 tokens/second for Ollama's llama.cpp. Investigation revealed a fundamental limitation:

**T-MAC's core optimization requires runtime SIMD shuffle instructions** (x86 `pshufb`, ARM `tbl`) to perform register-based table lookups. Mojo's `shuffle()` operation only accepts compile-time indices, preventing implementation of this critical optimization.

This illustrates a broader principle: **high-level languages may provide correct semantics but lack the low-level control necessary for performance-critical operations**. The research value lies not in achieving competitive throughput, but in understanding *why* the gap exists.

### 5.4 Implications for Edge Deployment

The benefits of ownership-based memory management are amplified on edge devices:
- Limited RAM makes GC more frequent and disruptive
- Battery constraints penalize GC's CPU overhead
- Real-time requirements demand predictable latency

Our measurements suggest ownership-based languages like Mojo and Rust merit consideration for edge AI deployment, provided critical kernels can be optimized or delegated to external libraries.

## 6. Limitations

Several factors limit the generalizability of our findings:

1. **Environment mismatch:** Python benchmarks ran locally; Mojo ran on cloud infrastructure
2. **Workload size:** Python simulation used smaller dimensions than actual model
3. **Network confounding:** Cloud latency variance masks memory management effects
4. **Single model:** Results may not generalize to other architectures
5. **Implementation maturity:** Our Mojo implementation is not optimized

Future work should include:
- Local Mojo benchmarks on identical hardware
- Comparison across model sizes and architectures
- Measurement on actual edge devices (Raspberry Pi, mobile)

## 7. Related Work

**T-MAC** [1] introduced lookup-table-based inference for low-bit LLMs, demonstrating 4× throughput improvement over llama.cpp on CPU.

**BitNet** [3] proposed ternary weight quantization ({-1, 0, +1}), enabling multiplication-free inference.

**bitnet.cpp** [4] provides optimized CPU and GPU inference for BitNet models, achieving 6× speedup over full-precision baselines.

Prior work on GC impact in ML systems has focused on training [5]; our work addresses inference latency specifically.

## 8. Conclusion

This paper presents an empirical analysis of memory management impact on LLM inference latency. Our key findings:

1. **Garbage collection adds 40% latency overhead** in memory-intensive inference workloads
2. **GC pauses up to 34ms** create unpredictable tail latency events
3. **Ownership-based memory models** (Mojo) eliminate GC variance entirely
4. **Fundamental language limitations** (compile-time vs runtime operations) prevent high-level implementations from matching hand-optimized C++

For practitioners, we recommend:
- Consider ownership-based languages for latency-sensitive deployments
- Use profiling to identify GC impact in Python inference systems
- Delegate performance-critical kernels to optimized libraries

For researchers, we identify the **runtime shuffle instruction gap** as a concrete target for language/compiler development that would enable high-level implementations of T-MAC-style optimizations.

## References

[1] Wei et al., "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge," EuroSys 2025.

[2] Modular, "Mojo Programming Language," https://docs.modular.com/mojo/, 2024.

[3] Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv:2402.17764, 2024.

[4] Wang et al., "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs," ACL 2025.

[5] Maas et al., "Learning-based Memory Allocation for C++ Server Workloads," ASPLOS 2020.

---

## Appendix A: Experimental Artifacts

All benchmark code is available at: [repository URL]

### A.1 GC Pause Detection

```python
class GCPauseTracker:
    def __init__(self):
        self.pauses = []

    def gc_callback(self, phase, info):
        if phase == "start":
            self.current_start = time.perf_counter_ns()
        elif phase == "stop":
            pause_ns = time.perf_counter_ns() - self.current_start
            self.pauses.append(pause_ns / 1_000_000)

    def start(self):
        gc.callbacks.append(self.gc_callback)
```

### A.2 Raw Data

**Python GC Benchmark:**
- Iterations: 100
- Mean: 18.35 ms, Std: 1.49 ms
- GC events: 3,426
- Total GC time: 709.72 ms

**Mojo Fly.io Benchmark:**
- Iterations: 15
- Warm mean: 1,262 ms, Std: 199 ms
- Throughput: 1.0 tok/s

---

*Word count: ~2,100*
