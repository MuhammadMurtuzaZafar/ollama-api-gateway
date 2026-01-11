# EdgeLLM - Claude Code Context

## Project Overview

**EdgeLLM** is a platform for fine-tuning, optimizing, and deploying custom LLMs to edge devices with deterministic real-time performance. Built with Mojo (no GC) + hybrid C FFI kernels.

**Vision**: Fine-tune once, deploy everywhere - from cloud to edge.

**Target Market**: Edge/IoT, real-time AI, privacy-focused deployments

## Key Technologies

- **Mojo** - Systems language with ownership model (no GC), Python-like syntax
- **T-MAC** - Table lookup-based inference (no multiplication)
- **BitNet** - 1.58-bit ternary weight quantization
- **C FFI** - AVX2/NEON kernels for critical path (pshufb/tbl)
- **QLoRA** - Efficient fine-tuning on consumer GPUs

## Performance Findings

### Current State vs Target

| Metric | Current | Target | Technique |
|--------|---------|--------|-----------|
| Throughput | 1 tok/s | 20-50 tok/s | C FFI kernel |
| Latency jitter | ~100ms | <10ms | No GC |
| Memory (1B model) | 800MB | 400MB | BitNet 1.58-bit |

### Bottleneck Analysis

```
Root Cause: LUT in RAM vs SIMD Registers

Current (slow):
  lut.get(g, pattern)  → Memory load → 100-300 cycles

T-MAC/bitnet.cpp (fast):
  pshufb(lut_reg, idx) → Register lookup → 1 cycle

Mojo limitation: shuffle() requires compile-time indices
Solution: C FFI for critical kernel
```

### Optimization Priority

| Priority | Optimization | Impact | Effort |
|----------|-------------|--------|--------|
| 1 | C FFI kernel (pshufb/tbl) | 50x | High |
| 2 | SIMD RMSNorm/Softmax | 8x | Low |
| 3 | LUT pre-build | 3x | Low |
| 4 | Prefetching | 1.3x | Medium |

## Important Files

| File | Purpose |
|------|---------|
| `IMPLEMENTATION_PLAN.md` | Full project roadmap |
| `src/bitnet_server.mojo` | Current inference server |
| `src/kernels/tmac_kernel.c` | C FFI kernel (to create) |
| `scripts/finetune/` | Fine-tuning pipeline |
| `scripts/quantize/` | Quantization tools |
| `cli/` | EdgeLLM CLI tool |
| `docs/cpu_register_optimization.md` | Deep technical analysis |
| `docs/optimization_roadmap.md` | Path to 50 tok/s |

## Architecture

### Hybrid Mojo + C FFI

```
┌─────────────────────────────────────────────────┐
│              Mojo Layer (95%)                   │
│  • Memory management (ownership, no GC)         │
│  • Control flow, model loading                  │
│  • SIMD ops (RMSNorm, Softmax)                 │
│  • LUT building, parallelization               │
└─────────────────────────────────────────────────┘
                      │
                  FFI Call
                      ↓
┌─────────────────────────────────────────────────┐
│           C Kernel Layer (5%)                   │
│  • tmac_matmul_avx2() - x86 pshufb             │
│  • tmac_matmul_neon() - ARM tbl                │
│  • Register-based LUT lookup                    │
└─────────────────────────────────────────────────┘
```

### Fine-Tuning → Deploy Pipeline

```
HuggingFace Model → QLoRA Fine-tune → Merge → Quantize → T-MAC → Deploy
                    (FREE Colab)              (BitNet)   (.tmac2)
```

## Target Hardware

| Device | RAM | Model Size | Expected Speed |
|--------|-----|------------|----------------|
| Pi Zero 2 W ($15) | 512MB | SmolLM-135M | 5-10 tok/s |
| Raspberry Pi 5 | 8GB | Llama-1B | 20-40 tok/s |
| Jetson Nano | 4GB | Qwen-0.5B | 15-25 tok/s |
| Mac M1/M2 | 8GB+ | Llama-3B | 40-60 tok/s |

## Mojo-Specific Notes

### Working Patterns
```mojo
# SIMD operations
alias simd_width = simdwidthof[DType.float32]()
var v = ptr.load[width=simd_width](offset)
var sum = v.reduce_add()

# Parallelization
@parameter
fn compute_row(row: Int):
    ...
parallelize[compute_row](num_rows)

# FFI to C
from sys.ffi import external_call, DLHandle
external_call["tmac_matmul_avx2", NoneType](output, weights, ...)
```

### Known Limitations
1. **shuffle() is compile-time only** - Can't do runtime SIMD shuffle
2. **No runtime intrinsics** - Need C FFI for pshufb/tbl
3. **Platform support** - Linux-64 and macOS-ARM64 only

### Common Fixes
- **Copyable error** → Add `Movable` trait with `__moveinit__`
- **Aliasing error** → Use temp buffer, don't reuse input as output
- **pow not found** → Use `**` operator

## CLI Commands (Target)

```bash
edgellm finetune --base-model smollm-135m --data ./data.jsonl
edgellm quantize --input ./model --format bitnet --output ./model.tmac2.bin
edgellm serve --model ./model.tmac2.bin --port 8080
edgellm benchmark --model ./model.tmac2.bin
```

## Build Commands

```bash
# Mojo runtime
pixi run mojo build -O3 src/edgellm/runtime/inference.mojo -o bin/edgellm

# C kernel (x86)
clang -O3 -mavx2 -shared -fPIC -o lib/libtmac_kernel.so src/kernels/tmac_kernel.c

# C kernel (ARM)
clang -O3 -shared -fPIC -o lib/libtmac_kernel.so src/kernels/tmac_kernel.c
```

## References

- [T-MAC Paper](https://arxiv.org/abs/2407.00088) - EuroSys 2025
- [BitNet Paper](https://arxiv.org/abs/2402.17764) - 1.58-bit LLMs
- [NoMAD-Attention](https://arxiv.org/abs/2403.01273) - NeurIPS 2024
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient fine-tuning
- [Mojo FFI Docs](https://docs.modular.com/mojo/stdlib/sys/ffi/)

## Research Findings

### GC Impact (from benchmarks)
- Python GC adds 40% latency overhead
- Max GC pause: 34ms (problematic for real-time)
- Mojo: 0ms GC pauses (deterministic)

### Memory Bandwidth
- LLM inference is memory-bound, not compute-bound
- BitNet 1.58-bit: 2.5x less memory bandwidth than INT4
- Theoretical max with BitNet: ~50-60 tok/s on DDR4

### Competitive Analysis
- Ollama: 39 tok/s (llama.cpp, C++)
- Our target: 20-50 tok/s (Mojo + C FFI)
- Differentiation: Edge-first, fine-tuning, deterministic latency
