# EdgeLLM - Claude Code Context

## Project Overview

**EdgeLLM** is a platform for fine-tuning, optimizing, and deploying custom LLMs to edge devices with deterministic real-time performance. Built with Mojo (no GC) + hybrid C FFI kernels.

**Vision**: Fine-tune once, deploy everywhere - from cloud to edge.

**Target Market**: Edge/IoT, real-time AI, privacy-focused deployments

## Current Status (January 2026)

### Completed
- C FFI kernel integration (AVX2/NEON)
- BitNet 1.58-bit quantization pipeline
- T-MAC lookup table inference
- Transformer forward pass (RoPE, KV cache, sampling)
- Ollama benchmark comparison
- Docker-based Mojo development environment
- Automated benchmark suite with JSON output

### In Progress
- Fly.io deployment for cloud benchmarking
- Paper-ready benchmark evaluation
- Multi-platform validation

## Key Technologies

- **Mojo** - Systems language with ownership model (no GC), Python-like syntax
- **T-MAC** - Table lookup-based inference (no multiplication)
- **BitNet** - 1.58-bit ternary weight quantization
- **C FFI** - AVX2/NEON kernels for critical path (pshufb/tbl)
- **QLoRA** - Efficient fine-tuning on consumer GPUs

## Performance Results (SmolLM-135M)

### Benchmark Comparison

| Metric | Ollama | EdgeLLM | Winner |
|--------|--------|---------|--------|
| Throughput | 156.7 tok/s | 38.4 tok/s (est.) | Ollama |
| Latency Jitter | 5566ms | <10ms (target) | **EdgeLLM** |
| Model Size | ~91 MB | 53.2 MB | **EdgeLLM** |
| Min Hardware | $800+ PC | $15 Pi Zero | **EdgeLLM** |

### Key Advantage: Deterministic Latency
```
Ollama:   ████████████████████████████████████████  5566 ms jitter
EdgeLLM:  █                                         <10 ms jitter (target)
```

## Important Files

| File | Purpose |
|------|---------|
| `src/bitnet_tmac_lut.mojo` | Main inference with T-MAC LUT |
| `src/kernels/tmac_kernel.c` | C FFI kernel (AVX2/NEON) |
| `src/edgellm/ffi/tmac_kernel.mojo` | Mojo FFI wrapper |
| `src/edgellm/ffi/test_ffi.mojo` | FFI integration test |
| `benchmarks/edgellm_benchmark.py` | Automated benchmark suite |
| `scripts/quantize/quantize_bitnet.py` | BitNet quantization |
| `Dockerfile.mojo` | Mojo development container |
| `Dockerfile.benchmark` | Benchmark container for fly.io |
| `PAPER_ROADMAP.md` | Research paper roadmap |
| `BENCHMARK_REPORT.md` | Benchmark comparison report |

## Architecture

### Hybrid Mojo + C FFI

```
┌─────────────────────────────────────────────────┐
│              Mojo Layer (95%)                   │
│  • Memory management (ownership, no GC)         │
│  • Control flow, model loading                  │
│  • Transformer forward pass                     │
│  • LUT building, parallelization               │
└─────────────────────────────────────────────────┘
                      │
                  FFI Call
                      ↓
┌─────────────────────────────────────────────────┐
│           C Kernel Layer (5%)                   │
│  • tmac_matmul_avx2() - x86 pshufb             │
│  • tmac_matmul_neon() - ARM tbl                │
│  • rmsnorm_avx2/neon() - SIMD normalization    │
│  • softmax_avx2/neon() - SIMD softmax          │
│  • build_lut() - LUT construction              │
└─────────────────────────────────────────────────┘
```

### Fine-Tuning → Deploy Pipeline

```
HuggingFace Model → QLoRA Fine-tune → Merge → Quantize → T-MAC → Deploy
                    (FREE Colab)              (BitNet)   (.tmac2)
```

## Build Commands

```bash
# Docker development (Intel Mac or Linux)
docker compose -f docker-compose.mojo.yml up --build

# C kernel build (x86)
clang -O3 -mavx2 -shared -fPIC -o lib/libtmac_kernel.so src/kernels/tmac_kernel.c

# C kernel build (ARM)
clang -O3 -shared -fPIC -o lib/libtmac_kernel.so src/kernels/tmac_kernel.c

# Mojo inference (in Docker)
pixi run mojo build -O3 src/bitnet_tmac_lut.mojo -o bin/edgellm

# Run benchmarks
python benchmarks/edgellm_benchmark.py --compare --runs 100
```

## Quantization Commands

```bash
# Quantize SmolLM-135M to BitNet format
python scripts/quantize/quantize_bitnet.py \
    --model HuggingFaceTB/SmolLM-135M \
    --output models/smollm-135m.tmac2.bin

# Verify quantization
python scripts/quantize/verify_bitnet.py models/smollm-135m.tmac2.bin
```

## Benchmark Commands

```bash
# Full comparison (EdgeLLM vs Ollama)
python benchmarks/edgellm_benchmark.py --compare --runs 100 -o results.json

# EdgeLLM only
python benchmarks/edgellm_benchmark.py --backend edgellm --model models/smollm-135m.tmac2.bin

# Ollama only
python benchmarks/edgellm_benchmark.py --backend ollama --model smollm:135m
```

## Mojo FFI Integration

### Working Pattern (OwnedDLHandle)
```mojo
from sys.ffi import OwnedDLHandle
from memory import UnsafePointer

fn main() raises:
    var handle = OwnedDLHandle("/path/to/libtmac_kernel.so")

    # Call C functions
    var features = handle.call["get_cpu_features", Int32]()

    # With pointers
    var input_data = List[Float32]()
    var input_ptr = input_data.unsafe_ptr()
    handle.call["rmsnorm_avx2", NoneType](output_ptr, input_ptr, weight_ptr, size, eps)
```

### Known Mojo API Changes
- `DLHandle` → `OwnedDLHandle`
- `UnsafePointer.alloc()` → `List[T]().unsafe_ptr()`
- `list.data` → `list.unsafe_ptr()`

## Target Hardware

| Device | Price | RAM | Model | Expected Speed |
|--------|-------|-----|-------|----------------|
| Pi Zero 2 W | **$15** | 512MB | SmolLM-135M | 5-10 tok/s |
| Pi 4 | $35 | 4GB | Qwen-0.5B | 8-15 tok/s |
| Pi 5 | $80 | 8GB | Llama-1B | 20-40 tok/s |
| Jetson Nano | $99 | 4GB | Phi-3-mini | 15-25 tok/s |

## Platform Support

| Platform | Mojo Native | Docker | Status |
|----------|-------------|--------|--------|
| Linux x86_64 | Yes | Yes | Full support |
| macOS ARM64 | Yes | Yes | Full support |
| macOS x86_64 | No | Yes | Docker only |
| ARM64 (Pi) | Coming | Yes | Docker only |

## References

- [T-MAC Paper](https://arxiv.org/abs/2407.00088) - EuroSys 2025
- [BitNet Paper](https://arxiv.org/abs/2402.17764) - 1.58-bit LLMs
- [NoMAD-Attention](https://arxiv.org/abs/2403.01273) - NeurIPS 2024
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient fine-tuning
- [Mojo FFI Docs](https://docs.modular.com/mojo/stdlib/sys/ffi/)

## Research Findings

### Key Insights
- LLM inference is memory-bound, not compute-bound
- BitNet 1.58-bit: 4.8x compression vs FP16
- Mojo: 0ms GC pauses (deterministic latency)
- T-MAC eliminates multiplications via lookup tables

### Competitive Positioning
- Ollama: Higher throughput, but variable latency
- EdgeLLM: Lower throughput, but deterministic latency + smaller models
- Use case: Real-time robotics, voice assistants, IoT automation
