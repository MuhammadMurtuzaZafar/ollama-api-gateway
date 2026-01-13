# EdgeLLM Engine

High-performance LLM inference engine optimized for edge GPUs. Achieves **80 tok/s** on T4 GPU with INT4 quantization - competitive with production systems at a fraction of the model size.

## Performance Highlights

| Metric | EdgeLLM (INT4) | Ollama (FP16) |
|--------|----------------|---------------|
| **Throughput** | **80 tok/s** | 60-70 tok/s |
| **Model Size** | **0.75 GB** | 3+ GB |
| **Memory** | 2 GB VRAM | 8+ GB VRAM |
| **Hardware** | T4 ($0.35/hr) | A100 ($2+/hr) |

Tested on Qwen2.5-1.5B with 200 token generation.

## Features

- **INT4 Quantization** - 4x smaller models with minimal quality loss
- **Optimized CUDA Kernels** - Custom multirow GEMV achieving 40% speedup
- **Edge GPU Support** - Runs on T4, Jetson, consumer GPUs
- **Low Latency** - Deterministic performance without GC pauses
- **Mojo Runtime** - Systems language with Python-like syntax

## Quick Start

### GPU Inference (Recommended)

```bash
# Clone repository
git clone https://github.com/umerkhan95/EdgeLLM.git
cd EdgeLLM/mojo-gateway

# Build CUDA kernels (requires CUDA toolkit)
cd src/kernels/cuda
make cuda  # For T4/consumer GPUs
# or: make cuda-jetson  # For Jetson devices

# Export and quantize a model
pip install torch transformers safetensors
python scripts/export_qwen_int4.py \
    --model Qwen/Qwen2.5-1.5B \
    --output models/qwen2.5-1.5b_int4.bin

# Run inference
./bin/edgellm_gpu_int4 \
    -m models/qwen2.5-1.5b_int4.bin \
    -z models/qwen2.5-1.5b_int4_tokenizer.bin \
    -n 50 -i "Explain quantum computing:"
```

### CPU Inference (BitNet)

For CPU-only deployment with ultra-low memory:

```bash
# Build C kernel
cd src/kernels && make

# Build Mojo binary
pixi run mojo build -O3 src/bitnet_tmac_lut.mojo -o bin/edgellm

# Run inference
./bin/edgellm models/smollm-135m.tm2.bin -n 32 -t 0.7
```

## Supported Models

| Model | INT4 Size | VRAM | Speed (T4) |
|-------|-----------|------|------------|
| Qwen2.5-0.5B | 0.3 GB | 1 GB | 120+ tok/s |
| Qwen2.5-1.5B | 0.75 GB | 2 GB | **80 tok/s** |
| Qwen2.5-3B | 1.5 GB | 4 GB | 50 tok/s |
| Llama-3.2-1B | 0.5 GB | 2 GB | 90 tok/s |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Mojo Runtime                        │
│  • Memory management (no GC)                        │
│  • Transformer forward pass                         │
│  • KV cache, RoPE, sampling                         │
└─────────────────────────────────────────────────────┘
                        │
                 Kernel Selector
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   CUDA   │  │  AVX2/   │  │   Pure   │
    │  (INT4)  │  │  NEON    │  │   Mojo   │
    │          │  │ (BitNet) │  │          │
    │ 80 tok/s │  │ 30 tok/s │  │ 8 tok/s  │
    └──────────┘  └──────────┘  └──────────┘
```

## CUDA Kernel Optimizations

The INT4 GEMV kernel includes several optimizations:

1. **Multirow Processing** - 8 rows per block reduces launch overhead (40% speedup)
2. **8-element Vectorization** - uint32 loads for 8 INT4 values at once
3. **FMA Instructions** - Fused multiply-add for better throughput
4. **Warp Reduction** - Efficient parallel reduction using shuffle intrinsics

### Kernel Performance (T4 GPU)

| Matrix Size | Baseline | Multirow | Speedup |
|-------------|----------|----------|---------|
| 1536×1536 | 21.1 μs | 11.8 μs | **1.8x** |
| 1536×8960 | 327 μs | 105 μs | **3.1x** |
| 8960×1536 | 119 μs | 44 μs | **2.7x** |

## Project Structure

```
mojo-gateway/
├── src/
│   ├── edgellm_gpu_int4.mojo    # GPU INT4 inference
│   ├── bitnet_tmac_lut.mojo     # CPU BitNet inference
│   └── kernels/
│       ├── cuda/
│       │   ├── int4_gemv.cu     # INT4 GEMV kernel
│       │   ├── int8_embedding.cu # INT8 embedding
│       │   └── cublas_matmul.cu  # cuBLAS integration
│       └── tmac_kernel.c         # CPU SIMD kernel
├── scripts/
│   ├── export_qwen_int4.py      # INT4 model export
│   └── quantize/                 # BitNet quantization
├── models/                       # Model files
└── docs/
    └── KERNEL_ARCHITECTURE_REVIEW.md
```

## Benchmarking

```bash
# Run GPU benchmark
./bin/edgellm_gpu_int4 \
    -m models/qwen2.5-1.5b_int4.bin \
    -z models/qwen2.5-1.5b_int4_tokenizer.bin \
    -n 200 -i "Write a detailed guide:"

# Expected output:
# Generated 200 tokens in 2489 ms
# Speed: 80 tokens/sec
```

## Hardware Requirements

### GPU (Recommended)
- NVIDIA GPU with compute capability 7.5+ (T4, RTX 20xx+)
- 2+ GB VRAM for 1.5B models
- CUDA 11.0+

### CPU (BitNet mode)
- x86_64 with AVX2 or ARM64 with NEON
- 512 MB+ RAM for small models

## Roadmap

- [x] INT4 multirow kernel (80 tok/s)
- [x] INT8 embedding quantization
- [ ] Tensor Core integration (target: 150+ tok/s)
- [ ] Flash Attention for long context
- [ ] Multi-GPU support

## References

- [Marlin Kernel](https://github.com/IST-DASLab/marlin) - Near-ideal INT4 speedup
- [PyTorch INT4](https://pytorch.org/blog/int4-decoding/) - CUDA optimizations
- [T-MAC Paper](https://arxiv.org/abs/2407.00088) - Table lookup inference
- [BitNet Paper](https://arxiv.org/abs/2402.17764) - 1.58-bit quantization

## License

MIT License
