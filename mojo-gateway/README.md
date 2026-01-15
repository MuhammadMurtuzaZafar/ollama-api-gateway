# EdgeLLM

Fast LLM inference on edge GPUs. **88 tok/s** on T4 with INT4 quantization.

## Quick Start

```bash
# Build
cd src/kernels/cuda && make edge

# Run
./bin/edge run qwen "What is the capital of France?"
```

## Commands

```bash
# Interactive chat
edge run qwen

# Single prompt
edge run qwen "Explain quantum computing"

# List models
edge models
```

## Supported Models

| Model | Command | Size | Speed (T4) |
|-------|---------|------|------------|
| Qwen2.5-1.5B | `edge run qwen` | 0.75 GB | 88 tok/s |
| Llama 3.2-1B | `edge run llama` | 0.5 GB | 90 tok/s |

## Export Your Own Model

```bash
# Install dependencies
pip install torch transformers safetensors

# Export Qwen
python scripts/export_qwen_int4.py \
    --model Qwen/Qwen2.5-1.5B \
    --output models/qwen2.5-1.5b_int4.bin
```

## Build from Source

```bash
# Requirements: CUDA toolkit, nvcc

cd src/kernels/cuda

# For T4/consumer GPUs (sm_75)
make edge

# For Jetson (sm_72)
make edge-jetson

# For RTX 30xx/40xx (sm_86)
make edge-rtx
```

## Performance

Tested on Tesla T4 GPU with Qwen2.5-1.5B INT4:

| Metric | EdgeLLM | Ollama |
|--------|---------|--------|
| Throughput | **88 tok/s** | 30-35 tok/s |
| Model Size | 0.75 GB | 3+ GB |
| VRAM Usage | 2 GB | 8+ GB |

## Project Structure

```
src/kernels/cuda/
├── edge_cli.cu          # Main CLI
├── int4_gemv.cu         # INT4 GEMV kernel
├── cublas_matmul.cu     # cuBLAS integration
└── Makefile
scripts/
├── export_qwen_int4.py  # Qwen export
└── export_llama_int4.py # Llama export
models/                  # Model files
```

## Hardware Requirements

- NVIDIA GPU (T4, RTX 20xx+, Jetson)
- 2+ GB VRAM
- CUDA 11.0+

## License

MIT
