# EdgeLLM Engine

High-performance LLM inference engine with deterministic latency for edge devices. Built with Mojo and C FFI kernels using BitNet 1.58-bit quantization.

## Features

- **15.5x lower latency jitter** than Ollama (373ms vs 5,799ms)
- **6.5x model compression** with BitNet 1.58-bit quantization
- **Deterministic performance** - no garbage collection pauses
- **Runs on $15 hardware** - Raspberry Pi Zero compatible
- **Offline capable** - no internet required

## Quick Start (Docker)

The easiest way to run EdgeLLM is with Docker:

```bash
# Clone the repository
git clone https://github.com/yourusername/edgellm.git
cd edgellm/mojo-gateway

# Build the Docker image
docker build -f Dockerfile.mojo -t edgellm-inference .

# Run inference (generates 20 tokens)
docker run --rm -v $(pwd)/models:/workspace/models \
    edgellm-inference \
    /workspace/bin/edgellm /workspace/models/smollm-135m.tm2.bin -n 20 -t 0.7
```

## Installation (Native)

### Prerequisites

**Linux (x86_64 or ARM64):**
```bash
# Install Mojo via pixi
curl -fsSL https://pixi.sh/install.sh | sh
pixi init -c https://conda.modular.com/max-nightly/ -c conda-forge
pixi add mojo python>=3.11
```

**macOS ARM64 (M1/M2/M3):**
```bash
# Native Mojo support
curl -fsSL https://pixi.sh/install.sh | sh
pixi init -c https://conda.modular.com/max-nightly/ -c conda-forge
pixi add mojo
```

**macOS Intel (x86_64):**
```bash
# Use Docker (native Mojo not supported on Intel Mac)
docker build -f Dockerfile.mojo -t edgellm-inference .
```

### Build from Source

```bash
# 1. Build C kernel
cd src/kernels
make clean all

# 2. Build Mojo inference binary
pixi run mojo build -O3 src/bitnet_tmac_lut.mojo -o bin/edgellm

# 3. Verify build
./bin/edgellm --help
```

## Model Preparation

### Option 1: Download Pre-quantized Model

```bash
# Create models directory
mkdir -p models

# Download SmolLM-135M (BitNet quantized)
# (Replace with actual download link when available)
wget -O models/smollm-135m.tm2.bin https://example.com/smollm-135m.tm2.bin
```

### Option 2: Quantize Your Own Model

```bash
# Install Python dependencies
pip install torch transformers safetensors numpy

# Step 1: Quantize HuggingFace model to TMAC format
python scripts/quantize/quantize_bitnet.py \
    --input HuggingFaceTB/SmolLM-135M \
    --output models/smollm-135m.tmac.bin

# Step 2: Convert TMAC to TM2 format (for Mojo runtime)
python scripts/convert_tmac_to_tm2.py \
    models/smollm-135m.tmac.bin \
    models/smollm-135m.tm2.bin
```

### Supported Models

| Model | Parameters | TM2 Size | Min RAM |
|-------|------------|----------|---------|
| SmolLM-135M | 135M | 40 MB | 256 MB |
| SmolLM-360M | 360M | 90 MB | 512 MB |
| Qwen2-0.5B | 500M | 125 MB | 1 GB |
| Llama-3.2-1B | 1B | 200 MB | 2 GB |

## Running Inference

### Basic Usage

```bash
# Generate 32 tokens with temperature 0.7
./bin/edgellm models/smollm-135m.tm2.bin -n 32 -t 0.7

# Greedy decoding (temperature 0)
./bin/edgellm models/smollm-135m.tm2.bin -n 50 -t 0

# Custom top-p sampling
./bin/edgellm models/smollm-135m.tm2.bin -n 32 -t 0.8 -p 0.95
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n <tokens>` | Number of tokens to generate | 32 |
| `-t <temp>` | Temperature (0 = greedy) | 0.7 |
| `-p <topp>` | Top-p nucleus sampling | 0.9 |

### Docker Usage

```bash
# Interactive mode
docker run -it --rm \
    -v $(pwd)/models:/workspace/models \
    edgellm-inference bash

# Inside container
/workspace/bin/edgellm /workspace/models/smollm-135m.tm2.bin -n 50

# One-liner inference
docker run --rm -v $(pwd)/models:/workspace/models \
    edgellm-inference \
    /workspace/bin/edgellm /workspace/models/smollm-135m.tm2.bin -n 32 -t 0.5
```

## Benchmarking

### Run Performance Benchmark

```bash
# EdgeLLM benchmark (in Docker)
docker run --rm \
    -v $(pwd)/models:/workspace/models \
    -v $(pwd)/results:/workspace/results \
    edgellm-inference \
    python3 /workspace/benchmarks/edgellm_benchmark.py \
        --backend edgellm \
        --model /workspace/models/smollm-135m.tm2.bin \
        --runs 30 \
        --output /workspace/results/benchmark.json

# View results
cat results/benchmark.json | python3 -m json.tool
```

### Compare with Ollama

```bash
# Start Ollama (if not running)
ollama serve &
ollama pull smollm:135m

# Run Ollama benchmark
python3 benchmarks/edgellm_benchmark.py \
    --backend ollama \
    --model smollm:135m \
    --runs 30 \
    --output results/ollama.json
```

### Expected Performance

| Metric | EdgeLLM | Ollama |
|--------|---------|--------|
| Throughput | 8 tok/s | 136 tok/s |
| Jitter | 373 ms | 5,799 ms |
| P99 Latency | 5.5 sec | 19.7 sec |
| Model Size | 40 MB | 91 MB |

**Note:** EdgeLLM prioritizes deterministic latency over raw throughput.

## Project Structure

```
mojo-gateway/
├── bin/                        # Compiled binaries
│   └── edgellm                 # Main inference binary
├── lib/                        # Shared libraries
│   └── libtmac_kernel.so       # C FFI kernel
├── models/                     # Model files (.tm2.bin)
├── results/                    # Benchmark results
├── src/
│   ├── bitnet_tmac_lut.mojo    # Main inference code
│   └── kernels/
│       ├── tmac_kernel.c       # AVX2/NEON SIMD kernel
│       └── Makefile
├── scripts/
│   ├── quantize/               # Quantization tools
│   │   └── quantize_bitnet.py
│   └── convert_tmac_to_tm2.py  # Format converter
├── benchmarks/
│   ├── edgellm_benchmark.py    # Performance benchmark
│   └── quality_metrics.py      # Output quality testing
├── Dockerfile.mojo             # Docker build file
└── BENCHMARK_REPORT.md         # Detailed benchmark results
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LD_LIBRARY_PATH` | Path to libtmac_kernel.so | `./lib` |
| `MOJO_ENABLE_STACK_TRACE_ON_ERROR` | Enable stack traces | unset |

### Memory Requirements

The model requires approximately:
- **Model weights:** Size of .tm2.bin file
- **KV Cache:** `2 × n_layers × seq_len × kv_dim × 4 bytes`
- **Runtime buffers:** ~10-20 MB

For SmolLM-135M with 2048 sequence length:
- Model: 40 MB
- KV Cache: ~70 MB
- Total: ~120 MB RAM

## Troubleshooting

### "Invalid model format" Error

The model file must be in TM2 format. Convert if needed:
```bash
python scripts/convert_tmac_to_tm2.py input.tmac.bin output.tm2.bin
```

### "Library not found" Error

Ensure the C kernel is built and in the library path:
```bash
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
```

### Docker Build Fails with "No space left"

Clean Docker cache:
```bash
docker system prune -f
docker builder prune -f
```

### Low Throughput

- Ensure running on native hardware (not emulated)
- Check CPU supports AVX2: `grep avx2 /proc/cpuinfo`
- Use Docker with `--cpuset-cpus` for CPU pinning

## Fine-Tuning (Optional)

To fine-tune your own model:

```bash
# 1. Prepare dataset (JSONL format)
cat > data.jsonl << 'EOF'
{"instruction": "Turn on the lights", "output": "Turning on living room lights"}
{"instruction": "Set temperature to 72", "output": "Setting thermostat to 72°F"}
EOF

# 2. Fine-tune with QLoRA (requires GPU)
python scripts/finetune/train_qlora.py \
    --base-model HuggingFaceTB/SmolLM-135M \
    --dataset data.jsonl \
    --output ./my-model

# 3. Merge LoRA weights
python scripts/finetune/merge_lora.py \
    --base-model HuggingFaceTB/SmolLM-135M \
    --lora-path ./my-model \
    --output ./my-model-merged

# 4. Quantize to BitNet
python scripts/quantize/quantize_bitnet.py \
    --input ./my-model-merged \
    --output ./my-model.tmac.bin

# 5. Convert to TM2
python scripts/convert_tmac_to_tm2.py \
    ./my-model.tmac.bin \
    ./my-model.tm2.bin
```

## API Reference

### Inference Binary

```
Usage: edgellm <model_path> [options]

Arguments:
  model_path    Path to .tm2.bin model file

Options:
  -n <int>      Number of tokens to generate (default: 32)
  -t <float>    Temperature for sampling (default: 0.7, 0 = greedy)
  -p <float>    Top-p for nucleus sampling (default: 0.9)
```

### Python Benchmark API

```python
from benchmarks.edgellm_benchmark import benchmark_edgellm

results = benchmark_edgellm(
    model_path="models/smollm-135m.tm2.bin",
    num_runs=30,
    tokens_per_run=32
)

print(f"Throughput: {results['throughput']['mean']:.1f} tok/s")
print(f"Jitter: {results['latency']['jitter']:.1f} ms")
```

## License

MIT License

## Acknowledgments

- [Modular](https://www.modular.com/) - Mojo language
- [T-MAC Paper](https://arxiv.org/abs/2407.00088) - Table lookup inference
- [BitNet Paper](https://arxiv.org/abs/2402.17764) - 1.58-bit quantization
- [llama2.c](https://github.com/karpathy/llama2.c) - Reference implementation
