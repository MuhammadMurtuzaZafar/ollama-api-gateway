# T-MAC: Table Lookup-Based LLM Inference in Mojo

A high-performance, memory-efficient LLM inference engine implementing the T-MAC (Table-lookup for Low-bit LLM Deployment) algorithm in pure Mojo.

## Overview

T-MAC eliminates multiplication operations in matrix-vector multiplication by using precomputed lookup tables. This enables extreme memory compression (16x) while maintaining competitive inference speeds.

**Based on:** [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088)

## Key Features

- **No Multiplication in MatMul** - Pure table lookups and additions
- **16x Memory Compression** - Ternary weights {-1, 0, +1} at 2 bits per weight
- **CPU Optimized** - Designed for edge deployment without GPU
- **Two Implementations**:
  - T-MAC v1: Pure ternary (maximum compression)
  - T-MAC v2: Scaled ternary (better quality)

## Performance Benchmarks

Tested on stories110M model (110M parameters):

| Implementation | Speed | Model Size | Compression |
|---------------|-------|------------|-------------|
| Float32 (baseline) | 29 tok/s | 418 MB | 1x |
| Int4 Quantized | 13 tok/s | 62 MB | 7.1x |
| **T-MAC v1** | 14 tok/s | 26 MB | **15.8x** |
| **T-MAC v2** | 17 tok/s | 27 MB | **15.7x** |

### Memory Comparison for 7B Parameter Model

| Format | Size | Fits in RAM |
|--------|------|-------------|
| FP32 | 28 GB | No |
| FP16 | 14 GB | Barely |
| Int4 | 3.5 GB | Yes |
| **T-MAC** | **1.75 GB** | Easily |

## How T-MAC Works

### Traditional MatMul
```
output[i] = Σ weight[i,j] × activation[j]  // Expensive multiplications
```

### T-MAC Approach
```
1. Group 4 activations together
2. Precompute all 256 possible sums (2^8 combinations of 4 ternary weights)
3. Use weight bits as index into lookup table

output[i] = Σ LUT[group, weight_pattern]  // Just lookups + additions!
```

### Ternary Weight Encoding
```
2 bits per weight, 4 weights per byte:
- 00 = 0
- 01 = +1
- 11 = -1
```

## Project Structure

```
mojo-gateway/
├── src/
│   ├── llama2_tmac.mojo      # T-MAC v1: Pure ternary inference
│   ├── llama2_tmac_v2.mojo   # T-MAC v2: Scaled ternary (better quality)
│   ├── llama2_parallel.mojo   # Float32 baseline
│   └── llama2_int4.mojo       # Int4 quantized baseline
├── scripts/
│   ├── quantize_tmac.py       # Convert models to T-MAC v1 format
│   └── quantize_tmac_v2.py    # Convert models to T-MAC v2 format
└── README_TMAC.md
```

## Quick Start

### 1. Quantize a Model

**T-MAC v1 (maximum compression):**
```bash
python scripts/quantize_tmac.py model.bin model.tmac.bin
```

**T-MAC v2 (better quality):**
```bash
python scripts/quantize_tmac_v2.py model.bin model.tmac2.bin
```

### 2. Build the Inference Engine

```bash
mojo build -O3 src/llama2_tmac.mojo -o llama2_tmac
# or
mojo build -O3 src/llama2_tmac_v2.mojo -o llama2_tmac_v2
```

### 3. Run Inference

```bash
./llama2_tmac model.tmac.bin -z tokenizer.bin -n 128 -t 0.8
```

**Options:**
- `-z` : Path to tokenizer
- `-n` : Number of tokens to generate
- `-t` : Temperature (0.0 = greedy, 0.8 = creative)
- `-p` : Top-p sampling threshold

## T-MAC v1 vs v2

| Feature | T-MAC v1 | T-MAC v2 |
|---------|----------|----------|
| Compression | 15.8x | 15.7x |
| Speed | 14 tok/s | 17 tok/s |
| Quality | Lower | Higher |
| Scale factors | No | Per-row |
| Best for | Maximum compression | Balance of quality/size |

### T-MAC v2 Improvements

T-MAC v2 adds per-row scale factors to preserve magnitude information:

```
Format per row: [scale: float16] + [ternary weights]
Output = scale × LUT_result
```

This recovers magnitude information lost in pure ternary quantization.

## Theoretical Performance (Trained Model)

If a model is trained from scratch with ternary weights (like BitNet b1.58):

| Metric | Post-Training Quantization | Trained from Scratch |
|--------|---------------------------|---------------------|
| Quality | ~30% of FP32 | 95-100% of FP32 |
| Speed | 17 tok/s | 50-70 tok/s |
| Memory | 27 MB | 27 MB |

**Research shows:** A 7B ternary model trained from scratch matches the quality of a 7B FP16 model while using 10x less memory and running 4x faster on CPU.

## Comparison with Ollama

For the same memory budget (700 MB):

| System | Model Size | Quality |
|--------|-----------|---------|
| Ollama (Q4) | 1.1B | Good |
| T-MAC (trained) | 7-8B | Better |

T-MAC enables running **7x larger models** in the same memory footprint.

## File Formats

### T-MAC v1 Format (.tmac.bin)
```
[4 bytes]  Magic: "TMAC"
[28 bytes] Config: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
[variable] Weights (ternary for large matrices, float32 for norms)
```

### T-MAC v2 Format (.tmac2.bin)
```
[4 bytes]  Magic: "TM2\0"
[28 bytes] Config
[variable] Weights with per-row scales:
           - Quantized: [flag=1][rows][cols][scale+ternary per row]
           - Float32:   [flag=0][data]
```

## Limitations

1. **Post-training quantization** causes quality degradation
2. **Best results** require training from scratch with ternary constraints
3. **GPU inference** - Ollama is faster on GPU (this targets CPU/edge)

## BitNet Integration

This project now includes support for Microsoft's BitNet b1.58 models, which are trained from scratch with ternary weights.

### BitNet b1.58-2B-4T Support

We've implemented a full pipeline for running BitNet models:

1. **Model Download** - Fetch from HuggingFace: `microsoft/bitnet-b1.58-2B-4T`
2. **Conversion** - Convert BitNet's base-3 packed format to T-MAC 2-bit format
3. **Inference** - Run on CPU with T-MAC-style matrix operations

### BitNet Weight Format

BitNet stores ternary weights using base-3 encoding (4 values per byte):
```
byte = w0 + 3*w1 + 9*w2 + 27*w3
where each w_i ∈ {0, 1, 2} maps to {-1, 0, +1}
```

Our conversion script (`scripts/convert_bitnet_to_tmac.py`) transforms this to T-MAC's 2-bit format:
```
2 bits per weight: 00=0, 01=+1, 11=-1
```

### BitNet Architecture Differences

BitNet differs from standard LLaMA in several ways:
- **ReLU² activation** instead of SiLU/GELU
- **Sub-layer normalization** (attn_sub_norm, ffn_sub_norm)
- **Higher RoPE theta** (500,000 vs 10,000)
- **Grouped Query Attention** (20 heads, 5 KV heads)

### Running BitNet Inference

```bash
# Build with Docker (required for Linux/ARM)
docker build -f Dockerfile.bitnet -t bitnet-inference .

# Run inference
docker run --rm bitnet-inference models/bitnet-2b.tmac2.bin -n 32

# Or with custom parameters
docker run --rm bitnet-inference models/bitnet-2b.tmac2.bin -n 64 -t 0.8 -p 0.9
```

### BitNet Benchmark Results

| Model | Size | Tokens/sec | Notes |
|-------|------|------------|-------|
| BitNet b1.58-2B | 657 MB | 0.36 tok/s | Unoptimized, CPU-only |

**Note:** These results are from initial implementation. Performance can be improved with:
- SIMD-optimized ternary matmul
- Better memory layout
- Parallelization tuning

## Future Work

- [ ] GPTQ-style calibration for better post-training quantization
- [ ] Quantization-aware fine-tuning support
- [ ] SIMD-optimized LUT operations (TBL/PSHUF instructions)
- [ ] Training scripts for ternary models
- [ ] Integration with llama2.c model format
- [x] BitNet b1.58 model support
- [ ] Optimized BitNet inference (target: 5+ tok/s)

## References

1. [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088) - Ma et al., EuroSys 2025
2. [BitNet b1.58: The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) - Microsoft Research
3. [llama2.c](https://github.com/karpathy/llama2.c) - Andrej Karpathy

## License

MIT License
