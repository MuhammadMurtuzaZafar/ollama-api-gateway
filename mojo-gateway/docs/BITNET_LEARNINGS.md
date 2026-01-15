# BitNet Integration Learnings

Technical documentation of integrating Microsoft BitNet b1.58-2B model with T-MAC inference.

## Key Technical Discoveries

### 1. BitNet Weight Format

BitNet uses a unique base-3 packed format for ternary weights:

```python
# BitNet encoding: 4 ternary values per byte
byte = w0 + 3*w1 + 9*w2 + 27*w3
# where w_i ∈ {0, 1, 2} maps to {-1, 0, +1}

# Decoding:
w0 = byte % 3
w1 = (byte // 3) % 3
w2 = (byte // 9) % 3
w3 = (byte // 27) % 3
ternary = w - 1  # Maps {0,1,2} -> {-1,0,+1}
```

This differs from our T-MAC 2-bit encoding:
```
00 = 0, 01 = +1, 11 = -1
```

### 2. BitNet Architecture Differences

BitNet b1.58 differs from standard LLaMA architecture:

| Component | LLaMA | BitNet |
|-----------|-------|--------|
| FFN Activation | SiLU | ReLU² (squared) |
| Normalization | Pre-norm only | + Sub-layer norms |
| RoPE theta | 10,000 | 500,000 |
| Attention | Standard MHA | Grouped Query (20h/5kv) |

### 3. Sub-Layer Normalization

BitNet adds normalization after attention and FFN outputs:
- `attn_sub_norm`: After attention, before output projection
- `ffn_sub_norm`: After FFN gate/up, before down projection

```mojo
# After attention weighted sum
rmsnorm(xb2, xb, attn_sub_norm_weights)  # BitNet specific
output_projection(xb, xb2)

# After FFN SwiGLU-like operation
rmsnorm(hb2, hb, ffn_sub_norm_weights)   # BitNet specific
down_projection(xb, hb2)
```

### 4. ReLU² Activation

BitNet uses squared ReLU instead of SiLU:
```mojo
# SiLU (standard LLaMA)
fn silu(x: Float32) -> Float32:
    return x / (1.0 + exp(-x))

# ReLU² (BitNet)
fn relu2(x: Float32) -> Float32:
    if x > 0:
        return x * x
    return 0.0
```

### 5. Mojo Language Learnings

#### Movable Trait
Structs containing Lists must implement `Movable`:
```mojo
struct FlatWeights(Movable):
    var data: List[UInt8]

    fn __moveinit__(out self, deinit other: Self):
        self.data = other.data^
```

#### Aliasing Issues
Mojo's borrow checker prevents aliased mutable arguments:
```mojo
# Error: aliased arguments
rmsnorm(state.x, state.x, weights)  # x is both output and input

# Solution: use temp buffer
rmsnorm(state.xb, state.x, weights)
for i in range(dim):
    state.x[i] = state.xb[i]
```

#### Deprecated Features
- `@value` decorator removed - use explicit trait conformances
- `owned` keyword deprecated - use `deinit`

### 6. Conversion Pipeline

Complete pipeline for BitNet to T-MAC:

```bash
# 1. Download model from HuggingFace
huggingface-cli download microsoft/bitnet-b1.58-2B-4T

# 2. Convert to T-MAC format
python scripts/convert_bitnet_to_tmac.py output.tmac2.bin models/bitnet-2b/

# 3. Build inference engine (requires ARM Mac or Linux)
docker build -f Dockerfile.bitnet -t bitnet-inference .

# 4. Run inference
docker run --rm bitnet-inference models/bitnet-2b.tmac2.bin -n 32
```

### 7. Platform Requirements

Mojo/MAX is only available for:
- Linux x86_64
- macOS ARM64 (Apple Silicon)

Intel Macs (x86_64 macOS) are NOT supported. Use Docker for cross-platform builds.

## Performance Observations

### Initial Results
- Model size: 657 MB (T-MAC format)
- Inference speed: 0.36 tok/s (unoptimized)
- Memory footprint: ~2 GB runtime

### Optimization Opportunities
1. SIMD-optimized ternary matmul
2. Better cache locality in weight layout
3. Parallelization tuning for GQA
4. Fused kernels for attention + norm

## Quality Considerations

### Post-Training Quantization
Converting float weights to ternary causes significant quality loss:
- Achieves ~30% of original quality
- Repetitive/degenerate outputs common

### Native Ternary Training
BitNet models trained from scratch maintain quality:
- 95-100% of FP16 quality
- No quantization artifacts

The microsoft/bitnet-b1.58-2B-4T model was trained with ternary constraints, so it should produce high-quality outputs once inference bugs are resolved.

## File Formats

### T-MAC v2 Format (.tmac2.bin)
```
Header:
  [4 bytes]  Magic: "TM2\0"
  [28 bytes] Config: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len

Per weight matrix:
  [1 byte]   Flag: 0=float32, 1=quantized

  If quantized:
    [4 bytes] rows (int32)
    [4 bytes] cols (int32)
    For each row:
      [2 bytes] scale (float16)
      [ceil(cols/4) bytes] packed ternary weights

  If float32:
    [size*4 bytes] raw float32 data
```

## References

1. [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764) - 1-bit LLM architecture
2. [T-MAC Paper](https://arxiv.org/abs/2407.00088) - Table lookup inference
3. [bitnet.cpp](https://github.com/microsoft/BitNet) - Reference implementation
4. [Mojo Documentation](https://docs.modular.com/mojo/) - Language reference
