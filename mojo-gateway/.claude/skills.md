# Mojo Gateway - Skills & Use Cases

> Skills are reusable task patterns for Claude Code. Use `/skill-name` to invoke.

---

## T-MAC Inference Skills

### /quantize-tmac
Convert a LLaMA model to T-MAC ternary format.

**Steps:**
1. Load model weights from .bin file
2. Quantize to ternary {-1, 0, +1} with per-row scales
3. Pack 4 weights per byte (2 bits each)
4. Save in T-MAC v2 format (.tmac2.bin)

**Usage:**
```bash
python scripts/quantize_tmac_v2.py model.bin output.tmac2.bin
```

**Files:** `scripts/quantize_tmac_v2.py`

---

### /build-tmac
Build T-MAC inference engine.

**Steps:**
1. Ensure pixi environment is set up
2. Build with optimizations
3. Test with sample prompt

**Commands:**
```bash
pixi install
pixi run mojo build -O3 src/llama2_tmac_v2.mojo -o bin/llama2_tmac
./bin/llama2_tmac model.tmac2.bin -z tokenizer.bin -n 32
```

**Files:** `src/llama2_tmac_v2.mojo`

---

### /run-inference
Run inference with any implementation.

**Variants:**
- `llama2_tmac_v2` - T-MAC scaled ternary (recommended)
- `llama2_parallel` - Float32 SIMD baseline
- `llama2_int4` - Int4 quantized

**Usage:**
```bash
./bin/{variant} model.bin -z tokenizer.bin -n 128 -t 0.8 -p 0.9
```

**Options:**
- `-z` : Tokenizer path
- `-n` : Token count
- `-t` : Temperature (0.0-1.0)
- `-p` : Top-p threshold

---

## BitNet Skills

### /convert-bitnet
Convert Microsoft BitNet model to T-MAC format.

**Prerequisites:**
- Download model: `huggingface-cli download microsoft/bitnet-b1.58-2B-4T`

**Steps:**
1. Load safetensors model
2. Unpack base-3 ternary weights
3. Repack to T-MAC 2-bit format
4. Save with per-row scales

**Usage:**
```bash
python scripts/convert_bitnet_to_tmac.py output.tmac2.bin models/bitnet-2b/
```

**Files:** `scripts/convert_bitnet_to_tmac.py`

**Key Conversion:**
```python
# BitNet base-3: byte = w0 + 3*w1 + 9*w2 + 27*w3
w0 = byte % 3
w1 = (byte // 3) % 3
# ... maps {0,1,2} -> {-1,0,+1}

# T-MAC 2-bit: 00=0, 01=+1, 11=-1
```

---

### /build-bitnet
Build BitNet inference engine using Docker.

**Why Docker:** Mojo only supports Linux-64 and macOS-ARM64.

**Steps:**
```bash
docker build -f Dockerfile.bitnet -t bitnet-inference .
docker run --rm bitnet-inference models/bitnet-2b.tmac2.bin -n 32
```

**Files:** `Dockerfile.bitnet`, `src/bitnet_simple.mojo`

---

### /bitnet-architecture
Understanding BitNet b1.58 architecture differences.

**Key Differences from LLaMA:**

| Component | LLaMA | BitNet |
|-----------|-------|--------|
| Activation | SiLU | ReLU² |
| Norms | Pre-norm | + Sub-norms |
| RoPE theta | 10,000 | 500,000 |
| Attention | MHA | GQA (20h/5kv) |

**Sub-norms:**
```mojo
# After attention
rmsnorm(xb, attn_output, attn_sub_norm)  # BitNet only
output_projection(result, xb)

# After FFN gate*up
rmsnorm(hb, ffn_intermediate, ffn_sub_norm)  # BitNet only
down_projection(result, hb)
```

---

## Mojo Development Skills

### /add-mojo-struct
Add a new Mojo struct with required traits.

**Template:**
```mojo
struct MyStruct(Movable):
    var data: List[Float32]
    var size: Int

    fn __init__(out self, size: Int):
        self.data = List[Float32]()
        self.size = size
        for _ in range(size):
            self.data.append(0.0)

    fn __moveinit__(out self, deinit other: Self):
        self.data = other.data^
        self.size = other.size
```

**Note:** Use `deinit` not `owned` (deprecated).

---

### /fix-aliasing
Fix Mojo aliasing errors.

**Error:**
```
argument allows reading a memory location previously writable through another aliased argument
```

**Solution:**
```mojo
# Bad: same buffer for input and output
rmsnorm(state.x, state.x, weights)

# Good: use temp buffer
rmsnorm(state.xb, state.x, weights)
for i in range(dim):
    state.x[i] = state.xb[i]
```

---

### /add-parallel
Add parallel computation.

**Template:**
```mojo
from algorithm import parallelize

fn process_data(mut data: List[Float32], size: Int):
    @parameter
    fn compute_item(i: Int):
        data[i] = data[i] * 2.0

    parallelize[compute_item](size)
```

---

## Quantization Skills

### /analyze-quantization
Analyze quantization quality.

**Metrics:**
- MSE between original and quantized
- Percentage of zeros
- Scale distribution

**Usage:**
```python
# In quantization script
original = weights.flatten()
quantized = quantize_ternary(original)
dequantized = dequantize(quantized, scales)
mse = np.mean((original - dequantized) ** 2)
print(f"MSE: {mse:.6f}")
```

---

### /calibration
Add calibration data for better quantization.

**Steps:**
1. Run model on calibration dataset
2. Collect activation statistics
3. Optimize scales to minimize error
4. Save calibrated model

**Note:** Not yet implemented - future work.

---

## Benchmark Skills

### /benchmark-all
Run comprehensive benchmarks.

**Script:**
```bash
./scripts/benchmark_all.sh model.bin tokenizer.bin
```

**Outputs:**
- Tokens/second for each implementation
- Memory usage
- Model size comparison

---

### /profile
Profile inference performance.

**Commands:**
```bash
# Time inference
time ./bin/llama2_tmac model.tmac2.bin -n 100

# Memory profile (Linux)
/usr/bin/time -v ./bin/llama2_tmac model.tmac2.bin -n 100
```

---

## Quick Reference

| Skill | Purpose |
|-------|---------|
| `/quantize-tmac` | Convert model to T-MAC format |
| `/build-tmac` | Build T-MAC inference |
| `/convert-bitnet` | Convert BitNet to T-MAC |
| `/build-bitnet` | Build BitNet inference (Docker) |
| `/run-inference` | Run any inference variant |
| `/add-mojo-struct` | Create Mojo struct |
| `/fix-aliasing` | Fix borrow checker errors |
| `/benchmark-all` | Run benchmarks |

---

## Architecture Notes

### Weight Formats

**T-MAC v2 (.tmac2.bin):**
```
Magic: "TM2\0"
Config: [dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len]
Weights: [flag][dims][scale+ternary per row] or [flag][float data]
```

**Ternary Encoding:**
```
2 bits per weight, 4 per byte:
00 = 0, 01 = +1, 11 = -1
```

### Compression Ratios

| Format | Bits/Weight | Compression |
|--------|-------------|-------------|
| Float32 | 32 | 1x |
| Int4 | 4 | 8x |
| Ternary | 2 | 16x |

---

## Platform Requirements

**Mojo/MAX Supported:**
- Linux x86_64
- macOS ARM64 (Apple Silicon)

**NOT Supported:**
- macOS x86_64 (Intel Mac) - use Docker

---

## File Structure

```
mojo-gateway/
├── src/
│   ├── llama2_tmac_v2.mojo     # T-MAC inference
│   ├── bitnet_simple.mojo      # BitNet inference
│   └── ...
├── scripts/
│   ├── quantize_tmac_v2.py     # T-MAC quantizer
│   ├── convert_bitnet_to_tmac.py  # BitNet converter
│   └── ...
├── models/
│   └── bitnet-2b/              # Downloaded models
├── docs/
│   └── BITNET_LEARNINGS.md     # Technical learnings
└── Dockerfile.bitnet           # Docker build
```
