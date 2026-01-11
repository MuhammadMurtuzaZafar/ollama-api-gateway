# Mojo Gateway - Claude Code Context

## Project Overview

High-performance LLM inference engine in Mojo featuring T-MAC (Table Lookup) inference for extreme memory efficiency. Includes support for Microsoft BitNet b1.58 models.

## Key Technologies

- **Mojo** - High-performance systems language from Modular
- **T-MAC** - Table lookup-based inference (no multiplication)
- **BitNet** - 1.58-bit ternary weight models from Microsoft
- **Docker** - Cross-platform builds (Mojo only on Linux/ARM Mac)

## Important Files

| File | Purpose |
|------|---------|
| `src/llama2_tmac_v2.mojo` | T-MAC inference engine |
| `src/bitnet_simple.mojo` | BitNet b1.58 inference |
| `scripts/quantize_tmac_v2.py` | Model quantizer |
| `scripts/convert_bitnet_to_tmac.py` | BitNet converter |
| `Dockerfile.bitnet` | Docker build for BitNet |

## Architecture

### T-MAC Approach
Eliminates multiplication by using lookup tables:
```
Traditional: output = Σ weight × activation
T-MAC: output = Σ LUT[weight_pattern]
```

### Weight Encoding
Ternary weights {-1, 0, +1} packed as 2 bits:
- `00` = 0
- `01` = +1
- `11` = -1

### BitNet Differences
- ReLU² activation (not SiLU)
- Sub-layer normalization
- RoPE theta = 500,000
- GQA: 20 heads, 5 KV heads

## Build Commands

```bash
# T-MAC (requires ARM Mac or Linux)
pixi run mojo build -O3 src/llama2_tmac_v2.mojo -o bin/llama2_tmac

# BitNet (Docker for any platform)
docker build -f Dockerfile.bitnet -t bitnet-inference .
docker run --rm bitnet-inference models/bitnet-2b.tmac2.bin -n 32
```

## Performance

| Implementation | Speed | Compression |
|---------------|-------|-------------|
| Float32 | 29 tok/s | 1x |
| T-MAC v2 | 17 tok/s | 15.7x |
| BitNet-2B | 0.36 tok/s | ~10x |

## Mojo-Specific Notes

1. **Movable Trait** - Structs with Lists need `__moveinit__`
2. **No Aliasing** - Can't use same buffer for input/output
3. **Use `deinit`** - `owned` is deprecated
4. **Platform** - Only Linux-64 and macOS-ARM64

## Common Tasks

### Add New Model
1. Create conversion script in `scripts/`
2. Create inference engine in `src/`
3. Add Docker support if needed
4. Update docs

### Fix Compilation Errors
- **Copyable error** → Add `Movable` trait
- **Aliasing error** → Use temp buffer
- **pow not found** → Use `**` operator

## References

- [T-MAC Paper](https://arxiv.org/abs/2407.00088)
- [BitNet Paper](https://arxiv.org/abs/2402.17764)
- [Mojo Docs](https://docs.modular.com/mojo/)
