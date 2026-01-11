# EdgeLLM - Skills & Use Cases

> Skills are reusable task patterns for Claude Code. Use `/skill-name` to invoke.

---

## Quick Start Skills

### /setup-dev
Set up development environment for EdgeLLM.

**Steps:**
```bash
# Clone and enter directory
cd mojo-gateway

# For Intel Mac or Linux (Docker required)
docker compose -f docker-compose.mojo.yml up --build

# For Apple Silicon Mac (native)
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
```

---

### /build-edgellm
Build EdgeLLM inference engine.

**Commands:**
```bash
# Build C kernels first
make -C src/kernels clean all

# Build Mojo inference (in Docker or native)
pixi run mojo build -O3 src/bitnet_tmac_lut.mojo -o bin/edgellm

# Verify build
./bin/edgellm --help
```

**Files:** `src/bitnet_tmac_lut.mojo`, `src/kernels/tmac_kernel.c`

---

### /run-inference
Run EdgeLLM inference.

**Usage:**
```bash
./bin/edgellm models/smollm-135m.tmac2.bin -n 32 -t 0.8 -p 0.9
```

**Options:**
- `-n` : Number of tokens to generate
- `-t` : Temperature (0.0 = deterministic)
- `-p` : Top-p threshold for sampling

---

## Quantization Skills

### /quantize-bitnet
Convert HuggingFace model to BitNet 1.58-bit format.

**Steps:**
```bash
python scripts/quantize/quantize_bitnet.py \
    --model HuggingFaceTB/SmolLM-135M \
    --output models/smollm-135m.tmac2.bin
```

**Result:**
- FP16 → 1.58-bit (4.8x compression)
- 256 MB → 53 MB for SmolLM-135M

**Files:** `scripts/quantize/quantize_bitnet.py`

---

### /verify-model
Verify quantized model integrity.

**Usage:**
```bash
python scripts/quantize/verify_bitnet.py models/smollm-135m.tmac2.bin
```

**Checks:**
- Magic bytes (TMAC)
- Config values
- Weight dimensions
- Scale distributions

---

## Benchmark Skills

### /benchmark-compare
Run EdgeLLM vs Ollama comparison benchmark.

**Prerequisites:**
```bash
# Ensure Ollama is running
ollama serve &
ollama pull smollm:135m
```

**Run benchmark:**
```bash
python benchmarks/edgellm_benchmark.py --compare --runs 100 -o results.json
```

**Output:**
- JSON file with detailed metrics
- Throughput (tok/s)
- Latency percentiles (P50, P95, P99)
- Jitter (standard deviation)

**Files:** `benchmarks/edgellm_benchmark.py`

---

### /benchmark-edgellm
Benchmark EdgeLLM only.

**Usage:**
```bash
python benchmarks/edgellm_benchmark.py \
    --backend edgellm \
    --model models/smollm-135m.tmac2.bin \
    --runs 100 \
    -o edgellm_results.json
```

---

### /benchmark-ollama
Benchmark Ollama only.

**Usage:**
```bash
python benchmarks/edgellm_benchmark.py \
    --backend ollama \
    --model smollm:135m \
    --runs 100 \
    -o ollama_results.json
```

---

## Docker Skills

### /docker-dev
Start Docker development environment.

**Usage:**
```bash
docker compose -f docker-compose.mojo.yml up --build
```

**Includes:**
- Mojo runtime via pixi
- C kernel compilation
- Python benchmarking tools

**Files:** `Dockerfile.mojo`, `docker-compose.mojo.yml`

---

### /docker-benchmark
Build and run benchmark Docker image.

**Usage:**
```bash
# Build
docker build -f Dockerfile.benchmark -t edgellm-benchmark .

# Run
docker run --rm -v $(pwd)/results:/workspace/results edgellm-benchmark
```

**Files:** `Dockerfile.benchmark`

---

## FFI Skills

### /test-ffi
Test C kernel FFI integration.

**Usage:**
```bash
# In Docker environment
pixi run mojo src/edgellm/ffi/test_ffi.mojo
```

**Tests:**
- CPU feature detection (AVX2, AVX512, NEON)
- RMSNorm performance
- Softmax performance
- LUT building

**Files:** `src/edgellm/ffi/test_ffi.mojo`

---

### /build-kernel
Build C kernels for current platform.

**x86 (AVX2):**
```bash
clang -O3 -mavx2 -shared -fPIC \
    -o lib/libtmac_kernel.so \
    src/kernels/tmac_kernel.c
```

**ARM (NEON):**
```bash
clang -O3 -shared -fPIC \
    -o lib/libtmac_kernel.so \
    src/kernels/tmac_kernel.c
```

**Files:** `src/kernels/tmac_kernel.c`, `src/kernels/tmac_kernel.h`

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

---

### /mojo-ffi-call
Make FFI call to C function.

**Pattern:**
```mojo
from sys.ffi import OwnedDLHandle

fn main() raises:
    var handle = OwnedDLHandle("lib/libtmac_kernel.so")

    # Simple call
    var result = handle.call["get_cpu_features", Int32]()

    # With pointers
    var data = List[Float32]()
    var ptr = data.unsafe_ptr()
    handle.call["rmsnorm_avx2", NoneType](out_ptr, ptr, weight_ptr, size, eps)
```

**Note:** Use `OwnedDLHandle` not `DLHandle` (deprecated).

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

## Research Skills

### /paper-roadmap
View research paper roadmap.

**File:** `PAPER_ROADMAP.md`

**Key Phases:**
1. Complete working system (Mojo inference)
2. Rigorous benchmarking (100+ runs)
3. Multi-platform validation (Pi 5, x86, ARM)
4. Paper writing
5. Artifact preparation

---

### /benchmark-report
View current benchmark report.

**File:** `BENCHMARK_REPORT.md`

**Key Results:**
- EdgeLLM: 38.4 tok/s (estimated), <10ms jitter
- Ollama: 156.7 tok/s, 5566ms jitter
- Model size: 53.2 MB (4.8x compression)

---

## Quick Reference

| Skill | Purpose |
|-------|---------|
| `/setup-dev` | Set up development environment |
| `/build-edgellm` | Build inference engine |
| `/run-inference` | Run model inference |
| `/quantize-bitnet` | Convert to BitNet format |
| `/benchmark-compare` | EdgeLLM vs Ollama benchmark |
| `/docker-dev` | Start Docker environment |
| `/test-ffi` | Test C kernel integration |
| `/paper-roadmap` | View research paper plan |

---

## File Structure

```
mojo-gateway/
├── src/
│   ├── bitnet_tmac_lut.mojo     # Main inference engine
│   ├── kernels/
│   │   ├── tmac_kernel.c        # C FFI kernel
│   │   └── tmac_kernel.h        # Kernel header
│   └── edgellm/
│       └── ffi/
│           ├── tmac_kernel.mojo # Mojo FFI wrapper
│           └── test_ffi.mojo    # FFI tests
├── benchmarks/
│   └── edgellm_benchmark.py     # Automated benchmarks
├── scripts/
│   └── quantize/
│       └── quantize_bitnet.py   # Quantization tool
├── models/                      # Model files (.tmac2.bin)
├── Dockerfile.mojo              # Dev container
├── Dockerfile.benchmark         # Benchmark container
├── BENCHMARK_REPORT.md          # Results
└── PAPER_ROADMAP.md             # Research roadmap
```

---

## Weight Format

### T-MAC v2 (.tmac2.bin)
```
Magic: "TMAC" (4 bytes)
Version: uint32
Hidden size: uint32
Num layers: uint32
Num heads: uint32
Vocab size: uint32
Bits: uint32
Group size: uint32
[Weights...]
```

### Ternary Encoding
```
2 bits per weight, 4 per byte:
00 = 0 (zero)
01 = +1 (positive)
11 = -1 (negative)
```

### Compression Ratio

| Format | Bits/Weight | Size (135M) |
|--------|-------------|-------------|
| FP16 | 16 | 256.6 MB |
| INT4 | 4 | ~80 MB |
| BitNet | 2 | 53.2 MB |

---

## Platform Requirements

| Platform | Mojo Native | Docker | Notes |
|----------|-------------|--------|-------|
| Linux x86_64 | Yes | Yes | Full support |
| macOS ARM64 | Yes | Yes | Apple Silicon |
| macOS x86_64 | No | Yes | Intel Mac - Docker only |
| ARM64 (Pi) | Coming | Yes | Raspberry Pi |
