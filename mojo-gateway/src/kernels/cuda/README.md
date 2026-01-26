# EdgeLLM CUDA Kernels

High-performance GPU kernels for BitNet 1.58-bit inference on NVIDIA devices.

## Supported Hardware

| Device | Compute Capability | Build Target |
|--------|-------------------|--------------|
| Jetson Nano | 5.3 | `make jetson-nano` |
| Jetson Xavier | 7.2 | Default |
| Jetson Orin | 8.7 | `make jetson-orin` |
| RTX 20 series | 7.5 | Default |
| RTX 30 series | 8.6 | `make rtx` |
| RTX 40 series | 8.9 | `make rtx` |

## Prerequisites

- NVIDIA CUDA Toolkit 11.0+ (12.x recommended)
- Compatible NVIDIA GPU
- nvcc compiler

```bash
# Check CUDA installation
nvcc --version

# Verify GPU is detected
nvidia-smi
```

## Build

```bash
# Build for detected architecture
make

# Build for specific hardware
make jetson-nano   # Jetson Nano
make jetson-orin   # Jetson Orin
make rtx           # RTX 30/40 series

# Debug build (with symbols)
make debug

# Profiling build (for Nsight)
make profile
```

## Output

After building:
- `lib/libtmac_kernel_cuda.so` (Linux)
- `lib/libtmac_kernel_cuda.dylib` (macOS)

## Test

```bash
make test
```

## Usage from Mojo

```mojo
from edgellm.ffi import select_backend, cuda_available, matmul

# Check CUDA availability
if cuda_available():
    print("CUDA is available!")

# Select best backend (prefers CUDA if available)
var backend = select_backend()

# Use unified matmul (auto-selects CUDA)
matmul(output, weights, activations, lut, scales, M, N, K, num_groups)
```

## Architecture

```
src/kernels/cuda/
    tmac_kernel.cu        # CUDA kernel implementations
    tmac_kernel_cuda.h    # C header for FFI
    test_cuda_kernel.cu   # Unit tests
    Makefile              # Build configuration

src/edgellm/ffi/
    cuda_kernel.mojo      # Mojo FFI wrapper
    kernel_selector.mojo  # Unified backend selection
    __init__.mojo         # Package exports
```

## Performance

Target performance on different hardware:

| Device | Expected Throughput | vs CPU |
|--------|--------------------:|-------:|
| Jetson Nano | 80-120 tok/s | 5-10x |
| Jetson Orin | 200-400 tok/s | 10-20x |
| RTX 3090 | 400-600 tok/s | 20-40x |
| RTX 4090 | 600-1000 tok/s | 30-50x |

## Kernel Details

### T-MAC Matmul

The T-MAC kernel uses table lookups instead of multiply-accumulate:
- Shared memory for lookup tables (fast access)
- Coalesced global memory reads
- Warp-level reductions (`__shfl_down_sync`)
- Minimal thread divergence

### RMSNorm

Block-parallel RMSNorm with:
- Shared memory reduction
- Fused normalization and weight application
- Single kernel launch

### Softmax

Numerically stable softmax:
- Two-pass algorithm (max-finding + normalization)
- Shared memory for intermediate results
- Single kernel launch per batch

## Troubleshooting

### CUDA not found
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Compute capability mismatch
Edit `Makefile` and update `CUDA_ARCH` to match your GPU.

### Out of memory
Reduce batch size or model dimensions in your application.
