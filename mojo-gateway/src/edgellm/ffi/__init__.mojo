"""
EdgeLLM FFI Package

Provides unified access to compute kernels:
- CUDA (GPU) for NVIDIA devices
- AVX2/NEON (CPU SIMD) for x86/ARM
- Pure Mojo fallback for any platform

Usage:
    from edgellm.ffi import select_backend, matmul, unified_rmsnorm, unified_softmax
"""

# Re-export kernel selector components
from .kernel_selector import (
    KernelBackend,
    select_backend,
    get_selected_backend,
    force_backend,
    cleanup_backend,
    matmul,
    unified_rmsnorm,
    unified_softmax,
    print_backend_info,
)

# Re-export CUDA-specific functions
from .cuda_kernel import (
    cuda_available,
    cuda_init,
    cuda_cleanup,
    cuda_device_name,
    cuda_sync,
    get_cuda_device_info,
    CUDADeviceInfo,
)

# Re-export CPU kernel functions
from .tmac_kernel import (
    tmac_matmul,
    rmsnorm,
    softmax,
    build_lut,
    has_avx2,
    has_neon,
    get_cpu_features,
)
