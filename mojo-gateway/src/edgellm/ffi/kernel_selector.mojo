"""
Kernel Selector - Unified Backend Selection

Automatically selects the best available compute backend:
1. CUDA (GPU) - Fastest on NVIDIA hardware
2. AVX2/NEON (CPU SIMD) - Fast on modern CPUs
3. Pure Mojo - Fallback for any platform

Usage:
    from edgellm.ffi.kernel_selector import KernelBackend, select_backend, matmul, rmsnorm, softmax
"""

from memory import UnsafePointer

# Import backend modules
from .cuda_kernel import (
    cuda_available,
    cuda_init,
    cuda_cleanup,
    tmac_matmul_cuda,
    rmsnorm_cuda,
    softmax_cuda,
    get_cuda_device_info,
)
from .tmac_kernel import (
    get_kernel,
    tmac_matmul,
    rmsnorm,
    softmax,
    build_lut,
    has_avx2,
    has_neon,
    rmsnorm_mojo,
    softmax_mojo,
)


# ============================================================================
# Backend Enumeration
# ============================================================================

@value
struct KernelBackend:
    """Available compute backends."""
    var value: Int

    alias CUDA = KernelBackend(0)
    alias AVX2 = KernelBackend(1)
    alias NEON = KernelBackend(2)
    alias MOJO = KernelBackend(3)
    alias NONE = KernelBackend(-1)

    fn __eq__(self, other: KernelBackend) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: KernelBackend) -> Bool:
        return self.value != other.value

    fn name(self) -> String:
        if self == KernelBackend.CUDA:
            return "CUDA"
        elif self == KernelBackend.AVX2:
            return "AVX2"
        elif self == KernelBackend.NEON:
            return "NEON"
        elif self == KernelBackend.MOJO:
            return "Pure Mojo"
        else:
            return "None"


# ============================================================================
# Backend Selection
# ============================================================================

var _selected_backend: KernelBackend = KernelBackend.NONE
var _backend_initialized: Bool = False


fn select_backend(
    prefer_gpu: Bool = True,
    max_weights_bytes: Int = 100_000_000,  # 100MB default
    max_activations: Int = 10_000_000,     # 10M elements
    max_output: Int = 10_000_000,          # 10M elements
) raises -> KernelBackend:
    """
    Select the best available compute backend.

    Priority order (when prefer_gpu=True):
    1. CUDA (if available and initialization succeeds)
    2. AVX2 (x86 with AVX2 support)
    3. NEON (ARM with NEON support)
    4. Pure Mojo (fallback)

    Args:
        prefer_gpu: If True, prefer GPU over CPU backends
        max_weights_bytes: Max weight buffer size for CUDA init
        max_activations: Max activation count for CUDA init
        max_output: Max output count for CUDA init

    Returns:
        Selected KernelBackend
    """
    if _backend_initialized:
        return _selected_backend

    # Try CUDA first if preferred
    if prefer_gpu and cuda_available():
        # Use no-arg init - buffers dynamically resize as needed
        if cuda_init():
            _selected_backend = KernelBackend.CUDA
            _backend_initialized = True
            return _selected_backend

    # Try CPU SIMD backends
    try:
        if has_avx2():
            _selected_backend = KernelBackend.AVX2
            _backend_initialized = True
            return _selected_backend
    except:
        pass

    try:
        if has_neon():
            _selected_backend = KernelBackend.NEON
            _backend_initialized = True
            return _selected_backend
    except:
        pass

    # Fallback to pure Mojo
    _selected_backend = KernelBackend.MOJO
    _backend_initialized = True
    return _selected_backend


fn get_selected_backend() -> KernelBackend:
    """Get the currently selected backend (or NONE if not selected)."""
    return _selected_backend


fn force_backend(backend: KernelBackend) raises:
    """
    Force a specific backend (for testing or user preference).

    Args:
        backend: Backend to use
    """
    if backend == KernelBackend.CUDA and not cuda_available():
        raise Error("CUDA not available")

    if backend == KernelBackend.AVX2 and not has_avx2():
        raise Error("AVX2 not available")

    if backend == KernelBackend.NEON and not has_neon():
        raise Error("NEON not available")

    _selected_backend = backend
    _backend_initialized = True


fn cleanup_backend() raises:
    """Cleanup backend resources."""
    if _selected_backend == KernelBackend.CUDA:
        cuda_cleanup()
    _backend_initialized = False
    _selected_backend = KernelBackend.NONE


# ============================================================================
# Unified Operations
# ============================================================================

fn matmul(
    output: UnsafePointer[Float32],
    weights: UnsafePointer[UInt8],
    activations: UnsafePointer[Float32],
    lut: UnsafePointer[Float32],
    scales: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
    num_groups: Int,
) raises:
    """
    Unified T-MAC matrix multiplication.

    Automatically uses the best available backend.

    Args:
        output: Output buffer [M * N] or [M] for single batch
        weights: Packed ternary weights [M * (K/4)]
        activations: Input activations [K * N] or [K] for single batch
        lut: Lookup tables [num_groups * 256] (ignored for CUDA)
        scales: Per-row scaling factors [M]
        M: Number of output rows
        N: Number of output columns (batch size)
        K: Inner dimension
        num_groups: Number of activation groups (for CPU LUT)
    """
    if not _backend_initialized:
        _ = select_backend()

    if _selected_backend == KernelBackend.CUDA:
        # CUDA handles LUT internally
        var weights_i8 = weights.bitcast[Int8]()
        var success = tmac_matmul_cuda(output, weights_i8, activations, scales, M, N, K)
        if not success:
            # Fallback to CPU on CUDA failure
            tmac_matmul(output, weights, lut, scales, M, K, num_groups)
    else:
        # CPU backends use pre-built LUT
        tmac_matmul(output, weights, lut, scales, M, K, num_groups)


fn unified_rmsnorm(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    batch_size: Int,
    size: Int,
    eps: Float32 = 1e-6,
) raises:
    """
    Unified RMSNorm.

    Automatically uses the best available backend.

    Args:
        output: Output buffer [batch_size * size]
        input: Input buffer [batch_size * size]
        weight: Weight buffer [size]
        batch_size: Number of batches
        size: Vector size per batch
        eps: Epsilon for numerical stability
    """
    if not _backend_initialized:
        _ = select_backend()

    if _selected_backend == KernelBackend.CUDA:
        var success = rmsnorm_cuda(output, input, weight, batch_size, size, eps)
        if not success:
            # Fallback
            for b in range(batch_size):
                var offset = b * size
                rmsnorm(
                    output.offset(offset),
                    input.offset(offset),
                    weight,
                    size,
                    eps
                )
    elif _selected_backend == KernelBackend.MOJO:
        for b in range(batch_size):
            var offset = b * size
            rmsnorm_mojo(
                output.offset(offset),
                input.offset(offset),
                weight,
                size,
                eps
            )
    else:
        # AVX2 or NEON
        for b in range(batch_size):
            var offset = b * size
            rmsnorm(
                output.offset(offset),
                input.offset(offset),
                weight,
                size,
                eps
            )


fn unified_softmax(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    batch_size: Int,
    size: Int,
) raises:
    """
    Unified Softmax.

    Automatically uses the best available backend.

    Args:
        output: Output buffer [batch_size * size]
        input: Input buffer [batch_size * size]
        batch_size: Number of batches
        size: Vector size per batch
    """
    if not _backend_initialized:
        _ = select_backend()

    if _selected_backend == KernelBackend.CUDA:
        var success = softmax_cuda(output, input, batch_size, size)
        if not success:
            # Fallback
            for b in range(batch_size):
                var offset = b * size
                softmax(output.offset(offset), input.offset(offset), size)
    elif _selected_backend == KernelBackend.MOJO:
        for b in range(batch_size):
            var offset = b * size
            softmax_mojo(output.offset(offset), input.offset(offset), size)
    else:
        # AVX2 or NEON
        for b in range(batch_size):
            var offset = b * size
            softmax(output.offset(offset), input.offset(offset), size)


# ============================================================================
# Info and Diagnostics
# ============================================================================

fn print_backend_info() raises:
    """Print information about available and selected backends."""
    print("EdgeLLM Kernel Backend Info")
    print("=" * 40)

    # Check CUDA
    var cuda_info = get_cuda_device_info()
    if cuda_info.available:
        print("CUDA: Available")
        print("  Device:", cuda_info.name)
        print("  Memory:", cuda_info.total_memory_gb, "GB")
        print("  Compute:", cuda_info.compute_major, ".", cuda_info.compute_minor)
    else:
        print("CUDA: Not available")

    # Check CPU SIMD
    try:
        print("AVX2:", "Available" if has_avx2() else "Not available")
    except:
        print("AVX2: Unknown (C kernel not loaded)")

    try:
        print("NEON:", "Available" if has_neon() else "Not available")
    except:
        print("NEON: Unknown (C kernel not loaded)")

    print("Pure Mojo: Always available")

    # Selected backend
    if _backend_initialized:
        print("\nSelected Backend:", _selected_backend.name())
    else:
        print("\nSelected Backend: Not yet selected")
