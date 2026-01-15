"""
T-MAC Kernel FFI Wrapper

Mojo wrapper for the high-performance C kernel.
This provides the critical path optimization for 30-50 tok/s inference.
"""

from sys.ffi import DLHandle, c_char
from memory import UnsafePointer


# Kernel library handle (lazy loaded)
var _kernel_handle: DLHandle = DLHandle()
var _kernel_loaded: Bool = False


fn _load_kernel() raises -> DLHandle:
    """Load the T-MAC kernel shared library."""
    # Try different paths
    var paths = List[String](
        "./lib/libtmac_kernel.dylib",
        "./lib/libtmac_kernel.so",
        "/usr/local/lib/libtmac_kernel.dylib",
        "/usr/local/lib/libtmac_kernel.so",
        "libtmac_kernel.dylib",
        "libtmac_kernel.so",
    )

    for i in range(len(paths)):
        try:
            return DLHandle(paths[i])
        except:
            continue

    raise Error("Failed to load T-MAC kernel library. Run 'make' in src/kernels/")


fn get_kernel() raises -> DLHandle:
    """Get or load the kernel library."""
    if not _kernel_loaded:
        _kernel_handle = _load_kernel()
        _kernel_loaded = True
    return _kernel_handle


fn tmac_matmul(
    output: UnsafePointer[Float32],
    weights: UnsafePointer[UInt8],
    lut: UnsafePointer[Float32],
    scales: UnsafePointer[Float32],
    rows: Int,
    cols: Int,
    num_groups: Int,
) raises:
    """
    T-MAC Matrix Multiplication using C FFI.

    This is the critical optimization - uses pshufb/tbl for register-based LUT.

    Args:
        output: Output buffer [rows]
        weights: Packed ternary weights [rows * cols/4]
        lut: Lookup tables [num_groups * 256]
        scales: Per-row scaling factors [rows]
        rows: Number of output rows
        cols: Number of columns (must be divisible by 4)
        num_groups: Number of activation groups
    """
    var kernel = get_kernel()

    # Call the appropriate kernel based on platform
    # The C code handles platform detection internally
    kernel.call["tmac_matmul_avx2", NoneType](
        output, weights, lut, scales, rows, cols, num_groups
    )


fn rmsnorm(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    size: Int,
    eps: Float32 = 1e-6,
) raises:
    """
    SIMD-accelerated RMSNorm.

    output[i] = (input[i] / rms) * weight[i]
    where rms = sqrt(mean(input^2) + eps)

    Args:
        output: Output buffer [size]
        input: Input buffer [size]
        weight: Weight buffer [size]
        size: Vector size
        eps: Epsilon for numerical stability
    """
    var kernel = get_kernel()
    kernel.call["rmsnorm_avx2", NoneType](output, input, weight, size, eps)


fn softmax(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
) raises:
    """
    SIMD-accelerated Softmax.

    Numerically stable: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        output: Output buffer [size]
        input: Input buffer [size]
        size: Vector size
    """
    var kernel = get_kernel()
    kernel.call["softmax_avx2", NoneType](output, input, size)


fn build_lut(
    lut: UnsafePointer[Float32],
    activations: UnsafePointer[Float32],
    size: Int,
    group_size: Int = 4,
) raises:
    """
    Build lookup tables for T-MAC.

    Pre-computes all possible dot products for groups of activations.

    Args:
        lut: Output LUT buffer [num_groups * 256]
        activations: Input activations [size]
        size: Activation size
        group_size: Number of activations per group (typically 4)
    """
    var kernel = get_kernel()
    kernel.call["build_lut", NoneType](lut, activations, size, group_size)


fn get_cpu_features() raises -> Int:
    """
    Check CPU features.

    Returns:
        Bitmask of supported features:
        - Bit 0: AVX2
        - Bit 1: AVX512
        - Bit 2: NEON
    """
    var kernel = get_kernel()
    return kernel.call["get_cpu_features", Int]()


fn has_avx2() raises -> Bool:
    """Check if AVX2 is supported."""
    return (get_cpu_features() & 1) != 0


fn has_avx512() raises -> Bool:
    """Check if AVX512 is supported."""
    return (get_cpu_features() & 2) != 0


fn has_neon() raises -> Bool:
    """Check if NEON is supported."""
    return (get_cpu_features() & 4) != 0


# ============================================================================
# Pure Mojo Fallbacks (for when C kernel is not available)
# ============================================================================

fn rmsnorm_mojo(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    size: Int,
    eps: Float32 = 1e-6,
):
    """Pure Mojo RMSNorm fallback."""
    # Compute sum of squares
    var ss: Float32 = 0.0

    # SIMD vectorized
    alias simd_width = simdwidthof[DType.float32]()
    var ss_vec = SIMD[DType.float32, simd_width](0)

    var i = 0
    while i + simd_width <= size:
        var v = input.load[width=simd_width](i)
        ss_vec += v * v
        i += simd_width

    ss = ss_vec.reduce_add()

    # Handle remainder
    while i < size:
        ss += input[i] * input[i]
        i += 1

    # Compute RMS scale
    var rms = 1.0 / (ss / size + eps).sqrt()

    # Apply normalization and weight
    i = 0
    var rms_vec = SIMD[DType.float32, simd_width](rms)
    while i + simd_width <= size:
        var inp = input.load[width=simd_width](i)
        var w = weight.load[width=simd_width](i)
        var out = inp * rms_vec * w
        output.store[width=simd_width](i, out)
        i += simd_width

    while i < size:
        output[i] = input[i] * rms * weight[i]
        i += 1


fn softmax_mojo(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
):
    """Pure Mojo softmax fallback."""
    # Find max
    var max_val: Float32 = input[0]
    for i in range(1, size):
        if input[i] > max_val:
            max_val = input[i]

    # Compute exp and sum
    var sum_val: Float32 = 0.0
    for i in range(size):
        output[i] = (input[i] - max_val).exp()
        sum_val += output[i]

    # Normalize
    var inv_sum = 1.0 / sum_val
    for i in range(size):
        output[i] *= inv_sum
