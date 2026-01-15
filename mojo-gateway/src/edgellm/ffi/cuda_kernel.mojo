"""
CUDA Kernel FFI Wrapper

Mojo wrapper for the high-performance CUDA kernel.
Provides GPU acceleration for BitNet inference on NVIDIA devices.

Target: Jetson Nano/Orin, RTX GPUs (80-400 tok/s)
"""

from sys.ffi import DLHandle, c_char
from memory import UnsafePointer


# Kernel library handle (lazy loaded)
var _cuda_handle: DLHandle = DLHandle()
var _cuda_loaded: Bool = False
var _cuda_available: Bool = False
var _cuda_checked: Bool = False


fn _load_cuda_kernel() raises -> DLHandle:
    """Load the CUDA kernel shared library."""
    var paths = List[String](
        "./lib/libtmac_kernel_cuda.dylib",
        "./lib/libtmac_kernel_cuda.so",
        "/usr/local/lib/libtmac_kernel_cuda.dylib",
        "/usr/local/lib/libtmac_kernel_cuda.so",
        "libtmac_kernel_cuda.dylib",
        "libtmac_kernel_cuda.so",
    )

    for i in range(len(paths)):
        try:
            return DLHandle(paths[i])
        except:
            continue

    raise Error("CUDA kernel not found. Build with 'make' in src/kernels/cuda/")


fn get_cuda_kernel() raises -> DLHandle:
    """Get or load the CUDA kernel library."""
    if not _cuda_loaded:
        _cuda_handle = _load_cuda_kernel()
        _cuda_loaded = True
    return _cuda_handle


fn cuda_available() -> Bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA device is available and kernel is loaded.
    """
    if _cuda_checked:
        return _cuda_available

    try:
        var kernel = get_cuda_kernel()
        _cuda_available = kernel.call["cuda_available", Int]() == 1
    except:
        _cuda_available = False

    _cuda_checked = True
    return _cuda_available


fn cuda_init() raises -> Bool:
    """
    Initialize CUDA context with default buffer sizes.

    Uses sensible defaults for models up to 7B parameters.
    Buffers are dynamically resized as needed.

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["cuda_init", Int]()
    return result == 0


fn cuda_init_sized(
    max_weights_bytes: Int,
    max_activations: Int,
    max_output: Int,
) raises -> Bool:
    """
    Initialize CUDA context with explicit buffer sizes.

    Args:
        max_weights_bytes: Initial size of weight buffer in bytes
        max_activations: Initial number of activation elements
        max_output: Initial number of output elements

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["cuda_init_sized", Int](
        max_weights_bytes, max_activations, max_output
    )
    return result == 0


fn cuda_cleanup() raises:
    """Cleanup CUDA resources."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_cleanup", NoneType]()


fn cuda_device_name() raises -> String:
    """
    Get CUDA device name.

    Returns:
        Device name string.
    """
    if not cuda_available():
        return "No CUDA device"

    var kernel = get_cuda_kernel()
    var name_ptr = kernel.call["cuda_device_name", UnsafePointer[c_char]]()
    # Convert C string to Mojo String
    return String(name_ptr)


fn cuda_sync() raises:
    """Synchronize CUDA device."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_sync", NoneType]()


fn tmac_matmul_cuda(
    output: UnsafePointer[Float32],
    weights: UnsafePointer[Int8],
    activations: UnsafePointer[Float32],
    scales: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
) raises -> Bool:
    """
    T-MAC Matrix Multiplication using CUDA.

    GPU-accelerated BitNet inference using table lookups.

    Args:
        output: Output buffer [M * N]
        weights: Packed ternary weights [M * (K/4)]
        activations: Input activations [K * N]
        scales: Per-row scaling factors [M]
        M: Number of output rows
        N: Number of output columns (batch size)
        K: Inner dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["tmac_matmul_cuda", Int](
        weights, activations, output, scales, M, N, K
    )
    return result == 0


fn rmsnorm_cuda(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    batch_size: Int,
    size: Int,
    eps: Float32 = 1e-6,
) raises -> Bool:
    """
    CUDA-accelerated RMSNorm.

    output[i] = (input[i] / rms) * weight[i]
    where rms = sqrt(mean(input^2) + eps)

    Args:
        output: Output buffer [batch_size * size]
        input: Input buffer [batch_size * size]
        weight: Weight buffer [size]
        batch_size: Number of batches
        size: Vector size per batch
        eps: Epsilon for numerical stability

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["rmsnorm_cuda", Int](
        output, input, weight, batch_size, size, eps
    )
    return result == 0


fn softmax_cuda(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    batch_size: Int,
    size: Int,
) raises -> Bool:
    """
    CUDA-accelerated Softmax.

    Numerically stable: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        output: Output buffer [batch_size * size]
        input: Input buffer [batch_size * size]
        batch_size: Number of batches
        size: Vector size per batch

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["softmax_cuda", Int](
        output, input, batch_size, size
    )
    return result == 0


# ============================================================================
# Device Info Utilities
# ============================================================================

struct CUDADeviceInfo:
    """CUDA device information."""
    var name: String
    var total_memory_gb: Float64
    var sm_count: Int
    var compute_major: Int
    var compute_minor: Int
    var available: Bool

    fn __init__(out self):
        self.name = "Unknown"
        self.total_memory_gb = 0.0
        self.sm_count = 0
        self.compute_major = 0
        self.compute_minor = 0
        self.available = False


fn get_cuda_device_info() raises -> CUDADeviceInfo:
    """
    Get comprehensive CUDA device information.

    Returns:
        CUDADeviceInfo struct with device details.
    """
    var info = CUDADeviceInfo()

    if not cuda_available():
        return info

    info.available = True
    info.name = cuda_device_name()

    var kernel = get_cuda_kernel()

    # Get device properties
    var total_mem = UnsafePointer[Int].alloc(1)
    var sm_count = UnsafePointer[Int].alloc(1)
    var major = UnsafePointer[Int].alloc(1)
    var minor = UnsafePointer[Int].alloc(1)

    var result = kernel.call["cuda_device_info", Int](
        total_mem, sm_count, major, minor
    )

    if result == 0:
        info.total_memory_gb = Float64(total_mem[0]) / (1024.0 * 1024.0 * 1024.0)
        info.sm_count = sm_count[0]
        info.compute_major = major[0]
        info.compute_minor = minor[0]

    total_mem.free()
    sm_count.free()
    major.free()
    minor.free()

    return info


fn print_cuda_info() raises:
    """Print CUDA device information."""
    var info = get_cuda_device_info()

    if not info.available:
        print("CUDA: Not available")
        return

    print("CUDA Device Information:")
    print("  Name:", info.name)
    print("  Memory:", info.total_memory_gb, "GB")
    print("  SM Count:", info.sm_count)
    print("  Compute:", info.compute_major, ".", info.compute_minor)


# ============================================================================
# Phase 1: Persistent GPU Memory API
# ============================================================================

fn cuda_load_weights(
    weights: UnsafePointer[Int8],
    scales: UnsafePointer[Float32],
    weight_bytes: Int,
    num_rows: Int,
) raises -> Bool:
    """
    Load model weights to GPU memory (one-time operation).

    Weights remain on GPU until cuda_unload_weights() or cuda_cleanup().
    Subsequent calls to tmac_matmul_cuda_persistent() skip weight transfer.

    Args:
        weights: Packed ternary weights (host memory)
        scales: Per-row scaling factors (host memory)
        weight_bytes: Size of weights in bytes
        num_rows: Number of rows (for scales)

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["cuda_load_weights", Int](
        weights, scales, weight_bytes, num_rows
    )
    return result == 0


fn cuda_unload_weights() raises:
    """Unload weights from GPU memory."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_unload_weights", NoneType]()


fn cuda_weights_loaded() raises -> Bool:
    """Check if weights are loaded on GPU."""
    if not _cuda_loaded:
        return False
    var kernel = get_cuda_kernel()
    return kernel.call["cuda_weights_loaded", Int]() == 1


fn tmac_matmul_cuda_persistent(
    output: UnsafePointer[Float32],
    activations: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
) raises -> Bool:
    """
    T-MAC MatMul with persistent weights (weights already on GPU).

    Args:
        output: Output buffer [M * N]
        activations: Input activations [K * N]
        M: Number of output rows
        N: Number of output columns (batch size)
        K: Inner dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["tmac_matmul_cuda_persistent", Int](
        activations, output, M, N, K
    )
    return result == 0


fn cuda_load_norm_weights(
    norm_weights: UnsafePointer[Float32],
    size: Int,
) raises -> Bool:
    """
    Load normalization weights to GPU (one-time operation).

    Args:
        norm_weights: Normalization weights (host memory)
        size: Weight vector size

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["cuda_load_norm_weights", Int](
        norm_weights, size
    )
    return result == 0


fn rmsnorm_cuda_persistent(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    batch_size: Int,
    size: Int,
    eps: Float32 = 1e-6,
) raises -> Bool:
    """
    RMSNorm using pre-loaded weights (CUDA).

    Args:
        output: Output buffer (host memory)
        input: Input buffer (host memory)
        batch_size: Number of batches
        size: Vector size per batch
        eps: Epsilon for numerical stability

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["rmsnorm_cuda_persistent", Int](
        output, input, batch_size, size, eps
    )
    return result == 0


# ============================================================================
# Phase 2.1: Optimized Kernels (No Atomics, True Fusion)
# ============================================================================

fn cuda_init_streams() raises -> Bool:
    """Initialize CUDA streams for async operations."""
    if not cuda_available():
        return False
    var kernel = get_cuda_kernel()
    return kernel.call["cuda_init_streams", Int]() == 0


fn cuda_cleanup_streams() raises:
    """Cleanup CUDA streams."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_cleanup_streams", NoneType]()


fn cuda_alloc_pinned(max_activations: Int, max_output: Int) raises -> Bool:
    """Allocate pinned (page-locked) host memory for faster transfers."""
    if not cuda_available():
        return False
    var kernel = get_cuda_kernel()
    return kernel.call["cuda_alloc_pinned", Int](max_activations, max_output) == 0


fn cuda_free_pinned() raises:
    """Free pinned host memory."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_free_pinned", NoneType]()


fn tmac_matmul_cuda_v3(
    output: UnsafePointer[Float32],
    activations: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
) raises -> Bool:
    """
    Optimized T-MAC MatMul with warp-private accumulation (Phase 2.1).

    No atomicAdd, warp-level shuffle reduction.

    Args:
        output: Output buffer [M * N]
        activations: Input activations [K * N]
        M: Number of output rows
        N: Number of output columns (batch size)
        K: Inner dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["tmac_matmul_cuda_v3", Int](
        activations, output, M, N, K
    )
    return result == 0


fn streaming_fused_rmsnorm_matmul_cuda(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    M: Int,
    K: Int,
    eps: Float32 = 1e-6,
) raises -> Bool:
    """
    Streaming Fused RMSNorm + T-MAC MatMul (Phase 2.1).

    True fusion: normalizes on-the-fly without intermediate storage.
    Best for batch_size=1 (single token generation).

    Args:
        output: Output buffer [M]
        input: Input activations [K]
        M: Output dimension
        K: Hidden size / input dimension
        eps: Epsilon for RMSNorm numerical stability

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["streaming_fused_rmsnorm_matmul_cuda", Int](
        input, output, M, K, eps
    )
    return result == 0


fn tmac_matmul_cuda_adaptive(
    output: UnsafePointer[Float32],
    activations: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
) raises -> Bool:
    """
    Adaptive T-MAC MatMul dispatch (Phase 2.1).

    Automatically chooses optimal kernel based on tensor size.

    Args:
        output: Output buffer [M * N]
        activations: Input activations [K * N]
        M: Number of output rows
        N: Number of output columns (batch size)
        K: Inner dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["tmac_matmul_cuda_adaptive", Int](
        activations, output, M, N, K
    )
    return result == 0


fn fused_rmsnorm_matmul_cuda_adaptive(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
    eps: Float32 = 1e-6,
) raises -> Bool:
    """
    Adaptive Fused RMSNorm + MatMul dispatch (Phase 2.1).

    Automatically chooses optimal kernel based on tensor size and batch.

    Args:
        output: Output buffer [M * N]
        input: Input activations [K * N]
        M: Number of output rows
        N: Number of output columns (batch size)
        K: Hidden size / input dimension
        eps: Epsilon for RMSNorm numerical stability

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["fused_rmsnorm_matmul_cuda_adaptive", Int](
        input, output, M, N, K, eps
    )
    return result == 0


# ============================================================================
# Phase 3: INT8 Tensor Core API
# ============================================================================

fn cuda_has_int8_tensorcore() raises -> Bool:
    """Check if INT8 Tensor Cores are available (requires sm_75+)."""
    if not cuda_available():
        return False
    var kernel = get_cuda_kernel()
    return kernel.call["cuda_has_int8_tensorcore", Int]() == 1


fn cuda_get_compute_capability() raises -> Int:
    """Get compute capability (e.g., 75 for sm_75)."""
    if not cuda_available():
        return 0
    var kernel = get_cuda_kernel()
    return kernel.call["cuda_get_compute_capability", Int]()


fn cuda_load_weights_int8_tc(
    packed_weights: UnsafePointer[Int8],
    scales: UnsafePointer[Float32],
    weight_bytes: Int,
    num_rows: Int,
    K: Int,
) raises -> Bool:
    """
    Load weights in INT8 Tensor Core format.

    Expands 2-bit packed ternary weights to full INT8 format.
    Memory usage: 4x the packed weight size.

    Args:
        packed_weights: Packed ternary weights [M * K/4]
        scales: Per-row scaling factors [M]
        weight_bytes: Size of packed weights in bytes
        num_rows: M dimension
        K: K dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["cuda_load_weights_int8_tc", Int](
        packed_weights, scales, weight_bytes, num_rows, K
    )
    return result == 0


fn cuda_unload_weights_int8_tc() raises:
    """Unload INT8 Tensor Core weights from GPU memory."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_unload_weights_int8_tc", NoneType]()


fn cuda_weights_int8_tc_loaded() raises -> Bool:
    """Check if INT8 TC weights are loaded."""
    if not _cuda_loaded:
        return False
    var kernel = get_cuda_kernel()
    return kernel.call["cuda_weights_int8_tc_loaded", Int]() == 1


fn tmac_matmul_cuda_int8_tc(
    output: UnsafePointer[Float32],
    activations: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
) raises -> Bool:
    """
    INT8 Tensor Core Matrix Multiplication.

    Requires: cuda_load_weights_int8_tc() called first.

    Args:
        output: FP32 output buffer [M * N]
        activations: FP32 input activations [K * N]
        M: Number of output rows
        N: Batch size
        K: Inner dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["tmac_matmul_cuda_int8_tc", Int](
        activations, output, M, N, K
    )
    return result == 0


fn fused_rmsnorm_matmul_cuda_adaptive_v2(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
    eps: Float32 = 1e-6,
) raises -> Bool:
    """
    Adaptive dispatch v2 with INT8 Tensor Core support.

    Automatically selects optimal kernel based on hardware and tensor size.

    Args:
        output: Output buffer [M * N]
        input: Input activations [K * N]
        M: Number of output rows
        N: Batch size
        K: Hidden size / input dimension
        eps: Epsilon for RMSNorm numerical stability

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["fused_rmsnorm_matmul_cuda_adaptive_v2", Int](
        input, output, M, N, K, eps
    )
    return result == 0


# ============================================================================
# Phase 4: Flash Attention API
# ============================================================================

# Flash Attention state (lazy loaded)
var _flash_attention_loaded: Bool = False
var _flash_attention_kv_ready: Bool = False


fn _load_flash_attention_lib() raises -> DLHandle:
    """Load the Flash Attention shared library."""
    var paths = List[String](
        "./lib/libflash_attention.dylib",
        "./lib/libflash_attention.so",
        "./lib/libtmac_kernel_cuda.dylib",  # FA is included in main lib
        "./lib/libtmac_kernel_cuda.so",
        "/usr/local/lib/libflash_attention.dylib",
        "/usr/local/lib/libflash_attention.so",
    )

    for i in range(len(paths)):
        try:
            return DLHandle(paths[i])
        except:
            continue

    raise Error("Flash Attention library not found. Build with 'make flash' in src/kernels/cuda/")


fn get_flash_attention_lib() raises -> DLHandle:
    """Get the Flash Attention library (uses same handle as CUDA kernel)."""
    return get_cuda_kernel()  # FA is linked into main CUDA lib


fn flash_attention_init(
    max_batch: Int,
    num_heads: Int,
    max_sequence_len: Int,
    head_dim: Int,
) raises -> Bool:
    """
    Initialize Flash Attention buffers.

    Allocates device memory for Q, K, V, O tensors.

    Args:
        max_batch: Maximum batch size
        num_heads: Number of attention heads
        max_sequence_len: Maximum sequence length
        head_dim: Dimension per head (typically 64 or 128)

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_flash_attention_lib()
    var result = kernel.call["flash_attention_init", Int](
        max_batch, num_heads, max_sequence_len, head_dim
    )
    if result == 0:
        _flash_attention_loaded = True
    return result == 0


fn flash_attention_init_kv_cache(
    max_batch: Int,
    num_heads: Int,
    max_cache_len: Int,
    head_dim: Int,
) raises -> Bool:
    """
    Initialize KV cache for inference/decoding.

    Allocates separate buffers for cached keys and values.
    Required for efficient autoregressive generation.

    Args:
        max_batch: Maximum batch size
        num_heads: Number of attention heads
        max_cache_len: Maximum cache length (max context window)
        head_dim: Dimension per head

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_flash_attention_lib()
    var result = kernel.call["flash_attention_init_kv_cache", Int](
        max_batch, num_heads, max_cache_len, head_dim
    )
    if result == 0:
        _flash_attention_kv_ready = True
    return result == 0


fn flash_attention_cleanup() raises:
    """Cleanup Flash Attention resources."""
    if _flash_attention_loaded:
        var kernel = get_flash_attention_lib()
        kernel.call["flash_attention_cleanup", NoneType]()
        _flash_attention_loaded = False
        _flash_attention_kv_ready = False


fn flash_attention_forward(
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    O: UnsafePointer[Float32],
    batch_heads: Int,
    seq_len: Int,
    head_dim: Int,
    causal: Bool = True,
) raises -> Bool:
    """
    Flash Attention Forward Pass.

    Computes: O = softmax(Q @ K^T / sqrt(d_k)) @ V

    Uses tiled computation with online softmax for O(N) memory.

    Args:
        Q: Query tensor [batch_heads, seq_len, head_dim]
        K: Key tensor [batch_heads, seq_len, head_dim]
        V: Value tensor [batch_heads, seq_len, head_dim]
        O: Output tensor [batch_heads, seq_len, head_dim]
        batch_heads: batch * num_heads
        seq_len: Sequence length
        head_dim: Head dimension
        causal: True for causal (decoder) attention

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_flash_attention_lib()
    var causal_int = 1 if causal else 0
    var result = kernel.call["flash_attention_forward", Int](
        Q, K, V, O, batch_heads, seq_len, head_dim, causal_int
    )
    return result == 0


fn flash_attention_decode(
    Q: UnsafePointer[Float32],
    K_new: UnsafePointer[Float32],
    V_new: UnsafePointer[Float32],
    O: UnsafePointer[Float32],
    batch_heads: Int,
    cache_pos: Int,
    head_dim: Int,
) raises -> Bool:
    """
    Flash Attention Decode - Single token with KV cache.

    Optimized for autoregressive generation where:
    - Q has shape [batch_heads, 1, head_dim] (single new token)
    - K_new, V_new are appended to cache
    - Attends over all cached tokens

    Args:
        Q: Query for new token [batch_heads, head_dim]
        K_new: New key to append to cache [batch_heads, head_dim]
        V_new: New value to append to cache [batch_heads, head_dim]
        O: Output [batch_heads, head_dim]
        batch_heads: batch * num_heads
        cache_pos: Current position in cache (0-indexed)
        head_dim: Head dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_flash_attention_lib()
    var result = kernel.call["flash_attention_decode", Int](
        Q, K_new, V_new, O, batch_heads, cache_pos, head_dim
    )
    return result == 0


fn flash_attention_update_kv_cache(
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    batch_heads: Int,
    start_pos: Int,
    num_tokens: Int,
    head_dim: Int,
) raises -> Bool:
    """
    Update KV cache directly.

    Used during prefill phase to populate cache with initial context.

    Args:
        K: Keys to cache [batch_heads, num_tokens, head_dim]
        V: Values to cache [batch_heads, num_tokens, head_dim]
        batch_heads: batch * num_heads
        start_pos: Starting position in cache
        num_tokens: Number of tokens to cache
        head_dim: Head dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_flash_attention_lib()
    var result = kernel.call["flash_attention_update_kv_cache", Int](
        K, V, batch_heads, start_pos, num_tokens, head_dim
    )
    return result == 0


struct FlashAttentionInfo:
    """Flash Attention status information."""
    var initialized: Bool
    var kv_cache_ready: Bool
    var max_seq_len: Int
    var head_dim: Int

    fn __init__(out self):
        self.initialized = False
        self.kv_cache_ready = False
        self.max_seq_len = 0
        self.head_dim = 0


fn flash_attention_info() raises -> FlashAttentionInfo:
    """
    Get Flash Attention status info.

    Returns:
        FlashAttentionInfo struct with current state.
    """
    var info = FlashAttentionInfo()

    if not cuda_available():
        return info

    var kernel = get_flash_attention_lib()

    var initialized = UnsafePointer[Int].alloc(1)
    var cache_ready = UnsafePointer[Int].alloc(1)
    var max_seq = UnsafePointer[Int].alloc(1)
    var head_d = UnsafePointer[Int].alloc(1)

    kernel.call["flash_attention_info", NoneType](
        initialized, cache_ready, max_seq, head_d
    )

    info.initialized = initialized[0] == 1
    info.kv_cache_ready = cache_ready[0] == 1
    info.max_seq_len = max_seq[0]
    info.head_dim = head_d[0]

    initialized.free()
    cache_ready.free()
    max_seq.free()
    head_d.free()

    return info


fn flash_attention_available() -> Bool:
    """Check if Flash Attention is available and initialized."""
    return _flash_attention_loaded


fn flash_attention_kv_cache_ready() -> Bool:
    """Check if KV cache is initialized."""
    return _flash_attention_kv_ready
