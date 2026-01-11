"""
SIMD-optimized RMSNorm for EdgeLLM.

Root Mean Square Layer Normalization:
    output[i] = (input[i] / rms) * weight[i]
    where rms = sqrt(mean(input^2) + eps)

This is a critical operation - called 4 times per transformer layer.
"""

from memory import UnsafePointer
from algorithm import vectorize, parallelize


alias DEFAULT_EPS: Float32 = 1e-6


fn rmsnorm(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    size: Int,
    eps: Float32 = DEFAULT_EPS,
):
    """
    SIMD-vectorized RMSNorm.

    Uses Mojo's vectorize for optimal SIMD utilization.

    Args:
        output: Output buffer [size]
        input: Input buffer [size]
        weight: Weight buffer [size]
        size: Vector size
        eps: Epsilon for numerical stability
    """
    alias simd_width = simdwidthof[DType.float32]()

    # Phase 1: Compute sum of squares
    var ss_vec = SIMD[DType.float32, simd_width](0)

    @parameter
    fn accumulate_squares[width: Int](i: Int):
        var v = input.load[width=width](i)
        ss_vec += (v * v).cast[DType.float32]()

    vectorize[accumulate_squares, simd_width](size)

    var ss = ss_vec.reduce_add()

    # Phase 2: Compute RMS scale
    var rms = 1.0 / (ss / size + eps).sqrt()

    # Phase 3: Apply normalization and weight
    var rms_vec = SIMD[DType.float32, simd_width](rms)

    @parameter
    fn apply_norm[width: Int](i: Int):
        var inp = input.load[width=width](i)
        var w = weight.load[width=width](i)
        var out = inp * rms_vec.slice[width]() * w
        output.store[width=width](i, out)

    vectorize[apply_norm, simd_width](size)


fn rmsnorm_inplace(
    x: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    size: Int,
    eps: Float32 = DEFAULT_EPS,
):
    """
    In-place RMSNorm (modifies input buffer).

    More memory-efficient when input is not needed after normalization.
    """
    alias simd_width = simdwidthof[DType.float32]()

    # Compute sum of squares
    var ss: Float32 = 0.0
    for i in range(size):
        ss += x[i] * x[i]

    # Compute RMS scale
    var rms = 1.0 / (ss / size + eps).sqrt()

    # Apply in-place
    var rms_vec = SIMD[DType.float32, simd_width](rms)

    var i = 0
    while i + simd_width <= size:
        var inp = x.load[width=simd_width](i)
        var w = weight.load[width=simd_width](i)
        x.store[width=simd_width](i, inp * rms_vec * w)
        i += simd_width

    # Handle remainder
    while i < size:
        x[i] = x[i] * rms * weight[i]
        i += 1


fn rmsnorm_parallel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    size: Int,
    num_threads: Int = 4,
    eps: Float32 = DEFAULT_EPS,
):
    """
    Parallelized RMSNorm for large vectors.

    Useful when size > 4096.
    """
    alias simd_width = simdwidthof[DType.float32]()
    var chunk_size = size // num_threads

    # Phase 1: Parallel sum of squares
    var partial_sums = UnsafePointer[Float32].alloc(num_threads)

    @parameter
    fn compute_partial_sum(thread_id: Int):
        var start = thread_id * chunk_size
        var end = start + chunk_size if thread_id < num_threads - 1 else size
        var local_sum: Float32 = 0.0
        for i in range(start, end):
            local_sum += input[i] * input[i]
        partial_sums[thread_id] = local_sum

    parallelize[compute_partial_sum](num_threads)

    # Reduce partial sums
    var ss: Float32 = 0.0
    for i in range(num_threads):
        ss += partial_sums[i]

    partial_sums.free()

    # Compute RMS scale
    var rms = 1.0 / (ss / size + eps).sqrt()

    # Phase 2: Parallel apply normalization
    @parameter
    fn apply_norm_parallel(thread_id: Int):
        var start = thread_id * chunk_size
        var end = start + chunk_size if thread_id < num_threads - 1 else size
        for i in range(start, end):
            output[i] = input[i] * rms * weight[i]

    parallelize[apply_norm_parallel](num_threads)


# ============================================================================
# Benchmark utilities
# ============================================================================

fn benchmark_rmsnorm(size: Int, iterations: Int = 1000) -> Float64:
    """
    Benchmark RMSNorm performance.

    Returns: Operations per second
    """
    var input = UnsafePointer[Float32].alloc(size)
    var weight = UnsafePointer[Float32].alloc(size)
    var output = UnsafePointer[Float32].alloc(size)

    # Initialize
    for i in range(size):
        input[i] = Float32(i) / size
        weight[i] = 1.0

    # Warmup
    for _ in range(10):
        rmsnorm(output, input, weight, size)

    # Benchmark
    from time import perf_counter_ns

    var start = perf_counter_ns()
    for _ in range(iterations):
        rmsnorm(output, input, weight, size)
    var end = perf_counter_ns()

    input.free()
    weight.free()
    output.free()

    var elapsed_ns = end - start
    var ops_per_sec = Float64(iterations) / (Float64(elapsed_ns) / 1e9)
    return ops_per_sec
