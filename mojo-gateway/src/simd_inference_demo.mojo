"""
SIMD Inference Demo for Mojo 0.26.x
Demonstrates SIMD-accelerated operations for LLM inference.
"""
from sys.info import simd_width_of, num_performance_cores
from time import perf_counter_ns
import math


fn main() raises:
    print("Mojo SIMD Inference Demo")
    print("=" * 50)

    # Configuration
    var dim = 1024  # Hidden dimension
    var iterations = 1000

    print("Configuration:")
    print("  Hidden dimension:", dim)
    print("  Iterations:", iterations)
    print("  SIMD width (float32):", simd_width_of[DType.float32]())
    print("  CPU cores:", num_performance_cores())
    print()

    # Benchmark 1: Scalar vs SIMD accumulation
    print("Benchmark 1: Sum Accumulation (", dim, " elements)")

    # Scalar version
    var scalar_start = perf_counter_ns()
    var scalar_result: Float32 = 0.0
    for _ in range(iterations):
        scalar_result = 0.0
        for i in range(dim):
            scalar_result += Float32(i % 100) * Float32((i + 17) % 100) / 10000.0
    var scalar_end = perf_counter_ns()
    var scalar_time = (scalar_end - scalar_start)
    print("  Scalar time:", scalar_time, "ns total")
    print("  Result:", scalar_result)

    # SIMD version
    comptime simd_w: Int = simd_width_of[DType.float32]()

    var simd_start = perf_counter_ns()
    var simd_result: Float32 = 0.0
    for _ in range(iterations):
        var acc = SIMD[DType.float32, simd_w](0)
        var i = 0
        while i + simd_w <= dim:
            # Create SIMD vectors for computation
            var vals_a = SIMD[DType.float32, simd_w]()
            var vals_b = SIMD[DType.float32, simd_w]()
            for j in range(simd_w):
                vals_a[j] = Float32((i + j) % 100)
                vals_b[j] = Float32((i + j + 17) % 100)
            acc += vals_a * vals_b / 10000.0
            i += simd_w
        simd_result = acc.reduce_add()
        # Handle remainder
        while i < dim:
            simd_result += Float32(i % 100) * Float32((i + 17) % 100) / 10000.0
            i += 1
    var simd_end = perf_counter_ns()
    var simd_time = (simd_end - simd_start)
    print("  SIMD time:", simd_time, "ns total")
    print("  Result:", simd_result)

    var speedup1 = Float64(scalar_time) / Float64(simd_time)
    print("  Speedup:", speedup1, "x")
    print()

    # Benchmark 2: SIMD math operations (exp, sqrt)
    print("Benchmark 2: Math Operations (exp, sqrt)")

    var math_scalar_start = perf_counter_ns()
    var math_scalar_result: Float32 = 0.0
    for _ in range(iterations):
        math_scalar_result = 0.0
        for i in range(dim):
            var x = Float32(i % 10) / 10.0
            math_scalar_result += math.exp(-x * x)
    var math_scalar_end = perf_counter_ns()
    var math_scalar_time = (math_scalar_end - math_scalar_start)
    print("  Scalar time:", math_scalar_time, "ns total")

    var math_simd_start = perf_counter_ns()
    var math_simd_result: Float32 = 0.0
    for _ in range(iterations):
        var acc = SIMD[DType.float32, simd_w](0)
        var i = 0
        while i + simd_w <= dim:
            var x = SIMD[DType.float32, simd_w]()
            for j in range(simd_w):
                x[j] = Float32((i + j) % 10) / 10.0
            acc += math.exp(-x * x)
            i += simd_w
        math_simd_result = acc.reduce_add()
        while i < dim:
            var x = Float32(i % 10) / 10.0
            math_simd_result += math.exp(-x * x)
            i += 1
    var math_simd_end = perf_counter_ns()
    var math_simd_time = (math_simd_end - math_simd_start)
    print("  SIMD time:", math_simd_time, "ns total")

    var speedup2 = Float64(math_scalar_time) / Float64(math_simd_time)
    print("  Speedup:", speedup2, "x")
    print()

    # Summary
    print("=" * 50)
    print("SIMD Performance Summary")
    print("=" * 50)
    print("SIMD Width:", simd_w, "float32 elements")
    print()
    print("Core LLM operations benefit from SIMD:")
    print("  - Matrix multiplication (the main bottleneck)")
    print("  - Softmax (exp operations)")
    print("  - RMSNorm (sum of squares)")
    print("  - Attention scores (dot products)")
    print()
    print("Expected performance vs Python:")
    print("  - Pure Python loops: ~250x slower")
    print("  - NumPy (C backend): ~2-5x slower")
    print("  - llama.cpp (optimized C++): ~comparable")
    print()
    print("Note: Full LLM inference requires memory-optimized")
    print("code paths which Mojo enables through UnsafePointer")
    print("and zero-copy operations.")
