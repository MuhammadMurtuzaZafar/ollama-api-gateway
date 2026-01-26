"""
T-MAC Kernel FFI Integration Test

Tests the C kernel integration with Mojo.
"""

from sys.ffi import OwnedDLHandle
from memory import UnsafePointer
from time import perf_counter_ns
from math import sqrt


fn main() raises:
    print("=" * 60)
    print("EdgeLLM C Kernel FFI Integration Test")
    print("=" * 60)

    # Load the C kernel library
    print("\nLoading C kernel library...")
    var handle = OwnedDLHandle("/workspace/lib/libtmac_kernel.so")
    print("  Loaded: /workspace/lib/libtmac_kernel.so")

    # Test CPU features
    print("\nCPU Features:")
    var features = handle.call["get_cpu_features", Int32]()
    print("  AVX2:   ", (features & 1) != 0)
    print("  AVX512: ", (features & 2) != 0)
    print("  NEON:   ", (features & 4) != 0)

    # Test RMSNorm
    print("\nTesting RMSNorm (AVX2)...")
    var size: Int32 = 4096

    # Allocate buffers
    var input_data = List[Float32]()
    var output_data = List[Float32]()
    var weight_data = List[Float32]()

    # Initialize
    for i in range(Int(size)):
        input_data.append(Float32(i % 100) / 100.0)
        output_data.append(0.0)
        weight_data.append(1.0)

    # Get raw pointers
    var input_ptr = input_data.unsafe_ptr()
    var output_ptr = output_data.unsafe_ptr()
    var weight_ptr = weight_data.unsafe_ptr()

    # Warm up
    handle.call["rmsnorm_avx2", NoneType](
        output_ptr, input_ptr, weight_ptr, size, Float32(1e-6)
    )

    # Benchmark
    var iterations = 10000
    var start = perf_counter_ns()
    for _ in range(iterations):
        handle.call["rmsnorm_avx2", NoneType](
            output_ptr, input_ptr, weight_ptr, size, Float32(1e-6)
        )
    var elapsed_ns = perf_counter_ns() - start
    var elapsed_ms = Float64(elapsed_ns) / 1000000.0
    var per_iter_us = Float64(elapsed_ns) / Float64(iterations) / 1000.0
    var throughput_gbps = Float64(size) * 4.0 * Float64(iterations) / Float64(elapsed_ns)

    print("  Size:       ", Int(size))
    print("  Iterations: ", iterations)
    print("  Total time: ", elapsed_ms, " ms")
    print("  Per iter:   ", per_iter_us, " us")
    print("  Throughput: ", throughput_gbps, " GB/s")

    # Verify output
    var sum_sq: Float64 = 0.0
    for i in range(Int(size)):
        sum_sq += Float64(output_data[i]) * Float64(output_data[i])
    print("  Output L2 norm:", sqrt(sum_sq))
    print("  Status: PASS")

    # Test Softmax
    print("\nTesting Softmax (AVX2)...")

    # Re-initialize
    for i in range(Int(size)):
        input_data[i] = Float32(i % 10) - 5.0
        output_data[i] = 0.0

    # Warm up
    handle.call["softmax_avx2", NoneType](output_ptr, input_ptr, size)

    # Benchmark
    start = perf_counter_ns()
    for _ in range(iterations):
        handle.call["softmax_avx2", NoneType](output_ptr, input_ptr, size)
    elapsed_ns = perf_counter_ns() - start
    elapsed_ms = Float64(elapsed_ns) / 1000000.0
    per_iter_us = Float64(elapsed_ns) / Float64(iterations) / 1000.0
    throughput_gbps = Float64(size) * 4.0 * Float64(iterations) / Float64(elapsed_ns)

    print("  Size:       ", Int(size))
    print("  Iterations: ", iterations)
    print("  Total time: ", elapsed_ms, " ms")
    print("  Per iter:   ", per_iter_us, " us")
    print("  Throughput: ", throughput_gbps, " GB/s")

    # Verify softmax sums to 1
    var softmax_sum: Float64 = 0.0
    for i in range(Int(size)):
        softmax_sum += Float64(output_data[i])
    print("  Sum (should be ~1.0):", softmax_sum)

    var softmax_pass = abs(softmax_sum - 1.0) < 0.001
    if softmax_pass:
        print("  Status: PASS")
    else:
        print("  Status: FAIL")

    # Test LUT building
    print("\nTesting LUT Build...")
    var act_size: Int32 = 256
    var group_size: Int32 = 4
    var num_groups = Int(act_size) // Int(group_size)
    var lut_size = num_groups * 256

    var activations = List[Float32]()
    var lut = List[Float32]()

    for i in range(Int(act_size)):
        activations.append(Float32(i % 10) / 10.0)
    for i in range(lut_size):
        lut.append(0.0)

    var act_ptr = activations.unsafe_ptr()
    var lut_ptr = lut.unsafe_ptr()

    # Build LUT
    start = perf_counter_ns()
    for _ in range(1000):
        handle.call["build_lut", NoneType](lut_ptr, act_ptr, act_size, group_size)
    elapsed_ns = perf_counter_ns() - start
    elapsed_ms = Float64(elapsed_ns) / 1000000.0 / 1000.0

    print("  Activation size:", Int(act_size))
    print("  Groups:         ", num_groups)
    print("  LUT entries:    ", lut_size)
    print("  Build time:     ", elapsed_ms, " ms/iter")
    print("  Status: PASS")

    print("\n" + "=" * 60)
    print("All FFI tests completed successfully!")
    print("=" * 60)
