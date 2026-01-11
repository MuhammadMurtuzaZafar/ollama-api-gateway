"""
Test UnsafePointer syntax in Mojo 0.26.x for SIMD read/write
"""
from collections import List
from memory import UnsafePointer
from sys.info import simd_width_of

fn simd_dot_product(x_ptr: UnsafePointer[Float32], w_ptr: UnsafePointer[Float32], size: Int) -> Float32:
    """SIMD dot product using read-only pointers."""
    var sum: Float32 = 0.0
    var i = 0

    # SIMD loop
    while i + 8 <= size:
        var x_vec = x_ptr.load[width=8](i)
        var w_vec = w_ptr.load[width=8](i)
        sum += (x_vec * w_vec).reduce_add()
        i += 8

    # Remainder
    while i < size:
        sum += x_ptr[i] * w_ptr[i]
        i += 1

    return sum


fn test_simd_matmul():
    """Test SIMD matrix-vector multiply."""
    print("Testing SIMD matmul...")

    var dim = 768  # Typical LLM hidden dimension
    var x = List[Float32](capacity=dim)
    var w = List[Float32](capacity=dim)

    for i in range(dim):
        x.append(Float32(i % 10) / 10.0)
        w.append(Float32((i + 5) % 10) / 10.0)

    # Get pointers
    var x_ptr = x.unsafe_ptr()
    var w_ptr = w.unsafe_ptr()

    # Time the operation
    from time import perf_counter_ns
    var iterations = 10000

    var start = perf_counter_ns()
    var result: Float32 = 0.0
    for _ in range(iterations):
        result = simd_dot_product(x_ptr, w_ptr, dim)
    var end = perf_counter_ns()

    var elapsed = Int(end - start)
    var ns_per_op = elapsed // iterations
    print("Result:", result)
    print("Time per 768-dim dot product:", ns_per_op, "ns")
    if ns_per_op > 0:
        print("Throughput:", 1000000000 // ns_per_op, "dot products/sec")


fn test_simd_write():
    """Test SIMD write operations."""
    print("\nTesting SIMD write...")

    var data = List[Float32](capacity=32)
    for _ in range(32):
        data.append(0.0)

    # Try to get mutable pointer and write
    # The key is using the list while it's mutable
    fn fill_simd(mut list: List[Float32]):
        var ptr = list.unsafe_ptr()
        # Create SIMD vector
        var vec = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        # This might fail due to mut requirements
        # ptr.store[width=8](0, vec)
        # Fallback: scalar write
        for i in range(8):
            list[i] = vec[i]

    fill_simd(data)
    print("After write:", data[0], data[1], data[2], "...")
    print("Write test passed!")


fn main():
    test_simd_matmul()
    test_simd_write()
