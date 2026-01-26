"""
EdgeLLM CUDA Benchmark - Multiple Model Configurations
Tests INT8 Flash Attention performance on T4 GPU for various model sizes.
"""
from sys.ffi import DLHandle
from memory import UnsafePointer
from collections import List
import time

fn benchmark_attention(
    flash_attn: DLHandle,
    name: String,
    n_heads: Int,
    head_dim: Int,
    max_cache: Int,
    iterations: Int
) raises:
    """Benchmark INT8 Flash Attention for a specific model config."""

    # Initialize
    var ret = flash_attn.call[
        "flash_attention_int8_init",
        Int32,
        Int32, Int32, Int32
    ](Int32(n_heads), Int32(max_cache), Int32(head_dim))

    if ret != 0:
        print(name, "- Init FAILED")
        return

    # Allocate test tensors
    var tensor_size = n_heads * head_dim
    var Q = List[Float32](capacity=tensor_size)
    var K = List[Float32](capacity=tensor_size)
    var V = List[Float32](capacity=tensor_size)
    var O = List[Float32](capacity=tensor_size)

    for i in range(tensor_size):
        Q.append(0.1)
        K.append(0.1)
        V.append(0.1)
        O.append(0.0)

    var q_ptr = Q.unsafe_ptr()
    var k_ptr = K.unsafe_ptr()
    var v_ptr = V.unsafe_ptr()
    var o_ptr = O.unsafe_ptr()

    # Reset cache
    flash_attn.call["flash_attention_int8_reset", NoneType]()

    # Warmup
    for i in range(10):
        _ = flash_attn.call[
            "flash_attention_int8_decode_fp32",
            Int32,
            UnsafePointer[Float32], UnsafePointer[Float32],
            UnsafePointer[Float32], UnsafePointer[Float32],
            Int32, Int32, Int32
        ](q_ptr, k_ptr, v_ptr, o_ptr, Int32(n_heads), Int32(i), Int32(head_dim))

    # Reset and benchmark
    flash_attn.call["flash_attention_int8_reset", NoneType]()

    var start = time.perf_counter_ns()
    for i in range(iterations):
        _ = flash_attn.call[
            "flash_attention_int8_decode_fp32",
            Int32,
            UnsafePointer[Float32], UnsafePointer[Float32],
            UnsafePointer[Float32], UnsafePointer[Float32],
            Int32, Int32, Int32
        ](q_ptr, k_ptr, v_ptr, o_ptr, Int32(n_heads), Int32(i), Int32(head_dim))
    var end = time.perf_counter_ns()

    var elapsed_us = Float64(end - start) / 1000.0
    var avg_us = elapsed_us / Float64(iterations)
    var tok_per_sec = 1e6 / avg_us

    print(name)
    print("  Config: heads=", n_heads, ", head_dim=", head_dim, ", cache=", max_cache)
    print("  Latency:", avg_us, "us/token")
    print("  Throughput:", tok_per_sec, "tok/s")
    print()

    # Cleanup
    flash_attn.call["flash_attention_int8_cleanup", NoneType]()


fn main() raises:
    print()
    print("=" * 70)
    print("EdgeLLM CUDA Benchmark - INT8 Flash Attention (__dp4a Tensor Cores)")
    print("=" * 70)
    print()

    # Load CUDA library
    var flash_attn = DLHandle("./lib/libflash_attention_int8.so")
    print("Loaded INT8 Flash Attention kernel")
    print()

    var iterations = 200
    print("Running", iterations, "iterations per model...")
    print("-" * 70)
    print()

    # SmolLM-135M: 576 hidden, 9 heads, 3 kv_heads, 64 head_dim
    benchmark_attention(flash_attn, "SmolLM-135M", 9, 64, 2048, iterations)

    # Qwen2-0.5B: 896 hidden, 14 heads, 2 kv_heads, 64 head_dim
    benchmark_attention(flash_attn, "Qwen2-0.5B", 14, 64, 4096, iterations)

    # Qwen2-1.5B: 1536 hidden, 12 heads, 2 kv_heads, 128 head_dim
    benchmark_attention(flash_attn, "Qwen2-1.5B", 12, 128, 4096, iterations)

    # LLaMA-1B: 2048 hidden, 32 heads, 8 kv_heads, 64 head_dim
    benchmark_attention(flash_attn, "LLaMA-1B", 32, 64, 8192, iterations)

    # LLaMA-3B: 3200 hidden, 32 heads, 32 kv_heads, 100 head_dim
    benchmark_attention(flash_attn, "LLaMA-3B", 32, 100, 8192, iterations)

    print("=" * 70)
    print("Benchmark Complete - EdgeLLM INT8 Flash Attention on Tesla T4")
    print("=" * 70)
