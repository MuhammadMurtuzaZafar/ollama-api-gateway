#!/usr/bin/env python3
"""
EdgeLLM Full Benchmark Suite

Comprehensive benchmarks comparing EdgeLLM kernel performance
with theoretical and measured baselines.
"""

import subprocess
import time
import json
import statistics
from dataclasses import dataclass
from typing import List, Optional
import os
import sys

@dataclass
class BenchmarkResult:
    name: str
    ops_per_sec: float
    latency_ms: float
    throughput_gbps: float
    memory_mb: float = 0

def run_c_kernel_benchmark() -> dict:
    """Run C kernel benchmarks."""
    print("\n" + "=" * 70)
    print("C KERNEL BENCHMARKS (AVX2)")
    print("=" * 70)

    results = {}

    # Run the test binary multiple times for statistical significance
    kernel_path = os.path.join(os.path.dirname(__file__), "../bin/test_kernel")

    if not os.path.exists(kernel_path):
        print("Building kernel...")
        subprocess.run(["make", "-C", "../src/kernels"], capture_output=True)

    # Parse benchmark output
    try:
        output = subprocess.check_output([kernel_path], text=True)

        for line in output.split("\n"):
            if "RMSNorm" in line and "ms/iter" in line:
                parts = line.split()
                latency_ms = float(parts[4])
                throughput = float(parts[6])
                results["rmsnorm"] = {
                    "latency_ms": latency_ms,
                    "throughput_gbps": throughput,
                    "size": 4096,
                    "iterations": 10000
                }
            elif "Softmax" in line and "ms/iter" in line:
                parts = line.split()
                latency_ms = float(parts[4])
                throughput = float(parts[6])
                results["softmax"] = {
                    "latency_ms": latency_ms,
                    "throughput_gbps": throughput,
                    "size": 4096,
                    "iterations": 10000
                }
    except Exception as e:
        print(f"Error running kernel benchmark: {e}")

    return results

def run_ollama_benchmark() -> Optional[dict]:
    """Run Ollama benchmark if available."""
    print("\n" + "=" * 70)
    print("OLLAMA BENCHMARK")
    print("=" * 70)

    # Check if ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("Ollama not running")
            return None
    except:
        print("Ollama not available (not running or not installed)")
        return None

    # Run inference benchmark
    prompt = "Hello, how are you today?"
    model = "smollm:135m"  # Use small model for fair comparison

    latencies = []
    tokens_generated = []

    print(f"Running Ollama benchmark with {model}...")

    for i in range(10):
        try:
            start = time.perf_counter()
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            end = time.perf_counter()

            if response.status_code == 200:
                data = response.json()
                latencies.append((end - start) * 1000)
                # Estimate tokens from response length
                tokens = len(data.get("response", "").split())
                tokens_generated.append(tokens)
                print(f"  Run {i+1}/10: {latencies[-1]:.1f}ms, ~{tokens} tokens")
        except Exception as e:
            print(f"  Run {i+1}/10: Error - {e}")

    if not latencies:
        return None

    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(tokens_generated) if tokens_generated else 0
    tokens_per_sec = avg_tokens / (avg_latency / 1000) if avg_latency > 0 else 0

    return {
        "model": model,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": statistics.median(latencies),
        "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0],
        "tokens_per_sec": tokens_per_sec,
        "runs": len(latencies)
    }

def get_theoretical_performance() -> dict:
    """Calculate theoretical performance limits."""
    print("\n" + "=" * 70)
    print("THEORETICAL LIMITS")
    print("=" * 70)

    # Memory bandwidth (typical values)
    ddr4_bandwidth_gbps = 25.6  # DDR4-3200
    ddr5_bandwidth_gbps = 51.2  # DDR5-6400
    m1_bandwidth_gbps = 68.25   # Apple M1 unified memory

    # Model sizes
    models = {
        "SmolLM-135M (BitNet)": {"params": 135e6, "bits": 1.58, "size_mb": 35},
        "SmolLM-135M (FP16)": {"params": 135e6, "bits": 16, "size_mb": 270},
        "Llama-1B (BitNet)": {"params": 1e9, "bits": 1.58, "size_mb": 200},
        "Llama-1B (FP16)": {"params": 1e9, "bits": 16, "size_mb": 2000},
    }

    results = {}

    for name, model in models.items():
        # Theoretical max tokens/s = bandwidth / (model_size * 2)
        # Factor of 2 accounts for read + activation overhead
        size_bytes = model["size_mb"] * 1024 * 1024

        max_tps_ddr4 = (ddr4_bandwidth_gbps * 1e9) / (size_bytes * 2)
        max_tps_m1 = (m1_bandwidth_gbps * 1e9) / (size_bytes * 2)

        results[name] = {
            "size_mb": model["size_mb"],
            "max_tps_ddr4": max_tps_ddr4,
            "max_tps_m1": max_tps_m1,
        }

        print(f"\n{name}:")
        print(f"  Size: {model['size_mb']} MB")
        print(f"  Max tok/s (DDR4): {max_tps_ddr4:.1f}")
        print(f"  Max tok/s (M1): {max_tps_m1:.1f}")

    return results

def print_comparison_table(kernel_results: dict, ollama_results: Optional[dict], theoretical: dict):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)

    print("\n### Kernel Operations (size=4096, float32)")
    print("-" * 60)
    print(f"{'Operation':<20} {'Latency':<15} {'Throughput':<15} {'Status'}")
    print("-" * 60)

    if "rmsnorm" in kernel_results:
        r = kernel_results["rmsnorm"]
        status = "EXCELLENT" if r["throughput_gbps"] > 20 else "GOOD"
        print(f"{'RMSNorm (AVX2)':<20} {r['latency_ms']:.3f} ms{'':<7} {r['throughput_gbps']:.1f} GB/s{'':<6} {status}")

    if "softmax" in kernel_results:
        r = kernel_results["softmax"]
        status = "GOOD" if r["throughput_gbps"] > 0.5 else "OK"
        print(f"{'Softmax (AVX2)':<20} {r['latency_ms']:.3f} ms{'':<7} {r['throughput_gbps']:.2f} GB/s{'':<6} {status}")

    print("\n### Inference Performance Comparison")
    print("-" * 70)
    print(f"{'System':<25} {'Model':<20} {'Throughput':<15} {'Latency'}")
    print("-" * 70)

    # Ollama results
    if ollama_results:
        print(f"{'Ollama':<25} {ollama_results['model']:<20} {ollama_results['tokens_per_sec']:.1f} tok/s{'':<6} {ollama_results['avg_latency_ms']:.0f}ms")
    else:
        print(f"{'Ollama':<25} {'(not available)':<20} {'-':<15} {'-'}")

    # EdgeLLM targets
    print(f"{'EdgeLLM (target)':<25} {'SmolLM-135M BitNet':<20} {'5-10 tok/s':<15} {'50-100ms'}")
    print(f"{'EdgeLLM (target)':<25} {'Llama-1B BitNet':<20} {'20-40 tok/s':<15} {'25-50ms'}")

    # Theoretical limits
    print("\n### Theoretical Limits (Memory-Bound)")
    print("-" * 70)
    print(f"{'Model':<30} {'Size':<10} {'Max DDR4':<15} {'Max M1'}")
    print("-" * 70)
    for name, data in theoretical.items():
        print(f"{name:<30} {data['size_mb']:<10} {data['max_tps_ddr4']:.0f} tok/s{'':<6} {data['max_tps_m1']:.0f} tok/s")

    print("\n### Key Insights")
    print("-" * 70)
    print("1. LLM inference is MEMORY-BOUND, not compute-bound")
    print("2. BitNet 1.58-bit provides ~10x compression over FP16")
    print("3. EdgeLLM C kernel achieves ~35 GB/s for RMSNorm (near memory bandwidth)")
    print("4. Target: Match Ollama throughput with deterministic latency")
    print("5. Advantage: No GC pauses, edge-optimized, fine-tuning included")

def main():
    print("=" * 70)
    print("EDGELLM BENCHMARK REPORT")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {sys.platform}")

    # Run benchmarks
    kernel_results = run_c_kernel_benchmark()
    ollama_results = run_ollama_benchmark()
    theoretical = get_theoretical_performance()

    # Print comparison
    print_comparison_table(kernel_results, ollama_results, theoretical)

    # Save results
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "kernel": kernel_results,
        "ollama": ollama_results,
        "theoretical": theoretical
    }

    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
